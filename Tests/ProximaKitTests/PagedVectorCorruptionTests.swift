// PagedVectorCorruptionTests.swift
// ProximaKitTests
//
// ADR-013 Stage 2 corruption matrix for the padded v3 vector section and the
// paged (`mmap`) reader (ADR-010 rule 5: a corruption test per format change).
// Every malformed input must surface a TYPED `PersistenceError` — never a trap,
// the same standard `PersistenceCorruptionTests` enforces for the base format.
//
// Covers:
//   • a section-table vector offset that is not 16 KiB-aligned  → typed error
//   • a section-table vector offset past EOF                    → typed error
//   • a paged open of an UNPADDED v3 base (Stage-1 shape)       → typed error,
//     while the same file still loads RESIDENT (backward compat)
//   • a paged open of a v2 base (no section table)              → typed error
//   • padded and unpadded v3 load resident to the SAME state

import XCTest
@testable import ProximaKit

final class PagedVectorCorruptionTests: XCTestCase {

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-corrupt-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0xBEEF &* UInt64(i + 1))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 10_000) / 10_000.0 })
    }
    private func uuid(_ i: Int) -> UUID {
        var b = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                 UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &b) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: b)
    }
    private func seededConfig() -> HNSWConfiguration {
        HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 20,
                          autoCompactionThreshold: nil, levelSeed: 0x1234)
    }

    // Trailer geometry (must mirror PersistenceEngine): fixed 96-byte trailer.
    //   sectionCount:UInt32 @ trailerStart
    //   sections[i]:(offset:UInt64,length:UInt64) @ trailerStart+4 + i*16
    //   vectors are section index 1
    private let trailerSize = 4 + 5 * 16 + 8 + 4
    private func vectorOffsetFieldPosition(fileSize: Int) -> Int {
        let trailerStart = fileSize - trailerSize
        return trailerStart + 4 + 1 * 16   // section index 1, offset field
    }
    private func patchUInt64(_ data: inout Data, at position: Int, to value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.replaceSubrange(position..<position + 8, with: $0) }
    }

    /// Builds a padded v3 base + empty WAL and returns their URLs.
    private func makePaddedBase(dir: URL, dim: Int, count: Int) async throws -> (base: URL, wal: URL) {
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")
        let idx = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<count { try await idx.add(vec(i, dim: dim), id: uuid(i)) }
        try await idx.checkpoint(baseURL: base, walURL: wal)   // padded v3
        await idx.closeJournal()
        return (base, wal)
    }

    private func expectCorrupted(
        _ body: () async throws -> Void, _ what: String,
        file: StaticString = #filePath, line: UInt = #line
    ) async {
        do {
            try await body()
            XCTFail("\(what): expected a typed PersistenceError, got success", file: file, line: line)
        } catch is PersistenceError {
            // Typed error, no trap — exactly the contract.
        } catch {
            XCTFail("\(what): expected PersistenceError, got \(error)", file: file, line: line)
        }
    }

    // ── 1. Non-16 KiB-aligned claimed vector offset → typed error ─────

    func testNonAlignedVectorOffsetRejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let (base, wal) = try await makePaddedBase(dir: dir, dim: 16, count: 120)

        var data = try Data(contentsOf: base)
        let pos = vectorOffsetFieldPosition(fileSize: data.count)
        let aligned = data.loadLE(UInt64.self, at: pos)
        XCTAssertEqual(aligned % UInt64(MappedVectorRegion.requiredAlignment), 0,
                       "sanity: the writer must have page-aligned the vector section")
        // Shift by 8 bytes: still in-bounds (node levels follow the vectors), but
        // no longer page-aligned, so it cannot be mapped.
        patchUInt64(&data, at: pos, to: aligned + 8)
        try data.write(to: base)

        await expectCorrupted({
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        }, "non-aligned vector offset")
    }

    // ── 2. Section-table vector offset past EOF → typed error ─────────

    func testVectorOffsetPastEOFRejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let (base, wal) = try await makePaddedBase(dir: dir, dim: 16, count: 120)

        var data = try Data(contentsOf: base)
        let pos = vectorOffsetFieldPosition(fileSize: data.count)
        patchUInt64(&data, at: pos, to: UInt64(data.count) + 1_000_000)
        try data.write(to: base)

        await expectCorrupted({
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        }, "vector offset past EOF (paged)")
        // The resident loader consults the same table for its padding jump, so it
        // must reject the same corruption rather than trap.
        await expectCorrupted({
            _ = try HNSWIndex.load(from: base)
        }, "vector offset past EOF (resident)")
    }

    // ── 3. Paged open of an UNPADDED v3 base → typed error, but the ───
    //      same file loads RESIDENT (Stage-1 backward compatibility)

    func testUnpaddedV3PagedRejectedButResidentLoads() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("unpadded.pxkt")
        let wal = dir.appendingPathComponent("unpadded.pxwal")
        let dim = 20, count = 100

        // Produce a Stage-1-shaped UNPADDED v3 file via the internal writer.
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<count { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        let snapshot = try await builder.persistenceSnapshot()
        try PersistenceEngine.saveHNSW(snapshot, generation: 7, to: base, padVectorSection: false)

        // Resident load MUST work and reproduce the state (backward compat).
        XCTAssertEqual(try PersistenceEngine.readGeneration(from: base), 7)
        let resident = try HNSWIndex.load(from: base)
        let residentCount = await resident.count
        XCTAssertEqual(residentCount, count)
        let refResults = await resident.search(query: vec(9001, dim: dim), k: 10)
        XCTAssertFalse(refResults.isEmpty)

        // Paged open MUST reject the unpadded base with a typed error (its vector
        // section is not page-aligned) — never a trap.
        await expectCorrupted({
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        }, "paged open of unpadded v3")
    }

    // ── 4. Padded and unpadded v3 load resident to the SAME state ─────

    func testPaddedAndUnpaddedV3LoadIdentically() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let padded = dir.appendingPathComponent("padded.pxkt")
        let unpadded = dir.appendingPathComponent("unpadded.pxkt")
        let dim = 20, count = 150

        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<count { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        let snapshot = try await builder.persistenceSnapshot()
        try PersistenceEngine.saveHNSW(snapshot, generation: 3, to: padded, padVectorSection: true)
        try PersistenceEngine.saveHNSW(snapshot, generation: 3, to: unpadded, padVectorSection: false)

        // The padded file must be strictly larger (it carries page padding) but
        // decode to the same searchable state.
        let paddedSize = try Data(contentsOf: padded).count
        let unpaddedSize = try Data(contentsOf: unpadded).count
        XCTAssertGreaterThan(paddedSize, unpaddedSize, "padding must add bytes")

        let a = try HNSWIndex.load(from: padded)
        let b = try HNSWIndex.load(from: unpadded)
        for q in 0..<15 {
            let ra = await a.search(query: vec(9100 + q, dim: dim), k: 10)
            let rb = await b.search(query: vec(9100 + q, dim: dim), k: 10)
            XCTAssertEqual(ra.map(\.id), rb.map(\.id), "query \(q) ids")
            XCTAssertEqual(ra.map { $0.distance.bitPattern }, rb.map { $0.distance.bitPattern },
                           "query \(q) distances")
        }
    }

    // ── 5. Paged open of a v2 base (no section table) → typed error ───

    func testPagedOpenOfV2Rejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("legacy.pxkt")
        let wal = dir.appendingPathComponent("legacy.pxwal")
        let dim = 16
        let idx = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<80 { try await idx.add(vec(i, dim: dim), id: uuid(i)) }
        try await idx.save(to: base)   // legacy v2, no trailer

        await expectCorrupted({
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        }, "paged open of v2 base")
    }
}
