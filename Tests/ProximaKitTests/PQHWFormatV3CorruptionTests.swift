// PQHWFormatV3CorruptionTests.swift
// ProximaKitTests
//
// ADR-014 Stage 1 — corruption matrix for the PQHW v3 trailer (ADR-010 rule 5:
// every new on-disk surface gets a typed-error test; recovery never traps).
// Covers: truncated trailer, bad trailer magic, wrong section count, section
// offset/length past EOF, broken contiguity (overlap / non-monotonic),
// flag/entry mismatch (both directions), wrong originals length, and the
// resident-load tolerance of an unaligned (unpadded) originals offset.
//
// The Stage-2 paged-open rejections (unaligned offset → paged error; flag-0 v3
// → "nothing to page" error) are deferred to Stage 2 with the paged read path.

import XCTest
@testable import ProximaKit

final class PQHWFormatV3CorruptionTests: XCTestCase {

    private let trailerSize = 4 + 7 * 16 + 8 + 4   // 128
    private func tempURL(_ ext: String = "qhnsw") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("pqhwv3c-\(UUID().uuidString)").appendingPathExtension(ext)
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    private func patch32(_ data: inout Data, at pos: Int, _ value: UInt32) {
        withUnsafeBytes(of: value.littleEndian) { data.replaceSubrange(pos..<pos + 4, with: $0) }
    }
    private func patch64(_ data: inout Data, at pos: Int, _ value: UInt64) {
        withUnsafeBytes(of: value.littleEndian) { data.replaceSubrange(pos..<pos + 8, with: $0) }
    }
    private func sectionOffsetPos(_ size: Int, _ i: Int) -> Int { size - trailerSize + 4 + i * 16 }
    private func sectionLengthPos(_ size: Int, _ i: Int) -> Int { sectionOffsetPos(size, i) + 8 }

    private func assertThrows(_ url: URL, _ what: String,
                              file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url), what, file: file, line: line) { e in
            XCTAssertTrue(e is PersistenceError, "\(what): expected PersistenceError, got \(e)",
                          file: file, line: line)
        }
    }

    private func makeTinyRetainedIndex() -> QuantizedHNSWIndex {
        makeTinyRetainedIndexWithIDs().index
    }

    private func makeTinyRetainedIndexWithIDs() -> (index: QuantizedHNSWIndex, ids: [UUID]) {
        let config = PQConfiguration(subspaceCount: 2)
        let codebook = [Float](repeating: 0.5, count: 256 * 2)
        let quantizer = ProductQuantizer(dimension: 4, config: config, codebooks: [codebook, codebook])
        let ids = [UUID(), UUID()]
        let index = QuantizedHNSWIndex(
            dimension: 4, hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            quantizer: quantizer, layers: [[[1], [0]]], nodeLevels: [0, 0],
            entryPointNode: 0, maxLevel: 0, codes: [[1, 2], [3, 4]],
            nodeToUUID: ids, uuidToNode: [ids[0]: 0, ids[1]: 1], metadata: [nil, nil],
            originals: [Vector([1, 2, 3, 4]), Vector([5, 6, 7, 8])])
        return (index, ids)
    }

    /// A padded v3, flag-1 file on disk.
    private func writePaddedV3(_ url: URL) async throws {
        try await makeTinyRetainedIndex().save(to: url, layout: .pagedV3)
    }

    // ── (1) truncated trailer ────────────────────────────────────────
    func testTruncatedTrailerThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        data = data.prefix(data.count - 10)   // cut into the trailer
        try data.write(to: url)
        assertThrows(url, "truncated trailer")
    }

    // ── (2) bad trailer magic ────────────────────────────────────────
    func testBadTrailerMagicThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        patch32(&data, at: data.count - 4, 0xDEAD_BEEF)
        try data.write(to: url)
        assertThrows(url, "bad trailer magic")
    }

    // ── (3) wrong section count ──────────────────────────────────────
    func testWrongSectionCountThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        patch32(&data, at: data.count - trailerSize, 5)   // must be 7
        try data.write(to: url)
        assertThrows(url, "sectionCount != 7")
    }

    // ── (4a) section offset past EOF ─────────────────────────────────
    func testSectionOffsetPastEOFThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        patch64(&data, at: sectionOffsetPos(data.count, 6), UInt64(data.count) + 1_000_000)
        try data.write(to: url)
        assertThrows(url, "originals offset past EOF")
    }

    // ── (4b) length overflow ─────────────────────────────────────────
    func testSectionLengthOverflowThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        patch64(&data, at: sectionLengthPos(data.count, 6), .max)   // offset+length overflows
        try data.write(to: url)
        assertThrows(url, "originals length overflow")
    }

    // ── (4c) broken contiguity (overlap / non-monotonic) ─────────────
    func testNonContiguousSectionsThrow() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        // Push the codes section (index 1) offset off the codebooks end.
        patch64(&data, at: sectionOffsetPos(data.count, 1), 999)
        try data.write(to: url)
        assertThrows(url, "non-contiguous section table")
    }

    // ── (6a) flag 1 but empty originals entry ────────────────────────
    func testFlag1WithEmptyOriginalsEntryThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        patch64(&data, at: sectionOffsetPos(data.count, 6), 0)
        patch64(&data, at: sectionLengthPos(data.count, 6), 0)
        try data.write(to: url)
        assertThrows(url, "flag 1 but empty originals entry")
    }

    func testEmptyOriginalsEntryWithNonzeroOffsetThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        let (idx, ids) = makeTinyRetainedIndexWithIDs()
        for id in ids {
            let removed = await idx.remove(id: id)
            XCTAssertTrue(removed, "fixture removes every live node")
        }
        try await idx.save(to: url, layout: .pagedV3)
        var data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3, "fixture must be v3")
        XCTAssertEqual(data.loadLE(UInt32.self, at: 48), 1, "fixture must be retaining")
        XCTAssertEqual(data.loadLE(UInt64.self, at: sectionOffsetPos(data.count, 6)), 0)
        XCTAssertEqual(data.loadLE(UInt64.self, at: sectionLengthPos(data.count, 6)), 0)

        let bodyEnd = UInt64(data.count - trailerSize)
        XCTAssertGreaterThan(bodyEnd, 0)
        patch64(&data, at: sectionOffsetPos(data.count, 6), bodyEnd)
        try data.write(to: url)

        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url),
                             "zero-length originals at nonzero offset") { error in
            guard case PersistenceError.corruptedData(let detail) = error else {
                return XCTFail("expected PersistenceError.corruptedData, got \(error)")
            }
            XCTAssertTrue(detail.contains("empty originals section must have offset 0"),
                          "unexpected error detail: \(detail)")
        }
    }

    // ── (6b) flag 0 but nonzero originals entry ──────────────────────
    func testFlag0WithNonzeroOriginalsEntryThrows() async throws {
        // A flag-0 v3 comes from upgrading a non-retained v2 base.
        let url = tempURL(); defer { cleanup(url) }
        let config = PQConfiguration(subspaceCount: 2)
        let codebook = [Float](repeating: 0.5, count: 256 * 2)
        let quantizer = ProductQuantizer(dimension: 4, config: config, codebooks: [codebook, codebook])
        let ids = [UUID(), UUID()]
        let bare = QuantizedHNSWIndex(
            dimension: 4, hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            quantizer: quantizer, layers: [[[1], [0]]], nodeLevels: [0, 0],
            entryPointNode: 0, maxLevel: 0, codes: [[1, 2], [3, 4]],
            nodeToUUID: ids, uuidToNode: [ids[0]: 0, ids[1]: 1], metadata: [nil, nil], originals: nil)
        try await bare.save(to: url)                       // v2, flag 0
        try QuantizedHNSWIndex.upgradeToV3(at: url)         // flag-0 v3, originals (0,0)
        var data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3, "upgrade produced v3")
        // Point the originals entry at real bytes even though the flag is 0.
        patch64(&data, at: sectionOffsetPos(data.count, 6), 56)
        patch64(&data, at: sectionLengthPos(data.count, 6), 32)
        try data.write(to: url)
        assertThrows(url, "flag 0 but nonzero originals entry")
    }

    // ── (7) wrong originals length ───────────────────────────────────
    func testWrongOriginalsLengthThrows() async throws {
        let url = tempURL(); defer { cleanup(url) }
        try await writePaddedV3(url)
        var data = try Data(contentsOf: url)
        // Keep it in-bounds (offset unchanged) but wrong length (16 != 32).
        patch64(&data, at: sectionLengthPos(data.count, 6), 16)
        try data.write(to: url)
        assertThrows(url, "originals length != count*dim*4")
    }

    // ── (5) unaligned (unpadded) originals offset loads resident ─────
    func testUnpaddedOriginalsOffsetLoadsResident() async throws {
        // The Stage-1 resident path tolerates a non-16-KiB originals offset;
        // only the (future) paged path requires alignment.
        let url = tempURL(); defer { cleanup(url) }
        try await makeTinyRetainedIndex().encodedV3(padOriginals: false).write(to: url)
        let loaded = try QuantizedHNSWIndex.load(from: url)
        let retains = await loaded.retainsOriginals
        let cnt = await loaded.count
        XCTAssertTrue(retains)
        XCTAssertEqual(cnt, 2)
    }
}
