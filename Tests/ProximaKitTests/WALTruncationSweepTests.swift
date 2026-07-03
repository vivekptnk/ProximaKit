// WALTruncationSweepTests.swift
// ProximaKitTests
//
// ADR-013 acceptance 1 (in-process, deterministic): exhaustively truncate a
// WAL at every record boundary across the whole log AND at every byte across
// the final record, and assert at each cut:
//   • loading never crashes and never throws (a torn tail is expected, not a
//     corrupt-file error);
//   • the recovered state == the base plus the LONGEST valid record prefix,
//     asserted by exact structural-fingerprint equality (not mere validity).
//
// The fixture is seeded (levelSeed + journaled levels) so every prefix has a
// single, reproducible recovered state.

import XCTest
@testable import ProximaKit

final class WALTruncationSweepTests: XCTestCase {

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("waltrunc-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0xBEEF &+ UInt64(i))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 1000) / 1000.0 })
    }
    private func uuid(_ i: Int) -> UUID {
        var b = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                 UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &b) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: b)
    }

    func testTruncationAtEveryBoundaryRecoversExactPrefix() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let walFull = dir.appendingPathComponent("full.pxwal")
        let dim = 4
        let recordCount = 12
        let config = HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 20,
                                       autoCompactionThreshold: nil, levelSeed: 0x5EED)

        // ── Build the full WAL over a checkpointed base ────────────────
        let producer = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
        try await producer.checkpoint(baseURL: base, walURL: walFull)
        for i in 0..<recordCount { try await producer.add(vec(i, dim: dim), id: uuid(i)) }
        try await producer.syncJournal()
        await producer.closeJournal()

        let fullBytes = try Data(contentsOf: walFull)
        // Framing is fixed (no metadata): payload = opcode + uuid + level + vector + metaLen.
        let payload = 1 + 16 + 4 + dim * 4 + 4
        let frame = 8 + payload
        let header = WALFormat.headerSize
        XCTAssertEqual(fullBytes.count, header + recordCount * frame,
                       "fixed-width framing assumption for the sweep")

        // Reference fingerprint for each valid prefix length (0...recordCount),
        // computed once by opening the base + a boundary-truncated WAL.
        func openFingerprint(prefixBytes: Int) async throws -> HNSWIndex.StructuralFingerprint {
            let w = dir.appendingPathComponent("ref-\(prefixBytes).pxwal")
            try fullBytes.prefix(prefixBytes).write(to: w)
            let idx = try await HNSWIndex.open(baseURL: base, walURL: w)
            let fp = await idx.structuralFingerprint
            await idx.closeJournal()
            cleanup(w)
            return fp
        }
        var reference: [HNSWIndex.StructuralFingerprint] = []
        for expected in 0...recordCount {
            reference.append(try await openFingerprint(prefixBytes: header + expected * frame))
        }

        // ── The sweep: every record boundary + every byte of the final record ──
        var cuts = Set<Int>()
        for i in 0...recordCount { cuts.insert(header + i * frame) }        // record boundaries
        for len in (fullBytes.count - frame)...fullBytes.count { cuts.insert(len) }  // final record, byte-by-byte

        for len in cuts.sorted() where len >= header && len <= fullBytes.count {
            let expected = min(recordCount, (len - header) / frame)
            let w = dir.appendingPathComponent("cut-\(len).pxwal")
            try fullBytes.prefix(len).write(to: w)

            // Must not crash and must not throw for any in-range cut.
            let recovered = try await HNSWIndex.open(baseURL: base, walURL: w)
            let live = await recovered.liveCount
            let fp = await recovered.structuralFingerprint
            let consistent = await recovered.reverseAdjacencyIsConsistent
            await recovered.closeJournal()
            cleanup(w)

            XCTAssertEqual(live, expected,
                           "cut @ \(len) bytes: recovered \(live) live, expected prefix of \(expected)")
            XCTAssertEqual(fp, reference[expected],
                           "cut @ \(len) bytes: recovered state must EXACTLY equal base + \(expected)-record prefix")
            XCTAssertTrue(consistent, "cut @ \(len) bytes: recovered graph must be internally consistent")
        }
    }

    /// A cut below the WAL header is a typed error, never a crash.
    func testSubHeaderTruncationThrowsTyped() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")
        let config = HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 20,
                                       autoCompactionThreshold: nil, levelSeed: 0x5EED)
        let idx = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
        try await idx.checkpoint(baseURL: base, walURL: wal)
        for i in 0..<4 { try await idx.add(vec(i, dim: 4), id: uuid(i)) }
        try await idx.syncJournal()
        await idx.closeJournal()

        let bytes = try Data(contentsOf: wal)
        let stub = dir.appendingPathComponent("stub.pxwal")
        try bytes.prefix(WALFormat.headerSize - 4).write(to: stub)
        do {
            _ = try await HNSWIndex.open(baseURL: base, walURL: stub)
            XCTFail("sub-header WAL must throw")
        } catch is PersistenceError {
            // Expected typed error (fileTooSmall).
        }
    }
}
