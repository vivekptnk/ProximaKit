// WALRecoveryTests.swift
// ProximaKitTests
//
// Stage-1 WAL (ADR-013) core recovery matrix — runs in CI:
//   • PXWL framing round-trip + CRC.
//   • Deterministic replay reproduces the EXACT state that wrote the log.
//   • Checkpoint folds the WAL into a fresh v3 base and truncates the WAL.
//   • Stale-generation rejection (typed error).
//   • CRC bit-flip detection (mid-stream) → prefix recovery, no crash.
//   • `.pxkt` v3 generation trailer round-trip + trailer corruption.
//   • Additive-API guarantee: save(to:)/load(from:) still write/read v2.
//
// Determinism: fixtures use HNSWConfiguration.levelSeed and the journaled
// level, so replay is byte-exact. No system RNG.

import XCTest
@testable import ProximaKit

final class WALRecoveryTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wal-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    /// Deterministic vector for index `i` (seeded, no RNG).
    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0xA11CE &+ UInt64(i))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 1000) / 1000.0 })
    }

    private func seededConfig() -> HNSWConfiguration {
        HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 20,
                          autoCompactionThreshold: nil, levelSeed: 0x5EED)
    }

    // ── 1. PXWL framing round-trip + CRC ──────────────────────────────

    func testRecordFramingRoundTrip() throws {
        let id = UUID()
        let add = WALRecord.add(id: id, level: 3, vector: [1, 2, 3, 4], metadata: Data([9, 8, 7]))
        let remove = WALRecord.remove(id: id)

        var image = WALFormat.encodeHeader(parentGeneration: 7, dimension: 4, metricRaw: 1)
        image.append(WALFormat.encodeRecord(add))
        image.append(WALFormat.encodeRecord(remove))

        let replay = try WALDecoder.decode(image, expectedGeneration: 7)
        XCTAssertEqual(replay.parentGeneration, 7)
        XCTAssertEqual(replay.dimension, 4)
        XCTAssertEqual(replay.trailingBytesDropped, 0)
        XCTAssertEqual(replay.records, [add, remove])
    }

    func testHeaderCRCMismatchThrows() throws {
        var image = WALFormat.encodeHeader(parentGeneration: 1, dimension: 4, metricRaw: 1)
        image.append(WALFormat.encodeRecord(.remove(id: UUID())))
        image[18] ^= 0xFF   // flip a header byte inside the CRC-covered region
        XCTAssertThrowsError(try WALDecoder.decode(image, expectedGeneration: nil)) { error in
            guard case PersistenceError.corruptedData = error else {
                return XCTFail("expected corruptedData, got \(error)")
            }
        }
    }

    func testBadMagicAndVersionThrow() throws {
        var image = WALFormat.encodeHeader(parentGeneration: 1, dimension: 4, metricRaw: 1)
        var badMagic = image
        badMagic[0] ^= 0xFF
        XCTAssertThrowsError(try WALDecoder.decode(badMagic, expectedGeneration: nil))

        // Bump version to an unsupported value (fix nothing else — magic ok).
        image.replaceSubrange(4..<8, with: withUnsafeBytes(of: UInt32(99).littleEndian) { Data($0) })
        XCTAssertThrowsError(try WALDecoder.decode(image, expectedGeneration: nil)) { error in
            guard case PersistenceError.unsupportedVersion(99) = error else {
                return XCTFail("expected unsupportedVersion(99), got \(error)")
            }
        }
    }

    // ── 2. Deterministic replay reproduces EXACT state ────────────────

    func testReplayReproducesExactState() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        // Producer: seeded, journaled. Checkpoint an empty base, then ingest.
        let producer = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        try await producer.checkpoint(baseURL: base, walURL: wal)
        for i in 0..<40 {
            try await producer.add(vec(i, dim: 8), id: uuid(i))
        }
        try await producer.syncJournal()
        let producerFP = await producer.structuralFingerprint
        await producer.closeJournal()

        // Recover a fresh index from base + WAL.
        let recovered = try await HNSWIndex.open(baseURL: base, walURL: wal)
        let recoveredFP = await recovered.structuralFingerprint

        XCTAssertEqual(recoveredFP, producerFP,
                       "clean WAL replay must reproduce the EXACT producing state")

        // Search parity on a held-out query.
        let q = vec(999, dim: 8)
        let a = await producer.search(query: q, k: 10)
        let b = await recovered.search(query: q, k: 10)
        XCTAssertEqual(a.map(\.id), b.map(\.id))
        XCTAssertEqual(a.map(\.distance), b.map(\.distance))
        await recovered.closeJournal()
    }

    // ── 2b. Replay reproduces EXACT state INCLUDING live tombstone slots ──
    //
    // `testReplayReproducesExactState` above journals adds only. ADR-013's
    // Stage-1 addendum claims replay reproduces the byte-exact producing state
    // "including tombstones". This exercises that: a seeded index journals a mix
    // of adds and removes, leaving live tombstone slots (liveCount < count) that
    // survive into recovery. Auto-compaction cannot erase them — `seededConfig`
    // sets no threshold, and a journaled index defers compaction to checkpoint
    // regardless (ADR-013 deviation 3). Recovery must reproduce the same
    // `structuralFingerprint`, tombstone slots and all.
    func testReplayReproducesExactStateWithTombstones() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        // Producer: seeded, journaled. Empty base, then a mix of adds + removes.
        let producer = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        try await producer.checkpoint(baseURL: base, walURL: wal)
        for i in 0..<40 { try await producer.add(vec(i, dim: 8), id: uuid(i)) }
        // Remove a distinct subset (no re-adds), journaled as `remove` records,
        // so each slot stays a tombstone: `count` unchanged, `liveCount` drops.
        let removed = [5, 11, 17, 23, 29, 35]
        for i in removed {
            let didRemove = await producer.remove(id: uuid(i))
            XCTAssertTrue(didRemove, "seeded id \(i) must be live before removal")
        }
        try await producer.syncJournal()

        let producerFP = await producer.structuralFingerprint
        let producerCount = await producer.count
        let producerLive = await producer.liveCount
        await producer.closeJournal()

        // The fingerprint under test must actually carry tombstones — otherwise
        // exact equality proves nothing about the tombstone component.
        let tombstones = producerCount - producerLive
        XCTAssertEqual(producerCount, 40)
        XCTAssertEqual(producerLive, 34)
        XCTAssertEqual(tombstones, removed.count,
                       "producing state must hold a non-trivial set of live tombstone slots")

        // Recover from base + WAL and compare the FULL fingerprint — adjacency,
        // levels, entry point, vectors, metadata, AND the tombstone slots.
        let recovered = try await HNSWIndex.open(baseURL: base, walURL: wal)
        let recoveredFP = await recovered.structuralFingerprint
        let recoveredCount = await recovered.count
        let recoveredLive = await recovered.liveCount

        XCTAssertEqual(recoveredFP, producerFP,
                       "WAL replay must reproduce the EXACT producing state, tombstones included")
        XCTAssertEqual(recoveredCount - recoveredLive, tombstones,
                       "recovery must reproduce the exact live tombstone-slot count")

        // Search parity on a held-out query (mirrors testReplayReproducesExactState).
        let q = vec(999, dim: 8)
        let a = await producer.search(query: q, k: 10)
        let b = await recovered.search(query: q, k: 10)
        XCTAssertEqual(a.map(\.id), b.map(\.id))
        XCTAssertEqual(a.map(\.distance), b.map(\.distance))
        await recovered.closeJournal()
    }

    // ── 3. Checkpoint folds WAL into base and truncates it ────────────

    func testCheckpointFoldsAndTruncatesWAL() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let idx = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        try await idx.checkpoint(baseURL: base, walURL: wal)   // gen 1
        let gen1 = await idx.currentGeneration
        XCTAssertEqual(gen1, 1)
        for i in 0..<30 { try await idx.add(vec(i, dim: 8), id: uuid(i)) }
        let recCount = await idx.journalRecordCount
        XCTAssertEqual(recCount, 30)

        let beforeFP = await idx.structuralFingerprint
        try await idx.checkpoint(baseURL: base, walURL: wal)   // gen 2, WAL reset
        let gen2 = await idx.currentGeneration
        let recCountAfter = await idx.journalRecordCount
        let fpAfter = await idx.structuralFingerprint
        XCTAssertEqual(gen2, 2)
        XCTAssertEqual(recCountAfter, 0, "checkpoint must truncate the WAL")
        XCTAssertEqual(fpAfter, beforeFP,
                       "checkpoint (no tombstones) must not change graph state")
        await idx.closeJournal()

        // Base now carries generation 2 and all 30 vectors; empty WAL replays clean.
        XCTAssertEqual(try PersistenceEngine.readGeneration(from: base), 2)
        let reopened = try await HNSWIndex.open(baseURL: base, walURL: wal)
        let reopenedCount = await reopened.liveCount
        let reopenedFP = await reopened.structuralFingerprint
        XCTAssertEqual(reopenedCount, 30)
        XCTAssertEqual(reopenedFP, beforeFP)
        await reopened.closeJournal()
    }

    // ── 4. Stale-generation rejection (typed error) ───────────────────

    func testStaleGenerationRejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let idx = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        try await idx.checkpoint(baseURL: base, walURL: wal)   // gen 1
        for i in 0..<10 { try await idx.add(vec(i, dim: 8), id: uuid(i)) }
        try await idx.syncJournal()
        let staleWAL = try Data(contentsOf: wal)               // a valid gen-1 WAL

        try await idx.checkpoint(baseURL: base, walURL: wal)   // base -> gen 2
        await idx.closeJournal()
        try staleWAL.write(to: wal)                            // restore the gen-1 WAL

        // base is gen 2, WAL header says gen 1 → typed mismatch, no silent discard.
        do {
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal)
            XCTFail("stale WAL must be rejected")
        } catch let PersistenceError.walGenerationMismatch(expected, found) {
            XCTAssertEqual(expected, 2)
            XCTAssertEqual(found, 1)
        }
    }

    // ── 4b. Crafted sidecar with wrong dimension/metric is rejected ───
    //
    // The decoder validates a WAL's own framing (header CRC, parent generation)
    // but not that the sidecar was written for *this* base index. A CRC-valid,
    // generation-matching WAL whose header claims a different dimension would
    // otherwise feed mismatched-length vectors straight into `insertNode` on
    // replay — past the public `add(_:id:)` dimension guard. `open` cross-checks
    // the header's dimension/metric against the base and rejects a mismatch with
    // a typed error, never a trap.
    func testCraftedWALDimensionOrMetricMismatchRejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        // A valid dim-8 euclidean base (gen 1) with a valid gen-1 WAL beside it.
        let idx = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        try await idx.checkpoint(baseURL: base, walURL: wal)          // gen 1
        for i in 0..<6 { try await idx.add(vec(i, dim: 8), id: uuid(i)) }
        try await idx.syncJournal()
        await idx.closeJournal()

        // Header layout (WALFormat): dimension @16, metricRaw @20, headerCRC @24
        // over bytes 0..<24. Sanity-check the pristine header stamps this base.
        let pristine = try Data(contentsOf: wal)
        XCTAssertEqual(pristine.loadLE(UInt32.self, at: 16), 8)
        XCTAssertEqual(pristine.loadLE(UInt32.self, at: 20), DistanceMetricType.euclidean.rawValue)

        // Byte-patches a CRC-covered header field and repairs the header CRC, so
        // the tampered sidecar stays framing-valid and generation-matching — only
        // its binding to *this* index is wrong.
        func craft(_ source: Data, patchField offset: Int, to value: UInt32) -> Data {
            var data = source
            withUnsafeBytes(of: value.littleEndian) { data.replaceSubrange(offset..<offset + 4, with: $0) }
            let crc = data.prefix(24).withUnsafeBytes { CRC32.checksum($0.bindMemory(to: UInt8.self)) }
            withUnsafeBytes(of: crc.littleEndian) { data.replaceSubrange(24..<28, with: $0) }
            return data
        }

        // (a) Wrong dimension (8 → 4): rejected before any record is replayed.
        try craft(pristine, patchField: 16, to: 4).write(to: wal)
        do {
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal)
            XCTFail("a WAL whose header dimension ≠ base must be rejected")
        } catch let PersistenceError.walDimensionMismatch(expected, found) {
            XCTAssertEqual(expected, 8)
            XCTAssertEqual(found, 4)
        }

        // (b) Wrong metric (euclidean → cosine), right dimension: also rejected.
        try craft(pristine, patchField: 20, to: DistanceMetricType.cosine.rawValue).write(to: wal)
        do {
            _ = try await HNSWIndex.open(baseURL: base, walURL: wal)
            XCTFail("a WAL whose header metric ≠ base must be rejected")
        } catch let PersistenceError.walMetricMismatch(expected, found) {
            XCTAssertEqual(expected, DistanceMetricType.euclidean.rawValue)
            XCTAssertEqual(found, DistanceMetricType.cosine.rawValue)
        }
    }

    // ── 5. CRC bit-flip mid-stream → prefix recovery, no crash ────────

    func testMidStreamBitFlipRecoversPrefix() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")
        let dim = 4

        let idx = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        try await idx.checkpoint(baseURL: base, walURL: wal)
        for i in 0..<6 { try await idx.add(vec(i, dim: dim), id: uuid(i)) }
        try await idx.syncJournal()
        await idx.closeJournal()

        // Frame layout (no metadata): payload = 1 + 16 + 4 + dim*4 + 4.
        let payload = 1 + 16 + 4 + dim * 4 + 4
        let frame = 8 + payload
        // Corrupt a payload byte of the record at index 3 → decoder stops there,
        // recovering records 0,1,2 (a 3-vector prefix).
        var data = try Data(contentsOf: wal)
        let flipAt = WALFormat.headerSize + 3 * frame + 8 + 10
        data[flipAt] ^= 0xFF
        try data.write(to: wal)

        let recovered = try await HNSWIndex.open(baseURL: base, walURL: wal)
        let recoveredCount = await recovered.liveCount
        let consistent = await recovered.reverseAdjacencyIsConsistent
        XCTAssertEqual(recoveredCount, 3,
                       "a mid-stream CRC failure truncates to the valid prefix")
        XCTAssertTrue(consistent)
        await recovered.closeJournal()
    }

    // ── 6. v3 generation trailer round-trip + trailer corruption ──────

    func testV3GenerationTrailerRoundTrip() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let idx = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<12 { try await idx.add(vec(i, dim: 8), id: uuid(i)) }
        try await idx.checkpoint(baseURL: base, walURL: wal)     // writes v3 base, gen 1
        await idx.closeJournal()

        XCTAssertEqual(try PersistenceEngine.readGeneration(from: base), 1)
        // v3 base still loads via the ordinary resident loader (body == v2).
        let loaded = try HNSWIndex.load(from: base)
        let loadedCount = await loaded.liveCount
        XCTAssertEqual(loadedCount, 12)

        // Corrupt the trailer magic → readGeneration throws typed, never traps.
        var data = try Data(contentsOf: base)
        data[data.count - 1] ^= 0xFF
        try data.write(to: base)
        XCTAssertThrowsError(try PersistenceEngine.readGeneration(from: base)) { error in
            guard case PersistenceError.corruptedData = error else {
                return XCTFail("expected corruptedData, got \(error)")
            }
        }
    }

    // ── 7. Additive API: legacy save/load stays v2 (byte-shape) ────────

    func testLegacySaveStillWritesV2() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("legacy.pxkt")
        let idx = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<10 { try await idx.add(vec(i, dim: 8), id: uuid(i)) }
        try await idx.save(to: url)

        let version = try Data(contentsOf: url).loadLE(UInt32.self, at: 4)
        XCTAssertEqual(version, 2, "legacy save(to:) must keep writing v2, unchanged")
        // v2 has no trailer → generation reads as 0.
        XCTAssertEqual(try PersistenceEngine.readGeneration(from: url), 0)
        let reloaded = try HNSWIndex.load(from: url)
        let reloadedCount = await reloaded.liveCount
        XCTAssertEqual(reloadedCount, 10)
    }

    // Deterministic UUIDs so producer/recovered use the same ids.
    private func uuid(_ i: Int) -> UUID {
        var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                     UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &bytes) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: bytes)
    }
}
