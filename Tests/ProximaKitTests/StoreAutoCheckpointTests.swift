// StoreAutoCheckpointTests.swift
// ProximaKitTests
//
// ADR-015 Stages A+B: store-level automatic checkpointing, HNSW paged load
// mirror, store-level dense residency plumbing, and shared IndexResidency names.

import XCTest
@testable import ProximaKit

private struct AutoCheckpointEmbedder: TextEmbedder {
    let dimension: Int

    func embed(_ text: String) async throws -> Vector {
        var seed: UInt64 = 0xCBF2_9CE4_8422_2325
        for byte in text.utf8 {
            seed ^= UInt64(byte)
            seed &*= 0x0000_0100_0000_01B3
        }
        var rng = SplitMix64(seed: seed)
        return Vector((0..<dimension).map { _ in
            Float(UInt32(truncatingIfNeeded: rng.next()) % 100_000) / 100_000.0
        })
    }
}

private func autoMeta(_ doc: String, _ idx: Int, _ text: String) -> ChunkMetadata {
    ChunkMetadata(documentId: doc, chunkIndex: idx, text: text)
}

private func autoBatch(doc: String, start: Int, count: Int) -> (chunks: [String], metadata: [ChunkMetadata]) {
    let chunks = (start..<(start + count)).map { "\(doc)-chunk-\($0)" }
    return (chunks, chunks.enumerated().map { autoMeta(doc, start + $0.offset, $0.element) })
}

private func autoVector(_ i: Int, dim: Int) -> Vector {
    var rng = SplitMix64(seed: 0xA015_5EED &+ UInt64(i))
    return Vector((0..<dim).map { _ in
        Float(UInt32(truncatingIfNeeded: rng.next()) % 100_000) / 100_000.0
    })
}

private func autoUUID(_ i: Int) -> UUID {
    var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                 UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    withUnsafeMutableBytes(of: &bytes) {
        $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self)
    }
    return UUID(uuid: bytes)
}

final class StoreAutoCheckpointTests: XCTestCase {
    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("StoreAutoCheckpoint-\(UUID().uuidString)")
    }

    override func tearDown() {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        super.tearDown()
    }

    private func vectorStore(
        _ name: String,
        checkpointAutomatically: WALCheckpointPolicy? = nil,
        dense: IndexResidency = .resident
    ) async throws -> VectorStore {
        try await VectorStore.open(
            name: name,
            embedder: AutoCheckpointEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: HNSWConfiguration(
                m: 4,
                efConstruction: 20,
                efSearch: 20,
                autoCompactionThreshold: nil,
                levelSeed: 0xA015
            ),
            checkpointAutomatically: checkpointAutomatically,
            dense: dense
        )
    }

    private func hybridStore(
        _ name: String,
        checkpointAutomatically: WALCheckpointPolicy? = nil,
        dense: IndexResidency = .resident
    ) async throws -> HybridVectorStore {
        try await HybridVectorStore.open(
            name: name,
            embedder: AutoCheckpointEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            hnswConfig: HNSWConfiguration(
                m: 4,
                efConstruction: 20,
                efSearch: 20,
                autoCompactionThreshold: nil,
                levelSeed: 0xA015
            ),
            checkpointAutomatically: checkpointAutomatically,
            dense: dense
        )
    }

    private func assertIdentical(
        _ lhs: [SearchResult],
        _ rhs: [SearchResult],
        _ message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.map(\.id), rhs.map(\.id), "\(message): ids differ", file: file, line: line)
        XCTAssertEqual(
            lhs.map { $0.distance.bitPattern },
            rhs.map { $0.distance.bitPattern },
            "\(message): distances differ",
            file: file,
            line: line
        )
    }

    func testVectorStoreAutomaticCheckpointShrinksWALAfterPolicyTrip() async throws {
        let policy = WALCheckpointPolicy(walBytesFractionOfBase: 1_000, maxOps: 3)
        let store = try await vectorStore("auto-vector", checkpointAutomatically: policy)

        let first = autoBatch(doc: "doc-auto", start: 0, count: 3)
        _ = try await store.addChunks(first.chunks, metadata: first.metadata)
        let bytesBeforeFold = await store.index.journalByteCount
        let recordsBeforeFold = await store.index.journalRecordCount
        XCTAssertEqual(recordsBeforeFold, 3, "precondition: policy has not tripped at exactly maxOps")

        let second = autoBatch(doc: "doc-auto", start: 3, count: 1)
        _ = try await store.addChunks(second.chunks, metadata: second.metadata)

        let bytesAfterFold = await store.index.journalByteCount
        let recordsAfterFold = await store.index.journalRecordCount
        let needsCheckpoint = await store.needsCheckpoint(policy: policy)
        XCTAssertLessThan(bytesAfterFold, bytesBeforeFold, "automatic checkpoint must shrink the WAL")
        XCTAssertEqual(recordsAfterFold, 0, "automatic checkpoint must reset records since last checkpoint")
        XCTAssertFalse(needsCheckpoint)
    }

    func testHybridStoreAutomaticCheckpointShrinksDenseWALAfterPolicyTrip() async throws {
        let policy = WALCheckpointPolicy(walBytesFractionOfBase: 1_000, maxOps: 3)
        let store = try await hybridStore("auto-hybrid", checkpointAutomatically: policy)

        let first = autoBatch(doc: "doc-hybrid", start: 0, count: 3)
        _ = try await store.addChunks(first.chunks, metadata: first.metadata)
        let bytesBeforeFold = await store.dense.journalByteCount
        let recordsBeforeFold = await store.dense.journalRecordCount
        XCTAssertEqual(recordsBeforeFold, 3, "precondition: policy has not tripped at exactly maxOps")

        let second = autoBatch(doc: "doc-hybrid", start: 3, count: 1)
        _ = try await store.addChunks(second.chunks, metadata: second.metadata)

        let bytesAfterFold = await store.dense.journalByteCount
        let recordsAfterFold = await store.dense.journalRecordCount
        let needsCheckpoint = await store.needsCheckpoint(policy: policy)
        XCTAssertLessThan(bytesAfterFold, bytesBeforeFold, "automatic checkpoint must shrink the dense WAL")
        XCTAssertEqual(recordsAfterFold, 0, "automatic checkpoint must reset dense records")
        XCTAssertFalse(needsCheckpoint)
    }

    func testNilAutomaticCheckpointKeepsManualCheckpointBehavior() async throws {
        let policy = WALCheckpointPolicy(walBytesFractionOfBase: 1_000, maxOps: 3)
        let store = try await vectorStore("manual-vector")

        let batch = autoBatch(doc: "doc-manual", start: 0, count: 4)
        _ = try await store.addChunks(batch.chunks, metadata: batch.metadata)
        try await store.save()

        let recordCount = await store.index.journalRecordCount
        let needsCheckpoint = await store.needsCheckpoint(policy: policy)
        XCTAssertEqual(recordCount, 4)
        XCTAssertTrue(needsCheckpoint)
    }

    func testAutomaticCheckpointFoldErrorsSurfaceThroughMutation() async throws {
        let policy = WALCheckpointPolicy(walBytesFractionOfBase: 1_000, maxOps: 1)
        let store = try await vectorStore("auto-error", checkpointAutomatically: policy)

        let docmapURL = tempDir.appendingPathComponent("auto-error/docmap.json")
        try FileManager.default.createDirectory(at: docmapURL, withIntermediateDirectories: false)

        let batch = autoBatch(doc: "doc-error", start: 0, count: 2)
        do {
            _ = try await store.addChunks(batch.chunks, metadata: batch.metadata)
            XCTFail("automatic checkpoint errors must surface through addChunks")
        } catch {
            XCTAssertFalse(String(describing: error).isEmpty)
        }

        // The triggering mutation is already committed; retrying addChunks here
        // would mint fresh UUIDs and double-add the same logical chunks.
        let liveAfterThrow = await store.liveCount
        let hitsAfterThrow = try await store.query("doc-error-chunk-0", k: 10)
        let walRecordsAfterThrow = await store.index.journalRecordCount
        XCTAssertEqual(liveAfterThrow, 2)
        XCTAssertEqual(hitsAfterThrow.count, 2)
        XCTAssertEqual(walRecordsAfterThrow, 0)
        await store.index.closeJournal()

        let reopened = try await vectorStore("auto-error")
        let reopenedLive = await reopened.liveCount
        let reopenedHits = try await reopened.query("doc-error-chunk-0", k: 10)
        let reopenedWALRecords = await reopened.index.journalRecordCount
        XCTAssertEqual(reopenedLive, 2)
        XCTAssertEqual(reopenedHits.count, 2)
        XCTAssertEqual(reopenedWALRecords, 0)
        await reopened.index.closeJournal()
    }

    func testHybridAutomaticCheckpointFoldErrorsSurfaceThroughMutation() async throws {
        let policy = WALCheckpointPolicy(walBytesFractionOfBase: 1_000, maxOps: 1)
        let store = try await hybridStore("auto-hybrid-error", checkpointAutomatically: policy)

        let mapURL = tempDir.appendingPathComponent("auto-hybrid-error/hybrid.json")
        try FileManager.default.createDirectory(at: mapURL, withIntermediateDirectories: false)

        let batch = autoBatch(doc: "doc-hybrid-error", start: 0, count: 2)
        do {
            _ = try await store.addChunks(batch.chunks, metadata: batch.metadata)
            XCTFail("automatic checkpoint errors must surface through addChunks")
        } catch {
            XCTAssertFalse(String(describing: error).isEmpty)
        }

        // The triggering mutation is already committed; retrying addChunks here
        // would mint fresh UUIDs and double-add the same logical chunks.
        let liveAfterThrow = await store.liveCount
        let hitsAfterThrow = try await store.query("doc-hybrid-error-chunk-0", k: 10)
        let walRecordsAfterThrow = await store.dense.journalRecordCount
        XCTAssertEqual(liveAfterThrow, 2)
        XCTAssertEqual(hitsAfterThrow.count, 2)
        XCTAssertEqual(walRecordsAfterThrow, 0)
        await store.dense.closeJournal()

        let reopened = try await hybridStore("auto-hybrid-error")
        let reopenedLive = await reopened.liveCount
        let reopenedHits = try await reopened.query("doc-hybrid-error-chunk-0", k: 10)
        let reopenedWALRecords = await reopened.dense.journalRecordCount
        XCTAssertEqual(reopenedLive, 2)
        XCTAssertEqual(reopenedHits.count, 2)
        XCTAssertEqual(reopenedWALRecords, 0)
        await reopened.dense.closeJournal()
    }

    func testHNSWLoadPagedMatchesJournaledPagedOpen() async throws {
        let dir = tempDir.appendingPathComponent("hnsw-load")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let config = HNSWConfiguration(
            m: 4,
            efConstruction: 30,
            efSearch: 20,
            autoCompactionThreshold: nil,
            levelSeed: 0xA015_BEEF
        )
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
        for i in 0..<80 {
            try await builder.add(autoVector(i, dim: dim), id: autoUUID(i))
        }
        try await builder.checkpoint(baseURL: base, walURL: wal)
        await builder.closeJournal()

        let legacyResident = try HNSWIndex.load(from: base)
        let mirrorResident = try HNSWIndex.load(from: base, mode: .resident)
        let mirrorResidentIsPaged = await mirrorResident.vectorsArePaged
        XCTAssertFalse(mirrorResidentIsPaged)
        for q in 0..<10 {
            let query = autoVector(1_000 + q, dim: dim)
            assertIdentical(
                await legacyResident.search(query: query, k: 8),
                await mirrorResident.search(query: query, k: 8),
                "resident load mirror query \(q)"
            )
        }

        let loadedPaged = try HNSWIndex.load(from: base, mode: .paged)
        let journaledPaged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        let loadedPagedIsPaged = await loadedPaged.vectorsArePaged
        let journaledPagedIsPaged = await journaledPaged.vectorsArePaged
        XCTAssertTrue(loadedPagedIsPaged)
        XCTAssertTrue(journaledPagedIsPaged)
        for q in 0..<10 {
            let query = autoVector(2_000 + q, dim: dim)
            assertIdentical(
                await loadedPaged.search(query: query, k: 8),
                await journaledPaged.search(query: query, k: 8),
                "paged load vs journaled paged open query \(q)"
            )
        }
        await journaledPaged.closeJournal()
    }

    func testStoreDensePagedOpenUsesMappedDenseLegs() async throws {
        let freshVector = try await vectorStore("fresh-dense-vector", dense: .paged)
        let freshVectorDenseIsPaged = await freshVector.index.vectorsArePaged
        XCTAssertTrue(freshVectorDenseIsPaged)
        await freshVector.index.closeJournal()

        var vector = try await vectorStore("dense-vector")
        let vectorBatch = autoBatch(doc: "doc-vector", start: 0, count: 8)
        _ = try await vector.addChunks(vectorBatch.chunks, metadata: vectorBatch.metadata)
        try await vector.checkpoint()
        await vector.index.closeJournal()

        vector = try await vectorStore("dense-vector", dense: .paged)
        let vectorDenseIsPaged = await vector.index.vectorsArePaged
        let vectorHits = try await vector.query("doc-vector-chunk-2", k: 3)
        XCTAssertTrue(vectorDenseIsPaged)
        XCTAssertFalse(vectorHits.isEmpty)
        await vector.index.closeJournal()

        let freshHybrid = try await hybridStore("fresh-dense-hybrid", dense: .paged)
        let freshHybridDenseIsPaged = await freshHybrid.dense.vectorsArePaged
        let freshHybridLiveCount = await freshHybrid.liveCount
        XCTAssertTrue(freshHybridDenseIsPaged)
        XCTAssertEqual(freshHybridLiveCount, 0)
        await freshHybrid.dense.closeJournal()

        var hybrid = try await hybridStore("dense-hybrid")
        let hybridBatch = autoBatch(doc: "doc-hybrid", start: 0, count: 8)
        _ = try await hybrid.addChunks(hybridBatch.chunks, metadata: hybridBatch.metadata)
        try await hybrid.checkpoint()
        await hybrid.dense.closeJournal()

        hybrid = try await hybridStore("dense-hybrid", dense: .paged)
        let hybridDenseIsPaged = await hybrid.dense.vectorsArePaged
        let hybridSparseCount = await hybrid.sparseCount
        let hybridLiveCount = await hybrid.liveCount
        let hybridHits = try await hybrid.query("doc-hybrid-chunk-2", k: 3)
        XCTAssertTrue(hybridDenseIsPaged)
        XCTAssertEqual(hybridSparseCount, hybridLiveCount)
        XCTAssertFalse(hybridHits.isEmpty)
        await hybrid.dense.closeJournal()
    }

    func testIndexResidencyAliasesCompileInOldAndNewSpellings() {
        let canonical: IndexResidency = .resident
        let oldHNSW: HNSWOpenMode = canonical
        let oldPQHW: PQHWOpenMode = .paged
        let newFromOld: IndexResidency = oldPQHW
        let oldFromOld: PQHWOpenMode = oldHNSW

        func describe(_ mode: IndexResidency) -> String {
            switch mode {
            case .resident:
                return "resident"
            case .paged:
                return "paged"
            }
        }

        XCTAssertEqual(describe(oldHNSW), "resident")
        XCTAssertEqual(describe(newFromOld), "paged")
        XCTAssertEqual(describe(oldFromOld), "resident")
    }
}
