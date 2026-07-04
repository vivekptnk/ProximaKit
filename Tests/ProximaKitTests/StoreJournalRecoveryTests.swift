// StoreJournalRecoveryTests.swift
// ProximaKitTests
//
// ADR-013 store-level journaling (closes deviation 5): in-process crash-recovery
// tests for the journaled `VectorStore` / `HybridVectorStore` lifecycle. The
// bar these enforce is the blocker bar from the mission brief:
//
//   after ANY crash-shaped recovery, the store is internally consistent —
//   every recovered vector is searchable AND mapped, and there is no phantom
//   mapping (a doc→UUID entry with no live vector) and no orphan (a live
//   vector with no mapping).
//
// The mechanism under test is derivation: the dense index + its WAL are the
// single source of truth; the document map (and, for hybrid, the whole sparse
// leg) are rebuilt from the recovered index on `open`, so a sidecar can never
// be newer or older than the index it describes. These tests exercise:
//   • index-ahead-of-a-stale-sidecar (WAL has records docmap.json never saw);
//   • sidecar-ahead-of-index (a hand-planted phantom docmap.json is ignored);
//   • a torn WAL tail under a store (prefix semantics, legs stay consistent);
//   • generation-mismatch typed rejection (stale WAL beside a newer base).
//
// Out-of-process SIGKILL is deliberately NOT re-run here: the only
// crash-relevant durability a journaled store adds is the dense-leg WAL, and
// that exact WAL on this exact durability path is already SIGKILL-hammered by
// `WALKillRecoveryTests`. The store's recovery step on top of it — rebuilding
// the map and sparse leg from `liveEntries()` — is pure in-memory computation
// with zero additional persistence, so dropping the store reference and
// reopening fully reproduces post-kill recovery. (Reasoning documented in the
// ADR-013 store-level journaling addendum.)

import XCTest
@testable import ProximaKit

// MARK: - Helpers

/// Deterministic embedder (same scheme as the other store test suites).
private struct JournalMockEmbedder: TextEmbedder {
    let dimension: Int
    func embed(_ text: String) async throws -> Vector {
        var hasher = Hasher()
        hasher.combine(text)
        let hash = abs(hasher.finalize())
        let base = Float(hash % 1000) / 1000.0
        return Vector((0..<dimension).map { base + Float($0) * 0.001 })
    }
}

private func meta(_ doc: String, _ idx: Int, _ text: String) -> ChunkMetadata {
    ChunkMetadata(documentId: doc, chunkIndex: idx, text: text)
}

private func batch(doc: String, count: Int, tag: String) -> (chunks: [String], metadata: [ChunkMetadata]) {
    let chunks = (0..<count).map { "\(tag)-\(doc)-\($0)" }
    return (chunks, chunks.enumerated().map { meta(doc, $0.offset, $0.element) })
}

// MARK: - VectorStore journaled recovery

final class StoreJournalVectorRecoveryTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("StoreJournalVec-\(UUID().uuidString)")
    }
    override func tearDown() {
        if let dir = tempDir { try? FileManager.default.removeItem(at: dir) }
        super.tearDown()
    }

    private func open(_ name: String, durability: WALDurability = .everyRecord) async throws -> VectorStore {
        try await VectorStore.open(
            name: name,
            embedder: JournalMockEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20,
                                      autoCompactionThreshold: nil, levelSeed: 0x5EED),
            durability: durability
        )
    }

    /// Asserts the blocker-bar invariant: the map's tracked id set is EXACTLY
    /// the index's live id set (bijection ⇒ no orphan, no phantom), and the
    /// index still answers queries.
    private func assertConsistent(_ store: VectorStore, expectedDocs: Set<String>) async {
        let liveIDs = Set(await store.index.liveEntries().map(\.id))
        let tracked = await store.trackedIDs
        XCTAssertEqual(liveIDs, tracked,
                       "index live-id set must equal document-map id set (no orphan / no phantom)")
        let live = await store.liveCount
        XCTAssertEqual(live, liveIDs.count)
        let docs = await store.documentIds
        XCTAssertEqual(docs, expectedDocs)
        if live > 0 {
            let hits = await (try? store.query("recover", k: live)) ?? []
            XCTAssertFalse(hits.isEmpty, "recovered index must be searchable")
            XCTAssertTrue(Set(hits.map(\.id)).isSubset(of: tracked),
                          "every search hit must be a tracked (mapped) id — no phantom result")
        }
    }

    /// Basic journaled round-trip: add, flush, drop the handle, reopen — the
    /// map is reconstructed from the WAL-recovered index.
    func testJournaledRoundTripRebuildsMapFromIndex() async throws {
        var store = try await open("roundtrip")
        let a = batch(doc: "doc-A", count: 5, tag: "r")
        let ids = try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.save()   // cheap WAL flush

        store = try await open("roundtrip")   // simulate process restart
        await assertConsistent(store, expectedDocs: ["doc-A"])
        let tracked = await store.trackedIDs
        XCTAssertEqual(tracked, Set(ids))
    }

    /// Index-ahead-of-sidecar: checkpoint captures doc-A into docmap.json, then
    /// doc-B is journaled but NEVER checkpointed. On reopen the WAL replays
    /// doc-B and the map is rebuilt to include it — the stale docmap.json (only
    /// doc-A) does not cap recovery.
    func testIndexAheadOfStaleSidecarRecoversBoth() async throws {
        var store = try await open("ahead")
        let a = batch(doc: "doc-A", count: 4, tag: "a")
        try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.checkpoint()   // docmap.json now == {doc-A}

        let b = batch(doc: "doc-B", count: 3, tag: "b")
        try await store.addChunks(b.chunks, metadata: b.metadata)
        try await store.save()         // doc-B durable in WAL, docmap.json still stale

        // Sanity: the on-disk cache really is stale (only doc-A).
        let cacheURL = tempDir.appendingPathComponent("ahead/docmap.json")
        let cache = try JSONDecoder().decode([String: [UUID]].self, from: Data(contentsOf: cacheURL))
        XCTAssertEqual(Set(cache.keys), ["doc-A"], "precondition: docmap.json is behind the WAL")

        store = try await open("ahead")
        await assertConsistent(store, expectedDocs: ["doc-A", "doc-B"])
    }

    /// Sidecar-ahead-of-index: a phantom docmap.json (a document + UUID that the
    /// index has never seen) must NOT resurrect a mapping-less mapping. The
    /// journaled open ignores the cache and rebuilds from the index.
    func testPhantomSidecarIsIgnored() async throws {
        var store = try await open("phantom")
        let a = batch(doc: "doc-A", count: 3, tag: "p")
        try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.checkpoint()

        // Plant a phantom mapping ahead of the index.
        let cacheURL = tempDir.appendingPathComponent("phantom/docmap.json")
        var cache = try JSONDecoder().decode([String: [UUID]].self, from: Data(contentsOf: cacheURL))
        cache["doc-PHANTOM"] = [UUID()]
        try JSONEncoder().encode(cache).write(to: cacheURL, options: .atomic)

        store = try await open("phantom")
        await assertConsistent(store, expectedDocs: ["doc-A"])   // phantom absent
        let phantomCount = await store.chunkCount(forDocument: "doc-PHANTOM")
        XCTAssertEqual(phantomCount, 0)
    }

    /// Torn WAL tail under a store: lop the last few bytes off the sidecar and
    /// reopen. Recovery must not throw, and the recovered map must still be an
    /// exact bijection with the recovered index (consistency holds at whatever
    /// prefix survived).
    func testTornWALTailStaysConsistent() async throws {
        var store = try await open("torn")
        let a = batch(doc: "doc-A", count: 10, tag: "t")
        try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.save()

        let walURL = tempDir.appendingPathComponent("torn/index.pxwal")
        let bytes = try Data(contentsOf: walURL)
        // Drop the last 5 bytes → the final record's frame runs past EOF.
        try bytes.prefix(bytes.count - 5).write(to: walURL, options: .atomic)

        store = try await open("torn")   // must not throw
        let liveIDs = Set(await store.index.liveEntries().map(\.id))
        let tracked = await store.trackedIDs
        XCTAssertEqual(liveIDs, tracked, "map must match the recovered index prefix exactly")
        XCTAssertLessThanOrEqual(liveIDs.count, 10)
        XCTAssertGreaterThanOrEqual(liveIDs.count, 9, "only the torn final record may be dropped")
    }

    /// A stale WAL beside a newer base (ADR-013 checkpoint crash window) is a
    /// typed `walGenerationMismatch`, never a trap or silent loss.
    func testGenerationMismatchIsTypedRejection() async throws {
        let store = try await open("genmix")
        let a = batch(doc: "doc-A", count: 3, tag: "g")
        try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.save()

        // Stash the current (generation-N) WAL, then checkpoint to bump the base
        // to generation N+1 with a fresh WAL, then restore the stale WAL.
        let walURL = tempDir.appendingPathComponent("genmix/index.pxwal")
        let stale = try Data(contentsOf: walURL)
        try await store.checkpoint()
        try stale.write(to: walURL, options: .atomic)

        do {
            _ = try await open("genmix")
            XCTFail("a stale WAL beside a newer base must be rejected")
        } catch let error as PersistenceError {
            guard case .walGenerationMismatch = error else {
                return XCTFail("expected walGenerationMismatch, got \(error)")
            }
        }
    }
}

// MARK: - HybridVectorStore journaled recovery

final class StoreJournalHybridRecoveryTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("StoreJournalHybrid-\(UUID().uuidString)")
    }
    override func tearDown() {
        if let dir = tempDir { try? FileManager.default.removeItem(at: dir) }
        super.tearDown()
    }

    private func open(_ name: String, durability: WALDurability = .everyRecord) async throws -> HybridVectorStore {
        try await HybridVectorStore.open(
            name: name,
            embedder: JournalMockEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20,
                                          autoCompactionThreshold: nil, levelSeed: 0x5EED),
            durability: durability
        )
    }

    /// The three-way consistency assertion for a hybrid store: the dense live
    /// set, the sparse live count, and the map's tracked set all agree, AND a
    /// keyword query hits the (rebuilt) sparse leg.
    private func assertConsistent(_ store: HybridVectorStore, expectedDocs: Set<String>) async {
        let liveIDs = Set(await store.dense.liveEntries().map(\.id))
        let tracked = await store.trackedIDs
        XCTAssertEqual(liveIDs, tracked, "dense live-id set must equal the map (no orphan / no phantom)")
        let denseLive = await store.liveCount
        let sparse = await store.sparseCount
        XCTAssertEqual(denseLive, sparse, "dense and sparse legs must hold the same document count")
        XCTAssertEqual(sparse, tracked.count, "sparse leg and map must agree — no leg diverges")
        let docs = await store.documentIds
        XCTAssertEqual(docs, expectedDocs)
    }

    /// The core hybrid blocker-bar test: the sparse leg has NO WAL, yet after a
    /// crash (drop + reopen) with NO checkpoint ever taken, the sparse leg is
    /// fully reconstructed from the dense WAL — every recovered id is present in
    /// both legs and the map, and a keyword search works.
    func testSparseLegReconstructedFromDenseWAL() async throws {
        var store = try await open("hy-recover")
        let a = batch(doc: "doc-A", count: 4, tag: "alpha")
        let b = batch(doc: "doc-B", count: 3, tag: "beta")
        _ = try await store.addChunks(a.chunks, metadata: a.metadata)
        _ = try await store.addChunks(b.chunks, metadata: b.metadata)
        try await store.save()   // flushes ONLY the dense WAL; sparse never persisted

        // No index.pxbm was ever written (no checkpoint) — prove it.
        let sparseFile = tempDir.appendingPathComponent("hy-recover/index.pxbm")
        XCTAssertFalse(FileManager.default.fileExists(atPath: sparseFile.path),
                       "precondition: sparse leg was never persisted to disk")

        store = try await open("hy-recover")
        await assertConsistent(store, expectedDocs: ["doc-A", "doc-B"])

        // The rebuilt sparse leg must actually answer keyword queries.
        let hits = try await store.query("alpha", k: 10)
        XCTAssertFalse(hits.isEmpty, "rebuilt sparse leg returned no keyword hits")
    }

    /// A torn dense-WAL tail must leave BOTH legs consistent (the sparse leg is
    /// rebuilt from whatever dense prefix survived, so it cannot diverge).
    func testTornDenseWALKeepsLegsConsistent() async throws {
        var store = try await open("hy-torn")
        let a = batch(doc: "doc-A", count: 10, tag: "g")
        _ = try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.save()

        let walURL = tempDir.appendingPathComponent("hy-torn/index.pxwal")
        let bytes = try Data(contentsOf: walURL)
        try bytes.prefix(bytes.count - 5).write(to: walURL, options: .atomic)

        store = try await open("hy-torn")   // must not throw
        let liveIDs = Set(await store.dense.liveEntries().map(\.id))
        let tracked = await store.trackedIDs
        let sparse = await store.sparseCount
        XCTAssertEqual(liveIDs, tracked)
        XCTAssertEqual(sparse, tracked.count, "sparse leg rebuilt from the dense prefix — legs agree")
        XCTAssertLessThanOrEqual(tracked.count, 10)
        XCTAssertGreaterThanOrEqual(tracked.count, 9)
    }

    /// Removals under journaling recover consistently too: a removed document is
    /// gone from the dense leg, the sparse leg, and the map after reopen.
    func testRemoveThenRecoverIsConsistent() async throws {
        var store = try await open("hy-remove")
        let a = batch(doc: "doc-A", count: 4, tag: "a")
        let b = batch(doc: "doc-B", count: 4, tag: "b")
        _ = try await store.addChunks(a.chunks, metadata: a.metadata)
        _ = try await store.addChunks(b.chunks, metadata: b.metadata)
        _ = try await store.removeDocument(id: "doc-A")
        try await store.save()

        store = try await open("hy-remove")
        await assertConsistent(store, expectedDocs: ["doc-B"])
        let bCount = await store.chunkCount(forDocument: "doc-B")
        XCTAssertEqual(bCount, 4)
    }

    /// Hybrid twin of `StoreJournalVectorRecoveryTests.testGenerationMismatchIsTypedRejection`:
    /// a stale WAL beside a newer base (the ADR-013 checkpoint crash window) is a
    /// typed `walGenerationMismatch`, never a trap or silent loss. It matters
    /// here because `HybridVectorStore.open` recovers its dense leg through the
    /// identical `HNSWIndex.open` → `WALDecoder` path the plain `VectorStore`
    /// uses, so the same typed rejection must surface unguarded to the caller.
    func testGenerationMismatchIsTypedRejection() async throws {
        let store = try await open("hy-genmix")
        let a = batch(doc: "doc-A", count: 3, tag: "g")
        _ = try await store.addChunks(a.chunks, metadata: a.metadata)
        try await store.save()

        // Stash the current (generation-N) WAL, then checkpoint to bump the base
        // to generation N+1 with a fresh WAL, then restore the stale WAL.
        let walURL = tempDir.appendingPathComponent("hy-genmix/index.pxwal")
        let stale = try Data(contentsOf: walURL)
        try await store.checkpoint()
        try stale.write(to: walURL, options: .atomic)

        do {
            _ = try await open("hy-genmix")
            XCTFail("a stale WAL beside a newer base must be rejected")
        } catch let error as PersistenceError {
            guard case .walGenerationMismatch = error else {
                return XCTFail("expected walGenerationMismatch, got \(error)")
            }
        }
    }
}
