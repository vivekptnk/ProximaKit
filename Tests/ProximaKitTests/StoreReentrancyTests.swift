// StoreReentrancyTests.swift
// ProximaKit
//
// Regression tests for actor-reentrancy hazards in VectorStore and
// HybridVectorStore (audit cluster: stores):
// - save() interleaved with addChunks() must not lose the write
//   (isDirty lost-update) or mark the store clean prematurely.
// - HybridVectorStore.save() must never persist diverged dense/sparse legs.
// - removeDocument() interleaved with addChunks() for the same document
//   must not orphan freshly added vectors in the index.
//
// The gated tests park an addChunks() call inside the embedder so a save()
// can be issued mid-flight deterministically; the stress tests fire compound
// operations concurrently and assert the store's cross-structure invariants.

import XCTest
@testable import ProximaKit

// MARK: - Embedder Helpers

/// Deterministic vector from a text hash (same scheme as VectorStoreTests).
private func deterministicVector(for text: String, dimension: Int) -> Vector {
    var hasher = Hasher()
    hasher.combine(text)
    let hash = abs(hasher.finalize())
    let base = Float(hash % 1000) / 1000.0
    let components = (0..<dimension).map { i in
        base + Float(i) * 0.001
    }
    return Vector(components)
}

/// Plain deterministic embedder for the stress tests.
private struct ReentrancyMockEmbedder: TextEmbedder {
    let dimension: Int

    func embed(_ text: String) async throws -> Vector {
        deterministicVector(for: text, dimension: dimension)
    }
}

/// Coordination point that lets a test park `embedBatch` mid-flight and
/// release it on demand, so a save() can be issued while an addChunks()
/// is provably suspended inside the store.
private actor EmbedderGate {
    private var isOpen = true
    private var entries = 0
    private var openWaiters: [CheckedContinuation<Void, Never>] = []
    private var entryWaiters: [(threshold: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func open() {
        isOpen = true
        let waiters = openWaiters
        openWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }

    func close() {
        isOpen = false
    }

    /// Called by the embedder at the top of `embedBatch`. Suspends while closed.
    func enter() async {
        entries += 1
        let reached = entries
        let ready = entryWaiters.filter { $0.threshold <= reached }
        entryWaiters.removeAll { $0.threshold <= reached }
        for waiter in ready { waiter.continuation.resume() }

        if !isOpen {
            await withCheckedContinuation { openWaiters.append($0) }
        }
    }

    /// Suspends until `embedBatch` has been entered at least `n` times.
    func waitForEntries(_ n: Int) async {
        if entries >= n { return }
        await withCheckedContinuation { entryWaiters.append((n, $0)) }
    }
}

/// Embedder whose `embedBatch` parks on an ``EmbedderGate`` while it is closed.
private struct GatedEmbedder: TextEmbedder {
    let dimension: Int
    let gate: EmbedderGate

    func embed(_ text: String) async throws -> Vector {
        deterministicVector(for: text, dimension: dimension)
    }

    func embedBatch(_ texts: [String]) async throws -> [Vector] {
        await gate.enter()
        return texts.map { deterministicVector(for: $0, dimension: dimension) }
    }
}

// MARK: - Shared Helpers

private func chunkMeta(_ doc: String, _ index: Int, _ text: String) -> ChunkMetadata {
    ChunkMetadata(documentId: doc, chunkIndex: index, text: text)
}

private func chunkBatch(doc: String, count: Int, tag: String) -> (chunks: [String], metadata: [ChunkMetadata]) {
    let chunks = (0..<count).map { "\(tag)-\(doc)-chunk-\($0)" }
    let metadata = chunks.enumerated().map { i, t in chunkMeta(doc, i, t) }
    return (chunks, metadata)
}

// MARK: - VectorStore Reentrancy Tests

final class VectorStoreReentrancyTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-StoreReentrancy-\(UUID().uuidString)")
    }

    override func tearDown() {
        super.tearDown()
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func makeStore(name: String, embedder: any TextEmbedder) throws -> VectorStore {
        try VectorStore(
            name: name,
            embedder: embedder,
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        )
    }

    /// Regression: pre-fix, a save() issued while an addChunks() was suspended
    /// in the embedder would run to completion first, persist only the old
    /// state, and set isDirty = false. The interleaved chunk then re-dirtied
    /// the store but was not on disk — the classic lost-update window.
    /// Post-fix, compound operations are serialized: the save runs after the
    /// in-flight addChunks, leaving the store clean AND chunk B persisted.
    func testSaveInterleavedWithAddChunksIsNotLost() async throws {
        let gate = EmbedderGate()
        let embedder = GatedEmbedder(dimension: dim, gate: gate)
        let store = try makeStore(name: "interleave", embedder: embedder)

        // Seed chunk A with the gate open so the store is dirty.
        try await store.addChunks(
            ["chunk A"],
            metadata: [chunkMeta("doc-A", 0, "chunk A")]
        )

        // Park a second addChunks inside embedBatch.
        await gate.close()
        let addTask = Task {
            try await store.addChunks(
                ["chunk B"],
                metadata: [chunkMeta("doc-B", 0, "chunk B")]
            )
        }
        await gate.waitForEntries(2)

        // Issue a save while the add is provably mid-flight, and give the
        // (pre-fix, unserialized) save ample time to run to completion.
        let saveTask = Task { try await store.save() }
        try await Task.sleep(nanoseconds: 100_000_000)

        // Release the add and let both operations finish.
        await gate.open()
        _ = try await addTask.value
        try await saveTask.value

        // The save must have run after the in-flight addChunks: store clean,
        // and chunk B on disk. Pre-fix the store ends dirty here (B unsaved).
        let dirty = await store.hasUnsavedChanges
        XCTAssertFalse(
            dirty,
            "save() must serialize after the in-flight addChunks and leave the store clean"
        )

        let reloaded = try makeStore(name: "interleave", embedder: embedder)
        try await reloaded.loadDocumentMap()
        let live = await reloaded.liveCount
        XCTAssertEqual(live, 2, "chunk B added during save() must be persisted")
        let docs = await reloaded.documentIds
        XCTAssertEqual(docs, Set(["doc-A", "doc-B"]))
    }

    /// Stress: concurrent save() + addChunks() must never silently lose
    /// chunks. After both complete, one more save() must leave the on-disk
    /// state containing every chunk whose addChunks call returned.
    /// Pre-fix, an interleaved save could clear isDirty after a mutation
    /// landed, making the final save a no-op and dropping chunks.
    func testConcurrentSaveAndAddChunksStress() async throws {
        let embedder = ReentrancyMockEmbedder(dimension: dim)

        for iteration in 0..<12 {
            let name = "stress-\(iteration)"
            let store = try makeStore(name: name, embedder: embedder)

            let seed = chunkBatch(doc: "doc-seed", count: 24, tag: "seed\(iteration)")
            try await store.addChunks(seed.chunks, metadata: seed.metadata)

            let fresh = chunkBatch(doc: "doc-new", count: 8, tag: "new\(iteration)")
            // Priorities skew scheduling so the add's continuations can win
            // the race against the save's resume (the pre-fix loss window).
            let addTask = Task(priority: .high) {
                try await store.addChunks(fresh.chunks, metadata: fresh.metadata)
            }
            let saveTask = Task(priority: .low) { try await store.save() }

            _ = try await addTask.value
            try await saveTask.value

            // Final save: persists anything that is still (correctly) dirty.
            try await store.save()

            let reloaded = try makeStore(name: name, embedder: embedder)
            try await reloaded.loadDocumentMap()
            let live = await reloaded.liveCount
            XCTAssertEqual(
                live, 32,
                "iteration \(iteration): chunks lost across interleaved save/addChunks"
            )
            let newCount = await reloaded.chunkCount(forDocument: "doc-new")
            XCTAssertEqual(
                newCount, 8,
                "iteration \(iteration): docmap lost interleaved document"
            )
        }
    }

    /// Regression: pre-fix, removeDocument() snapshotted the UUID set, looped
    /// over awaits, then dropped the documentMap entry wholesale. An
    /// addChunks() for the same document interleaving with the loop had its
    /// fresh UUIDs deleted from the map while the vectors stayed live in the
    /// index — permanently orphaned. Post-fix the operations serialize (and
    /// the removal subtracts only its snapshot), so index and map agree.
    func testRemoveDocumentInterleavedWithReAddDoesNotOrphan() async throws {
        let embedder = ReentrancyMockEmbedder(dimension: dim)
        let store = try makeStore(name: "orphan", embedder: embedder)

        for round in 0..<10 {
            let seed = chunkBatch(doc: "doc-X", count: 40, tag: "round\(round)")
            try await store.addChunks(seed.chunks, metadata: seed.metadata)

            // Delete-then-readd race: a realistic re-ingestion pattern.
            let removeTask = Task(priority: .low) {
                try await store.removeDocument(id: "doc-X")
            }
            let addTask = Task(priority: .high) {
                try await store.addChunks(
                    ["fresh-\(round)"],
                    metadata: [chunkMeta("doc-X", 0, "fresh-\(round)")]
                )
            }
            _ = try await removeTask.value
            _ = try await addTask.value

            // Invariant: every live vector is tracked by the document map.
            let live = await store.liveCount
            let mapped = await store.chunkCount(forDocument: "doc-X")
            XCTAssertEqual(
                live, mapped,
                "round \(round): orphaned vectors — live in index but untracked in documentMap"
            )

            // Reset for the next round (tolerate the all-removed case).
            if mapped > 0 {
                try await store.removeDocument(id: "doc-X")
            }
        }
    }
}

// MARK: - HybridVectorStore Reentrancy Tests

final class HybridVectorStoreReentrancyTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-HybridReentrancy-\(UUID().uuidString)")
    }

    override func tearDown() {
        super.tearDown()
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func makeStore(name: String, embedder: any TextEmbedder) throws -> HybridVectorStore {
        try HybridVectorStore(
            name: name,
            embedder: embedder,
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        )
    }

    /// Regression: pre-fix, a save() issued while an addChunks() was parked
    /// in the embedder persisted the old state of *both* legs and cleared
    /// isDirty; the interleaved chunk was then absent from disk despite a
    /// completed save. Post-fix the save serializes after the add.
    func testSaveInterleavedWithAddChunksIsNotLost() async throws {
        let gate = EmbedderGate()
        let embedder = GatedEmbedder(dimension: dim, gate: gate)
        let store = try makeStore(name: "hybrid-interleave", embedder: embedder)

        try await store.addChunks(
            ["alpha apple"],
            metadata: [chunkMeta("doc-A", 0, "alpha apple")]
        )

        await gate.close()
        let addTask = Task {
            try await store.addChunks(
                ["beta banana"],
                metadata: [chunkMeta("doc-B", 0, "beta banana")]
            )
        }
        await gate.waitForEntries(2)

        let saveTask = Task { try await store.save() }
        try await Task.sleep(nanoseconds: 100_000_000)

        await gate.open()
        _ = try await addTask.value
        try await saveTask.value

        let dirty = await store.hasUnsavedChanges
        XCTAssertFalse(
            dirty,
            "save() must serialize after the in-flight addChunks and leave the store clean"
        )

        // Both legs must contain both documents after reload.
        let reloaded = try makeStore(name: "hybrid-interleave", embedder: embedder)
        try await reloaded.loadDocumentMap()
        let denseLive = await reloaded.liveCount
        let sparse = await reloaded.sparseCount
        XCTAssertEqual(denseLive, 2, "dense leg lost the interleaved chunk")
        XCTAssertEqual(sparse, 2, "sparse leg lost the interleaved chunk")
        let docs = await reloaded.documentIds
        XCTAssertEqual(docs, Set(["doc-A", "doc-B"]))
    }

    /// Stress: the dense and sparse files on disk must always describe the
    /// same document set. Pre-fix, an addChunks() interleaving between the
    /// two leg writes of save() (or a lost dirty flag) could persist a torn
    /// pair where one leg misses documents the other has.
    func testConcurrentSaveAndAddChunksKeepsLegsConsistent() async throws {
        let embedder = ReentrancyMockEmbedder(dimension: dim)

        for iteration in 0..<12 {
            let name = "hybrid-stress-\(iteration)"
            let store = try makeStore(name: name, embedder: embedder)

            let seed = chunkBatch(doc: "doc-seed", count: 16, tag: "hseed\(iteration)")
            try await store.addChunks(seed.chunks, metadata: seed.metadata)

            let fresh = chunkBatch(doc: "doc-new", count: 6, tag: "hnew\(iteration)")
            let addTask = Task(priority: .high) {
                try await store.addChunks(fresh.chunks, metadata: fresh.metadata)
            }
            let saveTask = Task(priority: .low) { try await store.save() }

            _ = try await addTask.value
            try await saveTask.value
            try await store.save()

            let reloaded = try makeStore(name: name, embedder: embedder)
            try await reloaded.loadDocumentMap()
            let denseLive = await reloaded.liveCount
            let sparse = await reloaded.sparseCount
            XCTAssertEqual(
                denseLive, sparse,
                "iteration \(iteration): persisted legs diverged (dense \(denseLive) vs sparse \(sparse))"
            )
            XCTAssertEqual(
                denseLive, 22,
                "iteration \(iteration): chunks lost across interleaved save/addChunks"
            )
            let newCount = await reloaded.chunkCount(forDocument: "doc-new")
            XCTAssertEqual(newCount, 6, "iteration \(iteration): docmap lost interleaved document")
        }
    }

    /// Same orphan regression as the VectorStore variant, but the orphan
    /// would leak in *both* legs here.
    func testRemoveDocumentInterleavedWithReAddDoesNotOrphan() async throws {
        let embedder = ReentrancyMockEmbedder(dimension: dim)
        let store = try makeStore(name: "hybrid-orphan", embedder: embedder)

        for round in 0..<8 {
            let seed = chunkBatch(doc: "doc-X", count: 30, tag: "hround\(round)")
            try await store.addChunks(seed.chunks, metadata: seed.metadata)

            let removeTask = Task(priority: .low) {
                try await store.removeDocument(id: "doc-X")
            }
            let addTask = Task(priority: .high) {
                try await store.addChunks(
                    ["hfresh-\(round)"],
                    metadata: [chunkMeta("doc-X", 0, "hfresh-\(round)")]
                )
            }
            _ = try await removeTask.value
            _ = try await addTask.value

            let live = await store.liveCount
            let sparse = await store.sparseCount
            let mapped = await store.chunkCount(forDocument: "doc-X")
            XCTAssertEqual(
                live, mapped,
                "round \(round): orphaned vectors in dense leg — live but untracked in documentMap"
            )
            XCTAssertEqual(
                sparse, mapped,
                "round \(round): orphaned documents in sparse leg — live but untracked in documentMap"
            )

            if mapped > 0 {
                try await store.removeDocument(id: "doc-X")
            }
        }
    }
}
