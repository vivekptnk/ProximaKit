// HybridVectorStoreTests.swift
// ProximaKit
//
// End-to-end tests for HybridVectorStore:
// - Auto-embedding both legs from chunk text
// - Query retrieves expected chunks with metadata intact
// - removeDocument removes from both legs
// - Save/reload roundtrip preserves both legs + document map
// - Existing VectorStore contract still holds (no regression)

import XCTest
@testable import ProximaKit

// MARK: - Mock Embedder
//
// Same deterministic-hash pattern as VectorStoreTests, so the two stores
// share a reference-embedder style and behave comparably in parity tests.

private struct HybridMockEmbedder: TextEmbedder {
    let dimension: Int

    func embed(_ text: String) async throws -> Vector {
        var hasher = Hasher()
        hasher.combine(text)
        let hash = abs(hasher.finalize())
        let base = Float(hash % 1000) / 1000.0
        let components = (0..<dimension).map { i in
            base + Float(i) * 0.001
        }
        return Vector(components)
    }
}

// MARK: - Tests

final class HybridVectorStoreTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-Hybrid-\(UUID().uuidString)")
    }

    override func tearDown() {
        super.tearDown()
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func makeStore(name: String = "hybrid-test") throws -> HybridVectorStore {
        try HybridVectorStore(
            name: name,
            embedder: HybridMockEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        )
    }

    // MARK: - Add Chunks

    func testAddChunksWritesBothLegs() async throws {
        let store = try makeStore()

        let chunks = ["apple fruit red", "banana fruit yellow", "cherry fruit red"]
        let metadata = chunks.enumerated().map { i, text in
            ChunkMetadata(documentId: "doc-A", chunkIndex: i, text: text)
        }

        let ids = try await store.addChunks(chunks, metadata: metadata)
        XCTAssertEqual(ids.count, 3)

        let dense = await store.count
        let sparse = await store.sparseCount
        XCTAssertEqual(dense, 3)
        XCTAssertEqual(sparse, 3)
    }

    func testAddChunksMismatchThrows() async throws {
        let store = try makeStore()
        do {
            try await store.addChunks(
                ["a", "b"],
                metadata: [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "a")]
            )
            XCTFail("Expected VectorStoreError.chunkMetadataMismatch")
        } catch let error as VectorStoreError {
            guard case .chunkMetadataMismatch = error else {
                XCTFail("Unexpected error \(error)")
                return
            }
        }
    }

    func testAddEmptyThrows() async throws {
        let store = try makeStore()
        do {
            try await store.addChunks([], metadata: [])
            XCTFail("Expected VectorStoreError.emptyChunks")
        } catch let error as VectorStoreError {
            guard case .emptyChunks = error else {
                XCTFail("Unexpected error \(error)")
                return
            }
        }
    }

    // MARK: - Query

    func testQueryReturnsRelevantChunks() async throws {
        let store = try makeStore()

        let chunks = [
            "apple fruit",
            "banana fruit",
            "zebra animal",
        ]
        let metadata = chunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: t)
        }
        try await store.addChunks(chunks, metadata: metadata)

        let hits = try await store.query("apple fruit", k: 3)
        XCTAssertFalse(hits.isEmpty)

        // Apple-fruit chunk should outrank zebra-animal chunk under either strategy.
        let zebra = hits.firstIndex { ($0.decodeMetadata(as: ChunkMetadata.self)?.text ?? "").contains("zebra") }
        let apple = hits.firstIndex { ($0.decodeMetadata(as: ChunkMetadata.self)?.text ?? "").contains("apple") }
        if let z = zebra, let a = apple {
            XCTAssertLessThan(a, z, "Apple chunk should rank above zebra chunk for 'apple fruit'")
        }
    }

    func testQueryMetadataDecoding() async throws {
        let store = try makeStore()

        let meta = ChunkMetadata(
            documentId: "doc-X",
            chunkIndex: 2,
            text: "inspect metadata payload",
            extra: ["source": "hybrid-test"]
        )
        try await store.addChunks(["inspect metadata payload"], metadata: [meta])

        let hits = try await store.query("metadata", k: 1)
        XCTAssertEqual(hits.count, 1)

        let decoded = hits[0].decodeMetadata(as: ChunkMetadata.self)
        XCTAssertEqual(decoded?.documentId, "doc-X")
        XCTAssertEqual(decoded?.chunkIndex, 2)
        XCTAssertEqual(decoded?.extra?["source"], "hybrid-test")
    }

    // MARK: - Remove Document

    func testRemoveDocumentHitsBothLegs() async throws {
        let store = try makeStore()

        let aChunks = ["alpha one", "alpha two"]
        let aMeta = aChunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-A", chunkIndex: i, text: t)
        }
        let aIds = try await store.addChunks(aChunks, metadata: aMeta)

        let bChunks = ["beta only"]
        let bMeta = [ChunkMetadata(documentId: "doc-B", chunkIndex: 0, text: "beta only")]
        try await store.addChunks(bChunks, metadata: bMeta)

        let removed = try await store.removeDocument(id: "doc-A")
        XCTAssertEqual(removed, 2)

        let sparseCount = await store.sparseCount
        let live = await store.liveCount
        XCTAssertEqual(live, 1)
        XCTAssertEqual(sparseCount, 1)

        // Dense leg always returns available vectors for any query; the contract
        // is that the *removed* IDs must not leak back, not that results are empty.
        let removedSet = Set(aIds)
        let hits = try await store.query("alpha", k: 5)
        for hit in hits {
            XCTAssertFalse(removedSet.contains(hit.id),
                           "Removed chunk \(hit.id) leaked into fused results")
        }
    }

    func testRemoveNonexistentDocumentThrows() async throws {
        let store = try makeStore()
        do {
            try await store.removeDocument(id: "missing")
            XCTFail("Expected documentNotFound")
        } catch let error as VectorStoreError {
            guard case .documentNotFound = error else {
                XCTFail("Unexpected error \(error)")
                return
            }
        }
    }

    // MARK: - Persistence

    func testSaveAndReload() async throws {
        let store = try makeStore()

        let chunks = ["hello world", "foo bar baz"]
        let meta = chunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: t)
        }
        try await store.addChunks(chunks, metadata: meta)
        try await store.save()

        let reloaded = try makeStore()
        try await reloaded.loadDocumentMap()

        let reloadedLive = await reloaded.liveCount
        let reloadedSparse = await reloaded.sparseCount
        XCTAssertEqual(reloadedLive, 2)
        XCTAssertEqual(reloadedSparse, 2)
        let reloadedDocIds = await reloaded.documentIds
        XCTAssertEqual(reloadedDocIds, ["doc-1"])

        // Same query before/after reload gives the same result set.
        let pre = try await store.query("hello", k: 5)
        let post = try await reloaded.query("hello", k: 5)
        XCTAssertEqual(pre.map(\.id), post.map(\.id))
    }

    func testSaveSkipsWhenClean() async throws {
        let store = try makeStore()
        try await store.addChunks(
            ["seed"],
            metadata: [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "seed")]
        )
        let dirty1 = await store.hasUnsavedChanges
        XCTAssertTrue(dirty1)
        try await store.save()
        let dirty2 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty2)
        try await store.save()
        let dirty3 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty3)
    }

    // MARK: - Accessors

    func testChunkCountByDocument() async throws {
        let store = try makeStore()
        try await store.addChunks(
            ["one", "two", "three"],
            metadata: (0..<3).map { i in
                ChunkMetadata(documentId: "multi", chunkIndex: i, text: "c\(i)")
            }
        )
        let count = await store.chunkCount(forDocument: "multi")
        XCTAssertEqual(count, 3)
        let none = await store.chunkCount(forDocument: "missing")
        XCTAssertEqual(none, 0)
    }
}
