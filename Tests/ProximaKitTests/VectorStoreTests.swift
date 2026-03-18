// VectorStoreTests.swift
// ProximaKit
//
// Tests for VectorStore actor and ChunkMetadata.

import XCTest
@testable import ProximaKit

// MARK: - Mock Embedder

/// Deterministic mock embedder for testing.
/// Embeds text as a vector of [hash, hash, hash, ...] for reproducibility.
private struct MockEmbedder: TextEmbedder {
    let dimension: Int

    func embed(_ text: String) async throws -> Vector {
        // Create a deterministic vector from the text's hash.
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

// MARK: - ChunkMetadata Tests

final class ChunkMetadataTests: XCTestCase {

    func testCodableRoundtrip() throws {
        let meta = ChunkMetadata(
            documentId: "doc-42",
            chunkIndex: 3,
            text: "Hello world",
            extra: ["page": "7"]
        )

        let data = try JSONEncoder().encode(meta)
        let decoded = try JSONDecoder().decode(ChunkMetadata.self, from: data)

        XCTAssertEqual(decoded.documentId, "doc-42")
        XCTAssertEqual(decoded.chunkIndex, 3)
        XCTAssertEqual(decoded.text, "Hello world")
        XCTAssertEqual(decoded.extra?["page"], "7")
    }

    func testEquality() {
        let a = ChunkMetadata(documentId: "d1", chunkIndex: 0, text: "a")
        let b = ChunkMetadata(documentId: "d1", chunkIndex: 0, text: "a")
        let c = ChunkMetadata(documentId: "d2", chunkIndex: 0, text: "a")

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }

    func testExtraIsOptional() throws {
        let meta = ChunkMetadata(documentId: "d1", chunkIndex: 0, text: "t")
        XCTAssertNil(meta.extra)

        let data = try JSONEncoder().encode(meta)
        let decoded = try JSONDecoder().decode(ChunkMetadata.self, from: data)
        XCTAssertNil(decoded.extra)
    }
}

// MARK: - VectorStore Tests

final class VectorStoreTests: XCTestCase {

    private let dim = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-VectorStore-\(UUID().uuidString)")
    }

    override func tearDown() {
        super.tearDown()
        if let dir = tempDir {
            try? FileManager.default.removeItem(at: dir)
        }
    }

    private func makeStore(name: String = "test") throws -> VectorStore {
        try VectorStore(
            name: name,
            embedder: MockEmbedder(dimension: dim),
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        )
    }

    // MARK: - Add Chunks

    func testAddChunksReturnsUUIDs() async throws {
        let store = try makeStore()

        let chunks = ["chunk one", "chunk two", "chunk three"]
        let metadata = chunks.enumerated().map { i, text in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: text)
        }

        let ids = try await store.addChunks(chunks, metadata: metadata)

        XCTAssertEqual(ids.count, 3)
        XCTAssertEqual(Set(ids).count, 3, "All IDs should be unique")
        let count = await store.count
        XCTAssertEqual(count, 3)
    }

    func testAddChunksTracksDocumentMap() async throws {
        let store = try makeStore()

        let chunks1 = ["a", "b"]
        let meta1 = chunks1.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-A", chunkIndex: i, text: t)
        }
        let chunks2 = ["c"]
        let meta2 = [ChunkMetadata(documentId: "doc-B", chunkIndex: 0, text: "c")]

        try await store.addChunks(chunks1, metadata: meta1)
        try await store.addChunks(chunks2, metadata: meta2)

        let docIds = await store.documentIds
        XCTAssertEqual(docIds, ["doc-A", "doc-B"])
        let countA = await store.chunkCount(forDocument: "doc-A")
        XCTAssertEqual(countA, 2)
        let countB = await store.chunkCount(forDocument: "doc-B")
        XCTAssertEqual(countB, 1)
    }

    func testAddChunksMismatchThrows() async throws {
        let store = try makeStore()

        let chunks = ["a", "b"]
        let metadata = [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "a")]

        do {
            try await store.addChunks(chunks, metadata: metadata)
            XCTFail("Expected VectorStoreError.chunkMetadataMismatch")
        } catch let error as VectorStoreError {
            if case .chunkMetadataMismatch(let c, let m) = error {
                XCTAssertEqual(c, 2)
                XCTAssertEqual(m, 1)
            } else {
                XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testAddEmptyChunksThrows() async throws {
        let store = try makeStore()

        do {
            try await store.addChunks([], metadata: [])
            XCTFail("Expected VectorStoreError.emptyChunks")
        } catch let error as VectorStoreError {
            guard case .emptyChunks = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
        }
    }

    // MARK: - Query

    func testQueryReturnsResults() async throws {
        let store = try makeStore()

        let chunks = ["apple fruit", "banana fruit", "cherry fruit"]
        let metadata = chunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: t)
        }
        try await store.addChunks(chunks, metadata: metadata)

        let results = try await store.query("apple fruit", k: 3)

        XCTAssertFalse(results.isEmpty)
        XCTAssertLessThanOrEqual(results.count, 3)

        let firstMeta = results[0].decodeMetadata(as: ChunkMetadata.self)
        XCTAssertNotNil(firstMeta)
        XCTAssertEqual(firstMeta?.text, "apple fruit")
    }

    func testQueryMetadataDecoding() async throws {
        let store = try makeStore()

        let meta = ChunkMetadata(
            documentId: "doc-X",
            chunkIndex: 0,
            text: "test content",
            extra: ["source": "unit-test"]
        )
        try await store.addChunks(["test content"], metadata: [meta])

        let results = try await store.query("test content", k: 1)
        XCTAssertEqual(results.count, 1)

        let decoded = results[0].decodeMetadata(as: ChunkMetadata.self)
        XCTAssertEqual(decoded?.documentId, "doc-X")
        XCTAssertEqual(decoded?.chunkIndex, 0)
        XCTAssertEqual(decoded?.text, "test content")
        XCTAssertEqual(decoded?.extra?["source"], "unit-test")
    }

    // MARK: - Remove Document

    func testRemoveDocument() async throws {
        let store = try makeStore()

        let chunks = ["a", "b", "c"]
        let meta = chunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: t)
        }
        try await store.addChunks(chunks, metadata: meta)

        try await store.addChunks(
            ["d"],
            metadata: [ChunkMetadata(documentId: "doc-2", chunkIndex: 0, text: "d")]
        )

        let totalCount = await store.count
        XCTAssertEqual(totalCount, 4)

        let removed = try await store.removeDocument(id: "doc-1")
        XCTAssertEqual(removed, 3)
        let liveCount = await store.liveCount
        XCTAssertEqual(liveCount, 1)
        let docIds = await store.documentIds
        XCTAssertEqual(docIds, ["doc-2"])
    }

    func testRemoveNonexistentDocumentThrows() async throws {
        let store = try makeStore()

        do {
            try await store.removeDocument(id: "no-such-doc")
            XCTFail("Expected VectorStoreError.documentNotFound")
        } catch let error as VectorStoreError {
            guard case .documentNotFound(let id) = error else {
                XCTFail("Unexpected error: \(error)")
                return
            }
            XCTAssertEqual(id, "no-such-doc")
        }
    }

    // MARK: - Persistence

    func testSaveAndReload() async throws {
        let store = try makeStore()

        let chunks = ["hello world", "foo bar"]
        let meta = chunks.enumerated().map { i, t in
            ChunkMetadata(documentId: "doc-1", chunkIndex: i, text: t)
        }
        try await store.addChunks(chunks, metadata: meta)
        try await store.save()

        let reloaded = try makeStore()
        try await reloaded.loadDocumentMap()

        let reloadedCount = await reloaded.count
        XCTAssertEqual(reloadedCount, 2)
        let reloadedDocIds = await reloaded.documentIds
        XCTAssertEqual(reloadedDocIds, ["doc-1"])
        let reloadedChunkCount = await reloaded.chunkCount(forDocument: "doc-1")
        XCTAssertEqual(reloadedChunkCount, 2)
    }

    func testSaveSkipsWhenClean() async throws {
        let store = try makeStore()

        let chunks = ["test"]
        let meta = [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "test")]
        try await store.addChunks(chunks, metadata: meta)

        let dirty1 = await store.hasUnsavedChanges
        XCTAssertTrue(dirty1)
        try await store.save()
        let dirty2 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty2)

        try await store.save()
        let dirty3 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty3)
    }

    // MARK: - Dirty Flag

    func testDirtyFlagOnAdd() async throws {
        let store = try makeStore()
        let dirty1 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty1)

        try await store.addChunks(
            ["x"],
            metadata: [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "x")]
        )
        let dirty2 = await store.hasUnsavedChanges
        XCTAssertTrue(dirty2)
    }

    func testDirtyFlagOnRemove() async throws {
        let store = try makeStore()

        try await store.addChunks(
            ["x"],
            metadata: [ChunkMetadata(documentId: "d", chunkIndex: 0, text: "x")]
        )
        try await store.save()
        let dirty1 = await store.hasUnsavedChanges
        XCTAssertFalse(dirty1)

        try await store.removeDocument(id: "d")
        let dirty2 = await store.hasUnsavedChanges
        XCTAssertTrue(dirty2)
    }

    // MARK: - Edge Cases

    func testMultipleDocumentsSameStore() async throws {
        let store = try makeStore()

        for docIdx in 0..<5 {
            let chunks = (0..<3).map { "doc\(docIdx)-chunk\($0)" }
            let meta = chunks.enumerated().map { i, t in
                ChunkMetadata(documentId: "doc-\(docIdx)", chunkIndex: i, text: t)
            }
            try await store.addChunks(chunks, metadata: meta)
        }

        let totalCount = await store.count
        XCTAssertEqual(totalCount, 15)
        let docIdCount = await store.documentIds.count
        XCTAssertEqual(docIdCount, 5)

        try await store.removeDocument(id: "doc-2")
        let liveCount = await store.liveCount
        XCTAssertEqual(liveCount, 12)
        let docCount = await store.documentIds.count
        XCTAssertEqual(docCount, 4)
    }

    func testQueryWithFilter() async throws {
        let store = try makeStore()

        let ids1 = try await store.addChunks(
            ["target text"],
            metadata: [ChunkMetadata(documentId: "doc-A", chunkIndex: 0, text: "target text")]
        )
        try await store.addChunks(
            ["other text"],
            metadata: [ChunkMetadata(documentId: "doc-B", chunkIndex: 0, text: "other text")]
        )

        let targetId = ids1[0]
        let results = try await store.query(
            "target text",
            k: 10,
            filter: { $0 == targetId }
        )

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, targetId)
    }
}
