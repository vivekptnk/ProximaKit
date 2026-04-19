// HybridVectorStore.swift
// ProximaKit
//
// Hybrid (BM25 + dense) document store. Sibling of ``VectorStore`` — we do NOT
// mutate VectorStore, per CHA-107, so the v1.1 contract stays frozen and
// hybrid consumers opt in explicitly.
//
// Wraps a ``HybridIndex`` plus a ``TextEmbedder`` so callers pass raw chunk
// text and query strings; embeddings are computed internally. Persistence
// mirrors VectorStore: HNSW goes to `index.pxkt`, sparse goes to `index.pxbm`,
// and a small `hybrid.json` carries the document map.

import Foundation

/// A hybrid (dense + sparse) vector store with auto-embedding and persistence.
///
/// ``HybridVectorStore`` is the hybrid-retrieval analog of ``VectorStore``.
/// Each chunk is added to both legs of a ``HybridIndex`` under the same UUID;
/// queries run against both legs and are fused server-side.
///
/// ```swift
/// let store = try HybridVectorStore(
///     name: "notebook",
///     embedder: myEmbedder,
///     storageDirectory: appSupportURL
/// )
/// try await store.addChunks(texts, metadata: metas)
/// let hits = try await store.query("lanthanides", k: 10)
/// try await store.save()
/// ```
///
/// Lumen opt-in path: swap `VectorStore` for `HybridVectorStore` in the RAG
/// pipeline. Same public API shape (`addChunks`, `query`, `removeDocument`,
/// `save`) so the integration site is a one-line change.
public actor HybridVectorStore {

    // MARK: - Properties

    /// Collection name (used as the persistence subdirectory).
    public nonisolated let name: String

    /// Storage directory for this store.
    public nonisolated let storageURL: URL

    /// The underlying hybrid index. Exposed for advanced consumers that want
    /// to inspect per-leg counts or swap fusion strategies.
    public let index: HybridIndex

    /// Direct handle to the dense leg. Use for advanced diagnostics.
    public nonisolated var dense: HNSWIndex { _dense }
    /// Direct handle to the sparse leg. Use for advanced diagnostics.
    public nonisolated var sparse: SparseIndex { _sparse }

    private let _dense: HNSWIndex
    private let _sparse: SparseIndex

    private let embedder: any TextEmbedder
    private var documentMap: [String: Set<UUID>] = [:]
    private var isDirty: Bool = false

    // MARK: - Initialization

    /// Creates (or restores) a hybrid vector store at the given storage directory.
    ///
    /// If both `index.pxkt` and `index.pxbm` exist under `storageDirectory/name/`,
    /// they are loaded. Otherwise fresh empty legs are constructed.
    public init(
        name: String,
        embedder: any TextEmbedder,
        storageDirectory: URL,
        metric: any DistanceMetric = CosineDistance(),
        hnswConfig: HNSWConfiguration = HNSWConfiguration(),
        bm25Config: BM25Configuration = BM25Configuration(),
        tokenizer: any BM25Tokenizer = DefaultBM25Tokenizer(),
        fusion: HybridFusionStrategy = .rrf()
    ) throws {
        self.name = name
        self.embedder = embedder

        let storeDir = storageDirectory.appendingPathComponent(name)
        self.storageURL = storeDir

        let denseURL = storeDir.appendingPathComponent("index.pxkt")
        let sparseURL = storeDir.appendingPathComponent("index.pxbm")

        let dense: HNSWIndex
        if FileManager.default.fileExists(atPath: denseURL.path) {
            dense = try HNSWIndex.load(from: denseURL)
        } else {
            dense = HNSWIndex(
                dimension: embedder.dimension,
                metric: metric,
                config: hnswConfig
            )
        }

        let sparse: SparseIndex
        if FileManager.default.fileExists(atPath: sparseURL.path) {
            sparse = try SparseIndex.load(from: sparseURL, tokenizer: tokenizer)
        } else {
            sparse = SparseIndex(tokenizer: tokenizer, configuration: bm25Config)
        }

        self._dense = dense
        self._sparse = sparse
        self.index = HybridIndex(dense: dense, sparse: sparse, fusion: fusion)
    }

    /// Creates a HybridVectorStore wrapping an existing `HybridIndex` (testing hook).
    ///
    /// The dense and sparse legs of the provided index must be an `HNSWIndex`
    /// and a `SparseIndex` respectively; passing other conformers through this
    /// ctor is unsupported.
    public init(
        name: String,
        index: HybridIndex,
        dense: HNSWIndex,
        sparse: SparseIndex,
        embedder: any TextEmbedder,
        storageDirectory: URL
    ) {
        self.name = name
        self.index = index
        self._dense = dense
        self._sparse = sparse
        self.embedder = embedder
        self.storageURL = storageDirectory.appendingPathComponent(name)
    }

    // MARK: - Document-Level Operations

    /// Adds text chunks with metadata, embedding each chunk and writing it to both legs.
    @discardableResult
    public func addChunks(
        _ chunks: [String],
        metadata: [ChunkMetadata]
    ) async throws -> [UUID] {
        guard !chunks.isEmpty else {
            throw VectorStoreError.emptyChunks
        }
        guard chunks.count == metadata.count else {
            throw VectorStoreError.chunkMetadataMismatch(
                chunks: chunks.count,
                metadata: metadata.count
            )
        }

        let vectors = try await embedder.embedBatch(chunks)

        let encoder = JSONEncoder()
        var ids: [UUID] = []
        ids.reserveCapacity(chunks.count)

        for (i, vector) in vectors.enumerated() {
            let id = UUID()
            let metaData = try encoder.encode(metadata[i])

            try await index.add(
                text: chunks[i],
                vector: vector,
                id: id,
                metadata: metaData
            )

            documentMap[metadata[i].documentId, default: []].insert(id)
            ids.append(id)
        }

        isDirty = true
        return ids
    }

    /// Queries the store with a text string. Embeds once, fans out across both legs.
    public func query(
        _ text: String,
        k: Int = 10,
        candidatePoolK: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) async throws -> [SearchResult] {
        let queryVector = try await embedder.embed(text)
        return await index.search(
            queryText: text,
            queryVector: queryVector,
            k: k,
            candidatePoolK: candidatePoolK,
            filter: filter
        )
    }

    /// Removes all chunks associated with a document ID from both legs.
    @discardableResult
    public func removeDocument(id: String) async throws -> Int {
        guard let uuids = documentMap[id] else {
            throw VectorStoreError.documentNotFound(id)
        }

        var removed = 0
        for uuid in uuids {
            if await index.remove(id: uuid) {
                removed += 1
            }
        }

        documentMap.removeValue(forKey: id)
        if removed > 0 {
            isDirty = true
        }
        return removed
    }

    // MARK: - Persistence

    /// Persists both legs and the document map to disk. Skips writes when clean.
    public func save() async throws {
        guard isDirty else { return }

        let fm = FileManager.default
        if !fm.fileExists(atPath: storageURL.path) {
            try fm.createDirectory(at: storageURL, withIntermediateDirectories: true)
        }

        try await _dense.save(to: storageURL.appendingPathComponent("index.pxkt"))
        try await _sparse.save(to: storageURL.appendingPathComponent("index.pxbm"))

        let mapURL = storageURL.appendingPathComponent("hybrid.json")
        let mapData = try JSONEncoder().encode(
            documentMap.mapValues { Array($0) }
        )
        try mapData.write(to: mapURL)

        isDirty = false
    }

    /// Restores the document map from disk after a cold-load.
    public func loadDocumentMap() throws {
        let mapURL = storageURL.appendingPathComponent("hybrid.json")
        guard FileManager.default.fileExists(atPath: mapURL.path) else {
            return
        }
        let data = try Data(contentsOf: mapURL)
        let decoded = try JSONDecoder().decode([String: [UUID]].self, from: data)
        documentMap = decoded.mapValues { Set($0) }
    }

    // MARK: - Accessors

    public var count: Int {
        get async { await _dense.count }
    }

    public var liveCount: Int {
        get async { await _dense.liveCount }
    }

    public var sparseCount: Int {
        get async { await _sparse.count }
    }

    public var documentIds: Set<String> {
        Set(documentMap.keys)
    }

    public func chunkCount(forDocument id: String) -> Int {
        documentMap[id]?.count ?? 0
    }

    public var hasUnsavedChanges: Bool { isDirty }
}
