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

    /// Monotonic counter bumped on every committed mutation.
    /// ``save()`` snapshots this value before writing and only marks the
    /// store clean if no mutation landed while the write was in flight.
    private var mutationGeneration: UInt64 = 0

    /// The mutation generation last persisted to disk.
    private var savedGeneration: UInt64 = 0

    /// Whether the store has unsaved changes.
    private var isDirty: Bool { mutationGeneration != savedGeneration }

    /// Tail of the compound-operation chain (see ``serialized(_:)``).
    private var operationTail: Task<Void, Never>?

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

    // MARK: - Operation Serialization

    /// Runs `operation` after every previously enqueued compound operation
    /// has finished.
    ///
    /// Actors are reentrant: at every `await`, another call can interleave.
    /// Compound operations here are doubly exposed because every mutation
    /// touches *two* legs (dense + sparse) on two separate actors. Without
    /// serialization, a reentrant ``addChunks(_:metadata:)`` could land
    /// between ``save()``'s two leg writes, persisting a torn pair where one
    /// leg contains a document the other lacks; and a save interleaved with
    /// a mutation could mark the store clean while losing the write.
    /// Chaining keeps reads (``query(_:k:candidatePoolK:filter:)``)
    /// concurrent while guaranteeing mutators and saves never overlap.
    private func serialized<T: Sendable>(
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        let previous = operationTail
        let task = Task<T, Error> {
            await previous?.value
            return try await operation()
        }
        // Park the new tail. Errors are surfaced to the caller below and
        // must not break the chain for subsequent operations.
        operationTail = Task { _ = try? await task.value }
        return try await task.value
    }

    // MARK: - Document-Level Operations

    /// Adds text chunks with metadata, embedding each chunk and writing it to both legs.
    ///
    /// Runs serialized with other compound operations: it never interleaves
    /// with an in-flight ``save()`` or ``removeDocument(id:)``.
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

        return try await serialized {
            try await self.performAddChunks(chunks, metadata: metadata)
        }
    }

    /// Embeds and inserts chunks into both legs. Must only run inside ``serialized(_:)``.
    private func performAddChunks(
        _ chunks: [String],
        metadata: [ChunkMetadata]
    ) async throws -> [UUID] {
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

            // Bump the generation per committed chunk so even a partially
            // failed batch leaves the store marked dirty.
            documentMap[metadata[i].documentId, default: []].insert(id)
            mutationGeneration += 1
            ids.append(id)
        }

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
    ///
    /// Runs serialized with other compound operations: a concurrent
    /// ``addChunks(_:metadata:)`` for the same document cannot interleave
    /// with the removal loop, so freshly added chunks are never orphaned.
    @discardableResult
    public func removeDocument(id: String) async throws -> Int {
        try await serialized {
            try await self.performRemoveDocument(id: id)
        }
    }

    /// Removes a document's chunks from both legs. Must only run inside ``serialized(_:)``.
    private func performRemoveDocument(id: String) async throws -> Int {
        guard let uuids = documentMap[id] else {
            throw VectorStoreError.documentNotFound(id)
        }

        var removed = 0
        for uuid in uuids {
            if await index.remove(id: uuid) {
                removed += 1
                mutationGeneration += 1
            }
        }

        // Subtract only the snapshotted UUIDs rather than dropping the entry
        // wholesale: chunks inserted for the same document while the removal
        // loop was suspended must not be orphaned in either leg.
        documentMap[id]?.subtract(uuids)
        if documentMap[id]?.isEmpty == true {
            documentMap.removeValue(forKey: id)
        }
        return removed
    }

    // MARK: - Persistence

    /// Persists both legs and the document map to disk. Skips writes when clean.
    ///
    /// Runs serialized with other compound operations, so no mutation can
    /// land between the dense and sparse leg writes — the two files always
    /// describe the same document set. If the sparse write throws after the
    /// dense write succeeded, the store stays dirty and the next save
    /// rewrites both legs.
    public func save() async throws {
        try await serialized {
            try await self.performSave()
        }
    }

    /// Writes both legs and the document map. Must only run inside ``serialized(_:)``.
    private func performSave() async throws {
        guard isDirty else { return }

        // Snapshot the generation and document map before any suspension
        // point so the map written below matches the leg snapshots even if
        // a mutation were ever to interleave with this save.
        let generation = mutationGeneration
        let mapSnapshot = documentMap.mapValues { Array($0) }

        let fm = FileManager.default
        if !fm.fileExists(atPath: storageURL.path) {
            try fm.createDirectory(at: storageURL, withIntermediateDirectories: true)
        }

        try await _dense.save(to: storageURL.appendingPathComponent("index.pxkt"))
        try await _sparse.save(to: storageURL.appendingPathComponent("index.pxbm"))

        // `.atomic` so a crash mid-write cannot leave a truncated/corrupt
        // hybrid.json behind.
        let mapURL = storageURL.appendingPathComponent("hybrid.json")
        let mapData = try JSONEncoder().encode(mapSnapshot)
        try mapData.write(to: mapURL, options: .atomic)

        // Only mark clean if no mutation landed while the files were being
        // written; otherwise the store stays dirty and the next save
        // persists the newer state.
        if mutationGeneration == generation {
            savedGeneration = generation
        }
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
