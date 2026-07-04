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
/// try await store.loadDocumentMap()  // when reopening a persisted store
/// try await store.addChunks(texts, metadata: metas)
/// let hits = try await store.query("lanthanides", k: 10)
/// try await store.save()
/// ```
///
/// Lumen opt-in path: swap `VectorStore` for `HybridVectorStore` in the RAG
/// pipeline. Same public API shape (`addChunks`, `query`, `removeDocument`,
/// `save`) so the integration site is a one-line change.
///
/// ## Journaled vs. non-journaled
///
/// `HybridVectorStore` supports two lifecycles:
///
/// 1. **Non-journaled** (shown above): the synchronous
///    ``init(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:)``
///    or ``init(name:index:dense:sparse:embedder:storageDirectory:)``
///    initializers, paired with ``save()`` and ``loadDocumentMap()``. Each
///    ``save()`` rewrites both leg snapshots (O(corpus) per save), and
///    reopening a persisted store requires an explicit ``loadDocumentMap()``
///    call to restore the document → UUID map.
/// 2. **Journaled** (ADR-013): the async static
///    ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
///    factory. The dense leg streams mutations to a `.pxwal` sidecar, giving
///    O(change) saves via ``checkpoint()``/``needsCheckpoint(policy:)``, and
///    both the sparse leg and the document map are rebuilt automatically
///    from the recovered dense leg on open — no manual ``loadDocumentMap()``
///    call needed.
///
/// Prefer
/// ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
/// for continuous-mutation (agentic) workloads with frequent small writes.
/// The synchronous initializers remain fully supported for batch-style
/// workloads that build once and save occasionally.
///
/// ## Topics
///
/// ### Journaled Lifecycle
/// - ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
/// - ``save()``
/// - ``checkpoint()``
/// - ``needsCheckpoint(policy:)``
///
/// ### Non-journaled Lifecycle
/// - ``init(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:)``
/// - ``init(name:index:dense:sparse:embedder:storageDirectory:)``
/// - ``loadDocumentMap()``
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

    /// Whether the **dense** leg journals mutations to a `.pxwal` sidecar
    /// (ADR-013 store-level journaling). `false` for every store built through
    /// the two historical initializers. Only
    /// ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
    /// sets it `true`. The sparse leg has no WAL codec; under journaling it is
    /// reconstructed from the dense leg's metadata on open (see that factory).
    private let journaling: Bool

    /// WAL durability dial for the dense leg, used only when ``journaling``.
    private let durability: WALDurability

    /// Dense base snapshot path (`index.pxkt`).
    private var denseURL: URL { storageURL.appendingPathComponent("index.pxkt") }
    /// Dense write-ahead-log sidecar path (`index.pxwal`).
    private var walURL: URL { storageURL.appendingPathComponent("index.pxwal") }
    /// Sparse leg snapshot path (`index.pxbm`).
    private var sparseURL: URL { storageURL.appendingPathComponent("index.pxbm") }

    // MARK: - Initialization

    /// Creates (or restores) a hybrid vector store at the given storage directory.
    ///
    /// Each leg is restored **independently** from `storageDirectory/name/`:
    /// the dense leg loads from `index.pxkt` if that file exists (otherwise a
    /// fresh ``HNSWIndex`` is created), and the sparse leg loads from
    /// `index.pxbm` if that file exists (otherwise a fresh ``SparseIndex`` is
    /// created). If only one file is present, that leg is restored and the
    /// other starts empty.
    ///
    /// - Important: This initializer restores **only the index legs**. The
    ///   document → UUID map is persisted separately (`hybrid.json`) and is
    ///   NOT loaded here — call ``loadDocumentMap()`` after init when
    ///   reopening a persisted store. Until then, ``documentIds`` and
    ///   ``chunkCount(forDocument:)`` report an empty map, and
    ///   ``removeDocument(id:)`` throws
    ///   ``VectorStoreError/documentNotFound(_:)`` for documents that are
    ///   present in the legs. Vector-level queries work immediately.
    ///
    /// - Note: For continuous-mutation (agentic) workloads, prefer
    ///   ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
    ///   instead, which streams dense-leg mutations to a WAL for O(change)
    ///   saves and rebuilds the sparse leg and document map automatically.
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
        self.journaling = false
        self.durability = .everyBatch
    }

    /// Creates a HybridVectorStore wrapping an existing `HybridIndex` (testing hook).
    ///
    /// The dense and sparse legs of the provided index must be an `HNSWIndex`
    /// and a `SparseIndex` respectively; passing other conformers through this
    /// ctor is unsupported.
    ///
    /// For continuous-mutation (agentic) workloads, prefer the journaled
    /// ``open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
    /// factory instead, which gives O(change) saves and rebuilds the sparse
    /// leg and document map automatically on open.
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
        self.journaling = false
        self.durability = .everyBatch
    }

    /// Designated initializer for the journaled open path (private — journaling
    /// is established only through the async ``open`` factory, which recovers
    /// the dense leg from its WAL and rebuilds the sparse leg + document map
    /// before handing the fully-consistent state here).
    private init(
        name: String,
        index: HybridIndex,
        dense: HNSWIndex,
        sparse: SparseIndex,
        embedder: any TextEmbedder,
        storageDirectory: URL,
        journaling: Bool,
        durability: WALDurability,
        documentMap: [String: Set<UUID>]
    ) {
        self.name = name
        self.index = index
        self._dense = dense
        self._sparse = sparse
        self.embedder = embedder
        self.storageURL = storageDirectory.appendingPathComponent(name)
        self.journaling = journaling
        self.durability = durability
        self.documentMap = documentMap
    }

    // MARK: - Journaled Lifecycle (ADR-013 store-level journaling, opt-in)

    /// Opens a **journaled** hybrid store. The dense leg streams mutations to a
    /// `.pxwal` sidecar; the sparse (BM25) leg — which has no WAL codec — and
    /// the document map are **both reconstructed from the dense leg** on open.
    ///
    /// ## Why this is crash-consistent (the blocker bar)
    /// Each chunk's ``ChunkMetadata`` (carrying its `documentId` **and** its
    /// original `text`) is journaled into the dense leg's WAL record. On open,
    /// after the dense WAL is replayed to its longest valid prefix, the sparse
    /// leg is rebuilt by re-adding every live dense node's text, and the
    /// document map by indexing every live node's `documentId`. The dense
    /// index + WAL are therefore the **single source of truth**: the sparse
    /// leg and the map are pure projections of it, so after any crash every
    /// recovered id is searchable in *both* legs and mapped — a dense vector
    /// without a sparse entry or a mapping, or vice versa, is structurally
    /// impossible. A stale `index.pxbm` / `hybrid.json` left by a prior
    /// ``checkpoint()`` is ignored here.
    ///
    /// - Note: The sparse leg is reconstructed from `ChunkMetadata.text`, which
    ///   *is* the chunk text by that field's definition ("the original text of
    ///   this chunk"). In normal use it equals the string passed to
    ///   ``addChunks(_:metadata:)`` and the rebuilt sparse leg is identical to
    ///   the live one; even if a caller diverged the two, the blocker-bar
    ///   invariant (every id present in both legs and the map) still holds.
    ///
    /// - Parameter durability: dense-leg WAL flush policy (default `.everyBatch`).
    public static func open(
        name: String,
        embedder: any TextEmbedder,
        storageDirectory: URL,
        metric: any DistanceMetric = CosineDistance(),
        hnswConfig: HNSWConfiguration = HNSWConfiguration(),
        bm25Config: BM25Configuration = BM25Configuration(),
        tokenizer: any BM25Tokenizer = DefaultBM25Tokenizer(),
        fusion: HybridFusionStrategy = .rrf(),
        durability: WALDurability = .everyBatch
    ) async throws -> HybridVectorStore {
        let storeDir = storageDirectory.appendingPathComponent(name)
        let denseURL = storeDir.appendingPathComponent("index.pxkt")
        let walURL = storeDir.appendingPathComponent("index.pxwal")

        let dense: HNSWIndex
        if FileManager.default.fileExists(atPath: denseURL.path) {
            dense = try await HNSWIndex.open(baseURL: denseURL, walURL: walURL, durability: durability)
        } else {
            try FileManager.default.createDirectory(at: storeDir, withIntermediateDirectories: true)
            let fresh = HNSWIndex(dimension: embedder.dimension, metric: metric, config: hnswConfig)
            try await fresh.checkpoint(baseURL: denseURL, walURL: walURL, durability: durability)
            dense = fresh
        }

        // Rebuild the sparse leg and the document map from the recovered dense
        // leg — the single source of truth. Order is dense node order; the
        // sparse leg's internal numbering is UUID-keyed, so this is
        // deterministic and externally invisible.
        let sparse = SparseIndex(tokenizer: tokenizer, configuration: bm25Config)
        let decoder = JSONDecoder()
        var map: [String: Set<UUID>] = [:]
        for entry in await dense.liveEntries() {
            guard let data = entry.metadata,
                  let meta = try? decoder.decode(ChunkMetadata.self, from: data) else { continue }
            try await sparse.add(text: meta.text, id: entry.id, metadata: data)
            map[meta.documentId, default: []].insert(entry.id)
        }

        let index = HybridIndex(dense: dense, sparse: sparse, fusion: fusion)
        return HybridVectorStore(
            name: name,
            index: index,
            dense: dense,
            sparse: sparse,
            embedder: embedder,
            storageDirectory: storageDirectory,
            journaling: true,
            durability: durability,
            documentMap: map
        )
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

        // ── Journaled path (ADR-013): only the dense leg is durable-per-mutation
        // via its WAL; save() flushes that WAL. The sparse leg and the document
        // map are NOT persisted here — both are rebuilt from the dense leg on the
        // next open, so neither can fall behind (or ahead of) the WAL and present
        // a torn pair. This is what makes the sparse-leg-has-no-WAL asymmetry safe.
        if journaling {
            let generation = mutationGeneration
            try await _dense.syncJournal()
            if mutationGeneration == generation {
                savedGeneration = generation
            }
            return
        }

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

    /// Folds the dense WAL into a fresh base and refreshes both legs' on-disk
    /// caches. A full ``save()`` for a non-journaled store.
    ///
    /// Under journaling, ``save()`` is a cheap dense-WAL flush; `checkpoint()`
    /// is the periodic fold: it checkpoints the dense leg (compact → new v3
    /// base → `F_FULLFSYNC` → WAL reset), then rewrites the `index.pxbm` sparse
    /// snapshot and the `hybrid.json` map cache so a non-journaled reload still
    /// works. A crash between the dense commit and the cache rewrites is
    /// harmless: journaled ``open`` rebuilds both the sparse leg and the map
    /// from the dense leg, ignoring the stale caches.
    public func checkpoint() async throws {
        try await serialized {
            try await self.performCheckpoint()
        }
    }

    /// Whether the dense WAL warrants a ``checkpoint()`` under `policy`.
    /// Always `false` for a non-journaled store.
    public func needsCheckpoint(policy: WALCheckpointPolicy = WALCheckpointPolicy()) async -> Bool {
        guard journaling else { return false }
        return await _dense.needsCheckpoint(policy: policy)
    }

    /// Performs the fold. Must only run inside ``serialized(_:)``.
    private func performCheckpoint() async throws {
        guard journaling else {
            try await performSave()
            return
        }

        let generation = mutationGeneration
        let mapSnapshot = documentMap.mapValues { Array($0) }

        // Dense commit point: fold the WAL into a fresh v3 base + reset the WAL.
        try await _dense.checkpoint(baseURL: denseURL, walURL: walURL, durability: durability)

        // Refresh derived caches for the non-journaled reload path. Both are
        // rebuilt from the dense leg on a journaled open, so a crash before
        // they land cannot corrupt recovery.
        try await _sparse.save(to: sparseURL)
        let mapURL = storageURL.appendingPathComponent("hybrid.json")
        let mapData = try JSONEncoder().encode(mapSnapshot)
        try mapData.write(to: mapURL, options: .atomic)

        if mutationGeneration == generation {
            savedGeneration = generation
        }
    }

    /// Restores the document map from disk after a cold-load.
    ///
    /// - Note: A no-op for a journaled store (opened via ``open``): its map and
    ///   sparse leg are authoritative projections of the dense leg rebuilt at
    ///   open, so loading a possibly-stale `hybrid.json` would be a regression.
    public func loadDocumentMap() throws {
        guard !journaling else { return }
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

    /// Every UUID tracked by the document map (union over all documents).
    /// Internal diagnostic hook: recovery tests assert this equals both legs'
    /// live-id sets, proving the dense leg, sparse leg, and map never diverge.
    internal var trackedIDs: Set<UUID> { Set(documentMap.values.flatMap { $0 }) }
}
