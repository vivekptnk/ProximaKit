// VectorStore.swift
// ProximaKit
//
// A named, persistent vector collection with auto-embedding.
// Bundles an HNSWIndex with an embedder and document-level operations.
// Part of the ProximaKit ↔ Lumen RAG integration (ADR-006).

import Foundation

/// A named, persistent vector collection with auto-embedding.
///
/// `VectorStore` wraps an ``HNSWIndex`` with a ``TextEmbedder``,
/// providing document-level operations (add chunks, query by text,
/// remove by document ID) and automatic persistence.
///
/// ```swift
/// let store = try VectorStore(
///     name: "my-notebook",
///     embedder: myEmbedder,
///     storageDirectory: appSupportURL
/// )
/// let ids = try await store.addChunks(texts, metadata: metas)
/// let results = try await store.query("key findings", k: 5)
/// try await store.save()
/// ```
public actor VectorStore {

    // MARK: - Properties

    /// The name of this collection (used for persistence directory).
    public nonisolated let name: String

    /// The underlying vector index.
    public let index: HNSWIndex

    /// The embedding provider used for text operations.
    private let embedder: any TextEmbedder

    /// Directory where this store persists its index.
    public nonisolated let storageURL: URL

    /// Tracks which document IDs map to which vector UUIDs.
    /// Rebuilt from metadata on load.
    private var documentMap: [String: Set<UUID>] = [:]

    /// Whether the store has unsaved changes.
    private var isDirty: Bool = false

    // MARK: - Initialization

    /// Creates a new vector store, or loads an existing one from disk.
    ///
    /// If a persisted index exists at the storage path, it is loaded.
    /// Otherwise, a fresh index is created with the given configuration.
    ///
    /// - Parameters:
    ///   - name: The collection name (used as the directory name within `storageDirectory`).
    ///   - embedder: The text embedder to use for auto-embedding operations.
    ///   - storageDirectory: The parent directory for persistence. A subdirectory
    ///     named `name` is created automatically.
    ///   - metric: The distance metric for the underlying index.
    ///   - config: HNSW configuration parameters.
    public init(
        name: String,
        embedder: any TextEmbedder,
        storageDirectory: URL,
        metric: any DistanceMetric = CosineDistance(),
        config: HNSWConfiguration = HNSWConfiguration()
    ) throws {
        self.name = name
        self.embedder = embedder

        let storeDir = storageDirectory.appendingPathComponent(name)
        self.storageURL = storeDir

        let indexURL = storeDir.appendingPathComponent("index.pxkt")

        if FileManager.default.fileExists(atPath: indexURL.path) {
            self.index = try HNSWIndex.load(from: indexURL)
        } else {
            self.index = HNSWIndex(
                dimension: embedder.dimension,
                metric: metric,
                config: config
            )
        }
    }

    /// Creates a VectorStore wrapping an existing index (for testing or advanced use).
    ///
    /// - Parameters:
    ///   - name: The collection name.
    ///   - index: An existing HNSWIndex to wrap.
    ///   - embedder: The text embedder to use.
    ///   - storageDirectory: The parent directory for persistence.
    public init(
        name: String,
        index: HNSWIndex,
        embedder: any TextEmbedder,
        storageDirectory: URL
    ) {
        self.name = name
        self.index = index
        self.embedder = embedder
        self.storageURL = storageDirectory.appendingPathComponent(name)
    }

    // MARK: - Document-Level Operations

    /// Adds text chunks with metadata, embedding them automatically.
    ///
    /// Each chunk is embedded via the store's ``TextEmbedder`` and added
    /// to the underlying index with its metadata encoded as JSON.
    ///
    /// - Parameters:
    ///   - chunks: The text strings to embed and store.
    ///   - metadata: Metadata for each chunk. Must have the same count as `chunks`.
    /// - Returns: The UUIDs assigned to each chunk.
    /// - Throws: ``VectorStoreError/chunkMetadataMismatch(chunks:metadata:)`` if counts differ.
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

        // Batch embed all chunks.
        let vectors = try await embedder.embedBatch(chunks)

        let encoder = JSONEncoder()
        var ids: [UUID] = []
        ids.reserveCapacity(chunks.count)

        for (i, vector) in vectors.enumerated() {
            let id = UUID()
            let metaData = try encoder.encode(metadata[i])

            try await index.add(vector, id: id, metadata: metaData)

            // Track document → UUID mapping.
            let docId = metadata[i].documentId
            documentMap[docId, default: []].insert(id)

            ids.append(id)
        }

        isDirty = true
        return ids
    }

    /// Queries the store with a text string.
    ///
    /// Embeds the query text, searches the underlying index, and returns
    /// results with their stored metadata.
    ///
    /// - Parameters:
    ///   - text: The query text to embed and search for.
    ///   - k: The number of results to return.
    ///   - efSearch: Optional HNSW beam width override.
    ///   - filter: Optional predicate to filter results by UUID.
    /// - Returns: Up to `k` results, sorted by distance (ascending).
    public func query(
        _ text: String,
        k: Int = 10,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) async throws -> [SearchResult] {
        let queryVector = try await embedder.embed(text)
        return await index.search(
            query: queryVector,
            k: k,
            efSearch: efSearch,
            filter: filter
        )
    }

    /// Removes all chunks associated with a document ID.
    ///
    /// - Parameter id: The document identifier whose chunks should be removed.
    /// - Returns: The number of chunks removed.
    /// - Throws: ``VectorStoreError/documentNotFound(_:)`` if the document ID
    ///   is not tracked by this store.
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

    /// Persists the current state to disk.
    ///
    /// Creates the storage directory if needed and saves the index
    /// in ProximaKit's binary format. Also saves the document map
    /// as a separate JSON file for fast reload.
    ///
    /// Skips the write if no changes have been made since the last save.
    public func save() async throws {
        guard isDirty else { return }

        let fm = FileManager.default
        if !fm.fileExists(atPath: storageURL.path) {
            try fm.createDirectory(at: storageURL, withIntermediateDirectories: true)
        }

        // Save the HNSW index.
        let indexURL = storageURL.appendingPathComponent("index.pxkt")
        try await index.save(to: indexURL)

        // Save the document map for fast reload.
        let mapURL = storageURL.appendingPathComponent("docmap.json")
        let mapData = try JSONEncoder().encode(
            documentMap.mapValues { Array($0) }
        )
        try mapData.write(to: mapURL)

        isDirty = false
    }

    /// Loads the document map from disk if it exists.
    ///
    /// Call this after loading from persistence to restore
    /// the document → UUID mapping without scanning all metadata.
    public func loadDocumentMap() throws {
        let mapURL = storageURL.appendingPathComponent("docmap.json")
        guard FileManager.default.fileExists(atPath: mapURL.path) else {
            return
        }
        let data = try Data(contentsOf: mapURL)
        let decoded = try JSONDecoder().decode([String: [UUID]].self, from: data)
        documentMap = decoded.mapValues { Set($0) }
    }

    // MARK: - Accessors

    /// The number of chunks currently in the store.
    public var count: Int {
        get async { await index.count }
    }

    /// The number of live (non-tombstoned) chunks.
    public var liveCount: Int {
        get async { await index.liveCount }
    }

    /// The set of document IDs currently tracked by this store.
    public var documentIds: Set<String> {
        Set(documentMap.keys)
    }

    /// Returns the number of chunks for a given document ID.
    public func chunkCount(forDocument id: String) -> Int {
        documentMap[id]?.count ?? 0
    }

    /// Whether the store has unsaved changes.
    public var hasUnsavedChanges: Bool { isDirty }
}
