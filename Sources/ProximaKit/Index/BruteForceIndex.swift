// BruteForceIndex.swift
// ProximaKit
//
// Exact nearest-neighbor search. Compares query against every vector.
// O(n) per query, but uses batch vDSP_mmul for the distance computation
// so it's still fast for small datasets (< 1000 vectors).
//
// Primary use: accuracy baseline for HNSW, and production use for small collections.

import Foundation

/// An exact nearest-neighbor index that compares the query against every stored vector.
///
/// `BruteForceIndex` is an `actor`, guaranteeing thread safety for concurrent access.
/// Searches use batch distance computation via `vDSP_mmul` for performance.
///
/// Use this when:
/// - You have fewer than ~1000 vectors (exact search is fast enough)
/// - You need guaranteed 100% recall (no approximation)
/// - You want an accuracy baseline to validate HNSW results
///
/// ```swift
/// let index = BruteForceIndex(dimension: 384, metric: CosineDistance())
/// try await index.add(Vector([0.1, 0.2, ...]), id: UUID())
/// let results = try await index.search(query: queryVec, k: 10)
/// ```
public actor BruteForceIndex: VectorIndex {

    // ── Configuration ─────────────────────────────────────────────────

    /// The expected dimension for all vectors in this index.
    public let dimension: Int

    /// The distance metric used for ranking results.
    private let metric: any DistanceMetric

    // ── Storage ───────────────────────────────────────────────────────
    // Vectors are stored in parallel arrays for cache-friendly access.
    // The flat matrix enables batch distance via vDSP_mmul.

    /// Flat float array: all vectors laid out contiguously (row-major).
    /// vector[i] starts at offset i * dimension.
    private var vectorData: [Float] = []

    /// UUIDs for each stored vector, in insertion order.
    private var ids: [UUID] = []

    /// Metadata for each vector (JSON-encoded), in insertion order.
    private var metadataStore: [Data?] = []

    /// Fast lookup from UUID → index position.
    private var idToIndex: [UUID: Int] = [:]

    /// The number of vectors currently in the index.
    public var count: Int { ids.count }

    // ── Initialization ────────────────────────────────────────────────

    /// Creates a new empty brute-force index.
    ///
    /// - Parameters:
    ///   - dimension: The expected vector dimension. All added vectors must match.
    ///   - metric: The distance metric for ranking. Defaults to ``CosineDistance``.
    public init(dimension: Int, metric: any DistanceMetric = CosineDistance()) {
        precondition(dimension > 0, "Dimension must be positive")
        self.dimension = dimension
        self.metric = metric
    }

    /// Restores a BruteForce index from a persistence snapshot.
    public init(restoring snapshot: BruteForceSnapshot) {
        self.dimension = snapshot.dimension
        self.metric = snapshot.metricType.makeMetric()
        self.vectorData = snapshot.vectorData
        self.ids = snapshot.ids
        self.metadataStore = snapshot.metadataStore
        // Rebuild the reverse lookup.
        self.idToIndex = [:]
        for (i, id) in ids.enumerated() {
            idToIndex[id] = i
        }
    }

    // ── Persistence ───────────────────────────────────────────────────

    /// Returns a snapshot of this index's state for binary persistence.
    public func persistenceSnapshot() throws -> BruteForceSnapshot {
        guard let metricType = DistanceMetricType(metric: metric) else {
            throw PersistenceError.unserializableMetric
        }
        return BruteForceSnapshot(
            dimension: dimension,
            metricType: metricType,
            vectorData: vectorData,
            ids: ids,
            metadataStore: metadataStore
        )
    }

    /// Saves this index to a binary file.
    public func save(to url: URL) throws {
        let snapshot = try persistenceSnapshot()
        try PersistenceEngine.save(snapshot, to: url)
    }

    /// Loads a BruteForce index from a binary file.
    public static func load(from url: URL) throws -> BruteForceIndex {
        try PersistenceEngine.loadBruteForce(from: url)
    }

    // ── VectorIndex Conformance ───────────────────────────────────────

    public func add(_ vector: Vector, id: UUID, metadata: Data? = nil) throws {
        guard vector.dimension == dimension else {
            throw IndexError.dimensionMismatch(expected: dimension, got: vector.dimension)
        }

        // If this ID already exists, remove the old entry first.
        if idToIndex[id] != nil {
            _ = remove(id: id)
        }

        // Append vector data to the flat matrix.
        vectorData.append(contentsOf: vector.components)
        ids.append(id)
        metadataStore.append(metadata)
        idToIndex[id] = ids.count - 1
    }

    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        guard query.dimension == dimension else { return [] }
        guard !ids.isEmpty else { return [] }
        guard k > 0 else { return [] }

        // Use the flat-array overload directly — no intermediate Vector allocations.
        let distances = batchDistances(
            query: query, matrix: vectorData,
            vectorCount: ids.count, dimension: dimension, metric: metric
        )

        // Build results, applying filter if provided.
        var results: [SearchResult] = []
        results.reserveCapacity(min(k, ids.count))

        for (i, distance) in distances.enumerated() {
            let id = ids[i]
            if let filter = filter, !filter(id) {
                continue
            }
            results.append(SearchResult(
                id: id,
                distance: distance,
                metadata: metadataStore[i]
            ))
        }

        // Sort by distance (ascending) and take the top k.
        results.sort()
        if results.count > k {
            results = Array(results.prefix(k))
        }

        return results
    }

    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let index = idToIndex[id] else { return false }

        // Remove the vector data from the flat array.
        let dataStart = index * dimension
        let dataEnd = dataStart + dimension
        vectorData.removeSubrange(dataStart..<dataEnd)

        // Remove from parallel arrays.
        ids.remove(at: index)
        metadataStore.remove(at: index)

        // Rebuild the id→index mapping (indices shifted after removal).
        idToIndex.removeAll(keepingCapacity: true)
        for (i, storedId) in ids.enumerated() {
            idToIndex[storedId] = i
        }

        return true
    }
}

// ── Errors ────────────────────────────────────────────────────────────

/// Errors that can occur during index operations.
public enum IndexError: Error, LocalizedError {
    /// The vector's dimension doesn't match the index's expected dimension.
    case dimensionMismatch(expected: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expected, let got):
            return "Dimension mismatch: index expects \(expected), got \(got)"
        }
    }
}
