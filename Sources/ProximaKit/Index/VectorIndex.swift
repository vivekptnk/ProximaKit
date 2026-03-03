// VectorIndex.swift
// ProximaKit
//
// The protocol that all index types conform to.
// Actor-isolated for thread safety (see ADR-002).

import Foundation

/// A searchable collection of vectors with add, search, and remove operations.
///
/// All conforming types must be actors, which guarantees thread safety:
/// multiple tasks can search concurrently, and writes are serialized
/// automatically by the actor's queue.
///
/// ProximaKit ships two implementations:
/// - ``BruteForceIndex``: Exact search, O(n) per query. Good for < 1000 vectors.
/// - `HNSWIndex`: Approximate search, O(log n) per query. For production use.
///
/// ```swift
/// let index = BruteForceIndex(dimension: 384, metric: CosineDistance())
/// try await index.add(vector, id: UUID())
/// let results = try await index.search(query: queryVec, k: 10)
/// ```
public protocol VectorIndex: Actor {
    /// The dimension of vectors this index accepts.
    var dimension: Int { get }

    /// The number of vectors currently in the index.
    var count: Int { get }

    /// Adds a vector to the index.
    ///
    /// - Parameters:
    ///   - vector: The vector to add. Must match the index's dimension.
    ///   - id: A unique identifier for this vector.
    ///   - metadata: Optional metadata to store alongside the vector.
    ///     Encoded as JSON `Data` so it can cross actor boundaries safely.
    /// - Throws: If the vector dimension doesn't match the index dimension.
    func add(_ vector: Vector, id: UUID, metadata: Data?) throws

    /// Searches for the k nearest vectors to the query.
    ///
    /// - Parameters:
    ///   - query: The query vector. Must match the index's dimension.
    ///   - k: The number of results to return.
    ///   - efSearch: Optional beam width for HNSW. Ignored by BruteForceIndex.
    ///   - filter: Optional predicate to exclude vectors by ID.
    ///     Only vectors where `filter(id)` returns `true` are considered.
    /// - Returns: Up to `k` results, sorted by distance (ascending).
    func search(
        query: Vector,
        k: Int,
        efSearch: Int?,
        filter: (@Sendable (UUID) -> Bool)?
    ) -> [SearchResult]

    /// Removes a vector by its ID.
    ///
    /// - Parameter id: The ID of the vector to remove.
    /// - Returns: `true` if a vector was removed, `false` if the ID wasn't found.
    @discardableResult
    func remove(id: UUID) -> Bool
}

// Default parameter values via extension (protocols can't have defaults directly).
extension VectorIndex {
    /// Searches with default parameters (no efSearch override, no filter).
    public func search(
        query: Vector,
        k: Int
    ) -> [SearchResult] {
        search(query: query, k: k, efSearch: nil, filter: nil)
    }

    /// Searches with a filter but default efSearch.
    public func search(
        query: Vector,
        k: Int,
        filter: (@Sendable (UUID) -> Bool)?
    ) -> [SearchResult] {
        search(query: query, k: k, efSearch: nil, filter: filter)
    }
}
