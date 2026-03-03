// SearchResult.swift
// ProximaKit
//
// The value returned from index searches.

import Foundation

/// A single result from a vector similarity search.
///
/// Contains the ID of the matched vector, its distance from the query,
/// and optional metadata attached at insertion time.
///
/// Results are ordered by distance (ascending) — the first result is the
/// closest match. The distance value depends on the metric used:
/// - Cosine: 0 = identical, 2 = opposite
/// - Euclidean: 0 = identical, higher = farther
/// - Dot product: most negative = most similar
///
/// ```swift
/// let results = try await index.search(query: q, k: 5)
/// for result in results {
///     print("\(result.id): distance \(result.distance)")
/// }
/// ```
public struct SearchResult: Sendable, Identifiable {
    /// The UUID of the matched vector (assigned at insertion).
    public let id: UUID

    /// The distance from the query vector. Lower = more similar.
    public let distance: Float

    /// Optional metadata attached when the vector was added.
    /// Stored as opaque data since we can't make `any Codable` Sendable.
    public let metadata: Data?

    public init(id: UUID, distance: Float, metadata: Data? = nil) {
        self.id = id
        self.distance = distance
        self.metadata = metadata
    }

    /// Decodes the metadata as a specific Codable type.
    ///
    /// - Returns: The decoded value, or nil if no metadata or decoding fails.
    public func decodeMetadata<T: Decodable>(as type: T.Type) -> T? {
        guard let data = metadata else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }
}

extension SearchResult: Equatable {
    public static func == (lhs: SearchResult, rhs: SearchResult) -> Bool {
        lhs.id == rhs.id && lhs.distance == rhs.distance
    }
}

extension SearchResult: Comparable {
    /// Sorts by distance ascending (closest first).
    public static func < (lhs: SearchResult, rhs: SearchResult) -> Bool {
        lhs.distance < rhs.distance
    }
}
