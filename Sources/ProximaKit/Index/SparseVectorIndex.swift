// SparseVectorIndex.swift
// ProximaKit
//
// Protocol for sparse (lexical) indices. Peer of VectorIndex, not a subtype —
// sparse search accepts a query *string* rather than a dense vector, so the
// method signatures diverge.
//
// Ships with ``SparseIndex`` (BM25) as the sole implementation in v1.4.

import Foundation

/// A searchable collection of documents keyed by lexical features (tokens).
///
/// Unlike ``VectorIndex``, which takes a dense query `Vector`, a sparse index
/// takes a query `String`, tokenizes it, and scores documents using a lexical
/// scoring function (BM25 in this library).
///
/// All conforming types must be actors, which guarantees thread safety and
/// serializes writes without consumers needing their own locks.
///
/// ```swift
/// let index = SparseIndex()
/// try await index.add(text: "hybrid search rocks", id: UUID())
/// let results = await index.search(query: "hybrid", k: 5)
/// ```
public protocol SparseVectorIndex: Actor {
    /// The number of live (non-tombstoned) documents currently searchable.
    var count: Int { get }

    /// Adds a document identified by `id` with the given text and optional metadata.
    ///
    /// If a document with this `id` already exists, the existing entry is
    /// replaced (remove + add), matching ``VectorIndex`` semantics.
    ///
    /// - Parameters:
    ///   - text: The raw text. Tokenization is performed internally by the index.
    ///   - id: A unique identifier for this document.
    ///   - metadata: Optional JSON-encoded metadata to store alongside the document.
    func add(text: String, id: UUID, metadata: Data?) throws

    /// Searches for the top-k documents most relevant to the query text.
    ///
    /// Results are ordered by ascending ``SearchResult/distance``. To match
    /// ``VectorIndex`` semantics, "more relevant" is expressed as "smaller
    /// distance" — implementations convert their internal score (higher = better)
    /// via negation.
    ///
    /// - Parameters:
    ///   - query: The raw query text.
    ///   - k: The number of results to return.
    ///   - filter: Optional predicate to exclude documents by ID.
    /// - Returns: Up to `k` results, sorted by distance (ascending).
    func search(
        query: String,
        k: Int,
        filter: (@Sendable (UUID) -> Bool)?
    ) -> [SearchResult]

    /// Removes a document by its ID.
    ///
    /// - Returns: `true` if a document was removed, `false` if the ID wasn't found.
    @discardableResult
    func remove(id: UUID) -> Bool
}

extension SparseVectorIndex {
    /// Searches with no filter.
    public func search(query: String, k: Int) -> [SearchResult] {
        search(query: query, k: k, filter: nil)
    }
}
