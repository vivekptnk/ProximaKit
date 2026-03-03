// DistanceMetric.swift
// ProximaKit
//
// Protocol for pluggable distance functions.
// Indices are generic over their distance metric, so you can swap
// cosine for L2 without changing any search code.

import Accelerate

/// A protocol for measuring the distance (or dissimilarity) between two vectors.
///
/// Distance metrics are used by index types to rank search results.
/// Lower distance = more similar. All conforming types must be `Sendable`
/// so they can be stored inside actor-isolated indices.
///
/// ProximaKit ships three metrics:
/// - ``CosineDistance``: 1 - cosine_similarity. Best for text embeddings.
/// - ``EuclideanDistance``: L2 (straight-line) distance.
/// - ``DotProductDistance``: Negative dot product (for pre-normalized vectors).
///
/// ```swift
/// let metric = CosineDistance()
/// let d = metric.distance(vectorA, vectorB)  // 0.0 = identical, 2.0 = opposite
/// ```
public protocol DistanceMetric: Sendable {
    /// Computes the distance between two vectors of equal dimension.
    ///
    /// - Parameters:
    ///   - a: The first vector.
    ///   - b: The second vector.
    /// - Returns: A distance value where lower = more similar.
    func distance(_ a: Vector, _ b: Vector) -> Float
}

// ── Cosine Distance ──────────────────────────────────────────────────

/// Cosine distance: `1 - cosine_similarity(a, b)`.
///
/// Range: [0, 2] where 0 = identical direction, 1 = perpendicular, 2 = opposite.
///
/// This is the default metric for text embeddings (BERT, sentence-transformers,
/// NLEmbedding) because it measures direction regardless of magnitude.
/// We convert similarity → distance so that "lower = more similar" is consistent
/// across all metrics.
public struct CosineDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        // cosine_similarity returns [-1, 1].
        // Subtracting from 1 flips it to [0, 2] where 0 = identical.
        1.0 - a.cosineSimilarity(b)
    }
}

// ── Euclidean (L2) Distance ──────────────────────────────────────────

/// Euclidean (L2) distance: the straight-line distance between two points.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Good for when absolute position matters (image features, geographic
/// coordinates). Less common for text embeddings because it's sensitive
/// to vector magnitude.
public struct EuclideanDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        a.l2Distance(b)
    }
}

// ── Dot Product Distance ─────────────────────────────────────────────

/// Negative dot product distance: `-dot(a, b)`.
///
/// For **pre-normalized** (unit) vectors, this is equivalent to cosine distance
/// but faster — it skips the magnitude computation.
///
/// Range: [-1, 1] for unit vectors, where -1 = identical, 1 = opposite.
/// (Negated so that lower = more similar, consistent with other metrics.)
///
/// **Important:** Only use this with normalized vectors. For unnormalized vectors,
/// use ``CosineDistance`` instead.
public struct DotProductDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        // Negate because higher dot product = more similar,
        // but our convention is lower distance = more similar.
        -a.dot(b)
    }
}
