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
/// ProximaKit ships five metrics:
/// - ``CosineDistance``: 1 - cosine_similarity. Best for text embeddings.
/// - ``EuclideanDistance``: L2 (straight-line) distance.
/// - ``DotProductDistance``: Negative dot product (for pre-normalized vectors).
/// - ``ManhattanDistance``: L1 (taxicab) distance. Good for sparse data.
/// - ``HammingDistance``: Count of differing positions. For binary/quantized vectors.
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

// ── Manhattan (L1) Distance ────────────────────────────────────────────

/// Manhattan (L1) distance: the sum of absolute differences between components.
///
/// Range: [0, ∞) where 0 = identical vectors.
///
/// Also known as "taxicab" or "city block" distance. Good for sparse data,
/// grid-based problems, or when outliers should have proportional (not squared)
/// influence on distance. Uses vDSP for SIMD-accelerated computation.
public struct ManhattanDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        let count = vDSP_Length(a.dimension)

        // Step 1: Compute difference vector (b - a)
        var difference = [Float](repeating: 0, count: a.dimension)
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                vDSP_vsub(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &difference, 1, count)
            }
        }

        // Step 2: Absolute values in-place
        var absDiff = [Float](repeating: 0, count: a.dimension)
        vDSP_vabs(difference, 1, &absDiff, 1, count)

        // Step 3: Sum the absolute differences
        var sum: Float = 0
        vDSP_sve(absDiff, 1, &sum, count)

        return sum
    }
}

// ── Hamming Distance ───────────────────────────────────────────────────

/// Hamming distance: the count of positions where two vectors differ.
///
/// Range: [0, dimension] where 0 = identical vectors.
///
/// For float vectors, two components are considered equal if they have
/// the same bit pattern (exact equality). This metric is most useful for
/// binary or quantized vectors where elements take on discrete values
/// (e.g., 0.0/1.0 binary codes).
public struct HammingDistance: DistanceMetric, Sendable {
    public init() {}

    public func distance(_ a: Vector, _ b: Vector) -> Float {
        precondition(a.dimension == b.dimension,
                     "Dimension mismatch: \(a.dimension) vs \(b.dimension)")

        var count: Float = 0
        a.components.withUnsafeBufferPointer { aPtr in
            b.components.withUnsafeBufferPointer { bPtr in
                for i in 0..<a.dimension {
                    if aPtr[i] != bPtr[i] {
                        count += 1
                    }
                }
            }
        }
        return count
    }
}
