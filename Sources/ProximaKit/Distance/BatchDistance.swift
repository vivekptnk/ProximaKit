// BatchDistance.swift
// ProximaKit
//
// Batch distance computation using vDSP_mmul (matrix multiply).
// This is the critical optimization for search: computing query-vs-N-vectors
// in a single Accelerate call instead of N separate distance calls.

import Accelerate

/// Computes distances from a query vector to all vectors in a flat matrix,
/// using a single `vDSP_mmul` call for the dot products.
///
/// ## Why This Is Fast
///
/// Instead of N separate dot products (each a vDSP_dotpr call with function
/// overhead), we pack all vectors into a matrix and compute all dot products
/// in one `vDSP_mmul`. This lets Accelerate optimize memory access patterns
/// and use wider SIMD operations.
///
/// ## Memory Layout
///
/// The matrix must be row-major: vector i occupies elements
/// `[i * dimension ..< (i + 1) * dimension]` in the flat array.
///
/// ```swift
/// let distances = batchDotProducts(query: queryVec, matrix: flatVectors,
///                                  vectorCount: 10000, dimension: 384)
/// // distances[i] = dot(query, vectors[i])
/// ```
///
/// - Parameters:
///   - query: The query vector.
///   - matrix: A flat array of floats containing `vectorCount` vectors laid
///     out contiguously (row-major). Length must be `vectorCount * dimension`.
///   - vectorCount: The number of vectors in the matrix.
///   - dimension: The dimension of each vector.
/// - Returns: An array of `vectorCount` dot products.
public func batchDotProducts(
    query: Vector,
    matrix: [Float],
    vectorCount: Int,
    dimension: Int
) -> [Float] {
    precondition(query.dimension == dimension, "Query dimension mismatch")
    precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")

    // vDSP_mmul computes: C = A × B
    // We want: [N×1] result = [N×D] matrix × [D×1] query
    //
    // A = matrix      (N rows × D cols, row-major)
    // B = query       (D rows × 1 col)
    // C = result      (N rows × 1 col)

    var results = [Float](repeating: 0, count: vectorCount)

    matrix.withUnsafeBufferPointer { matPtr in
        query.components.withUnsafeBufferPointer { qPtr in
            vDSP_mmul(
                matPtr.baseAddress!, 1,     // A: the N×D matrix
                qPtr.baseAddress!, 1,       // B: the D×1 query column
                &results, 1,                // C: the N×1 result column
                vDSP_Length(vectorCount),    // M: rows of A (number of vectors)
                1,                           // N: cols of B (1, since query is a column)
                vDSP_Length(dimension)       // K: cols of A = rows of B (dimension)
            )
        }
    }

    return results
}

/// Computes batch distances using a given metric.
///
/// For dot-product-based metrics (cosine, dot product), this uses `vDSP_mmul`
/// for the dot product step and then applies metric-specific post-processing.
///
/// - Parameters:
///   - query: The query vector.
///   - vectors: The vectors to compare against.
///   - metric: The distance metric to use.
/// - Returns: An array of distances, one per vector.
public func batchDistances(
    query: Vector,
    vectors: [Vector],
    metric: some DistanceMetric
) -> [Float] {
    guard !vectors.isEmpty else { return [] }

    let dimension = query.dimension
    let count = vectors.count

    // Fast path: use matrix multiply for cosine and dot product metrics.
    // These are both based on dot products, so we can batch them.
    if metric is CosineDistance || metric is DotProductDistance {
        // Flatten vectors into a contiguous matrix (row-major)
        var matrix = [Float]()
        matrix.reserveCapacity(count * dimension)
        for v in vectors {
            precondition(v.dimension == dimension, "Dimension mismatch in batch")
            matrix.append(contentsOf: v.components)
        }

        // Compute all dot products in one call
        let dots = batchDotProducts(
            query: query, matrix: matrix,
            vectorCount: count, dimension: dimension
        )

        if metric is DotProductDistance {
            // DotProductDistance = -dot(a, b)
            var negated = [Float](repeating: 0, count: count)
            var minusOne: Float = -1.0
            dots.withUnsafeBufferPointer { dotsPtr in
                vDSP_vsmul(
                    dotsPtr.baseAddress!, 1,
                    &minusOne,
                    &negated, 1,
                    vDSP_Length(count)
                )
            }
            return negated
        }

        if metric is CosineDistance {
            // CosineDistance = 1 - dot(a,b) / (|a| * |b|)
            let queryMag = query.magnitude
            guard queryMag > 0 else {
                return [Float](repeating: 0, count: count)
            }
            return dots.enumerated().map { i, dot in
                let vecMag = vectors[i].magnitude
                guard vecMag > 0 else { return 0 }
                return 1.0 - dot / (queryMag * vecMag)
            }
        }
    }

    // Fallback: compute distances one at a time (for EuclideanDistance or custom metrics)
    return vectors.map { metric.distance(query, $0) }
}
