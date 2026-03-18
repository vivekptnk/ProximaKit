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

/// Computes batch L2 (Euclidean) distances from a query to all vectors in a flat matrix.
///
/// Uses vDSP to compute: `distance[i] = sqrt( |query|² + |vec_i|² - 2·dot(query, vec_i) )`
///
/// This avoids N separate subtraction+sum-of-squares passes by exploiting
/// the identity: `|a - b|² = |a|² + |b|² - 2·dot(a, b)`.
///
/// - Parameters:
///   - query: The query vector.
///   - matrix: A flat row-major matrix of `vectorCount × dimension` floats.
///   - vectorCount: Number of vectors in the matrix.
///   - dimension: Dimension of each vector.
/// - Returns: An array of `vectorCount` Euclidean distances.
public func batchL2Distances(
    query: Vector,
    matrix: [Float],
    vectorCount: Int,
    dimension: Int
) -> [Float] {
    precondition(query.dimension == dimension, "Query dimension mismatch")
    precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")

    let count = vDSP_Length(vectorCount)
    let dim = vDSP_Length(dimension)

    // Step 1: |query|² (scalar, computed once)
    var queryNormSq: Float = 0
    query.components.withUnsafeBufferPointer { qPtr in
        vDSP_svesq(qPtr.baseAddress!, 1, &queryNormSq, vDSP_Length(query.dimension))
    }

    // Step 2: |vec_i|² for each vector (batch via vDSP_svesq per row)
    var vecNormsSq = [Float](repeating: 0, count: vectorCount)
    matrix.withUnsafeBufferPointer { matPtr in
        for i in 0..<vectorCount {
            vDSP_svesq(
                matPtr.baseAddress! + i * dimension, 1,
                &vecNormsSq[i],
                dim
            )
        }
    }

    // Step 3: dot(query, vec_i) for all vectors via vDSP_mmul
    let dots = batchDotProducts(query: query, matrix: matrix,
                                vectorCount: vectorCount, dimension: dimension)

    // Step 4: |a - b|² = |a|² + |b|² - 2·dot(a,b), then sqrt
    // All via vDSP: vecNormsSq + queryNormSq - 2*dots
    var results = [Float](repeating: 0, count: vectorCount)

    // results = vecNormsSq + queryNormSq
    var qNorm = queryNormSq
    vDSP_vsadd(vecNormsSq, 1, &qNorm, &results, 1, count)

    // results = results - 2*dots
    var minusTwo: Float = -2.0
    var scaledDots = [Float](repeating: 0, count: vectorCount)
    dots.withUnsafeBufferPointer { dotsPtr in
        vDSP_vsmul(dotsPtr.baseAddress!, 1, &minusTwo, &scaledDots, 1, count)
    }
    vDSP_vadd(results, 1, scaledDots, 1, &results, 1, count)

    // Clamp negative values (floating-point rounding) and sqrt
    for i in 0..<vectorCount {
        results[i] = sqrt(max(results[i], 0))
    }

    return results
}

/// Computes batch magnitudes for vectors in a flat row-major matrix.
///
/// Uses `vDSP_svesq` per row to compute sum-of-squares, then `vvsqrtf`
/// to batch the square roots.
///
/// - Parameters:
///   - matrix: A flat row-major matrix of `vectorCount × dimension` floats.
///   - vectorCount: Number of vectors.
///   - dimension: Dimension of each vector.
/// - Returns: An array of `vectorCount` magnitudes.
func batchMagnitudes(
    matrix: [Float],
    vectorCount: Int,
    dimension: Int
) -> [Float] {
    var sumOfSquares = [Float](repeating: 0, count: vectorCount)
    let dim = vDSP_Length(dimension)

    matrix.withUnsafeBufferPointer { matPtr in
        for i in 0..<vectorCount {
            vDSP_svesq(
                matPtr.baseAddress! + i * dimension, 1,
                &sumOfSquares[i],
                dim
            )
        }
    }

    // Batch sqrt via vvsqrtf (vectorized square root)
    var magnitudes = [Float](repeating: 0, count: vectorCount)
    var n = Int32(vectorCount)
    vvsqrtf(&magnitudes, sumOfSquares, &n)

    return magnitudes
}

/// Computes batch distances from a query to all vectors in a pre-built flat matrix.
///
/// This is the fast path for callers that already store vectors in a flat row-major
/// layout (e.g. ``BruteForceIndex``), avoiding the cost of constructing intermediate
/// `Vector` objects just to flatten them again.
///
/// - Parameters:
///   - query: The query vector.
///   - matrix: A flat row-major array of `vectorCount × dimension` floats.
///   - vectorCount: Number of vectors in the matrix.
///   - dimension: Dimension of each vector.
///   - metric: The distance metric to use.
/// - Returns: An array of `vectorCount` distances.
public func batchDistances(
    query: Vector,
    matrix: [Float],
    vectorCount: Int,
    dimension: Int,
    metric: some DistanceMetric
) -> [Float] {
    guard vectorCount > 0 else { return [] }
    precondition(query.dimension == dimension, "Query dimension mismatch")
    precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")

    // Fast path: Euclidean distance using batch L2
    if metric is EuclideanDistance {
        return batchL2Distances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension
        )
    }

    // Fast path: use matrix multiply for cosine and dot product metrics.
    if metric is CosineDistance || metric is DotProductDistance {
        let dots = batchDotProducts(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension
        )

        if metric is DotProductDistance {
            var negated = [Float](repeating: 0, count: vectorCount)
            var minusOne: Float = -1.0
            dots.withUnsafeBufferPointer { dotsPtr in
                vDSP_vsmul(
                    dotsPtr.baseAddress!, 1,
                    &minusOne,
                    &negated, 1,
                    vDSP_Length(vectorCount)
                )
            }
            return negated
        }

        if metric is CosineDistance {
            let queryMag = query.magnitude
            guard queryMag > 0 else {
                return [Float](repeating: 0, count: vectorCount)
            }
            let vecMags = batchMagnitudes(
                matrix: matrix, vectorCount: vectorCount, dimension: dimension
            )
            var results = [Float](repeating: 0, count: vectorCount)
            for i in 0..<vectorCount {
                let denominator = queryMag * vecMags[i]
                results[i] = denominator > 0 ? 1.0 - dots[i] / denominator : 0
            }
            return results
        }
    }

    // Fallback: reconstruct Vectors for custom metrics
    var distances = [Float]()
    distances.reserveCapacity(vectorCount)
    for i in 0..<vectorCount {
        let start = i * dimension
        let vec = Vector(Array(matrix[start..<start + dimension]))
        distances.append(metric.distance(query, vec))
    }
    return distances
}

/// Computes batch distances using a given metric.
///
/// For dot-product-based metrics (cosine, dot product), this uses `vDSP_mmul`
/// for the dot product step and then applies metric-specific post-processing.
/// For Euclidean distance, uses the optimized `batchL2Distances` path.
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

    // Build the flat row-major matrix and delegate to the flat-array overload.
    var matrix = [Float]()
    matrix.reserveCapacity(count * dimension)
    for v in vectors {
        precondition(v.dimension == dimension, "Dimension mismatch in batch")
        matrix.append(contentsOf: v.components)
    }

    return batchDistances(
        query: query, matrix: matrix,
        vectorCount: count, dimension: dimension, metric: metric
    )
}
