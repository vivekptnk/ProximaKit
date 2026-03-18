// ProductQuantizer.swift
// ProximaKit
//
// Product quantization (PQ) for vector compression.
// Splits D-dimensional vectors into M subspaces, clusters each with k-means,
// and stores centroid IDs (UInt8) instead of Float32 components.
//
// Memory reduction: D * 4 bytes → M bytes (e.g., 384 * 4 = 1536B → 48B = 32×)
//
// Reference: "Product Quantization for Nearest Neighbor Search"
// (Jégou, Douze, Schmid, 2011)

import Accelerate
import Foundation

/// Configuration for product quantization.
public struct PQConfiguration: Sendable, Codable, Equatable {
    /// Number of subspaces (M). The vector dimension must be divisible by this.
    public let subspaceCount: Int

    /// Number of centroids per subspace (K). Fixed at 256 to fit in UInt8.
    public let centroidsPerSubspace: Int

    /// Number of k-means iterations during training.
    public let trainingIterations: Int

    public init(
        subspaceCount: Int,
        trainingIterations: Int = 25
    ) {
        precondition(subspaceCount > 0, "subspaceCount must be positive")
        precondition(trainingIterations > 0, "trainingIterations must be positive")
        self.subspaceCount = subspaceCount
        self.centroidsPerSubspace = 256
        self.trainingIterations = trainingIterations
    }
}

/// A trained product quantizer that can encode, decode, and compute distances
/// on compressed vector representations.
///
/// ## Usage
///
/// ```swift
/// // Train on a representative sample of vectors
/// let pq = try ProductQuantizer.train(
///     vectors: trainingVectors,
///     dimension: 384,
///     config: PQConfiguration(subspaceCount: 48)
/// )
///
/// // Encode vectors to compact codes
/// let codes = pq.encode(vector)  // 48 bytes instead of 1536
///
/// // Asymmetric distance: full-precision query vs quantized database
/// let table = pq.buildDistanceTable(query: queryVector, metric: .euclidean)
/// let distance = pq.asymmetricDistance(table: table, codes: codes)
/// ```
public struct ProductQuantizer: Sendable {

    // ── Configuration ────────────────────────────────────────────────

    /// The full vector dimension this quantizer was trained for.
    public let dimension: Int

    /// PQ configuration (subspace count, centroids per subspace, etc).
    public let config: PQConfiguration

    /// Dimension of each subspace: `dimension / subspaceCount`.
    public let subspaceDimension: Int

    // ── Codebooks ────────────────────────────────────────────────────
    //
    // Flat storage: codebooks[m] is a contiguous [Float] of length K * ds,
    // where K = centroidsPerSubspace and ds = subspaceDimension.
    // Centroid j of subspace m starts at codebooks[m][j * ds].
    //
    // Flat layout enables vDSP batch distance computation against all
    // centroids of a subspace in a single pass.

    /// Codebooks for each subspace. `codebooks[m]` has `K * subspaceDimension` floats.
    public let codebooks: [[Float]]

    // ── Initialization ───────────────────────────────────────────────

    /// Creates a ProductQuantizer from pre-trained codebooks.
    ///
    /// - Parameters:
    ///   - dimension: The full vector dimension.
    ///   - config: PQ configuration.
    ///   - codebooks: M codebooks, each `K * subspaceDimension` floats in row-major order.
    public init(dimension: Int, config: PQConfiguration, codebooks: [[Float]]) {
        precondition(dimension > 0, "Dimension must be positive")
        precondition(
            dimension % config.subspaceCount == 0,
            "Dimension (\(dimension)) must be divisible by subspaceCount (\(config.subspaceCount))"
        )
        let ds = dimension / config.subspaceCount
        precondition(
            codebooks.count == config.subspaceCount,
            "Expected \(config.subspaceCount) codebooks, got \(codebooks.count)"
        )
        for (i, cb) in codebooks.enumerated() {
            precondition(
                cb.count == config.centroidsPerSubspace * ds,
                "Codebook \(i) has \(cb.count) floats, expected \(config.centroidsPerSubspace * ds)"
            )
        }
        self.dimension = dimension
        self.config = config
        self.subspaceDimension = ds
        self.codebooks = codebooks
    }

    // ── Training ─────────────────────────────────────────────────────

    /// Trains a product quantizer on a set of vectors using k-means clustering.
    ///
    /// - Parameters:
    ///   - vectors: Training vectors as a flat row-major `[Float]` of length `n * dimension`.
    ///   - vectorCount: Number of training vectors.
    ///   - dimension: Vector dimension.
    ///   - config: PQ configuration.
    /// - Returns: A trained `ProductQuantizer`.
    /// - Throws: `ProductQuantizerError` if inputs are invalid.
    public static func train(
        vectors: [Float],
        vectorCount: Int,
        dimension: Int,
        config: PQConfiguration
    ) throws -> ProductQuantizer {
        guard vectorCount > 0 else {
            throw ProductQuantizerError.emptyTrainingSet
        }
        guard dimension > 0, dimension % config.subspaceCount == 0 else {
            throw ProductQuantizerError.dimensionNotDivisible(
                dimension: dimension, subspaceCount: config.subspaceCount
            )
        }
        guard vectors.count == vectorCount * dimension else {
            throw ProductQuantizerError.invalidVectorData(
                expected: vectorCount * dimension, got: vectors.count
            )
        }

        let M = config.subspaceCount
        let K = config.centroidsPerSubspace
        let ds = dimension / M

        // Train each subspace independently.
        var codebooks: [[Float]] = []
        codebooks.reserveCapacity(M)

        for m in 0..<M {
            // Extract subspace m from all training vectors.
            // subVectors is row-major: vectorCount rows × ds columns.
            var subVectors = [Float](repeating: 0, count: vectorCount * ds)
            let subOffset = m * ds
            for i in 0..<vectorCount {
                let srcStart = i * dimension + subOffset
                let dstStart = i * ds
                for d in 0..<ds {
                    subVectors[dstStart + d] = vectors[srcStart + d]
                }
            }

            // Run k-means on this subspace.
            let centroids = kmeans(
                data: subVectors,
                vectorCount: vectorCount,
                dimension: ds,
                k: min(K, vectorCount),
                iterations: config.trainingIterations
            )

            codebooks.append(centroids)
        }

        return ProductQuantizer(
            dimension: dimension,
            config: config,
            codebooks: codebooks
        )
    }

    /// Convenience overload that accepts `[Vector]` instead of flat floats.
    public static func train(
        vectors: [Vector],
        config: PQConfiguration
    ) throws -> ProductQuantizer {
        guard let first = vectors.first else {
            throw ProductQuantizerError.emptyTrainingSet
        }
        let dimension = first.dimension
        var flat = [Float]()
        flat.reserveCapacity(vectors.count * dimension)
        for v in vectors {
            precondition(v.dimension == dimension, "All training vectors must have the same dimension")
            flat.append(contentsOf: v.components)
        }
        return try train(
            vectors: flat,
            vectorCount: vectors.count,
            dimension: dimension,
            config: config
        )
    }

    // ── Encoding ─────────────────────────────────────────────────────

    /// Encodes a vector into PQ codes (M bytes).
    ///
    /// For each subspace, finds the nearest centroid and stores its index.
    ///
    /// - Parameter vector: A flat `[Float]` of length `dimension`.
    /// - Returns: An array of M `UInt8` centroid indices.
    public func encode(_ vector: [Float]) -> [UInt8] {
        precondition(vector.count == dimension, "Vector dimension mismatch")

        let M = config.subspaceCount
        let K = config.centroidsPerSubspace
        let ds = subspaceDimension
        var codes = [UInt8](repeating: 0, count: M)

        for m in 0..<M {
            let subOffset = m * ds

            // Find nearest centroid for this subspace using L2.
            var bestDist: Float = .greatestFiniteMagnitude
            var bestIdx: Int = 0

            codebooks[m].withUnsafeBufferPointer { cbPtr in
                for j in 0..<K {
                    let centroidStart = j * ds
                    var dist: Float = 0

                    // L2 squared distance: sum((v[d] - c[d])^2)
                    for d in 0..<ds {
                        let diff = vector[subOffset + d] - cbPtr[centroidStart + d]
                        dist += diff * diff
                    }

                    if dist < bestDist {
                        bestDist = dist
                        bestIdx = j
                    }
                }
            }

            codes[m] = UInt8(bestIdx)
        }

        return codes
    }

    /// Encodes a `Vector` into PQ codes.
    public func encode(_ vector: Vector) -> [UInt8] {
        encode(Array(vector.components))
    }

    // ── Decoding ─────────────────────────────────────────────────────

    /// Decodes PQ codes back to an approximate vector.
    ///
    /// Concatenates the centroid vectors for each subspace code.
    ///
    /// - Parameter codes: M `UInt8` centroid indices.
    /// - Returns: The reconstructed vector as `[Float]`.
    public func decode(_ codes: [UInt8]) -> [Float] {
        precondition(codes.count == config.subspaceCount, "Code length mismatch")

        let ds = subspaceDimension
        var result = [Float](repeating: 0, count: dimension)

        for m in 0..<config.subspaceCount {
            let centroidIdx = Int(codes[m])
            let centroidStart = centroidIdx * ds
            let resultStart = m * ds

            codebooks[m].withUnsafeBufferPointer { cbPtr in
                for d in 0..<ds {
                    result[resultStart + d] = cbPtr[centroidStart + d]
                }
            }
        }

        return result
    }

    /// Decodes PQ codes to a `Vector`.
    public func decodeToVector(_ codes: [UInt8]) -> Vector {
        Vector(decode(codes))
    }

    // ── Asymmetric Distance Computation ──────────────────────────────
    //
    // ADC: compute distance from a full-precision query to a PQ-encoded
    // database vector. Precompute a distance table (M × K) from the query
    // to all centroids, then sum M table lookups per database vector.
    //
    // This is the standard PQ search approach: O(M*K*ds) to build the table
    // once, then O(M) per database vector — much faster than decoding.

    /// A precomputed distance lookup table for asymmetric distance computation.
    ///
    /// `table[m][k]` = distance from query subvector m to centroid k of subspace m.
    /// Flat storage: `table[m]` has K entries.
    public typealias DistanceTable = [[Float]]

    /// Builds a distance table from a query vector to all centroids.
    ///
    /// Supports L2 (squared Euclidean) distance for the table entries.
    /// The table enables O(M) distance computation per database vector.
    ///
    /// - Parameter query: The full-precision query vector.
    /// - Returns: An M × K distance table.
    public func buildDistanceTable(query: [Float]) -> DistanceTable {
        precondition(query.count == dimension, "Query dimension mismatch")

        let M = config.subspaceCount
        let K = config.centroidsPerSubspace
        let ds = subspaceDimension

        var table = [[Float]](repeating: [Float](repeating: 0, count: K), count: M)

        for m in 0..<M {
            let queryOffset = m * ds

            codebooks[m].withUnsafeBufferPointer { cbPtr in
                for j in 0..<K {
                    let centroidStart = j * ds
                    var dist: Float = 0

                    // L2 squared distance
                    for d in 0..<ds {
                        let diff = query[queryOffset + d] - cbPtr[centroidStart + d]
                        dist += diff * diff
                    }

                    table[m][j] = dist
                }
            }
        }

        return table
    }

    /// Builds a distance table from a `Vector`.
    public func buildDistanceTable(query: Vector) -> DistanceTable {
        buildDistanceTable(query: Array(query.components))
    }

    /// Computes the asymmetric distance from a precomputed table to a PQ code.
    ///
    /// This is the inner loop of PQ search: sum M table lookups.
    /// Returns L2 squared distance (take sqrt for actual L2 if needed).
    ///
    /// - Parameters:
    ///   - table: A distance table built with `buildDistanceTable(query:)`.
    ///   - codes: The PQ codes for a database vector (M bytes).
    /// - Returns: The summed (squared L2) distance.
    @inline(__always)
    public func asymmetricDistance(table: DistanceTable, codes: [UInt8]) -> Float {
        var distance: Float = 0
        for m in 0..<config.subspaceCount {
            distance += table[m][Int(codes[m])]
        }
        return distance
    }

    /// Batch asymmetric distance computation.
    ///
    /// Computes distances from a precomputed table to N PQ code vectors.
    ///
    /// - Parameters:
    ///   - table: A distance table.
    ///   - codeMatrix: A flat array of `N * M` UInt8 codes. Row i is codes for vector i.
    ///   - vectorCount: Number of vectors (N).
    /// - Returns: N distances.
    public func batchAsymmetricDistances(
        table: DistanceTable,
        codeMatrix: [UInt8],
        vectorCount: Int
    ) -> [Float] {
        let M = config.subspaceCount
        precondition(codeMatrix.count == vectorCount * M, "Code matrix size mismatch")

        var distances = [Float](repeating: 0, count: vectorCount)

        codeMatrix.withUnsafeBufferPointer { codesPtr in
            for i in 0..<vectorCount {
                let rowStart = i * M
                var dist: Float = 0
                for m in 0..<M {
                    dist += table[m][Int(codesPtr[rowStart + m])]
                }
                distances[i] = dist
            }
        }

        return distances
    }

    // ── Memory Statistics ────────────────────────────────────────────

    /// Memory used by codebooks in bytes.
    public var codebookMemoryBytes: Int {
        config.subspaceCount * config.centroidsPerSubspace * subspaceDimension * MemoryLayout<Float>.size
    }

    /// Bytes per encoded vector (= subspaceCount).
    public var bytesPerCode: Int {
        config.subspaceCount
    }

    /// Bytes per original Float32 vector (= dimension * 4).
    public var bytesPerOriginalVector: Int {
        dimension * MemoryLayout<Float>.size
    }

    /// Compression ratio (original / compressed).
    public var compressionRatio: Float {
        Float(bytesPerOriginalVector) / Float(bytesPerCode)
    }
}

// MARK: - K-Means Clustering

/// Standard k-means clustering for PQ codebook training.
///
/// Uses random initialization (not k-means++) for simplicity.
/// With 256 centroids and typical embedding dimensions, random init
/// converges well within 25 iterations.
///
/// - Parameters:
///   - data: Flat row-major training data, `vectorCount * dimension` floats.
///   - vectorCount: Number of training vectors.
///   - dimension: Dimension of each vector.
///   - k: Number of clusters (centroids).
///   - iterations: Number of iterations.
/// - Returns: Flat row-major centroids, `k * dimension` floats.
func kmeans(
    data: [Float],
    vectorCount: Int,
    dimension: Int,
    k: Int,
    iterations: Int
) -> [Float] {
    precondition(vectorCount >= k, "Need at least k training vectors")

    // Initialize centroids by sampling k random vectors (without replacement).
    var usedIndices = Set<Int>()
    var centroids = [Float](repeating: 0, count: k * dimension)

    for j in 0..<k {
        var idx: Int
        repeat {
            idx = Int.random(in: 0..<vectorCount)
        } while usedIndices.contains(idx)
        usedIndices.insert(idx)

        let srcStart = idx * dimension
        let dstStart = j * dimension
        for d in 0..<dimension {
            centroids[dstStart + d] = data[srcStart + d]
        }
    }

    // Scratch buffers.
    var assignments = [Int](repeating: 0, count: vectorCount)
    var clusterSums = [Float](repeating: 0, count: k * dimension)
    var clusterCounts = [Int](repeating: 0, count: k)

    for _ in 0..<iterations {
        // Assignment step: find nearest centroid for each vector.
        data.withUnsafeBufferPointer { dataPtr in
            centroids.withUnsafeBufferPointer { centPtr in
                for i in 0..<vectorCount {
                    let vecStart = i * dimension
                    var bestDist: Float = .greatestFiniteMagnitude
                    var bestJ: Int = 0

                    for j in 0..<k {
                        let centStart = j * dimension
                        var dist: Float = 0
                        for d in 0..<dimension {
                            let diff = dataPtr[vecStart + d] - centPtr[centStart + d]
                            dist += diff * diff
                        }
                        if dist < bestDist {
                            bestDist = dist
                            bestJ = j
                        }
                    }

                    assignments[i] = bestJ
                }
            }
        }

        // Update step: recompute centroids as cluster means.
        for i in 0..<clusterSums.count { clusterSums[i] = 0 }
        for i in 0..<clusterCounts.count { clusterCounts[i] = 0 }

        data.withUnsafeBufferPointer { dataPtr in
            for i in 0..<vectorCount {
                let j = assignments[i]
                clusterCounts[j] += 1
                let vecStart = i * dimension
                let centStart = j * dimension
                for d in 0..<dimension {
                    clusterSums[centStart + d] += dataPtr[vecStart + d]
                }
            }
        }

        for j in 0..<k {
            let count = clusterCounts[j]
            let centStart = j * dimension
            if count > 0 {
                let scale = 1.0 / Float(count)
                for d in 0..<dimension {
                    centroids[centStart + d] = clusterSums[centStart + d] * scale
                }
            }
            // Empty clusters keep their previous centroids (no reinit).
        }
    }

    return centroids
}
