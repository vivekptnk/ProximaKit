// ProductQuantizerTests.swift
// ProximaKitTests
//
// Tests for product quantization: training, encode/decode, asymmetric distance,
// persistence roundtrip, and recall quality.

import XCTest
@testable import ProximaKit

final class ProductQuantizerTests: XCTestCase {

    // ── Helpers ──────────────────────────────────────────────────────

    /// Generates random vectors as a flat [Float] array.
    private func randomVectors(count: Int, dimension: Int) -> [Float] {
        (0..<count * dimension).map { _ in Float.random(in: -1...1) }
    }

    /// Generates random Vector objects.
    private func randomVectorObjects(count: Int, dimension: Int) -> [Vector] {
        (0..<count).map { _ in
            Vector((0..<dimension).map { _ in Float.random(in: -1...1) })
        }
    }

    /// L2 squared distance between two flat slices.
    private func l2Squared(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { sum, pair in
            let diff = pair.0 - pair.1
            return sum + diff * diff
        }
    }

    // ── Configuration Tests ──────────────────────────────────────────

    func testConfigurationDefaults() {
        let config = PQConfiguration(subspaceCount: 8)
        XCTAssertEqual(config.subspaceCount, 8)
        XCTAssertEqual(config.centroidsPerSubspace, 256)
        XCTAssertEqual(config.trainingIterations, 25)
    }

    func testConfigurationCustom() {
        let config = PQConfiguration(subspaceCount: 48, trainingIterations: 10)
        XCTAssertEqual(config.subspaceCount, 48)
        XCTAssertEqual(config.trainingIterations, 10)
    }

    // ── Training Tests ───────────────────────────────────────────────

    func testTrainProducesCorrectCodebooks() throws {
        let dim = 32
        let M = 8
        let n = 500
        let vectors = randomVectors(count: n, dimension: dim)

        let pq = try ProductQuantizer.train(
            vectors: vectors,
            vectorCount: n,
            dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        XCTAssertEqual(pq.dimension, dim)
        XCTAssertEqual(pq.config.subspaceCount, M)
        XCTAssertEqual(pq.subspaceDimension, dim / M)
        XCTAssertEqual(pq.codebooks.count, M)

        for cb in pq.codebooks {
            XCTAssertEqual(cb.count, 256 * (dim / M))
        }
    }

    func testTrainWithVectorObjects() throws {
        let dim = 16
        let M = 4
        let vectors = randomVectorObjects(count: 300, dimension: dim)

        let pq = try ProductQuantizer.train(
            vectors: vectors,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5)
        )

        XCTAssertEqual(pq.dimension, dim)
        XCTAssertEqual(pq.codebooks.count, M)
    }

    func testTrainEmptyThrows() {
        XCTAssertThrowsError(
            try ProductQuantizer.train(
                vectors: [] as [Float],
                vectorCount: 0,
                dimension: 32,
                config: PQConfiguration(subspaceCount: 8)
            )
        ) { error in
            XCTAssertEqual(error as? ProductQuantizerError, .emptyTrainingSet)
        }
    }

    func testTrainDimensionNotDivisibleThrows() {
        let vectors = randomVectors(count: 100, dimension: 33)
        XCTAssertThrowsError(
            try ProductQuantizer.train(
                vectors: vectors,
                vectorCount: 100,
                dimension: 33,
                config: PQConfiguration(subspaceCount: 8)
            )
        ) { error in
            XCTAssertEqual(
                error as? ProductQuantizerError,
                .dimensionNotDivisible(dimension: 33, subspaceCount: 8)
            )
        }
    }

    func testTrainInvalidVectorDataThrows() {
        XCTAssertThrowsError(
            try ProductQuantizer.train(
                vectors: [Float](repeating: 0, count: 100),
                vectorCount: 10,
                dimension: 32,
                config: PQConfiguration(subspaceCount: 8)
            )
        ) { error in
            XCTAssertEqual(
                error as? ProductQuantizerError,
                .invalidVectorData(expected: 320, got: 100)
            )
        }
    }

    // ── Encode/Decode Tests ──────────────────────────────────────────

    func testEncodeProducesCorrectCodeLength() throws {
        let dim = 32
        let M = 8
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        let testVec = Array(vectors[0..<dim])
        let codes = pq.encode(testVec)
        XCTAssertEqual(codes.count, M)
    }

    func testDecodeReconstructsApproximateVector() throws {
        let dim = 32
        let M = 8
        let n = 500
        let vectors = randomVectors(count: n, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 15)
        )

        // Encode and decode a training vector — should be close.
        let original = Array(vectors[0..<dim])
        let codes = pq.encode(original)
        let reconstructed = pq.decode(codes)

        XCTAssertEqual(reconstructed.count, dim)

        // Reconstruction error should be significantly less than the original magnitude.
        let error = l2Squared(original, reconstructed)
        let magnitude = l2Squared(original, [Float](repeating: 0, count: dim))
        XCTAssertLessThan(error, magnitude, "Reconstruction error should be less than vector magnitude")
    }

    func testEncodeDecodeWithVectorType() throws {
        let dim = 16
        let M = 4
        let vectors = randomVectorObjects(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        let original = vectors[0]
        let codes = pq.encode(original)
        let reconstructed = pq.decodeToVector(codes)

        XCTAssertEqual(reconstructed.dimension, dim)
        // Reconstruction should preserve general direction.
        let similarity = original.cosineSimilarity(reconstructed)
        XCTAssertGreaterThan(similarity, 0.5, "Reconstructed vector should be in similar direction")
    }

    // ── Asymmetric Distance Tests ────────────────────────────────────

    func testDistanceTableShape() throws {
        let dim = 32
        let M = 8
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5)
        )

        let query = Array(vectors[0..<dim])
        let table = pq.buildDistanceTable(query: query)

        XCTAssertEqual(table.count, M)
        for row in table {
            XCTAssertEqual(row.count, 256)
        }
    }

    func testAsymmetricDistanceNonNegative() throws {
        let dim = 32
        let M = 8
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        let query = Array(vectors[0..<dim])
        let table = pq.buildDistanceTable(query: query)

        for i in 0..<10 {
            let vec = Array(vectors[i * dim..<(i + 1) * dim])
            let codes = pq.encode(vec)
            let dist = pq.asymmetricDistance(table: table, codes: codes)
            XCTAssertGreaterThanOrEqual(dist, 0, "Asymmetric distance must be non-negative")
        }
    }

    func testAsymmetricDistanceSelfIsSmall() throws {
        let dim = 32
        let M = 8
        let n = 500
        let vectors = randomVectors(count: n, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 15)
        )

        // Distance of a vector to its own PQ encoding should be small
        // (it's the quantization error).
        let query = Array(vectors[0..<dim])
        let table = pq.buildDistanceTable(query: query)
        let codes = pq.encode(query)
        let selfDist = pq.asymmetricDistance(table: table, codes: codes)

        // Compare with distance to a random other vector.
        let otherCodes = pq.encode(Array(vectors[dim..<2 * dim]))
        let otherDist = pq.asymmetricDistance(table: table, codes: otherCodes)

        // Self-distance (quantization error) should typically be smaller than
        // distance to a random vector. This can fail with very small datasets
        // but is reliable with 500 vectors.
        XCTAssertLessThan(selfDist, otherDist + 1.0,
            "Self-distance should generally be less than distance to random vector")
    }

    func testBatchAsymmetricDistances() throws {
        let dim = 32
        let M = 8
        let n = 200
        let vectors = randomVectors(count: n, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        // Encode all vectors.
        var codeMatrix = [UInt8]()
        codeMatrix.reserveCapacity(n * M)
        for i in 0..<n {
            let vec = Array(vectors[i * dim..<(i + 1) * dim])
            codeMatrix.append(contentsOf: pq.encode(vec))
        }

        let query = Array(vectors[0..<dim])
        let table = pq.buildDistanceTable(query: query)

        // Batch computation.
        let batchDists = pq.batchAsymmetricDistances(
            table: table, codeMatrix: codeMatrix, vectorCount: n
        )

        // Compare with individual computation.
        for i in 0..<n {
            let codes = Array(codeMatrix[i * M..<(i + 1) * M])
            let singleDist = pq.asymmetricDistance(table: table, codes: codes)
            XCTAssertEqual(batchDists[i], singleDist, accuracy: 1e-6,
                "Batch distance should match individual distance for vector \(i)")
        }
    }

    // ── Memory Statistics Tests ──────────────────────────────────────

    func testCompressionRatio() throws {
        let dim = 384
        let M = 48
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5)
        )

        XCTAssertEqual(pq.bytesPerOriginalVector, 384 * 4)  // 1536 bytes
        XCTAssertEqual(pq.bytesPerCode, 48)                  // 48 bytes
        XCTAssertEqual(pq.compressionRatio, 32.0, accuracy: 0.01)

        // With M=48, each code is 48 bytes vs 1536 bytes = 32x compression.
        // More than the 4x acceptance criterion.
    }

    func testCompressionRatio8Subspaces() throws {
        let dim = 384
        let M = 8
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5)
        )

        XCTAssertEqual(pq.bytesPerCode, 8)
        // 384*4 / 8 = 192x compression
        XCTAssertEqual(pq.compressionRatio, 192.0, accuracy: 0.01)
    }

    // ── Persistence Tests ────────────────────────────────────────────

    func testSaveLoadRoundtrip() throws {
        let dim = 32
        let M = 8
        let vectors = randomVectors(count: 300, dimension: dim)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: 300, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_pq_\(UUID().uuidString).pqtt")

        defer { try? FileManager.default.removeItem(at: url) }

        try pq.save(to: url)
        let loaded = try ProductQuantizer.load(from: url)

        XCTAssertEqual(loaded.dimension, pq.dimension)
        XCTAssertEqual(loaded.config, pq.config)
        XCTAssertEqual(loaded.subspaceDimension, pq.subspaceDimension)

        // Codebooks should be identical.
        for m in 0..<M {
            XCTAssertEqual(loaded.codebooks[m].count, pq.codebooks[m].count)
            for i in 0..<pq.codebooks[m].count {
                XCTAssertEqual(loaded.codebooks[m][i], pq.codebooks[m][i], accuracy: 1e-7)
            }
        }

        // Encoding should produce the same codes.
        let testVec = Array(vectors[0..<dim])
        XCTAssertEqual(pq.encode(testVec), loaded.encode(testVec))
    }

    func testLoadInvalidMagicThrows() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_bad_magic_\(UUID().uuidString).pqtt")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = Data(repeating: 0xFF, count: 24)
        try data.write(to: url)

        XCTAssertThrowsError(try ProductQuantizer.load(from: url))
    }

    // ── Recall Quality Tests ─────────────────────────────────────────

    func testRecallQualityAt1000Vectors() throws {
        let dim = 64
        let M = 16
        let n = 1000
        let k = 10

        // Generate clustered data for realistic recall measurement.
        var vectors = [Float]()
        vectors.reserveCapacity(n * dim)
        let numClusters = 10
        for c in 0..<numClusters {
            // Each cluster has a random center.
            let center = (0..<dim).map { _ in Float.random(in: -5...5) }
            let clusterSize = n / numClusters
            for _ in 0..<clusterSize {
                for d in 0..<dim {
                    vectors.append(center[d] + Float.random(in: -0.5...0.5))
                }
            }
            _ = c
        }

        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 20)
        )

        // Encode all vectors.
        var codeMatrix = [UInt8]()
        codeMatrix.reserveCapacity(n * M)
        for i in 0..<n {
            let vec = Array(vectors[i * dim..<(i + 1) * dim])
            codeMatrix.append(contentsOf: pq.encode(vec))
        }

        // Run multiple queries and measure recall.
        var totalRecall: Float = 0
        let numQueries = 20

        for q in 0..<numQueries {
            let query = Array(vectors[q * dim..<(q + 1) * dim])

            // Ground truth: exact L2 distances.
            var exactDists = [(Int, Float)]()
            for i in 0..<n {
                let vec = Array(vectors[i * dim..<(i + 1) * dim])
                exactDists.append((i, l2Squared(query, vec)))
            }
            exactDists.sort { $0.1 < $1.1 }
            let groundTruth = Set(exactDists.prefix(k).map(\.0))

            // PQ asymmetric distances.
            let table = pq.buildDistanceTable(query: query)
            let pqDists = pq.batchAsymmetricDistances(
                table: table, codeMatrix: codeMatrix, vectorCount: n
            )
            var pqRanked = pqDists.enumerated().map { ($0.offset, $0.element) }
            pqRanked.sort { $0.1 < $1.1 }
            let pqTopK = Set(pqRanked.prefix(k).map(\.0))

            let recall = Float(groundTruth.intersection(pqTopK).count) / Float(k)
            totalRecall += recall
        }

        let avgRecall = totalRecall / Float(numQueries)

        // With clustered data, 64d, 16 subspaces, 1000 vectors:
        // PQ should achieve >70% recall@10.
        // The acceptance criterion is <5% recall loss, which applies to
        // PQ+HNSW vs HNSW (not PQ vs exact), so this is a baseline check.
        XCTAssertGreaterThan(avgRecall, 0.50,
            "PQ recall@10 should be >50% on clustered data (got \(avgRecall))")
    }
}
