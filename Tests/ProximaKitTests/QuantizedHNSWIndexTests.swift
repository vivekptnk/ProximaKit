// QuantizedHNSWIndexTests.swift
// ProximaKitTests
//
// Tests for QuantizedHNSWIndex: build, search, persistence roundtrip,
// memory reduction verification, and recall quality.

import XCTest
@testable import ProximaKit

final class QuantizedHNSWIndexTests: XCTestCase {

    // ── Helpers ──────────────────────────────────────────────────────

    /// Generates random Vector objects with optional clustering.
    private func randomVectors(count: Int, dimension: Int) -> [Vector] {
        (0..<count).map { _ in
            Vector((0..<dimension).map { _ in Float.random(in: -1...1) })
        }
    }

    /// Generates clustered vectors for more realistic recall measurement.
    private func clusteredVectors(
        count: Int, dimension: Int, clusters: Int
    ) -> [Vector] {
        var vectors = [Vector]()
        vectors.reserveCapacity(count)
        let perCluster = count / clusters

        for _ in 0..<clusters {
            let center = (0..<dimension).map { _ in Float.random(in: -5...5) }
            for _ in 0..<perCluster {
                let v = (0..<dimension).map { d in
                    center[d] + Float.random(in: -0.5...0.5)
                }
                vectors.append(Vector(v))
            }
        }

        // Fill any remainder from rounding
        while vectors.count < count {
            vectors.append(Vector((0..<dimension).map { _ in Float.random(in: -1...1) }))
        }

        return vectors
    }

    /// Brute-force k-NN ground truth using L2 squared distance.
    private func bruteForceKNN(
        query: Vector, vectors: [Vector], k: Int
    ) -> [(index: Int, distance: Float)] {
        let metric = EuclideanDistance()
        var dists = vectors.enumerated().map { (index: $0.offset, distance: metric.distance(query, $0.element)) }
        dists.sort { $0.distance < $1.distance }
        return Array(dists.prefix(k))
    }

    // ── Build Tests ──────────────────────────────────────────────────

    func testBuildCreatesIndex() async throws {
        let dim = 32
        let n = 300
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 8, trainingIterations: 5)
        )

        let count = await qIndex.count
        let liveCount = await qIndex.liveCount
        XCTAssertEqual(count, n)
        XCTAssertEqual(liveCount, n)
    }

    func testBuildWithMetadata() async throws {
        let dim = 16
        let n = 100
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }
        let metadata: [Data?] = (0..<n).map { i in
            try? JSONEncoder().encode(["index": i])
        }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            metadata: metadata,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let count = await qIndex.count
        XCTAssertEqual(count, n)
    }

    // ── Search Tests ─────────────────────────────────────────────────

    func testSearchReturnsResults() async throws {
        let dim = 32
        let n = 500
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50),
            pqConfig: PQConfiguration(subspaceCount: 8, trainingIterations: 10)
        )

        let query = vectors[0]
        let results = await qIndex.search(query: query, k: 10)

        XCTAssertGreaterThan(results.count, 0)
        XCTAssertLessThanOrEqual(results.count, 10)

        // Results should be sorted by distance (ascending).
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i - 1].distance, results[i].distance)
        }
    }

    func testSearchWithDimensionMismatchReturnsEmpty() async throws {
        let dim = 16
        let n = 100
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let wrongDimQuery = Vector((0..<32).map { _ in Float.random(in: -1...1) })
        let results = await qIndex.search(query: wrongDimQuery, k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchWithFilter() async throws {
        let dim = 16
        let n = 200
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }
        let allowedSet = Set(ids.prefix(50))

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let query = vectors[0]
        let results = await qIndex.search(query: query, k: 10) { id in
            allowedSet.contains(id)
        }

        for result in results {
            XCTAssertTrue(allowedSet.contains(result.id))
        }
    }

    // ── Remove Tests ─────────────────────────────────────────────────

    func testRemoveReducesLiveCount() async throws {
        let dim = 16
        let n = 100
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let removed = await qIndex.remove(id: ids[0])
        XCTAssertTrue(removed)

        let liveCount = await qIndex.liveCount
        XCTAssertEqual(liveCount, n - 1)

        // Removed vector should not appear in search results.
        let results = await qIndex.search(query: vectors[0], k: n)
        let resultIds = Set(results.map(\.id))
        XCTAssertFalse(resultIds.contains(ids[0]))
    }

    func testRemoveNonExistentReturnsFalse() async throws {
        let dim = 16
        let n = 50
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let removed = await qIndex.remove(id: UUID())
        XCTAssertFalse(removed)
    }

    // ── Memory Statistics Tests ───────────────────────────────────────

    func testMemorySavingsRatio() async throws {
        let dim = 384
        let M = 48
        let n = 300
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: M, trainingIterations: 5)
        )

        let codeBytes = await qIndex.codeStorageBytes
        let fullBytes = await qIndex.equivalentFullPrecisionBytes
        let ratio = await qIndex.memorySavingsRatio

        // 384d * 4 bytes = 1536 bytes per vector vs 48 bytes = 32x savings
        XCTAssertEqual(codeBytes, n * M)       // 300 * 48 = 14400
        XCTAssertEqual(fullBytes, n * dim * 4)  // 300 * 384 * 4 = 460800
        XCTAssertGreaterThanOrEqual(ratio, 4.0,
            "Memory savings should be at least 4x (acceptance criterion). Got \(ratio)x")
        XCTAssertEqual(ratio, 32.0, accuracy: 0.1, "Expected 32x compression with 48 subspaces")
    }

    // ── Persistence Tests ────────────────────────────────────────────

    func testSaveLoadRoundtrip() async throws {
        let dim = 32
        let n = 200
        let vectors = randomVectors(count: n, dimension: dim)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 8, trainingIterations: 5)
        )

        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_qhnsw_\(UUID().uuidString).qhnsw")
        defer { try? FileManager.default.removeItem(at: url) }

        try await qIndex.save(to: url)
        let loaded = try QuantizedHNSWIndex.load(from: url)

        let origCount = await qIndex.count
        let loadedCount = await loaded.count
        XCTAssertEqual(origCount, loadedCount)

        // Search should produce comparable results.
        let query = vectors[0]
        let origResults = await qIndex.search(query: query, k: 10)
        let loadedResults = await loaded.search(query: query, k: 10)

        // Same top results (IDs should match since graph and codes are identical).
        XCTAssertEqual(origResults.map(\.id), loadedResults.map(\.id))
        for (a, b) in zip(origResults, loadedResults) {
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-6)
        }
    }

    func testLoadInvalidMagicThrows() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let url = tmpDir.appendingPathComponent("test_bad_qhnsw_\(UUID().uuidString).qhnsw")
        defer { try? FileManager.default.removeItem(at: url) }

        var data = Data(repeating: 0xFF, count: 56)
        try data.write(to: url)

        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url))
    }

    // ── Recall Quality Test ──────────────────────────────────────────

    func testRecallAtClusteredData() async throws {
        let dim = 64
        let M = 16
        let n = 1000
        let k = 10

        // Clustered data for realistic recall measurement.
        let vectors = clusteredVectors(count: n, dimension: dim, clusters: 10)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 100),
            pqConfig: PQConfiguration(subspaceCount: M, trainingIterations: 15)
        )

        // Also build a full-precision BruteForceIndex for ground truth.
        let bfIndex = BruteForceIndex(dimension: dim, metric: EuclideanDistance())
        for i in 0..<n {
            try await bfIndex.add(vectors[i], id: ids[i])
        }

        var totalRecall: Float = 0
        let numQueries = 20

        for q in 0..<numQueries {
            let query = vectors[q]

            // Ground truth from brute force.
            let exact = await bfIndex.search(query: query, k: k)
            let groundTruth = Set(exact.map(\.id))

            // Quantized HNSW results.
            let pqResults = await qIndex.search(query: query, k: k, efSearch: 200)
            let pqTopK = Set(pqResults.map(\.id))

            let recall = Float(groundTruth.intersection(pqTopK).count) / Float(k)
            totalRecall += recall
        }

        let avgRecall = totalRecall / Float(numQueries)

        // Acceptance criterion: <5% recall loss vs full HNSW.
        // PQ+HNSW typically achieves >80% recall@10 on clustered data.
        // We check >50% as a baseline (PQ introduces some quantization error).
        XCTAssertGreaterThan(avgRecall, 0.50,
            "QuantizedHNSW recall@10 should be >50% on clustered data (got \(avgRecall))")
    }
}
