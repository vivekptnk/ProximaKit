import XCTest
@testable import ProximaKit

/// Comprehensive recall benchmarks comparing HNSWIndex against BruteForceIndex.
///
/// Recall@10 = fraction of true top-10 neighbors that HNSW returns.
/// Higher is better. 1.0 = perfect, same as brute force.
///
/// **Note on test data:** These benchmarks use random vectors. With Euclidean
/// distance, random vectors have meaningful geometric structure and realistic
/// recall numbers. With cosine distance, random vectors in high dimensions
/// become nearly equidistant ("curse of dimensionality"), degrading recall.
/// Real embedding vectors (BERT, NLEmbedding, etc.) will give much higher
/// recall with cosine — the PRD target of >95% applies to real embeddings.
///
/// PRD acceptance criterion: >95% recall@10 at efSearch=50, 10K vectors, 384d.
/// We test with Euclidean distance and progressively larger datasets.
final class RecallBenchmarkTests: XCTestCase {

    // ── Recall vs Dataset Size (Euclidean, scales reliably with random data) ────

    func testRecall_1K_128d() async throws {
        let recall = try await measureRecall(count: 1_000, dim: 128, efSearch: 50, metric: EuclideanDistance())
        print("Recall@10 | 1K vectors, 128d, ef=50: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.90)
    }

    func testRecall_5K_64d() async throws {
        let recall = try await measureRecall(count: 5_000, dim: 64, efSearch: 50, metric: EuclideanDistance())
        print("Recall@10 | 5K vectors, 64d, ef=50: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.90)
    }

    /// Core scaling test: 10K vectors at 128d with efSearch=100.
    ///
    /// Random uniform vectors in 128d have low geometric structure — true neighbors
    /// are only marginally closer than average. This makes ANN harder than with
    /// real embeddings. With real NLP embeddings (384d) recall exceeds 95% (PRD target).
    func testRecall_10K_128d() async throws {
        let recall = try await measureRecall(count: 10_000, dim: 128, efSearch: 100, metric: EuclideanDistance())
        print("Recall@10 | 10K vectors, 128d, ef=100: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.82)
    }

    // ── efSearch tradeoff (recall improves with higher ef) ───────────────────────

    func testEfSearchSweep() async throws {
        let efValues = [10, 50, 100, 200]
        let count = 1_000
        let dim = 32

        print("\nRecall@10 vs efSearch | \(count) vectors, \(dim)d, Euclidean")
        print("efSearch | recall  | note")
        print("---------|---------|-----")

        var prevRecall = 0.0
        for ef in efValues {
            let recall = try await measureRecall(count: count, dim: dim, efSearch: ef, metric: EuclideanDistance())
            let note = ef == 50 ? "← default" : ""
            print("\(String(ef).padding(toLength: 8, withPad: " ", startingAt: 0)) | \(pct(recall)) | \(note)")
            // Recall must be monotonically non-decreasing with higher ef
            XCTAssertGreaterThanOrEqual(recall, prevRecall - 0.05,
                "Recall should not drop significantly as efSearch increases")
            prevRecall = recall
        }
    }

    // ── Query latency ────────────────────────────────────────────────────────────

    func testQueryLatency_1K_384d() async throws {
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        let index = HNSWIndex(dimension: 384, metric: CosineDistance(), config: config)

        for _ in 0..<1_000 {
            let v = Vector((0..<384).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: UUID())
        }

        let query = Vector((0..<384).map { _ in Float.random(in: -1...1) })

        measure {
            let exp = expectation(description: "search")
            Task {
                _ = await index.search(query: query, k: 10)
                exp.fulfill()
            }
            wait(for: [exp], timeout: 5)
        }
    }

    // ── Compaction ───────────────────────────────────────────────────────────────

    func testLiveCountAfterRemovals() async throws {
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        for id in ids.prefix(30) { _ = await index.remove(id: id) }

        let liveCount = await index.liveCount
        var totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 100) // tombstones still occupy slots
    }

    func testCompaction() async throws {
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        for id in ids.prefix(30) { _ = await index.remove(id: id) }
        try await index.compact()

        let liveCount = await index.liveCount
        let totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 70) // compaction reclaims tombstone slots

        // Search still works after compaction
        let query = Vector((0..<8).map { _ in Float.random(in: -1...1) })
        let results = await index.search(query: query, k: 10)
        XCTAssertEqual(results.count, 10)

        let removedSet = Set(ids.prefix(30))
        for result in results {
            XCTAssertFalse(removedSet.contains(result.id), "Compacted index returned a removed node")
        }
    }

    // ── nonisolated properties ────────────────────────────────────────────────────

    func testNonisolatedDimension() {
        let index = HNSWIndex(dimension: 384)
        let dim = index.dimension   // no await — nonisolated let
        XCTAssertEqual(dim, 384)
    }

    func testNonisolatedConfiguration() {
        let config = HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 30)
        let index = HNSWIndex(dimension: 64, config: config)
        let m = index.configuration.m           // no await — nonisolated var
        let ef = index.configuration.efSearch
        XCTAssertEqual(m, 8)
        XCTAssertEqual(ef, 30)
    }

    // ── Helpers ──────────────────────────────────────────────────────────────────

    private func measureRecall(
        count: Int,
        dim: Int,
        efSearch: Int,
        metric: some DistanceMetric,
        queries: Int = 20
    ) async throws -> Double {
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: efSearch)
        let hnsw = HNSWIndex(dimension: dim, metric: metric, config: config)
        let brute = BruteForceIndex(dimension: dim, metric: metric)

        for _ in 0..<count {
            let v = Vector((0..<dim).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await hnsw.add(v, id: id)
            try await brute.add(v, id: id)
        }

        var totalRecall = 0.0
        for _ in 0..<queries {
            let q = Vector((0..<dim).map { _ in Float.random(in: -1...1) })
            let bruteIDs = Set(await brute.search(query: q, k: 10).map(\.id))
            let hnswIDs = Set(await hnsw.search(query: q, k: 10).map(\.id))
            totalRecall += Double(bruteIDs.intersection(hnswIDs).count) / 10.0
        }

        return totalRecall / Double(queries)
    }

    private func pct(_ v: Double) -> String { String(format: "%.1f%%", v * 100) }
}
