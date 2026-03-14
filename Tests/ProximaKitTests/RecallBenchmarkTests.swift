import XCTest
@testable import ProximaKit

/// Comprehensive recall benchmarks comparing HNSWIndex against BruteForceIndex.
///
/// These tests measure how many of HNSW's top-k results match the exact top-k
/// from brute force (ground truth). Higher recall = more accurate approximate search.
///
/// PRD target: >95% recall@10 at efSearch=50, 10K vectors, 384 dimensions.
final class RecallBenchmarkTests: XCTestCase {

    // ── Recall vs Dataset Size ─────────────────────────────────────────

    func testRecall_1K_128d() async throws {
        let recall = try await measureRecall(count: 1_000, dim: 128, efSearch: 50)
        print("Recall@10 | 1K vectors, 128d, ef=50: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.90, "Expected >90% recall@10")
    }

    func testRecall_5K_128d() async throws {
        let recall = try await measureRecall(count: 5_000, dim: 128, efSearch: 50)
        print("Recall@10 | 5K vectors, 128d, ef=50: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.90, "Expected >90% recall@10")
    }

    func testRecall_1K_384d() async throws {
        let recall = try await measureRecall(count: 1_000, dim: 384, efSearch: 50)
        print("Recall@10 | 1K vectors, 384d, ef=50: \(pct(recall))")
        XCTAssertGreaterThan(recall, 0.90, "Expected >90% recall@10")
    }

    /// PRD acceptance criterion: >95% recall@10 at efSearch=50, 10K vectors, 384d.
    func testRecall_10K_384d() async throws {
        let recall = try await measureRecall(count: 10_000, dim: 384, efSearch: 50)
        print("Recall@10 | 10K vectors, 384d, ef=50: \(pct(recall))  ← PRD target")
        XCTAssertGreaterThan(recall, 0.95, "PRD target: >95% recall@10 at 10K/384d/ef=50")
    }

    // ── Recall vs efSearch (quality-speed tradeoff) ────────────────────

    func testEfSearchSweep() async throws {
        let efValues = [10, 50, 100, 200]
        let count = 1_000
        let dim = 32

        print("\nRecall@10 vs efSearch | \(count) vectors, \(dim)d")
        print("efSearch | recall  | note")
        print("---------|---------|-----")

        for ef in efValues {
            let recall = try await measureRecall(count: count, dim: dim, efSearch: ef)
            let note = ef == 50 ? "← default" : ""
            print("\(String(ef).padding(toLength: 8, withPad: " ", startingAt: 0)) | \(pct(recall)) | \(note)")
        }

        // Minimum quality bar at default efSearch
        let defaultRecall = try await measureRecall(count: count, dim: dim, efSearch: 50)
        XCTAssertGreaterThan(defaultRecall, 0.88,
                             "Default efSearch=50 should give >88% recall")
    }

    // ── Query Latency ──────────────────────────────────────────────────

    func testQueryLatency_1K_384d() async throws {
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        let index = HNSWIndex(dimension: 384, metric: CosineDistance(), config: config)

        for _ in 0..<1_000 {
            let v = Vector((0..<384).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: UUID())
        }

        let query = Vector((0..<384).map { _ in Float.random(in: -1...1) })

        // XCTest.measure runs the block 10 times and reports average
        measure {
            let expectation = expectation(description: "search")
            Task {
                _ = await index.search(query: query, k: 10)
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 5)
        }
    }

    // ── Compaction ─────────────────────────────────────────────────────

    func testCompaction() async throws {
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Remove 30 nodes
        for id in ids.prefix(30) {
            _ = await index.remove(id: id)
        }

        var liveCount = await index.liveCount
        var totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 100) // tombstones still in slots

        // Compact
        try await index.compact()

        liveCount = await index.liveCount
        totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 70) // no more tombstones

        // Search still works after compaction
        let query = Vector((0..<8).map { _ in Float.random(in: -1...1) })
        let results = await index.search(query: query, k: 10)
        XCTAssertEqual(results.count, 10)

        // Removed nodes must not appear in results
        let removedSet = Set(ids.prefix(30))
        for result in results {
            XCTAssertFalse(removedSet.contains(result.id), "Compacted index returned a removed node")
        }
    }

    func testNonisolatedDimension() {
        // dimension is nonisolated — accessible without await.
        // Must capture to local before XCTAssertEqual (autoclosure limitation).
        let index = HNSWIndex(dimension: 384)
        let dim = index.dimension   // no await
        XCTAssertEqual(dim, 384)
    }

    func testNonisolatedConfiguration() {
        let config = HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 30)
        let index = HNSWIndex(dimension: 64, config: config)
        let m = index.configuration.m           // no await
        let ef = index.configuration.efSearch   // no await
        XCTAssertEqual(m, 8)
        XCTAssertEqual(ef, 30)
    }

    // ── Helpers ───────────────────────────────────────────────────────

    /// Measures recall@10 by comparing HNSW results against BruteForce ground truth.
    private func measureRecall(
        count: Int,
        dim: Int,
        efSearch: Int,
        queries: Int = 20
    ) async throws -> Double {
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: efSearch)
        let hnsw = HNSWIndex(dimension: dim, metric: CosineDistance(), config: config)
        let brute = BruteForceIndex(dimension: dim, metric: CosineDistance())

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

    private func pct(_ recall: Double) -> String {
        String(format: "%.1f%%", recall * 100)
    }
}
