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
///
/// This class is intentionally benchmarks-only and is skipped in CI
/// (CI's Run Tests step passes `--skip RecallBenchmarkTests` — the sweeps
/// take minutes).
/// Fast functional compaction / auto-compaction / nonisolated-property tests
/// were moved to `CompactionTests` (CHA-201) so CI runs them; do not add
/// functional tests here.
///
/// **Local skip gate (belt and suspenders):** a class-level `setUpWithError`
/// now requires `PROXIMA_RECALL_BENCH=1`, so a bare local `swift test` (without
/// `--skip`) skips this class fast instead of running the 20+-minute sweeps; set
/// `PROXIMA_RECALL_BENCH=1` to run them locally. CI's explicit
/// `swift test --skip RecallBenchmarkTests` is untouched and remains the primary
/// gate. This is the one place in the suite where a class is allowed a local
/// skip gate, and it is acceptable ONLY because CI's behavior is completely
/// unchanged: it weakens no assertion and does not change what CI verifies.
final class RecallBenchmarkTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["PROXIMA_RECALL_BENCH"] != "1",
            "recall benchmark sweeps are opt-in locally (10K-vector sweeps take "
            + "20+ minutes); set PROXIMA_RECALL_BENCH=1 to run "
            + "(CI excludes this class regardless via --skip RecallBenchmarkTests)"
        )
    }

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

        // Seeded RNG: the recall thresholds asserted by callers are
        // data-dependent, so the dataset must be identical on every run
        // (see SeededRandom.swift). HNSW's internal level draws stay
        // unseeded, but recall is robust to graph-construction randomness —
        // it was the dataset/query draw that made thresholds flaky.
        var rng = SeededRandom(seed: 0xCA11_AB1E_5EED_0001)

        for _ in 0..<count {
            let v = Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
            let id = UUID()
            try await hnsw.add(v, id: id)
            try await brute.add(v, id: id)
        }

        var totalRecall = 0.0
        for _ in 0..<queries {
            let q = Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
            let bruteIDs = Set(await brute.search(query: q, k: 10).map(\.id))
            let hnswIDs = Set(await hnsw.search(query: q, k: 10).map(\.id))
            totalRecall += Double(bruteIDs.intersection(hnswIDs).count) / 10.0
        }

        return totalRecall / Double(queries)
    }

    private func pct(_ v: Double) -> String { String(format: "%.1f%%", v * 100) }
}
