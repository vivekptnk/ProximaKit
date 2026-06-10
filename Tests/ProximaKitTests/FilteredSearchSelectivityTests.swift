// FilteredSearchSelectivityTests.swift
// ProximaKitTests
//
// Selectivity acceptance tests for graph-aware filtered search in HNSWIndex
// (ADR-008 addendum). The retired post-filter strategy applied the predicate
// only at result materialization: with efSearch = 50 layer-0 candidates and a
// 1%-selective predicate, ~0.5 candidates pass on average, so searches came
// back with far fewer than k results. The graph-aware beam applies the
// predicate DURING traversal (rejected nodes route, never occupy result
// candidacy) and adaptively widens ef, so selective filters still fill k.
//
// Corpus: 2000 seeded vectors. Both the data (SeededRandom) and the graph
// topology (levelSeed) are deterministic, per the CHA-105 flake policy —
// every threshold asserted here measures the same dataset on every run.
// Ground truth: BruteForceIndex with the same filter (exact under any
// filter, ADR-008).

import XCTest
@testable import ProximaKit

final class FilteredSearchSelectivityTests: XCTestCase {

    // ── Fixture ───────────────────────────────────────────────────────

    private static let corpusSize = 2000
    private static let dimension = 32
    private static let k = 10
    private static let efSearch = 50
    private static let queryCount = 20

    private struct Fixture: Sendable {
        let hnsw: HNSWIndex
        let brute: BruteForceIndex
        /// IDs in insertion order — selectivity sets are index-based slices.
        let ids: [UUID]
        let queries: [Vector]

        /// ~10% pass rate: every 10th insertion (200 of 2000).
        var matching10: Set<UUID> { matching(stride: 10) }
        /// ~1% pass rate: every 100th insertion (20 of 2000).
        var matching1: Set<UUID> { matching(stride: 100) }
        /// ~0.1% pass rate: exactly 2 of 2000.
        var matching01: Set<UUID> { [ids[500], ids[1500]] }

        private func matching(stride strideBy: Int) -> Set<UUID> {
            Set(stride(from: 0, to: ids.count, by: strideBy).map { ids[$0] })
        }
    }

    /// Built once for the whole class — a 2000-vector HNSW build per test
    /// would dominate runtime. The fixture is never mutated by tests.
    private static let fixtureTask = Task<Fixture, Error> {
        let config = HNSWConfiguration(
            m: 16,
            efConstruction: 200,
            efSearch: efSearch,
            levelSeed: 0xF117_E2ED_BEA3_0001
        )
        let metric = EuclideanDistance()
        let hnsw = HNSWIndex(dimension: dimension, metric: metric, config: config)
        let brute = BruteForceIndex(dimension: dimension, metric: metric)

        var rng = SeededRandom(seed: 0xF117_E25E_1EC7_0001)
        var ids: [UUID] = []
        for _ in 0..<corpusSize {
            let v = Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
            let id = UUID()
            ids.append(id)
            try await hnsw.add(v, id: id)
            try await brute.add(v, id: id)
        }

        let queries = (0..<queryCount).map { _ in
            Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
        }
        return Fixture(hnsw: hnsw, brute: brute, ids: ids, queries: queries)
    }

    private func fixture() async throws -> Fixture {
        try await Self.fixtureTask.value
    }

    // ── ~10% selectivity: fills k, high recall ────────────────────────

    func testTenPercentFilterFillsKWithHighRecall() async throws {
        let fixture = try await fixture()
        let matching = fixture.matching10
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }

        var recallSum = 0.0
        for query in fixture.queries {
            let results = await fixture.hnsw.search(query: query, k: Self.k, filter: filter)

            // (a) Fills k: 200 live matching vectors >> k.
            XCTAssertEqual(results.count, min(Self.k, matching.count),
                           "10% filter must fill k = \(Self.k)")
            // (b) Every returned id satisfies the filter.
            for result in results {
                XCTAssertTrue(matching.contains(result.id), "non-matching id in filtered results")
            }

            // (c) Recall@k vs brute-force-filtered ground truth.
            let truth = Set(await fixture.brute.search(query: query, k: Self.k, filter: filter).map(\.id))
            recallSum += Double(truth.intersection(results.map(\.id)).count) / Double(truth.count)
        }

        let recall = recallSum / Double(fixture.queries.count)
        XCTAssertGreaterThanOrEqual(recall, 0.9,
            "recall@\(Self.k) at ~10% selectivity must be >= 0.9 (got \(recall))")
    }

    // ── ~1% selectivity: fills k where post-filter under-filled ───────

    func testOnePercentFilterFillsKWithHighRecall() async throws {
        let fixture = try await fixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }

        var recallSum = 0.0
        for query in fixture.queries {
            let results = await fixture.hnsw.search(query: query, k: Self.k, filter: filter)

            // (a) Fills k: 20 live matching vectors >= k = 10.
            XCTAssertEqual(results.count, min(Self.k, matching.count),
                           "1% filter must fill k = \(Self.k)")
            // (b) Every returned id satisfies the filter.
            for result in results {
                XCTAssertTrue(matching.contains(result.id), "non-matching id in filtered results")
            }

            // (c) Recall@k vs brute-force-filtered ground truth.
            let truth = Set(await fixture.brute.search(query: query, k: Self.k, filter: filter).map(\.id))
            recallSum += Double(truth.intersection(results.map(\.id)).count) / Double(truth.count)
        }

        let recall = recallSum / Double(fixture.queries.count)
        XCTAssertGreaterThanOrEqual(recall, 0.9,
            "recall@\(Self.k) at ~1% selectivity must be >= 0.9 (got \(recall))")
    }

    /// Control documenting the improvement over the retired strategy:
    /// post-filter took the ef(=50) unfiltered layer-0 candidates and applied
    /// the predicate at materialization — exactly what this loop emulates.
    /// At ~1% selectivity that demonstrably under-fills k on EVERY seeded
    /// query, while the graph-aware path (asserted above) fills all of them.
    func testOnePercentControlPostFilterUnderfillsK() async throws {
        let fixture = try await fixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }

        for query in fixture.queries {
            // The retired pipeline: ef = 50 beam, filter applied afterwards.
            // Asking for k = ef with no filter returns the full candidate set.
            let unfilteredBeam = await fixture.hnsw.search(
                query: query, k: Self.efSearch, efSearch: Self.efSearch
            )
            XCTAssertEqual(unfilteredBeam.count, Self.efSearch,
                           "control needs the full ef candidate set")
            let survivors = unfilteredBeam.filter { matching.contains($0.id) }
            XCTAssertLessThan(survivors.count, Self.k,
                "post-filter emulation should under-fill at ~1% selectivity — "
                + "if this ever fills k, the control no longer documents an improvement")

            let graphAware = await fixture.hnsw.search(query: query, k: Self.k, filter: filter)
            XCTAssertGreaterThan(graphAware.count, survivors.count,
                "graph-aware filtering must beat the post-filter result count")
            XCTAssertEqual(graphAware.count, Self.k)
        }
    }

    // ── ~0.1% selectivity: exact set ──────────────────────────────────

    func testPointOnePercentFilterReturnsExactMatchingSet() async throws {
        let fixture = try await fixture()
        let matching = fixture.matching01
        XCTAssertEqual(matching.count, 2, "0.1% of 2000 = 2 matching vectors")
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }

        for query in fixture.queries {
            let results = await fixture.hnsw.search(query: query, k: Self.k, filter: filter)

            // min(k, live matching) = 2, and it must be EXACTLY the
            // matching set — the widened beam explores until the predicate's
            // entire reachable support is found.
            XCTAssertEqual(results.count, min(Self.k, matching.count))
            XCTAssertEqual(Set(results.map(\.id)), matching,
                           "0.1% filter must return exactly the matching set")

            // Order must agree with the exact brute-force-filtered ranking.
            let truth = await fixture.brute.search(query: query, k: Self.k, filter: filter)
            XCTAssertEqual(results.map(\.id), truth.map(\.id),
                           "results must be sorted by distance, matching brute force")
        }
    }

    // ── Filter × tombstone interplay, and the empty-filter edge ───────

    /// Liveness is checked before the predicate inside the beam: a removed
    /// matching vector must vanish from filtered results, leaving exactly
    /// the surviving match. (Separate small index — the shared fixture is
    /// never mutated.)
    func testFilteredSearchSkipsTombstonedMatches() async throws {
        let config = HNSWConfiguration(
            m: 16,
            efConstruction: 100,
            efSearch: 32,
            autoCompactionThreshold: nil,  // keep the tombstone in the graph
            levelSeed: 0xF117_E2DE_AD00_0001
        )
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: config)
        var rng = SeededRandom(seed: 0xF117_E2DE_AD5E_0001)
        var ids: [UUID] = []
        for _ in 0..<200 {
            let id = UUID()
            ids.append(id)
            try await index.add(
                Vector((0..<8).map { _ in Float.random(in: -1...1, using: &rng) }), id: id
            )
        }

        let matching: Set<UUID> = [ids[40], ids[160]]
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let query = Vector((0..<8).map { _ in Float.random(in: -1...1, using: &rng) })

        let before = await index.search(query: query, k: 5, filter: filter)
        XCTAssertEqual(Set(before.map(\.id)), matching)

        await index.remove(id: ids[40])

        let after = await index.search(query: query, k: 5, filter: filter)
        XCTAssertEqual(after.map(\.id), [ids[160]],
                       "tombstoned match must not surface; the live match must")
    }

    func testFilterMatchingNothingReturnsEmpty() async throws {
        let fixture = try await fixture()
        let results = await fixture.hnsw.search(
            query: fixture.queries[0], k: Self.k, filter: { _ in false }
        )
        XCTAssertTrue(results.isEmpty, "a filter matching nothing must return []")
    }
}
