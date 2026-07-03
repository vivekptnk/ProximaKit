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

    // ══════════════════════════════════════════════════════════════════
    // MARK: - Graph-aware filtering on the quantized indexes (ADR-008 2nd)
    //
    // Ports of the beam above onto QuantizedHNSWIndex (ADC scoring) and
    // ScalarQuantizedHNSWIndex (dequantize scoring). Recall floors below are
    // NOT copied from the full-precision 0.9 target — they are the honest
    // values measured from the seeded release-build runs recorded in this
    // wave's output, asserted with margin. Pure-ADC PQ trails full precision
    // (lossy 32×-compressed distances reorder near-ties); the rerank-enabled
    // PQ path recovers to the full-precision band; SQ (only ~4× lossy) sits
    // just under it.
    // ══════════════════════════════════════════════════════════════════

    private static let pqSubspaceCount = 8
    private static let pqSeed: UInt64 = 0xF117_E2AD_C0DE_0001
    private static let rerankDepth = 4 * k

    private struct QuantFixture: Sendable {
        let pq: QuantizedHNSWIndex          // built with retainOriginals: true
        let sq: ScalarQuantizedHNSWIndex
        let brute: BruteForceIndex          // exact filtered ground truth
        let ids: [UUID]
        let vectorByID: [UUID: Vector]
        let queries: [Vector]

        var matching10: Set<UUID> { matching(stride: 10) }   // 200 of 2000
        var matching1: Set<UUID> { matching(stride: 100) }   // 20 of 2000
        var matching01: Set<UUID> { [ids[500], ids[1500]] }  // exactly 2

        private func matching(stride strideBy: Int) -> Set<UUID> {
            Set(stride(from: 0, to: ids.count, by: strideBy).map { ids[$0] })
        }
    }

    /// Built once for the whole class — same corpus shape as the HNSW
    /// fixture (2000 × 32d Euclidean), fully seeded (SeededRandom data,
    /// levelSeed topology, PQ training seed). Retains originals so the same
    /// index serves both the pure-ADC path (`rerankDepth: 0`) and the
    /// rerank-enabled path (default `4·k`).
    private static let quantFixtureTask = Task<QuantFixture, Error> {
        let config = HNSWConfiguration(
            m: 16,
            efConstruction: 200,
            efSearch: efSearch,
            levelSeed: 0xF117_E2ED_BEA3_0002
        )
        var rng = SeededRandom(seed: 0xF117_E25E_1EC7_0002)
        var vectors: [Vector] = []
        var ids: [UUID] = []
        for _ in 0..<corpusSize {
            vectors.append(Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }))
            ids.append(UUID())
        }
        let queries = (0..<queryCount).map { _ in
            Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
        }

        let brute = BruteForceIndex(dimension: dimension, metric: EuclideanDistance())
        for i in 0..<corpusSize { try await brute.add(vectors[i], id: ids[i]) }

        let pq = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dimension,
            hnswConfig: config,
            pqConfig: PQConfiguration(
                subspaceCount: pqSubspaceCount, trainingIterations: 20, seed: pqSeed
            ),
            retainOriginals: true
        )
        let sq = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dimension,
            hnswConfig: config, metric: .euclidean
        )

        let vectorByID = Dictionary(uniqueKeysWithValues: zip(ids, vectors))
        return QuantFixture(
            pq: pq, sq: sq, brute: brute, ids: ids,
            vectorByID: vectorByID, queries: queries
        )
    }

    private func quantFixture() async throws -> QuantFixture {
        try await Self.quantFixtureTask.value
    }

    /// Mean recall@k of `results` vs the exact brute-force-filtered ranking,
    /// asserting fill and predicate-membership per query along the way.
    private func meanRecall(
        _ fixture: QuantFixture,
        matching: Set<UUID>,
        expectedCount: Int,
        search: (Vector) async throws -> [SearchResult]
    ) async throws -> Double {
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        var recallSum = 0.0
        for query in fixture.queries {
            let results = try await search(query)
            XCTAssertEqual(results.count, expectedCount,
                           "expected \(expectedCount) filled results")
            for result in results {
                XCTAssertTrue(matching.contains(result.id), "non-matching id in filtered results")
            }
            let truth = Set(await fixture.brute.search(query: query, k: Self.k, filter: filter).map(\.id))
            recallSum += Double(truth.intersection(results.map(\.id)).count) / Double(truth.count)
        }
        return recallSum / Double(fixture.queries.count)
    }

    // ── PQ pure-ADC: fills k at 10% / 1% ──────────────────────────────

    func testQuantizedPureADCTenPercentFillsK() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching10
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            try await fixture.pq.search(query: q, k: Self.k, rerankDepth: 0, filter: filter)
        }
        // Measured (seeded, debug == release): 0.745. Floor 0.65 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.65,
            "PQ pure-ADC recall@\(Self.k) at ~10% — below full-precision 0.9 as expected (got \(recall))")
    }

    func testQuantizedPureADCOnePercentFillsK() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            try await fixture.pq.search(query: q, k: Self.k, rerankDepth: 0, filter: filter)
        }
        // Measured (seeded, debug == release): 0.870. Floor 0.78 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.78,
            "PQ pure-ADC recall@\(Self.k) at ~1% — below full-precision 0.9 as expected (got \(recall))")
    }

    // ── PQ rerank recovers close to full precision ────────────────────

    func testQuantizedRerankTenPercentRecoversRecall() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching10
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        // Default search auto-reranks (4·k) because originals are retained.
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            await fixture.pq.search(query: q, k: Self.k, filter: filter)
        }
        // Measured (seeded, debug == release): 0.995. Floor 0.95 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.95,
            "PQ rerank recall@\(Self.k) at ~10% must recover near full precision (got \(recall))")
    }

    func testQuantizedRerankOnePercentRecoversRecall() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            await fixture.pq.search(query: q, k: Self.k, filter: filter)
        }
        // Measured (seeded, debug == release): 1.000. Floor 0.95 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.95,
            "PQ rerank recall@\(Self.k) at ~1% must recover near full precision (got \(recall))")
    }

    // ── SQ: fills k at 10% / 1% ───────────────────────────────────────

    func testScalarQuantizedTenPercentFillsK() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching10
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            await fixture.sq.search(query: q, k: Self.k, filter: filter)
        }
        // Measured (seeded, debug == release): 1.000. Floor 0.95 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.95,
            "SQ recall@\(Self.k) at ~10% — ~4x-lossy SQ sits just under full precision (got \(recall))")
    }

    func testScalarQuantizedOnePercentFillsK() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let recall = try await meanRecall(fixture, matching: matching, expectedCount: Self.k) { q in
            await fixture.sq.search(query: q, k: Self.k, filter: filter)
        }
        // Measured (seeded, debug == release): 1.000. Floor 0.95 (margin).
        XCTAssertGreaterThanOrEqual(recall, 0.95,
            "SQ recall@\(Self.k) at ~1% — ~4x-lossy SQ sits just under full precision (got \(recall))")
    }

    // ── 0.1% selectivity: exact matching set on both indexes ──────────

    func testQuantizedPointOnePercentReturnsExactSet() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching01
        XCTAssertEqual(matching.count, 2)
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        for query in fixture.queries {
            let pqADC = try await fixture.pq.search(query: query, k: Self.k, rerankDepth: 0, filter: filter)
            XCTAssertEqual(Set(pqADC.map(\.id)), matching,
                           "PQ pure-ADC 0.1% must return exactly the matching set")
            let pqRerank = await fixture.pq.search(query: query, k: Self.k, filter: filter)
            XCTAssertEqual(Set(pqRerank.map(\.id)), matching)
            // Rerank distances are exact, so order matches the brute ranking.
            let truth = await fixture.brute.search(query: query, k: Self.k, filter: filter)
            XCTAssertEqual(pqRerank.map(\.id), truth.map(\.id),
                           "PQ rerank 0.1% order must match brute force")

            let sq = await fixture.sq.search(query: query, k: Self.k, filter: filter)
            XCTAssertEqual(Set(sq.map(\.id)), matching,
                           "SQ 0.1% must return exactly the matching set")
        }
    }

    // ── Rerank composition: filtered candidates only, counted to depth ─

    /// Requirement M3-F31 #2. The graph-aware filtered rerank must compose
    /// with `rerankDepth` exactly as post-filter did: the exact re-score
    /// sees the top-`rerankDepth` FILTERED candidates by ADC (never depth
    /// slots wasted on non-matching nodes), then truncates to k.
    ///
    /// Oracle: the pure-ADC filtered beam (`rerankDepth: 0`) surfaces the
    /// same filtered candidate pool by ADC order. Re-scoring its top
    /// `rerankDepth` exactly (against the known corpus vectors) and taking k
    /// must reproduce the index's own rerank result — id-for-id and with
    /// identical exact distances.
    func testQuantizedRerankComposesWithDepthLikePostFilter() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching10          // 200 matches ≫ rerankDepth
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }
        let metric = EuclideanDistance()
        let depth = Self.rerankDepth

        for query in fixture.queries {
            // Index rerank path.
            let reranked = try await fixture.pq.search(
                query: query, k: Self.k, rerankDepth: depth, filter: filter
            )

            // Oracle: filtered candidates by ADC (same beam, target = depth),
            // top-`depth` re-scored exactly, then top-k.
            let adcPool = try await fixture.pq.search(
                query: query, k: depth, rerankDepth: 0, filter: filter
            )
            let oracle = adcPool.prefix(depth).map { r in
                (id: r.id, dist: metric.distance(query, fixture.vectorByID[r.id]!))
            }
            .sorted { $0.dist < $1.dist }
            .prefix(Self.k)

            XCTAssertEqual(reranked.map(\.id), oracle.map(\.id),
                "rerank must consume ONLY filtered candidates, top-\(depth) by ADC, then top-k")
            for r in reranked {
                // Distances are exact L2 — proves the re-score ran on the
                // filtered set, not raw ADC approximations.
                XCTAssertEqual(r.distance, metric.distance(query, fixture.vectorByID[r.id]!), accuracy: 1e-4,
                               "reranked distance must be exact L2 on the filtered candidate")
                XCTAssertTrue(matching.contains(r.id))
            }
        }
    }

    // ── Under-fill control: post-filter could not deliver full k ──────

    /// Requirement M3-F31 #3 control, mirroring `testOnePercentControlPost…`.
    /// Emulating the retired post-filter pipeline on both quantized indexes
    /// (apply the predicate to the unfiltered ef-beam) under-fills k on every
    /// seeded query at ~1% selectivity, while the graph-aware beam fills all.
    func testQuantizedOnePercentControlPostFilterUnderfillsK() async throws {
        let fixture = try await quantFixture()
        let matching = fixture.matching1
        let filter: @Sendable (UUID) -> Bool = { matching.contains($0) }

        for query in fixture.queries {
            // PQ post-filter emulation: unfiltered ADC beam of ef, filtered after.
            let pqBeam = try await fixture.pq.search(
                query: query, k: Self.efSearch, efSearch: Self.efSearch, rerankDepth: 0
            )
            XCTAssertEqual(pqBeam.count, Self.efSearch, "control needs the full ef candidate set")
            let pqSurvivors = pqBeam.filter { matching.contains($0.id) }
            XCTAssertLessThan(pqSurvivors.count, Self.k,
                "PQ post-filter emulation must under-fill at ~1% selectivity")
            let pqGraph = try await fixture.pq.search(
                query: query, k: Self.k, rerankDepth: 0, filter: filter
            )
            XCTAssertEqual(pqGraph.count, Self.k, "graph-aware PQ must fill k")
            XCTAssertGreaterThan(pqGraph.count, pqSurvivors.count)

            // SQ post-filter emulation.
            let sqBeam = await fixture.sq.search(
                query: query, k: Self.efSearch, efSearch: Self.efSearch
            )
            XCTAssertEqual(sqBeam.count, Self.efSearch)
            let sqSurvivors = sqBeam.filter { matching.contains($0.id) }
            XCTAssertLessThan(sqSurvivors.count, Self.k,
                "SQ post-filter emulation must under-fill at ~1% selectivity")
            let sqGraph = await fixture.sq.search(query: query, k: Self.k, filter: filter)
            XCTAssertEqual(sqGraph.count, Self.k, "graph-aware SQ must fill k")
            XCTAssertGreaterThan(sqGraph.count, sqSurvivors.count)
        }
    }

    // ── filter == nil fast path is byte-identical (zero behavior change) ─

    /// The unfiltered path must be structurally untouched: a `nil` filter
    /// must return exactly what the pre-port beam did. We pin that by
    /// comparing `filter: nil` against an all-true predicate — the former
    /// takes the original filter-blind beam, the latter the new graph-aware
    /// beam that accepts everything; at full acceptance the adaptive width
    /// collapses to `ef`, so both must agree id-for-id and distance-for-
    /// distance.
    func testQuantizedNilFilterMatchesAllTruePredicate() async throws {
        let fixture = try await quantFixture()
        let all: @Sendable (UUID) -> Bool = { _ in true }
        for query in fixture.queries.prefix(5) {
            let pqNil = try await fixture.pq.search(query: query, k: Self.k, rerankDepth: 0, filter: nil)
            let pqAll = try await fixture.pq.search(query: query, k: Self.k, rerankDepth: 0, filter: all)
            XCTAssertEqual(pqNil.map(\.id), pqAll.map(\.id),
                           "PQ nil-filter fast path must equal all-true graph-aware path")
            XCTAssertEqual(pqNil.map(\.distance), pqAll.map(\.distance))

            let sqNil = await fixture.sq.search(query: query, k: Self.k, filter: nil)
            let sqAll = await fixture.sq.search(query: query, k: Self.k, filter: all)
            XCTAssertEqual(sqNil.map(\.id), sqAll.map(\.id),
                           "SQ nil-filter fast path must equal all-true graph-aware path")
            XCTAssertEqual(sqNil.map(\.distance), sqAll.map(\.distance))
        }
    }
}
