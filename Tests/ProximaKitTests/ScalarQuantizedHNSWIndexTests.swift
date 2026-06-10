// ScalarQuantizedHNSWIndexTests.swift
// ProximaKitTests
//
// Tests for ScalarQuantizedHNSWIndex: build, search across metrics, recall
// bounds vs BruteForceIndex (the ADR-007 ~1-2% degradation claim), memory
// accounting (no full-precision vectors retained), removal, and the
// identity-based tombstone liveness scenarios.

import XCTest
@testable import ProximaKit

final class ScalarQuantizedHNSWIndexTests: XCTestCase {

    // ── Seeded RNG (deterministic corpus generation) ─────────────────


    // ── Helpers ──────────────────────────────────────────────────────

    private func randomVectors(
        count: Int, dimension: Int, rng: inout SeededRandom
    ) -> [Vector] {
        (0..<count).map { _ in
            Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
        }
    }

    /// Generates clustered vectors for realistic recall measurement.
    /// Deterministic: identical output for the same seed.
    private func clusteredVectors(
        count: Int, dimension: Int, clusters: Int, rng: inout SeededRandom
    ) -> [Vector] {
        var vectors = [Vector]()
        vectors.reserveCapacity(count)
        let perCluster = count / clusters

        for _ in 0..<clusters {
            let center = (0..<dimension).map { _ in Float.random(in: -2...2, using: &rng) }
            for _ in 0..<perCluster {
                let v = (0..<dimension).map { d in
                    center[d] + Float.random(in: -0.5...0.5, using: &rng)
                }
                vectors.append(Vector(v))
            }
        }

        // Fill any remainder from rounding
        while vectors.count < count {
            vectors.append(Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }))
        }

        return vectors
    }

    /// Average Recall@k of the scalar-quantized index against BruteForceIndex
    /// ground truth, both using the same metric.
    private func measureRecall(
        vectors: [Vector],
        ids: [UUID],
        metric: DistanceMetricType,
        k: Int,
        numQueries: Int,
        efSearch: Int
    ) async throws -> Float {
        let dim = vectors[0].dimension

        // levelSeed makes graph construction deterministic, so the measured
        // recall is a constant for a given seed — not a per-run sample.
        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(
                m: 16, efConstruction: 200, efSearch: 100, levelSeed: 0x5EED_60AF
            ),
            metric: metric
        )

        let bfIndex = BruteForceIndex(dimension: dim, metric: metric.makeMetric())
        for i in 0..<vectors.count {
            try await bfIndex.add(vectors[i], id: ids[i])
        }

        var totalRecall: Float = 0
        // Stride across the whole corpus so queries probe every cluster —
        // vectors[0..<numQueries] would all land in cluster 0 (clusteredVectors
        // lays clusters out contiguously).
        let queryStride = max(1, vectors.count / numQueries)
        for q in 0..<numQueries {
            let query = vectors[q * queryStride]

            let exact = await bfIndex.search(query: query, k: k)
            let groundTruth = Set(exact.map(\.id))

            let sqResults = await sqIndex.search(query: query, k: k, efSearch: efSearch)
            let sqTopK = Set(sqResults.map(\.id))

            totalRecall += Float(groundTruth.intersection(sqTopK).count) / Float(k)
        }
        return totalRecall / Float(numQueries)
    }

    // ── Build Tests ──────────────────────────────────────────────────

    func testBuildCreatesIndex() async throws {
        let dim = 32
        let n = 300
        var rng = SeededRandom(seed: 1)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            metric: .euclidean
        )

        let count = await sqIndex.count
        let liveCount = await sqIndex.liveCount
        XCTAssertEqual(count, n)
        XCTAssertEqual(liveCount, n)
        XCTAssertEqual(sqIndex.metricType, .euclidean)
    }

    /// Duplicate ids follow HNSWIndex's replace-on-duplicate semantics:
    /// the LAST vector wins and the built index has one node per distinct id.
    func testBuildWithDuplicateIdsKeepsLastVector() async throws {
        let idA = UUID()
        let idB = UUID()
        let vectors = [Vector([100, 100]), Vector([50, 50]), Vector([1, 1])]
        let ids = [idA, idB, idA]  // idA appears twice — [1, 1] must win

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: 2,
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            metric: .euclidean
        )

        let count = await sqIndex.count
        let liveCount = await sqIndex.liveCount
        XCTAssertEqual(count, 2, "one node per distinct id")
        XCTAssertEqual(liveCount, 2)

        let results = await sqIndex.search(query: Vector([1, 1]), k: 2)
        XCTAssertEqual(results.first?.id, idA)
        // Quantization error at maxAbs = 1 is at most scale/2 ≈ 0.004 per axis.
        XCTAssertEqual(results.first?.distance ?? -1, 0, accuracy: 0.01,
                       "must match the re-added vector [1,1], not the stale [100,100]")
    }

    // ── Search Tests ─────────────────────────────────────────────────

    func testSearchReturnsSortedResults() async throws {
        let dim = 32
        let n = 500
        var rng = SeededRandom(seed: 2)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50),
            metric: .euclidean
        )

        let results = await sqIndex.search(query: vectors[0], k: 10)

        XCTAssertGreaterThan(results.count, 0)
        XCTAssertLessThanOrEqual(results.count, 10)
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i - 1].distance, results[i].distance)
        }
        // The query is a database vector — it must come back first, with only
        // quantization error in the distance.
        XCTAssertEqual(results.first?.id, ids[0])
    }

    func testSearchWithDimensionMismatchReturnsEmpty() async throws {
        let dim = 16
        let n = 100
        var rng = SeededRandom(seed: 3)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            metric: .cosine
        )

        let wrongDimQuery = Vector((0..<32).map { Float($0) })
        let results = await sqIndex.search(query: wrongDimQuery, k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchWithFilter() async throws {
        let dim = 16
        let n = 200
        var rng = SeededRandom(seed: 4)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }
        let allowedSet = Set(ids.prefix(50))

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50),
            metric: .euclidean
        )

        let results = await sqIndex.search(query: vectors[0], k: 10) { id in
            allowedSet.contains(id)
        }

        XCTAssertFalse(results.isEmpty)
        for result in results {
            XCTAssertTrue(allowedSet.contains(result.id))
        }
    }

    // ── Recall Quality Tests (ADR-007 accuracy claim) ────────────────

    /// Euclidean Recall@10 vs BruteForceIndex on 1000 clustered vectors must
    /// stay >= 0.95 — the "~1-2% degradation" roadmap claim with headroom for
    /// the HNSW graph's own (full-precision) approximation.
    func testRecallAt10EuclideanOnClusteredData() async throws {
        var rng = SeededRandom(seed: 0x5EED_0001)
        let vectors = clusteredVectors(count: 1000, dimension: 64, clusters: 10, rng: &rng)
        let ids = (0..<1000).map { _ in UUID() }

        let recall = try await measureRecall(
            vectors: vectors, ids: ids, metric: .euclidean,
            k: 10, numQueries: 25, efSearch: 250
        )
        XCTAssertGreaterThanOrEqual(recall, 0.95,
            "ScalarQuantized HNSW Recall@10 (euclidean) must be >= 0.95, got \(recall)")
    }

    /// Cosine Recall@10 — the metric PQ's ADC cannot serve — must stay >= 0.93.
    func testRecallAt10CosineOnClusteredData() async throws {
        var rng = SeededRandom(seed: 0x5EED_0002)
        let vectors = clusteredVectors(count: 1000, dimension: 64, clusters: 10, rng: &rng)
        let ids = (0..<1000).map { _ in UUID() }

        let recall = try await measureRecall(
            vectors: vectors, ids: ids, metric: .cosine,
            k: 10, numQueries: 25, efSearch: 250
        )
        XCTAssertGreaterThanOrEqual(recall, 0.93,
            "ScalarQuantized HNSW Recall@10 (cosine) must be >= 0.93, got \(recall)")
    }

    // ── Memory Tests (the ≈4× claim, structurally) ───────────────────

    /// Code storage must be exactly dimension + 4 bytes per slot, and the
    /// savings ratio vs Float32 must validate the ≈3.96× math at 384d.
    func testMemoryAccountingMatchesFourXClaim() async throws {
        let dim = 384
        let n = 200
        var rng = SeededRandom(seed: 5)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            metric: .euclidean
        )

        let codeBytes = await sqIndex.codeStorageBytes
        let fullBytes = await sqIndex.equivalentFullPrecisionBytes
        let ratio = await sqIndex.memorySavingsRatio

        XCTAssertEqual(codeBytes, n * (dim + 4))    // 200 * 388 = 77600
        XCTAssertEqual(fullBytes, n * dim * 4)      // 200 * 1536 = 307200
        XCTAssertGreaterThan(ratio, 3.9,
            "Memory savings should be ≈3.96x at 384d (ADR-007). Got \(ratio)x")
    }

    /// Structural guarantee behind the memory claim: after build, the index
    /// must retain NO full-precision vector storage — only Int8 codes and
    /// Float32 scales.
    func testNoFullPrecisionVectorsRetainedAfterBuild() async throws {
        let dim = 16
        let n = 50
        var rng = SeededRandom(seed: 6)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            metric: .euclidean
        )

        let mirror = Mirror(reflecting: sqIndex)
        let children = Array(mirror.children)
        XCTAssertFalse(children.isEmpty, "actor reflection should expose stored properties")
        for child in children {
            XCTAssertFalse(child.value is [Vector],
                "stored property '\(child.label ?? "?")' retains [Vector] — "
                + "full-precision vectors must be discarded at build time")
            XCTAssertFalse(child.value is Vector,
                "stored property '\(child.label ?? "?")' retains a Vector")
        }

        // And the codes really are one byte per dimension per slot.
        let codes = await sqIndex.codes
        XCTAssertEqual(codes.count, n)
        XCTAssertTrue(codes.allSatisfy { $0.count == dim })
    }

    // ── Remove Tests ─────────────────────────────────────────────────

    func testRemoveReducesLiveCount() async throws {
        let dim = 16
        let n = 100
        var rng = SeededRandom(seed: 7)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            metric: .euclidean
        )

        let removed = await sqIndex.remove(id: ids[0])
        XCTAssertTrue(removed)

        let liveCount = await sqIndex.liveCount
        let count = await sqIndex.count
        XCTAssertEqual(liveCount, n - 1)
        XCTAssertEqual(count, n, "removal tombstones the slot — count is unchanged")

        let results = await sqIndex.search(query: vectors[0], k: n)
        XCTAssertFalse(Set(results.map(\.id)).contains(ids[0]))
    }

    func testRemoveNonExistentReturnsFalse() async throws {
        let dim = 16
        let n = 50
        var rng = SeededRandom(seed: 8)
        let vectors = randomVectors(count: n, dimension: dim, rng: &rng)
        let ids = (0..<n).map { _ in UUID() }

        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            metric: .euclidean
        )

        let removed = await sqIndex.remove(id: UUID())
        XCTAssertFalse(removed)
    }

    // ── Tombstone Liveness Regression (TombstoneLivenessTests scenarios) ──

    /// Entry-point recovery in remove() must use identity-based liveness when
    /// scanning for a replacement entry point — no tombstoned ids may surface.
    func testEntryPointRecoverySkipsTombstones() async throws {
        var rng = SeededRandom(seed: 9)
        var vectors: [Vector] = []
        var ids: [UUID] = []
        for i in 0..<64 {
            vectors.append(Vector((0..<8).map { _ in
                Float.random(in: -1...1, using: &rng) + Float(i) * 0.01
            }))
            ids.append(UUID())
        }
        let sqIndex = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: 8,
            metric: .euclidean
        )

        // Remove half the nodes, including (eventually) the entry point.
        for id in ids.prefix(32) {
            await sqIndex.remove(id: id)
        }

        let live = await sqIndex.liveCount
        XCTAssertEqual(live, 32)
        let results = await sqIndex.search(query: vectors[40], k: 5)
        XCTAssertFalse(results.isEmpty, "search must still work after heavy removal")
        let removed = Set(ids.prefix(32))
        XCTAssertTrue(results.allSatisfy { !removed.contains($0.id) },
                      "no tombstoned ids in results")
    }

    /// The 20-trial re-add repro from TombstoneLivenessTests, applied to this
    /// index via build's replace-on-duplicate semantics: after removing the
    /// other node (forcing entry-point recovery), search must find the NEW
    /// vector at distance ~0 — never the stale tombstoned body.
    func testReAddThenRemoveEntryPointReturnsLiveVector() async throws {
        for trial in 0..<20 {
            let idA = UUID()
            let idB = UUID()
            let sqIndex = try await ScalarQuantizedHNSWIndex.build(
                vectors: [Vector([100, 100]), Vector([50, 50]), Vector([1, 1])],
                ids: [idA, idB, idA],  // re-add of A — [1, 1] wins
                metadata: [Data("old".utf8), nil, Data("new".utf8)],
                dimension: 2,
                hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
                metric: .euclidean
            )
            // Remove B; if B was the entry point, recovery must pick A's live node.
            await sqIndex.remove(id: idB)

            let results = await sqIndex.search(query: Vector([1, 1]), k: 2)
            XCTAssertEqual(results.count, 1, "trial \(trial): only A is live")
            XCTAssertEqual(results.first?.id, idA, "trial \(trial)")
            XCTAssertEqual(results.first?.distance ?? -1, 0, accuracy: 0.01,
                           "trial \(trial): must match the re-added vector [1,1], not the stale [100,100]")
            XCTAssertEqual(results.first?.metadata, Data("new".utf8),
                           "trial \(trial): metadata must come from the live slot")
        }
    }
}
