// HybridIndexTests.swift
// ProximaKit
//
// Tests for HybridIndex:
// - Concurrent fan-out and ID consistency across legs
// - RRF invariant: top-k ⊇ (dense top-k ∩ sparse top-k) on constructed cases
// - Weighted-sum strategy behavior
// - Filter + tombstone behavior across both legs
// - Cross-leg remove

import XCTest
@testable import ProximaKit

// MARK: - Construction Helpers

private let testDim = 4

private func denseVector(_ values: [Float]) -> Vector {
    Vector(values)
}

// MARK: - Basic Hybrid Tests

final class HybridIndexTests: XCTestCase {

    private func makeHybrid(
        fusion: HybridFusionStrategy = .rrf()
    ) -> HybridIndex {
        let dense = BruteForceIndex(dimension: testDim, metric: EuclideanDistance())
        let sparse = SparseIndex()
        return HybridIndex(dense: dense, sparse: sparse, fusion: fusion)
    }

    func testUnifiedAddPopulatesBothLegs() async throws {
        let hybrid = makeHybrid()
        let id = UUID()
        try await hybrid.add(
            text: "hybrid retrieval",
            vector: denseVector([1, 0, 0, 0]),
            id: id
        )
        let dc = await hybrid.denseCount
        let sc = await hybrid.sparseCount
        XCTAssertEqual(dc, 1)
        XCTAssertEqual(sc, 1)
    }

    func testRemoveHitsBothLegs() async throws {
        let hybrid = makeHybrid()
        let a = UUID()
        let b = UUID()
        try await hybrid.add(text: "alpha keyword", vector: denseVector([1, 0, 0, 0]), id: a)
        try await hybrid.add(text: "beta keyword",  vector: denseVector([0, 1, 0, 0]), id: b)

        let removed = await hybrid.remove(id: a)
        XCTAssertTrue(removed)

        let dc = await hybrid.denseCount
        let sc = await hybrid.sparseCount
        XCTAssertEqual(dc, 1)
        XCTAssertEqual(sc, 1)

        let hits = await hybrid.search(
            queryText: "keyword",
            queryVector: denseVector([0, 1, 0, 0]),
            k: 5
        )
        let ids = Set(hits.map(\.id))
        XCTAssertFalse(ids.contains(a))
        XCTAssertTrue(ids.contains(b))
    }

    // MARK: - RRF Invariant
    //
    // The plan's hybrid invariant is:
    //   fused top-k ⊇ (dense top-k ∩ sparse top-k)
    //
    // Any document ranked in both legs' top-k lists should also appear in the
    // fused top-k (assuming candidatePoolK ≥ k). RRF is strict about this —
    // co-occurring docs accumulate two reciprocal contributions, so their fused
    // score cannot be beaten by a doc appearing in only one list with the same
    // rank.

    func testRRFInvariantOnConstructedCase() async throws {
        let hybrid = makeHybrid(fusion: .rrf(k: 60))

        // Build a deliberate overlap: ids A, B, C are strong in both legs;
        // ids D, E, F are dense-only; G, H, I are sparse-only.
        let shared = [UUID(), UUID(), UUID()]
        let denseOnly = [UUID(), UUID(), UUID()]
        let sparseOnly = [UUID(), UUID(), UUID()]

        // Shared docs: close dense vectors AND strong term hits.
        for (i, id) in shared.enumerated() {
            try await hybrid.add(
                text: "alpha bravo charlie shared\(i)",
                vector: denseVector([1 - Float(i) * 0.01, 0, 0, 0]),
                id: id
            )
        }

        // Dense-only docs: close dense vectors, weak/irrelevant text.
        for (i, id) in denseOnly.enumerated() {
            try await hybrid.add(
                text: "zzz yyy noise\(i)",
                vector: denseVector([0.99 - Float(i) * 0.01, 0, 0, 0.01]),
                id: id
            )
        }

        // Sparse-only docs: matching text, distant dense vectors.
        for (i, id) in sparseOnly.enumerated() {
            try await hybrid.add(
                text: "alpha bravo charlie extra\(i)",
                vector: denseVector([0, 0, 0, 1 + Float(i) * 0.05]),
                id: id
            )
        }

        // Query: matches "alpha bravo" lexically AND points at x-axis dense-wise.
        let denseResults = await hybrid.dense.search(
            query: denseVector([1, 0, 0, 0]),
            k: 5
        )
        let sparseResults = await hybrid.sparse.search(
            query: "alpha bravo charlie",
            k: 5
        )

        let denseSet = Set(denseResults.map(\.id))
        let sparseSet = Set(sparseResults.map(\.id))
        let intersection = denseSet.intersection(sparseSet)
        XCTAssertFalse(intersection.isEmpty, "Fixture should produce cross-leg overlap")

        let fused = await hybrid.search(
            queryText: "alpha bravo charlie",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 5,
            candidatePoolK: 10
        )
        let fusedSet = Set(fused.map(\.id))

        for id in intersection {
            XCTAssertTrue(
                fusedSet.contains(id),
                "RRF invariant violated: id \(id) in both top-k but missing from fused top-k"
            )
        }
    }

    func testRRFFavorsCoOccurrences() async throws {
        let hybrid = makeHybrid(fusion: .rrf(k: 60))

        let both = UUID()
        let denseOnly = UUID()
        let sparseOnly = UUID()

        try await hybrid.add(
            text: "alpha bravo",
            vector: denseVector([1, 0, 0, 0]),
            id: both
        )
        try await hybrid.add(
            text: "no match zzz",
            vector: denseVector([0.99, 0.01, 0, 0]),
            id: denseOnly
        )
        try await hybrid.add(
            text: "alpha bravo",
            vector: denseVector([0, 0, 0, 1]),
            id: sparseOnly
        )

        let results = await hybrid.search(
            queryText: "alpha bravo",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 3,
            candidatePoolK: 10
        )
        XCTAssertFalse(results.isEmpty)
        XCTAssertEqual(
            results[0].id,
            both,
            "Doc appearing in BOTH legs should outrank either single-leg entry under RRF"
        )
    }

    // MARK: - Weighted Sum

    func testWeightedSumAlphaOneDegeneratesToDense() async throws {
        let hybrid = makeHybrid(fusion: .weightedSum(alpha: 1.0))

        let a = UUID()  // dense-close, sparse-miss
        let b = UUID()  // sparse-hit, dense-far
        try await hybrid.add(text: "noise zzz", vector: denseVector([1, 0, 0, 0]), id: a)
        try await hybrid.add(text: "target words", vector: denseVector([0, 0, 0, 1]), id: b)

        let hits = await hybrid.search(
            queryText: "target words",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 2
        )
        XCTAssertEqual(hits.first?.id, a, "alpha=1.0 should rank by dense alone")
    }

    func testWeightedSumAlphaZeroDegeneratesToSparse() async throws {
        let hybrid = makeHybrid(fusion: .weightedSum(alpha: 0.0))

        let a = UUID()
        let b = UUID()
        try await hybrid.add(text: "noise", vector: denseVector([1, 0, 0, 0]), id: a)
        try await hybrid.add(text: "target words match", vector: denseVector([0, 0, 0, 1]), id: b)

        let hits = await hybrid.search(
            queryText: "target words",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 2
        )
        XCTAssertEqual(hits.first?.id, b, "alpha=0.0 should rank by sparse alone")
    }

    // MARK: - Filter

    func testFilterAppliesAcrossBothLegs() async throws {
        let hybrid = makeHybrid()
        let keep = UUID()
        let drop = UUID()
        try await hybrid.add(
            text: "keep this one",
            vector: denseVector([1, 0, 0, 0]),
            id: keep
        )
        try await hybrid.add(
            text: "drop that one",
            vector: denseVector([0.99, 0.01, 0, 0]),
            id: drop
        )

        let results = await hybrid.search(
            queryText: "one",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 5,
            filter: { $0 == keep }
        )
        XCTAssertEqual(results.map(\.id), [keep])
    }

    // MARK: - Tombstone Consistency

    func testTombstonedDocsAreAbsentFromFusedResults() async throws {
        let hybrid = makeHybrid()

        let ids = (0..<10).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            try await hybrid.add(
                text: "topic\(i) alpha bravo",
                vector: denseVector([1 - Float(i) * 0.05, 0, 0, 0]),
                id: id
            )
        }

        // Remove half of them.
        for id in ids.prefix(5) {
            _ = await hybrid.remove(id: id)
        }

        let hits = await hybrid.search(
            queryText: "alpha bravo",
            queryVector: denseVector([1, 0, 0, 0]),
            k: 10
        )
        let returned = Set(hits.map(\.id))
        for removedId in ids.prefix(5) {
            XCTAssertFalse(returned.contains(removedId),
                           "Removed id \(removedId) leaked into fused results")
        }
    }

    // MARK: - Pure Fusion Math (no live legs)

    func testFuseEmptyLists() {
        let fused = HybridIndex.fuse(
            dense: [],
            sparse: [],
            strategy: .rrf(),
            k: 5
        )
        XCTAssertTrue(fused.isEmpty)
    }

    func testFuseSingleLeg() {
        let a = UUID(), b = UUID()
        let dense = [
            SearchResult(id: a, distance: 0.1),
            SearchResult(id: b, distance: 0.3),
        ]
        let fused = HybridIndex.fuse(
            dense: dense,
            sparse: [],
            strategy: .rrf(k: 60),
            k: 5
        )
        XCTAssertEqual(fused.map(\.id), [a, b])
    }

    func testFuseRRFScoreSanity() {
        // Doc X is rank 1 on both lists; Y is rank 1 on sparse, 3 on dense.
        let x = UUID()
        let y = UUID()
        let other = UUID()

        let dense = [
            SearchResult(id: x, distance: 0.1),
            SearchResult(id: other, distance: 0.2),
            SearchResult(id: y, distance: 0.3),
        ]
        let sparse = [
            SearchResult(id: y, distance: -1.0),
            SearchResult(id: x, distance: -0.5),
        ]

        let fused = HybridIndex.fuse(
            dense: dense,
            sparse: sparse,
            strategy: .rrf(k: 60),
            k: 3
        )

        // Under RRF(k=60):
        //   X: 1/61 + 1/62 ≈ 0.03269
        //   Y: 1/63 + 1/61 ≈ 0.03226
        // X should rank above Y.
        XCTAssertEqual(fused.first?.id, x)
        XCTAssertEqual(fused[1].id, y)
    }

    // MARK: - Weighted Sum Fusion Math (non-degenerate alpha, hand-computed)
    //
    // The live-leg tests above only cover the degenerate alphas (0.0 and 1.0).
    // These pin the actual blend math at alpha = 0.3 / 0.5 / 0.7 on one shared
    // fixture, hand-computed through minMaxNormalize:
    //
    //   dense leg (distance, lower = better):  A: 0.0   B: 0.2   C: 0.8
    //     negate → min-max over range 0.8:     A: 1.0   B: 0.75  C: 0.0
    //   sparse leg (negated BM25 scores):      C: -1.0  A: -0.4  B: -0.1
    //     negate → min-max over range 0.9:     C: 1.0   A: 1/3   B: 0.0
    //
    //   fused(id) = alpha * dense_norm + (1 - alpha) * sparse_norm:
    //     A = alpha + (1 - alpha)/3      B = 0.75 * alpha      C = 1 - alpha
    //
    //   alpha = 0.3:  C (0.700) > A (0.533) > B (0.225)
    //   alpha = 0.5:  A (0.667) > C (0.500) > B (0.375)
    //   alpha = 0.7:  A (0.800) > B (0.525) > C (0.300)
    //
    // Each alpha yields a *different* total order, so a sign/weight bug in
    // either leg's contribution cannot pass all three.

    private static let wsA = UUID()
    private static let wsB = UUID()
    private static let wsC = UUID()

    private var weightedSumFixture: (dense: [SearchResult], sparse: [SearchResult]) {
        (
            dense: [
                SearchResult(id: Self.wsA, distance: 0.0),
                SearchResult(id: Self.wsB, distance: 0.2),
                SearchResult(id: Self.wsC, distance: 0.8),
            ],
            sparse: [
                SearchResult(id: Self.wsC, distance: -1.0),
                SearchResult(id: Self.wsA, distance: -0.4),
                SearchResult(id: Self.wsB, distance: -0.1),
            ]
        )
    }

    func testWeightedSumAlpha03Ordering() {
        let (dense, sparse) = weightedSumFixture
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.3), k: 3
        )
        XCTAssertEqual(fused.map(\.id), [Self.wsC, Self.wsA, Self.wsB],
            "alpha=0.3: expected C (0.700) > A (0.533) > B (0.225)")
    }

    func testWeightedSumAlpha05Ordering() {
        let (dense, sparse) = weightedSumFixture
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.5), k: 3
        )
        XCTAssertEqual(fused.map(\.id), [Self.wsA, Self.wsC, Self.wsB],
            "alpha=0.5: expected A (0.667) > C (0.500) > B (0.375)")
    }

    func testWeightedSumAlpha07Ordering() {
        let (dense, sparse) = weightedSumFixture
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.7), k: 3
        )
        XCTAssertEqual(fused.map(\.id), [Self.wsA, Self.wsB, Self.wsC],
            "alpha=0.7: expected A (0.800) > B (0.525) > C (0.300)")
    }

    // MARK: - Weighted Sum Normalization Edge Cases

    func testWeightedSumIdenticalScoresInOneLegNormalizeToMidpoint() {
        // All dense distances tied → min-max range is 0 → every dense doc
        // gets the 0.5 midpoint (not NaN, not 0, not 1). Ordering must then
        // be decided entirely by the sparse leg.
        let a = UUID(), b = UUID()
        let dense = [
            SearchResult(id: a, distance: 0.5),
            SearchResult(id: b, distance: 0.5),
        ]
        let sparse = [
            SearchResult(id: b, distance: -1.0),  // sparse-best
            SearchResult(id: a, distance: -0.2),
        ]
        // fused(a) = 0.5*0.5 + 0.5*0.0 = 0.25; fused(b) = 0.25 + 0.5 = 0.75.
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.5), k: 2
        )
        XCTAssertEqual(fused.map(\.id), [b, a],
            "With the dense leg fully tied, the sparse leg must decide the order")
    }

    func testWeightedSumBothLegsFullyTiedYieldMidpointScores() {
        // Both legs tied → every doc scores alpha*0.5 + (1-alpha)*0.5 = 0.5,
        // i.e. fused distance is exactly -0.5 for all. Order among ties is
        // unspecified, so assert the score values and membership, not order.
        let a = UUID(), b = UUID()
        let dense = [
            SearchResult(id: a, distance: 1.0),
            SearchResult(id: b, distance: 1.0),
        ]
        let sparse = [
            SearchResult(id: a, distance: -2.0),
            SearchResult(id: b, distance: -2.0),
        ]
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.3), k: 2
        )
        XCTAssertEqual(Set(fused.map(\.id)), [a, b])
        for result in fused {
            XCTAssertEqual(result.distance, -0.5,
                "Fully tied legs must fuse to the 0.5 midpoint, not NaN/0/1")
        }
    }

    func testWeightedSumSingleElementLegNormalizesToMidpoint() {
        // A single-element leg has min == max → range 0 → midpoint 0.5,
        // regardless of how large its raw distance is. The raw 42.0 must NOT
        // leak into the fused score.
        let x = UUID(), y = UUID(), z = UUID()
        let dense = [SearchResult(id: x, distance: 42.0)]
        let sparse = [
            SearchResult(id: y, distance: -3.0),
            SearchResult(id: z, distance: -1.0),
        ]
        // alpha = 0.5:
        //   x = 0.5 * 0.5             = 0.25
        //   y = 0.5 * 1.0 (sparse-best) = 0.5
        //   z = 0.5 * 0.0             = 0.0
        let fused = HybridIndex.fuse(
            dense: dense, sparse: sparse,
            strategy: .weightedSum(alpha: 0.5), k: 3
        )
        XCTAssertEqual(fused.map(\.id), [y, x, z],
            "Single-element dense leg should contribute the 0.5 midpoint, ranking x between y and z")
    }
}
