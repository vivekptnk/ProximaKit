// HNSWGraphRepairTests.swift
// ProximaKitTests
//
// Regression tests for HNSW remove() graph repair:
//   1. Reverse-edge sweep — insertion-time pruning is one-sided, so a live
//      node can hold an edge to a removed node that the removed node never
//      listed back. remove() must leave NO edges into tombstoned slots.
//   2. Neighbor reconnection — the removed node's former neighbors must be
//      bridged so connectivity that ran through the removed node survives.
//
// Also locks in the documented count/liveCount tombstone semantics, the
// search() empty-return contract on dimension mismatch, and weightedSum
// alpha validation/clamping (HybridFusionAlphaTests below).

import XCTest
@testable import ProximaKit

final class HNSWGraphRepairTests: XCTestCase {

    // ── Fixtures ──────────────────────────────────────────────────────

    /// A line graph: v_i = [i, 0] with m = 2 (mMax0 = 4). With a beam wide
    /// enough to see every node, each insertion links only to its nearest
    /// neighbors, so every layer-0 edge spans at most ±4 positions. Removing
    /// a contiguous block wider than that window MUST sever layer 0 unless
    /// remove() reconnects the survivors.
    private func makeLineIndex(count: Int) async throws -> (HNSWIndex, [UUID]) {
        let config = HNSWConfiguration(
            m: 2,
            efConstruction: 64,
            efSearch: 32,
            autoCompactionThreshold: nil  // keep tombstones visible to the tests
        )
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)
        var ids: [UUID] = []
        for i in 0..<count {
            let id = UUID()
            ids.append(id)
            try await index.add(Vector([Float(i), 0]), id: id)
        }
        return (index, ids)
    }

    // ── Reverse-edge sweep ────────────────────────────────────────────

    func testRemoveLeavesNoDanglingEdges() async throws {
        // Dense deterministic cluster with tiny m → heavy one-sided pruning,
        // which historically left edges pointing at tombstoned slots.
        let config = HNSWConfiguration(
            m: 2, efConstruction: 64, efSearch: 32, autoCompactionThreshold: nil
        )
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
        var ids: [UUID] = []
        for i in 0..<40 {
            let id = UUID()
            ids.append(id)
            let vector = Vector([
                Float(i % 8), Float(i / 8), Float((i * 3) % 5), 0
            ])
            try await index.add(vector, id: id)
        }

        for i in stride(from: 0, to: 40, by: 4) {
            await index.remove(id: ids[i])
        }

        let dangling = await index.hasDanglingEdges
        XCTAssertFalse(
            dangling,
            "remove() must sweep ALL incoming edges, including one-sided edges left by pruning"
        )
    }

    // ── Neighbor reconnection ─────────────────────────────────────────

    func testRemoveRepairsLayer0Connectivity() async throws {
        let (index, ids) = try await makeLineIndex(count: 31)

        // Remove a contiguous block wider than the ±4 layer-0 edge window.
        // Without reconnection this splits layer 0 into {0...9} and {21...30}.
        for i in 10...20 {
            await index.remove(id: ids[i])
        }

        let connected = await index.isLayer0Connected
        XCTAssertTrue(
            connected,
            "remove() must bridge the removed node's former neighbors so layer 0 stays connected"
        )

        // The far end must remain findable end-to-end.
        let results = await index.search(query: Vector([30, 0]), k: 1, efSearch: 64)
        XCTAssertEqual(results.first?.id, ids[30])
    }

    func testSearchAfterRemovalsReturnsOnlyLiveIds() async throws {
        let (index, ids) = try await makeLineIndex(count: 20)
        let removed = Set(stride(from: 0, to: 20, by: 2).map { ids[$0] })
        for id in removed {
            await index.remove(id: id)
        }

        let results = await index.search(query: Vector([10, 0]), k: 20, efSearch: 64)
        XCTAssertFalse(results.isEmpty)
        for result in results {
            XCTAssertFalse(removed.contains(result.id), "tombstoned id resurfaced in results")
        }
    }

    // ── count / liveCount / isEmpty tombstone semantics ───────────────

    func testCountIncludesTombstonesLiveCountDoesNot() async throws {
        let (index, ids) = try await makeLineIndex(count: 10)

        await index.remove(id: ids[0])
        await index.remove(id: ids[1])

        // Documented contract: HNSWIndex.count includes tombstoned slots
        // (unlike SparseIndex.count, which is live-only).
        let count = await index.count
        let liveCount = await index.liveCount
        let isEmpty = await index.isEmpty
        XCTAssertEqual(count, 10)
        XCTAssertEqual(liveCount, 8)
        XCTAssertFalse(isEmpty)
    }

    func testIsEmptyTrueWhenOnlyTombstonesRemain() async throws {
        let config = HNSWConfiguration(m: 2, autoCompactionThreshold: nil)
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)
        let id = UUID()
        try await index.add(Vector([1, 1]), id: id)
        await index.remove(id: id)

        let isEmpty = await index.isEmpty
        let count = await index.count
        XCTAssertTrue(isEmpty, "isEmpty tracks live vectors, not tombstoned slots")
        XCTAssertEqual(count, 1, "the tombstoned slot still occupies count")
    }

    // ── Configuration & search contracts ──────────────────────────────

    func testMinimumValidMConfigurationWorks() async throws {
        // m = 1 is rejected at construction (would trap on the first add via
        // an infinite level multiplier); m = 2 is the smallest valid value
        // and must work end to end.
        let config = HNSWConfiguration(m: 2)
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)
        let id = UUID()
        try await index.add(Vector([1, 0]), id: id)
        let results = await index.search(query: Vector([1, 0]), k: 1)
        XCTAssertEqual(results.first?.id, id)
    }

    func testSearchWrongDimensionReturnsEmpty() async throws {
        // Documented contract: search() returns [] on a query-dimension
        // mismatch (unlike add(), which throws IndexError.dimensionMismatch).
        let index = HNSWIndex(dimension: 3)
        try await index.add(Vector([1, 2, 3]), id: UUID())
        let results = await index.search(query: Vector([1, 2]), k: 5)
        XCTAssertTrue(results.isEmpty)
    }
}

// MARK: - HybridFusionAlphaTests

final class HybridFusionAlphaTests: XCTestCase {

    /// Out-of-range alpha must behave exactly like the nearest bound instead
    /// of assigning a negative weight that inverts a leg's ranking.
    /// (HybridIndex also preconditions alpha at init/setFusion; the static
    /// fuse hook clamps as defense in depth — this exercises the clamp.)
    func testWeightedSumAlphaAboveOneClampsToOne() {
        let idA = UUID()
        let idB = UUID()
        let dense = [
            SearchResult(id: idA, distance: 0.1),
            SearchResult(id: idB, distance: 0.9),
        ]
        let sparse = [
            SearchResult(id: idB, distance: 0.2),
            SearchResult(id: idA, distance: 0.8),
        ]

        let clamped = HybridIndex.fuse(
            dense: dense, sparse: sparse, strategy: .weightedSum(alpha: 1.5), k: 2
        )
        let exact = HybridIndex.fuse(
            dense: dense, sparse: sparse, strategy: .weightedSum(alpha: 1.0), k: 2
        )

        XCTAssertEqual(clamped.map(\.id), exact.map(\.id))
        XCTAssertEqual(clamped.map(\.distance), exact.map(\.distance))
    }

    func testWeightedSumNegativeAlphaClampsToZero() {
        let idA = UUID()
        let idB = UUID()
        let dense = [
            SearchResult(id: idA, distance: 0.1),
            SearchResult(id: idB, distance: 0.9),
        ]
        let sparse = [
            SearchResult(id: idB, distance: 0.2),
            SearchResult(id: idA, distance: 0.8),
        ]

        let clamped = HybridIndex.fuse(
            dense: dense, sparse: sparse, strategy: .weightedSum(alpha: -0.5), k: 2
        )
        let exact = HybridIndex.fuse(
            dense: dense, sparse: sparse, strategy: .weightedSum(alpha: 0.0), k: 2
        )

        XCTAssertEqual(clamped.map(\.id), exact.map(\.id))
        XCTAssertEqual(clamped.map(\.distance), exact.map(\.distance))
    }

    func testInRangeAlphaIsAcceptedByHybridIndex() async {
        // Boundary values must pass the init precondition.
        let dense = HNSWIndex(dimension: 4)
        let sparse = SparseIndex()
        _ = HybridIndex(dense: dense, sparse: sparse, fusion: .weightedSum(alpha: 0.0))
        _ = HybridIndex(dense: dense, sparse: sparse, fusion: .weightedSum(alpha: 1.0))

        let hybrid = HybridIndex(dense: dense, sparse: sparse, fusion: .rrf())
        await hybrid.setFusion(.weightedSum(alpha: 0.5))
        let fusion = await hybrid.fusion
        XCTAssertEqual(fusion, .weightedSum(alpha: 0.5))
    }
}
