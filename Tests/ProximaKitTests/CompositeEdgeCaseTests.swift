// CompositeEdgeCaseTests.swift
// ProximaKitTests
//
// Empty / degenerate query paths on the composite (HybridIndex) surface:
// - one leg empty (dense-only docs, sparse-only docs), both legs empty
// - k = 0 and k > document count
// - dimension-mismatch query vector (documented to degrade to sparse-only,
//   NOT throw — see the `- Important:` note on HybridIndex.search)
//
// These paths previously had no coverage (CHA-201 audit, tests dimension).

import XCTest
@testable import ProximaKit

final class CompositeEdgeCaseTests: XCTestCase {

    private let dim = 4

    private func makeHybrid(
        fusion: HybridFusionStrategy = .rrf()
    ) -> HybridIndex {
        let dense = BruteForceIndex(dimension: dim, metric: EuclideanDistance())
        let sparse = SparseIndex()
        return HybridIndex(dense: dense, sparse: sparse, fusion: fusion)
    }

    // ── Empty legs ────────────────────────────────────────────────────

    func testSearchWithBothLegsEmptyReturnsEmpty() async throws {
        let hybrid = makeHybrid()
        let hits = await hybrid.search(
            queryText: "anything",
            queryVector: Vector([1, 0, 0, 0]),
            k: 10
        )
        XCTAssertTrue(hits.isEmpty, "Empty hybrid index must return [] — not crash or pad")
    }

    func testSearchWithEmptyDenseLegFallsBackToSparseRanking() async throws {
        // Populate ONLY the sparse leg via the documented advanced accessor;
        // the dense leg stays empty, so fusion sees one ranked list.
        let hybrid = makeHybrid()
        let a = UUID()
        let b = UUID()
        try await hybrid.sparse.add(text: "alpha bravo charlie", id: a, metadata: nil)
        try await hybrid.sparse.add(text: "alpha unrelated noise", id: b, metadata: nil)

        let hits = await hybrid.search(
            queryText: "alpha bravo charlie",
            queryVector: Vector([1, 0, 0, 0]),
            k: 5
        )
        XCTAssertEqual(hits.map(\.id), [a, b],
            "Empty dense leg: fused output must equal the sparse-only ranking")
    }

    func testSearchWithEmptySparseLegFallsBackToDenseRanking() async throws {
        // Populate ONLY the dense leg; the sparse leg stays empty.
        let hybrid = makeHybrid()
        let near = UUID()
        let far = UUID()
        try await hybrid.dense.add(Vector([1, 0, 0, 0]), id: near, metadata: nil)
        try await hybrid.dense.add(Vector([0, 0, 0, 1]), id: far, metadata: nil)

        let hits = await hybrid.search(
            queryText: "no sparse documents exist",
            queryVector: Vector([1, 0, 0, 0]),
            k: 5
        )
        XCTAssertEqual(hits.map(\.id), [near, far],
            "Empty sparse leg: fused output must equal the dense-only ranking")
    }

    // ── k boundaries ──────────────────────────────────────────────────

    func testSearchWithKZeroReturnsEmpty() async throws {
        let hybrid = makeHybrid()
        try await hybrid.add(text: "doc one", vector: Vector([1, 0, 0, 0]), id: UUID())
        try await hybrid.add(text: "doc two", vector: Vector([0, 1, 0, 0]), id: UUID())

        let hits = await hybrid.search(
            queryText: "doc",
            queryVector: Vector([1, 0, 0, 0]),
            k: 0
        )
        XCTAssertTrue(hits.isEmpty, "k = 0 must return [] even on a populated index")
    }

    func testSearchWithKGreaterThanCountReturnsAllDocuments() async throws {
        let hybrid = makeHybrid()
        let ids = (0..<3).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            try await hybrid.add(
                text: "common term doc\(i)",
                vector: Vector([1 - Float(i) * 0.1, 0, 0, 0]),
                id: id
            )
        }

        let hits = await hybrid.search(
            queryText: "common term",
            queryVector: Vector([1, 0, 0, 0]),
            k: 100
        )
        XCTAssertEqual(hits.count, 3, "k > count must return exactly count results, no padding")
        XCTAssertEqual(Set(hits.map(\.id)), Set(ids))
    }

    // ── Dimension mismatch ────────────────────────────────────────────

    func testDenseLegReturnsEmptyForDimensionMismatchedQuery() async throws {
        // Documented behavior: search returns [] on dimension mismatch rather
        // than throwing (unlike add, which throws IndexError.dimensionMismatch).
        let hybrid = makeHybrid()
        try await hybrid.add(text: "some doc", vector: Vector([1, 0, 0, 0]), id: UUID())

        let direct = await hybrid.dense.search(
            query: Vector([1, 0, 0]),   // 3d query against a 4d leg
            k: 5,
            efSearch: nil,
            filter: nil
        )
        XCTAssertTrue(direct.isEmpty, "Dense leg must return [] for a mismatched query dimension")
    }

    func testHybridSearchWithMismatchedVectorDegradesToSparseOnly() async throws {
        // Per the `- Important:` doc on HybridIndex.search: a mismatched
        // queryVector silently empties the dense leg's contribution and the
        // fused output degrades to sparse-only ranking with no diagnostic.
        let hybrid = makeHybrid()
        let sparseBest = UUID()
        let denseBest = UUID()
        try await hybrid.add(
            text: "target words match strongly",
            vector: Vector([0, 0, 0, 1]),
            id: sparseBest
        )
        try await hybrid.add(
            text: "zzz noise",
            vector: Vector([1, 0, 0, 0]),
            id: denseBest
        )

        let hits = await hybrid.search(
            queryText: "target words",
            queryVector: Vector([1, 0, 0]),   // 3d query against a 4d dense leg
            k: 5
        )
        XCTAssertEqual(hits.map(\.id), [sparseBest],
            "Mismatched dense query must degrade fusion to the sparse-only ranking")
    }

    func testHybridSearchWithMismatchedVectorAndEmptySparseReturnsEmpty() async throws {
        // Mismatched dense query AND no sparse hits → both lists empty → [].
        let hybrid = makeHybrid()
        try await hybrid.add(text: "alpha", vector: Vector([1, 0, 0, 0]), id: UUID())

        let hits = await hybrid.search(
            queryText: "completely unmatched query terms",
            queryVector: Vector([1, 0, 0]),   // 3d query against a 4d dense leg
            k: 5
        )
        XCTAssertTrue(hits.isEmpty)
    }
}
