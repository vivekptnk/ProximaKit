// QuantizedBuildAlignmentTests.swift
// ProximaKitTests
//
// Regression tests for QuantizedHNSWIndex.build alignment between the PQ
// code/metadata arrays and the graph node indices.
//
// Background: build() constructs the graph through a full-precision HNSWIndex,
// which replace-on-duplicate tombstones earlier slots; persistenceSnapshot()
// then COMPACTS, renumbering every node. Codes/metadata encoded positionally
// from the *input* therefore silently shift by one for every compacted slot.
// The fix derives codes/metadata from the snapshot's node order instead.

import XCTest
@testable import ProximaKit

final class QuantizedBuildAlignmentTests: XCTestCase {

    // ── Fixtures ──────────────────────────────────────────────────────

    /// Deterministic, well-separated vectors: scaled one-hot directions so
    /// nearest-neighbor identity stays unambiguous even after PQ quantization.
    private func separatedVectors(count: Int, dimension: Int) -> [Vector] {
        (0..<count).map { i in
            var values = [Float](repeating: 0, count: dimension)
            values[i % dimension] = 10.0 + Float(i)
            return Vector(values)
        }
    }

    private func metadataPayloads(count: Int) -> [Data?] {
        (0..<count).map { "meta-\($0)".data(using: .utf8) }
    }

    private func buildIndex(
        vectors: [Vector],
        ids: [UUID],
        metadata: [Data?]
    ) async throws -> QuantizedHNSWIndex {
        try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            metadata: metadata,
            dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )
    }

    // ── Duplicate-id alignment ────────────────────────────────────────

    func testBuildWithDuplicateIdsKeepsCountConsistent() async throws {
        let n = 20
        let vectors = separatedVectors(count: n, dimension: 16)
        var ids = (0..<n).map { _ in UUID() }
        ids[n - 1] = ids[0]  // duplicate: last occurrence must win

        let qIndex = try await buildIndex(
            vectors: vectors, ids: ids, metadata: metadataPayloads(count: n)
        )

        // 19 distinct ids → 19 nodes. Before the fix, codes were encoded from
        // the 20 raw inputs while the snapshot compacted to 19 nodes, so
        // count (codes.count) disagreed with the graph and with liveCount.
        let count = await qIndex.count
        let liveCount = await qIndex.liveCount
        XCTAssertEqual(count, n - 1)
        XCTAssertEqual(liveCount, n - 1)
        XCTAssertEqual(count, liveCount, "build() must not leave tombstone skew")
    }

    func testBuildWithDuplicateIdsAlignsCodesIdsAndMetadata() async throws {
        let n = 20
        let vectors = separatedVectors(count: n, dimension: 16)
        var ids = (0..<n).map { _ in UUID() }
        ids[n - 1] = ids[0]  // duplicate: last occurrence must win

        let qIndex = try await buildIndex(
            vectors: vectors, ids: ids, metadata: metadataPayloads(count: n)
        )

        // Every non-duplicated vector must come back under its own id with
        // its own metadata. Before the fix, the compaction of the replaced
        // slot shifted every code/metadata entry by one position.
        for i in 1..<(n - 1) {
            let results = await qIndex.search(query: vectors[i], k: 1)
            XCTAssertEqual(results.count, 1, "probe \(i) returned no results")
            XCTAssertEqual(results.first?.id, ids[i], "id misaligned for probe \(i)")
            XCTAssertEqual(
                results.first?.metadata,
                "meta-\(i)".data(using: .utf8),
                "metadata misaligned for probe \(i)"
            )
        }
    }

    func testBuildWithDuplicateIdsLastVectorWins() async throws {
        let n = 20
        let vectors = separatedVectors(count: n, dimension: 16)
        var ids = (0..<n).map { _ in UUID() }
        ids[n - 1] = ids[0]

        let qIndex = try await buildIndex(
            vectors: vectors, ids: ids, metadata: metadataPayloads(count: n)
        )

        // The duplicated id must resolve to the LAST vector/metadata passed
        // for it (HNSWIndex replace-on-duplicate semantics).
        let results = await qIndex.search(query: vectors[n - 1], k: 1)
        XCTAssertEqual(results.first?.id, ids[0])
        XCTAssertEqual(results.first?.metadata, "meta-\(n - 1)".data(using: .utf8))
    }

    // ── No-duplicate sanity ───────────────────────────────────────────

    func testBuildWithoutDuplicatesRemainsAligned() async throws {
        let n = 20
        let vectors = separatedVectors(count: n, dimension: 16)
        let ids = (0..<n).map { _ in UUID() }

        let qIndex = try await buildIndex(
            vectors: vectors, ids: ids, metadata: metadataPayloads(count: n)
        )

        let count = await qIndex.count
        XCTAssertEqual(count, n)

        for i in 0..<n {
            let results = await qIndex.search(query: vectors[i], k: 1)
            XCTAssertEqual(results.first?.id, ids[i])
            XCTAssertEqual(results.first?.metadata, "meta-\(i)".data(using: .utf8))
        }
    }
}
