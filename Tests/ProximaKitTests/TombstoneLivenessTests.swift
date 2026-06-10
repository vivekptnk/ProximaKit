import XCTest
@testable import ProximaKit

/// Regression tests for the tombstone liveness bug (CHA-201 audit, critical).
///
/// `add()` replaces an existing UUID by tombstoning the old node slot. Liveness
/// of a slot must be checked by identity (`uuidToNode[uuid] == node`), not
/// presence (`uuidToNode[uuid] != nil`): after a re-add the UUID maps to the
/// NEW node, so presence checks wrongly treat the old tombstoned slot as live.
/// Symptoms before the fix: searches returned stale vectors/metadata, entry
/// point recovery could select a disconnected tombstone, and `compact()`
/// resurrected deleted vector bodies.
final class TombstoneLivenessTests: XCTestCase {

    // MARK: - HNSWIndex

    /// The auditor's 20/20 repro: with auto-compaction disabled (so the
    /// tombstone persists), re-add A with a new vector, remove the entry
    /// point, and verify search finds the NEW vector at distance ~0.
    /// Repeated 20× because entry-point levels are randomized.
    func testReAddThenRemoveEntryPointReturnsLiveVector() async throws {
        for trial in 0..<20 {
            let index = HNSWIndex(
                dimension: 2,
                metric: EuclideanDistance(),
                config: HNSWConfiguration(autoCompactionThreshold: nil)
            )
            let idA = UUID()
            let idB = UUID()

            try await index.add(Vector([100, 100]), id: idA, metadata: Data("old".utf8))
            try await index.add(Vector([50, 50]), id: idB)
            // Re-add A with a different vector — tombstones the old slot.
            try await index.add(Vector([1, 1]), id: idA, metadata: Data("new".utf8))
            // Remove B; if B was the entry point, recovery must not pick A's tombstone.
            await index.remove(id: idB)

            let results = await index.search(query: Vector([1, 1]), k: 2)
            XCTAssertEqual(results.count, 1, "trial \(trial): only A is live")
            XCTAssertEqual(results.first?.id, idA, "trial \(trial)")
            XCTAssertEqual(results.first?.distance ?? -1, 0, accuracy: 1e-5,
                           "trial \(trial): must match the re-added vector [1,1], not the stale [100,100]")
            XCTAssertEqual(results.first?.metadata, Data("new".utf8),
                           "trial \(trial): metadata must come from the live slot")
        }
    }

    /// Default config, index large enough that the live ratio stays above the
    /// auto-compaction threshold — the tombstone persists and must never
    /// surface a stale vector or a duplicate id in results.
    func testReAddDoesNotReturnStaleResults() async throws {
        for trial in 0..<20 {
            let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())
            var ids: [UUID] = []
            for i in 0..<10 {
                let id = UUID()
                ids.append(id)
                try await index.add(Vector([Float(i * 10 + 50), Float(i * 10 + 50)]), id: id)
            }
            // Re-add ids[0] near the origin. Ratio 10/11 ≈ 0.91 → no auto-compact.
            try await index.add(Vector([1, 1]), id: ids[0])

            let results = await index.search(query: Vector([1, 1]), k: 3)
            let aHits = results.filter { $0.id == ids[0] }
            XCTAssertEqual(aHits.count, 1, "trial \(trial): re-added id must appear exactly once")
            XCTAssertEqual(aHits.first?.distance ?? -1, 0, accuracy: 1e-3,
                           "trial \(trial): distance must reflect the live vector [1,1]")

            // Entry-point collapse leg: remove another node (ratio 9/11 ≈ 0.82,
            // still no auto-compact) and verify the graph stays fully searchable.
            await index.remove(id: ids[5])
            let results2 = await index.search(query: Vector([1, 1]), k: 5)
            XCTAssertEqual(results2.count, 5,
                           "trial \(trial): graph must remain connected after entry-point recovery")
            let aHits2 = results2.filter { $0.id == ids[0] }
            XCTAssertEqual(aHits2.first?.distance ?? -1, 0, accuracy: 1e-3,
                           "trial \(trial): live vector must survive entry-point recovery")
        }
    }

    /// compact() must drop the tombstoned body and keep exactly the live set.
    func testCompactAfterReAddKeepsOnlyLiveVectors() async throws {
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())
        let idA = UUID()
        let idB = UUID()
        try await index.add(Vector([100, 100]), id: idA)
        try await index.add(Vector([50, 50]), id: idB)
        try await index.add(Vector([1, 1]), id: idA) // re-add

        try await index.compact()

        let count = await index.count
        let liveCount = await index.liveCount
        XCTAssertEqual(count, 2)
        XCTAssertEqual(liveCount, 2)

        let results = await index.search(query: Vector([1, 1]), k: 2)
        XCTAssertEqual(Set(results.map(\.id)), Set([idA, idB]))
        XCTAssertEqual(results.first?.id, idA)
        XCTAssertEqual(results.first?.distance ?? -1, 0, accuracy: 1e-5)
    }

    /// Persistence snapshots compact first — a re-added UUID must round-trip
    /// with its live vector only.
    func testPersistenceRoundTripAfterReAdd() async throws {
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())
        let idA = UUID()
        try await index.add(Vector([100, 100]), id: idA)
        try await index.add(Vector([1, 1]), id: idA) // re-add

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tombstone-roundtrip-\(UUID().uuidString).proxima")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url)
        let loaded = try HNSWIndex.load(from: url)

        let count = await loaded.count
        XCTAssertEqual(count, 1)
        let results = await loaded.search(query: Vector([1, 1]), k: 1)
        XCTAssertEqual(results.first?.id, idA)
        XCTAssertEqual(results.first?.distance ?? -1, 0, accuracy: 1e-5)
    }

    // MARK: - QuantizedHNSWIndex

    /// Entry-point recovery in QuantizedHNSWIndex.remove() must use identity-based
    /// liveness when scanning for a replacement entry point.
    func testQuantizedEntryPointRecoverySkipsTombstones() async throws {
        var vectors: [Vector] = []
        var ids: [UUID] = []
        for i in 0..<64 {
            vectors.append(Vector((0..<8).map { _ in Float.random(in: -1...1) + Float(i) * 0.01 }))
            ids.append(UUID())
        }
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: 8,
            pqConfig: PQConfiguration(subspaceCount: 2, trainingIterations: 5)
        )

        // Remove half the nodes, including (eventually) the entry point.
        for id in ids.prefix(32) {
            await index.remove(id: id)
        }

        let live = await index.liveCount
        XCTAssertEqual(live, 32)
        let results = await index.search(query: vectors[40], k: 5)
        XCTAssertFalse(results.isEmpty, "search must still work after heavy removal")
        let removed = Set(ids.prefix(32))
        XCTAssertTrue(results.allSatisfy { !removed.contains($0.id) },
                      "no tombstoned ids in results")
    }

    // MARK: - SparseIndex

    /// SparseIndex compaction must keep exactly one copy of a re-added document.
    func testSparseCompactionAfterReAdd() async throws {
        let index = SparseIndex()
        let idA = UUID()
        try await index.add(text: "old stale document text", id: idA)
        try await index.add(text: "completely different fresh words", id: idA) // re-add
        for i in 0..<3 {
            try await index.add(text: "filler document number \(i)", id: UUID())
        }

        await index.compact()

        let results = await index.search(query: "fresh words", k: 4)
        let aHits = results.filter { $0.id == idA }
        XCTAssertEqual(aHits.count, 1, "A must appear exactly once after compaction")

        let stale = await index.search(query: "stale", k: 4)
        XCTAssertTrue(stale.allSatisfy { $0.id != idA },
                      "stale tokens must not match the re-added document")
    }
}
