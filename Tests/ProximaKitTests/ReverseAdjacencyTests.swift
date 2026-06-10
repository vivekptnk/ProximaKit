// ReverseAdjacencyTests.swift
// ProximaKitTests
//
// Tests for the reverse-adjacency (incoming-edge) map that makes
// HNSWIndex.remove() O(in-degree) instead of a full per-layer edge sweep.
//
// The map is derived state, so two things must hold everywhere:
//   1. Invariant: `inEdges` is the exact transpose of `layers` after every
//      mutation — add, insertion-time pruning, remove-time reconnection,
//      compact(), and auto-compaction (`reverseAdjacencyIsConsistent`).
//   2. Persistence: the map is NOT persisted (the snapshot format is
//      unchanged, ADR-010); init(restoring:) rebuilds it from the
//      snapshot's layers.
//
// Equivalence is asserted against a brute-force-maintained expectation: a
// plain id → vector dictionary updated by the same seeded workload. With a
// query ef >= the whole graph, the layer-0 beam visits every reachable
// node, so HNSW results must EXACTLY equal the brute-force ranking as long
// as remove() keeps layer 0 connected — any dangling-edge or repair
// regression from the new bookkeeping surfaces as a mismatch here.

import XCTest
@testable import ProximaKit

final class ReverseAdjacencyTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    private func randomVector(dim: Int, rng: inout SeededRandom) -> Vector {
        Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    /// The brute-force-maintained expectation: rank the live id → vector
    /// map by the same metric and take the top k.
    private func bruteForceTopK(
        live: [UUID: Vector], query: Vector, k: Int, metric: some DistanceMetric
    ) -> [UUID] {
        live
            .map { (id: $0.key, distance: metric.distance(query, $0.value)) }
            .sorted { $0.distance < $1.distance }
            .prefix(k)
            .map(\.id)
    }

    // ── Equivalence under a seeded add/remove/re-add workload ─────────

    func testSeededChurnWorkloadMatchesBruteForceExpectation() async throws {
        let dim = 16
        let metric = EuclideanDistance()
        let config = HNSWConfiguration(
            m: 8,
            efConstruction: 100,
            efSearch: 64,
            autoCompactionThreshold: nil,  // keep tombstones in play
            levelSeed: 0x2EAD_7ACE_0000_0001
        )
        let index = HNSWIndex(dimension: dim, metric: metric, config: config)
        var rng = SeededRandom(seed: 0x2EAD_7ACE_5EED_0001)

        var live: [UUID: Vector] = [:]
        var liveIDs: [UUID] = []

        // Phase 1: 300 adds.
        for _ in 0..<300 {
            let v = randomVector(dim: dim, rng: &rng)
            let id = UUID()
            liveIDs.append(id)
            live[id] = v
            try await index.add(v, id: id)
        }

        // Phase 2: 120 seeded removals.
        for _ in 0..<120 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            live[victim] = nil
            await index.remove(id: victim)
        }

        // Phase 3: 30 re-adds of surviving ids with NEW vectors (exercises
        // the replace-tombstone path inside add()) plus 50 fresh adds.
        for _ in 0..<30 {
            let id = liveIDs[Int.random(in: 0..<liveIDs.count, using: &rng)]
            let v = randomVector(dim: dim, rng: &rng)
            live[id] = v
            try await index.add(v, id: id)
        }
        for _ in 0..<50 {
            let v = randomVector(dim: dim, rng: &rng)
            let id = UUID()
            liveIDs.append(id)
            live[id] = v
            try await index.add(v, id: id)
        }

        // The graph must be searchable end to end: with ef covering every
        // node, results must match the brute-force-maintained expectation
        // exactly (ids AND order — seeded float data makes ties measure-zero).
        let connected = await index.isLayer0Connected
        XCTAssertTrue(connected, "repair must keep layer 0 connected — exact equality depends on it")

        for q in 0..<10 {
            let query = randomVector(dim: dim, rng: &rng)
            let got = await index.search(query: query, k: 10, efSearch: 1024).map(\.id)
            let want = bruteForceTopK(live: live, query: query, k: 10, metric: metric)
            XCTAssertEqual(got, want, "query \(q): full-beam HNSW must match brute force exactly")
        }

        // Structural health after the workload.
        let consistent = await index.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistent, "inEdges must be the exact transpose of layers")
        let dangling = await index.hasDanglingEdges
        XCTAssertFalse(dangling, "remove() must still clear every incoming edge")
    }

    // ── Invariant under heavy churn (auto-compaction enabled) ─────────

    func testInvariantHoldsThroughHeavyChurnAndAutoCompaction() async throws {
        let dim = 8
        let config = HNSWConfiguration(
            m: 4,
            efConstruction: 48,
            efSearch: 32,
            autoCompactionThreshold: 0.7,  // let compaction fire mid-churn
            levelSeed: 0x2EAD_7ACE_C0DE_0002
        )
        let index = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
        var rng = SeededRandom(seed: 0x2EAD_7ACE_5EED_0002)
        var liveIDs: [UUID] = []

        for op in 0..<600 {
            // ~60% adds, ~40% removals of a seeded live pick.
            if liveIDs.isEmpty || Int.random(in: 0..<10, using: &rng) < 6 {
                let id = UUID()
                liveIDs.append(id)
                try await index.add(randomVector(dim: dim, rng: &rng), id: id)
            } else {
                let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
                await index.remove(id: victim)
            }

            // Checking the full transpose is O(E) — sample every 50 ops.
            if op % 50 == 49 {
                let consistent = await index.reverseAdjacencyIsConsistent
                XCTAssertTrue(consistent, "inEdges out of sync with layers after op \(op)")
            }
        }

        // Explicit compaction must rebuild a consistent map too.
        try await index.compact()
        let consistentAfterCompact = await index.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistentAfterCompact, "compact() must leave a consistent map")

        let count = await index.count
        let liveCount = await index.liveCount
        XCTAssertEqual(count, liveCount, "compact() reclaims all tombstones")
        XCTAssertEqual(liveCount, liveIDs.count, "bookkeeping drifted from the index")
    }

    // ── Restore from snapshot: rebuilt, not persisted ─────────────────

    func testRestoreRebuildsReverseAdjacencyAndStaysConsistent() async throws {
        let dim = 8
        let metric = EuclideanDistance()
        let config = HNSWConfiguration(
            m: 6,
            efConstruction: 64,
            efSearch: 32,
            autoCompactionThreshold: nil,
            levelSeed: 0x2EAD_7ACE_D15C_0003
        )
        let index = HNSWIndex(dimension: dim, metric: metric, config: config)
        var rng = SeededRandom(seed: 0x2EAD_7ACE_5EED_0003)

        var liveIDs: [UUID] = []
        for _ in 0..<150 {
            let id = UUID()
            liveIDs.append(id)
            try await index.add(randomVector(dim: dim, rng: &rng), id: id)
        }
        // Churn so the pre-save graph carries repair history (tombstones are
        // compacted away by persistenceSnapshot(), but the surviving edges
        // reflect remove()-time reconnection).
        for _ in 0..<40 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            await index.remove(id: victim)
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("reverse-adjacency-\(UUID().uuidString).proxima")
        defer { try? FileManager.default.removeItem(at: url) }

        // save() snapshots (compacting the original in place) — afterwards
        // the original and the loaded index hold identical layers, so this
        // is a structural-equivalence comparison, not an approximate one.
        try await index.save(to: url)
        let loaded = try HNSWIndex.load(from: url)

        // The map is rebuilt on load (it is not in the snapshot — the
        // HNSWSnapshot type has no such field, keeping the ADR-010 format
        // unchanged) and must satisfy the transpose invariant immediately.
        let loadedConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(loadedConsistent, "init(restoring:) must rebuild inEdges from layers")

        // Identical layers ⇒ identical full-beam results, id for id.
        for _ in 0..<5 {
            let query = randomVector(dim: dim, rng: &rng)
            let original = await index.search(query: query, k: 10, efSearch: 512)
            let restored = await loaded.search(query: query, k: 10, efSearch: 512)
            XCTAssertEqual(original.map(\.id), restored.map(\.id))
            XCTAssertEqual(original.map(\.distance), restored.map(\.distance))
        }

        // The rebuilt map must support O(in-degree) removal correctly:
        // remove on the LOADED index, then re-verify structure and search.
        let removed = liveIDs.removeFirst()
        let didRemove = await loaded.remove(id: removed)
        XCTAssertTrue(didRemove)

        let stillConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(stillConsistent, "remove() on a restored index must keep the map in sync")
        let dangling = await loaded.hasDanglingEdges
        XCTAssertFalse(dangling, "remove() on a restored index must clear every incoming edge")

        let results = await loaded.search(
            query: randomVector(dim: dim, rng: &rng), k: 200, efSearch: 512
        )
        XCTAssertFalse(results.map(\.id).contains(removed), "removed id resurfaced after restore")
        XCTAssertEqual(results.count, liveIDs.count, "all surviving vectors stay reachable")
    }
}
