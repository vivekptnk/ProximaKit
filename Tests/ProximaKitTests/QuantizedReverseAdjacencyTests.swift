// QuantizedReverseAdjacencyTests.swift
// ProximaKitTests
//
// Tests for the reverse-adjacency (incoming-edge) map that makes
// QuantizedHNSWIndex.remove() and ScalarQuantizedHNSWIndex.remove() O(in-degree)
// instead of a full per-layer edge sweep — the mission-3 port of the map that
// HNSWIndex already maintains (see ReverseAdjacencyTests).
//
// Both quantized graphs are built once by a full-precision HNSWIndex and never
// grow their own edges afterward: the ONLY neighbor-mutating path is remove().
// So the map is derived state and two things must hold:
//   1. Differential equivalence: the new O(in-degree) remove() leaves the graph
//      byte-identical to the retired O(E_l) full sweep. Each index keeps that
//      sweep internally (`removeUsingFullSweep`) so a control index driven only
//      by it is the ground truth. Two indexes built from the same seeded corpus
//      are byte-identical, so applying the same removal sequence — subject via
//      remove(), control via the sweep — and comparing `graphFingerprint` after
//      every op is an exact proof, not an approximate one.
//   2. Persistence: the map is NOT persisted (the on-disk format is unchanged —
//      no new field). save() compacts tombstones and renumbers; after load()
//      the map is re-materialized lazily from the restored layers (NOT eagerly
//      in the memberwise initializer, which must stay a dumb store so the
//      corruption suite can route deliberately out-of-bounds adjacency to
//      save()/load() — see the inEdges property doc). It must satisfy the
//      transpose invariant on first inspection and drive a correct removal
//      afterward.
//
// Unlike HNSWIndex, the quantized remove does NOT reconnect former neighbors
// (the full vectors the diversity heuristic needs were discarded at build
// time), so layer 0 can fracture under churn — full reachability is therefore
// NOT asserted. Equivalence is proven against the full-sweep control (identical
// graph ⇒ identical search) and, across save/load, against the pre-save index's
// full-beam id-set (renumbering-safe).

import XCTest
@testable import ProximaKit

final class QuantizedReverseAdjacencyTests: XCTestCase {

    // ── Seeded helpers ────────────────────────────────────────────────

    private func randomVector(dim: Int, rng: inout SeededRandom) -> Vector {
        Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    /// A deterministic UUID drawn from the seeded stream, so a corpus (and thus
    /// two independently-built indexes) is byte-reproducible — the precondition
    /// for a differential comparison.
    private func seededUUID(rng: inout SeededRandom) -> UUID {
        let bytes = (0..<16).map { _ in UInt8.random(in: 0...255, using: &rng) }
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }

    /// A fully seeded (vectors, ids) corpus. Independent from the churn RNG.
    private func seededCorpus(
        count: Int, dim: Int, seed: UInt64
    ) -> (vectors: [Vector], ids: [UUID]) {
        var rng = SeededRandom(seed: seed)
        var vectors: [Vector] = []
        var ids: [UUID] = []
        for _ in 0..<count {
            vectors.append(randomVector(dim: dim, rng: &rng))
            ids.append(seededUUID(rng: &rng))
        }
        return (vectors, ids)
    }

    // ── PQ: differential equivalence through seeded churn ─────────────

    func testPQRemoveIsGraphIdenticalToFullSweepThroughSeededChurn() async throws {
        let dim = 24
        let n = 260
        let (vectors, ids) = seededCorpus(count: n, dim: dim, seed: 0x9A11_7ACE_0000_0011)
        let hnswConfig = HNSWConfiguration(
            m: 8, efConstruction: 90, efSearch: 48, levelSeed: 0x9A11_7ACE_C0DE_0011
        )
        let pqConfig = PQConfiguration(
            subspaceCount: 8, trainingIterations: 8, seed: 0x9A11_7ACE_5EED_0011
        )

        func buildIndex() async throws -> QuantizedHNSWIndex {
            try await QuantizedHNSWIndex.build(
                vectors: vectors, ids: ids, dimension: dim,
                hnswConfig: hnswConfig, pqConfig: pqConfig
            )
        }

        let subject = try await buildIndex()   // new inEdges-based remove()
        let control = try await buildIndex()   // retired full sweep

        // Deterministic builds ⇒ byte-identical starting graphs, and the map
        // is the exact transpose right after build.
        let subjectStart = await subject.graphFingerprint
        let controlStart = await control.graphFingerprint
        XCTAssertEqual(subjectStart, controlStart,
            "seeded builds must be identical for a differential comparison")
        let consistentAtBuild = await subject.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistentAtBuild, "inEdges must transpose layers right after build")

        var rng = SeededRandom(seed: 0x9A11_7ACE_DEAD_0011)
        var liveIDs = ids

        for op in 0..<140 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            let subjectRemoved = await subject.remove(id: victim)
            let controlRemoved = await control.removeUsingFullSweep(id: victim)
            XCTAssertEqual(subjectRemoved, controlRemoved,
                "remove() return must agree with the full sweep (op \(op))")

            // Core proof: identical graph state after each remove.
            let subjectFP = await subject.graphFingerprint
            let controlFP = await control.graphFingerprint
            XCTAssertEqual(subjectFP, controlFP,
                "inEdges remove diverged from the full sweep at op \(op)")

            // The map stays the exact transpose and leaves no dangling edge.
            let consistent = await subject.reverseAdjacencyIsConsistent
            XCTAssertTrue(consistent, "inEdges out of sync with layers after op \(op)")
            let dangling = await subject.hasDanglingEdges
            XCTAssertFalse(dangling, "remove() left a dangling incoming edge at op \(op)")

            // Interleaved searches must agree with the control, id for id.
            if op % 10 == 9 {
                let query = randomVector(dim: dim, rng: &rng)
                let got = await subject.search(query: query, k: 10, efSearch: 128).map(\.id)
                let want = await control.search(query: query, k: 10, efSearch: 128).map(\.id)
                XCTAssertEqual(got, want, "search diverged from the control at op \(op)")
            }
        }
    }

    // ── Scalar: differential equivalence through seeded churn ─────────

    func testScalarRemoveIsGraphIdenticalToFullSweepThroughSeededChurn() async throws {
        let dim = 20
        let n = 240
        let (vectors, ids) = seededCorpus(count: n, dim: dim, seed: 0x5CA1_7ACE_0000_0033)
        let hnswConfig = HNSWConfiguration(
            m: 8, efConstruction: 80, efSearch: 48, levelSeed: 0x5CA1_7ACE_C0DE_0033
        )

        func buildIndex() async throws -> ScalarQuantizedHNSWIndex {
            try await ScalarQuantizedHNSWIndex.build(
                vectors: vectors, ids: ids, dimension: dim,
                hnswConfig: hnswConfig, metric: .euclidean
            )
        }

        let subject = try await buildIndex()   // new inEdges-based remove()
        let control = try await buildIndex()   // retired full sweep

        let subjectStart = await subject.graphFingerprint
        let controlStart = await control.graphFingerprint
        XCTAssertEqual(subjectStart, controlStart,
            "seeded builds must be identical for a differential comparison")
        let consistentAtBuild = await subject.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistentAtBuild, "inEdges must transpose layers right after build")

        var rng = SeededRandom(seed: 0x5CA1_7ACE_DEAD_0033)
        var liveIDs = ids

        for op in 0..<130 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            let subjectRemoved = await subject.remove(id: victim)
            let controlRemoved = await control.removeUsingFullSweep(id: victim)
            XCTAssertEqual(subjectRemoved, controlRemoved,
                "remove() return must agree with the full sweep (op \(op))")

            let subjectFP = await subject.graphFingerprint
            let controlFP = await control.graphFingerprint
            XCTAssertEqual(subjectFP, controlFP,
                "inEdges remove diverged from the full sweep at op \(op)")

            let consistent = await subject.reverseAdjacencyIsConsistent
            XCTAssertTrue(consistent, "inEdges out of sync with layers after op \(op)")
            let dangling = await subject.hasDanglingEdges
            XCTAssertFalse(dangling, "remove() left a dangling incoming edge at op \(op)")

            if op % 10 == 9 {
                let query = randomVector(dim: dim, rng: &rng)
                let got = await subject.search(query: query, k: 10, efSearch: 128).map(\.id)
                let want = await control.search(query: query, k: 10, efSearch: 128).map(\.id)
                XCTAssertEqual(got, want, "search diverged from the control at op \(op)")
            }
        }
    }

    // ── PQ: rebuilt on load, then drives a correct removal ────────────

    func testPQRebuildsReverseAdjacencyOnLoadAndRemovesCorrectly() async throws {
        let dim = 16
        let n = 200
        let (vectors, ids) = seededCorpus(count: n, dim: dim, seed: 0x9A11_7ACE_0000_0022)
        let hnswConfig = HNSWConfiguration(
            m: 6, efConstruction: 80, efSearch: 40, levelSeed: 0x9A11_7ACE_C0DE_0022
        )
        let pqConfig = PQConfiguration(
            subspaceCount: 8, trainingIterations: 8, seed: 0x9A11_7ACE_5EED_0022
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: hnswConfig, pqConfig: pqConfig
        )

        var rng = SeededRandom(seed: 0x9A11_7ACE_DEAD_0022)
        var liveIDs = ids
        for _ in 0..<60 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            await index.remove(id: victim)
        }
        let consistentBeforeSave = await index.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistentBeforeSave, "map must be consistent after churn")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("q-reverse-adjacency-\(UUID().uuidString).pqhnsw")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url)
        let loaded = try QuantizedHNSWIndex.load(from: url)

        // The map is rebuilt on load (it is not in the file — the format is
        // unchanged) and must be the exact transpose of the restored,
        // save-compacted layers immediately.
        let loadedConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(loadedConsistent, "load must rebuild inEdges from the snapshot layers")
        let loadedDangling = await loaded.hasDanglingEdges
        XCTAssertFalse(loadedDangling, "a freshly loaded index must carry no dangling edges")

        // save() renumbers the live subgraph without reconnecting it, so a
        // full-beam query (ef ≥ live count) surfaces the same reachable id-set
        // before and after — a renumbering-safe structural equivalence.
        for _ in 0..<5 {
            let query = randomVector(dim: dim, rng: &rng)
            let before = Set(await index.search(query: query, k: n, efSearch: n).map(\.id))
            let after = Set(await loaded.search(query: query, k: n, efSearch: n).map(\.id))
            XCTAssertEqual(before, after, "restored index must surface the same live vectors")
        }

        // The rebuilt map must drive a correct O(in-degree) removal.
        let liveBefore = await loaded.liveCount
        let removed = liveIDs.removeFirst()
        let didRemove = await loaded.remove(id: removed)
        XCTAssertTrue(didRemove)

        let stillConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(stillConsistent, "remove() on a restored index must keep the map in sync")
        let dangling = await loaded.hasDanglingEdges
        XCTAssertFalse(dangling, "remove() on a restored index must clear every incoming edge")

        let liveAfter = await loaded.liveCount
        XCTAssertEqual(liveAfter, liveBefore - 1, "remove() must drop exactly one live vector")
        let results = await loaded.search(query: randomVector(dim: dim, rng: &rng), k: n, efSearch: n)
        XCTAssertFalse(results.map(\.id).contains(removed), "removed id resurfaced after restore")
    }

    // ── Scalar: rebuilt on load, then drives a correct removal ────────

    func testScalarRebuildsReverseAdjacencyOnLoadAndRemovesCorrectly() async throws {
        let dim = 16
        let n = 200
        let (vectors, ids) = seededCorpus(count: n, dim: dim, seed: 0x5CA1_7ACE_0000_0044)
        let hnswConfig = HNSWConfiguration(
            m: 6, efConstruction: 80, efSearch: 40, levelSeed: 0x5CA1_7ACE_C0DE_0044
        )
        let index = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: hnswConfig, metric: .euclidean
        )

        var rng = SeededRandom(seed: 0x5CA1_7ACE_DEAD_0044)
        var liveIDs = ids
        for _ in 0..<60 {
            let victim = liveIDs.remove(at: Int.random(in: 0..<liveIDs.count, using: &rng))
            await index.remove(id: victim)
        }
        let consistentBeforeSave = await index.reverseAdjacencyIsConsistent
        XCTAssertTrue(consistentBeforeSave, "map must be consistent after churn")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sq-reverse-adjacency-\(UUID().uuidString).sqhnsw")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url)
        let loaded = try ScalarQuantizedHNSWIndex.load(from: url)

        let loadedConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(loadedConsistent, "load must rebuild inEdges from the snapshot layers")
        let loadedDangling = await loaded.hasDanglingEdges
        XCTAssertFalse(loadedDangling, "a freshly loaded index must carry no dangling edges")

        for _ in 0..<5 {
            let query = randomVector(dim: dim, rng: &rng)
            let before = Set(await index.search(query: query, k: n, efSearch: n).map(\.id))
            let after = Set(await loaded.search(query: query, k: n, efSearch: n).map(\.id))
            XCTAssertEqual(before, after, "restored index must surface the same live vectors")
        }

        let liveBefore = await loaded.liveCount
        let removed = liveIDs.removeFirst()
        let didRemove = await loaded.remove(id: removed)
        XCTAssertTrue(didRemove)

        let stillConsistent = await loaded.reverseAdjacencyIsConsistent
        XCTAssertTrue(stillConsistent, "remove() on a restored index must keep the map in sync")
        let dangling = await loaded.hasDanglingEdges
        XCTAssertFalse(dangling, "remove() on a restored index must clear every incoming edge")

        let liveAfter = await loaded.liveCount
        XCTAssertEqual(liveAfter, liveBefore - 1, "remove() must drop exactly one live vector")
        let results = await loaded.search(query: randomVector(dim: dim, rng: &rng), k: n, efSearch: n)
        XCTAssertFalse(results.map(\.id).contains(removed), "removed id resurfaced after restore")
    }
}
