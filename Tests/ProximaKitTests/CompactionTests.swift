// CompactionTests.swift
// ProximaKitTests
//
// Functional compaction / auto-compaction / nonisolated-property tests.
//
// Moved verbatim from RecallBenchmarkTests (CHA-201 audit): CI skips that
// whole class (`swift test --skip RecallBenchmarkTests`) because the recall
// sweeps take minutes, which silently skipped these fast functional tests
// too. Housing them here means CI exercises them on every PR.

import XCTest
@testable import ProximaKit

final class CompactionTests: XCTestCase {

    // ── Compaction ───────────────────────────────────────────────────────────────

    func testLiveCountAfterRemovals() async throws {
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        for id in ids.prefix(30) { _ = await index.remove(id: id) }

        let liveCount = await index.liveCount
        let totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 100) // tombstones still occupy slots
    }

    func testCompaction() async throws {
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        for id in ids.prefix(30) { _ = await index.remove(id: id) }
        try await index.compact()

        let liveCount = await index.liveCount
        let totalCount = await index.count
        XCTAssertEqual(liveCount, 70)
        XCTAssertEqual(totalCount, 70) // compaction reclaims tombstone slots

        // Search still works after compaction
        let query = Vector((0..<8).map { _ in Float.random(in: -1...1) })
        let results = await index.search(query: query, k: 10)
        XCTAssertEqual(results.count, 10)

        let removedSet = Set(ids.prefix(30))
        for result in results {
            XCTAssertFalse(removedSet.contains(result.id), "Compacted index returned a removed node")
        }
    }

    // ── Auto-Compaction Threshold ──────────────────────────────────────────────────

    func testAutoCompactionTriggersAtThreshold() async throws {
        // Default threshold is 0.7 — removing >30% should trigger auto-compact.
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<20 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Remove 6 of 20 (leaves 14/20 = 70%, NOT below threshold yet).
        for id in ids.prefix(6) { _ = await index.remove(id: id) }
        let countAt70 = await index.count
        // At exactly 70% live ratio, auto-compact should NOT fire (requires strictly < 0.7).
        XCTAssertEqual(countAt70, 20, "At exactly threshold ratio, auto-compact should not trigger")

        // Remove one more — 13/20 = 65%, below the 70% threshold.
        _ = await index.remove(id: ids[6])
        let countAfterAutoCompact = await index.count
        let liveAfterAutoCompact = await index.liveCount
        XCTAssertEqual(liveAfterAutoCompact, 13)
        XCTAssertEqual(countAfterAutoCompact, 13, "Auto-compact should reclaim all tombstones")
    }

    func testAutoCompactionCustomThreshold() async throws {
        // Use a lower threshold (0.5) so auto-compact triggers later.
        let config = HNSWConfiguration(
            m: 4, efConstruction: 20, efSearch: 10,
            autoCompactionThreshold: 0.5
        )
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
        var ids: [UUID] = []

        for _ in 0..<20 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Remove 7 of 20 (leaves 13/20 = 65%, above 0.5 threshold).
        for id in ids.prefix(7) { _ = await index.remove(id: id) }
        let countAt65 = await index.count
        XCTAssertEqual(countAt65, 20, "At 65% live ratio with 50% threshold, no auto-compact")

        // Remove 4 more — 9/20 = 45%, below the 50% threshold.
        for id in ids[7..<11] { _ = await index.remove(id: id) }
        let countAfterAutoCompact = await index.count
        let liveAfterAutoCompact = await index.liveCount
        XCTAssertEqual(liveAfterAutoCompact, 9)
        XCTAssertEqual(countAfterAutoCompact, 9, "Auto-compact should trigger at custom threshold")
    }

    func testAutoCompactionDisabled() async throws {
        // Passing nil disables auto-compaction entirely.
        let config = HNSWConfiguration(
            m: 4, efConstruction: 20, efSearch: 10,
            autoCompactionThreshold: nil
        )
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
        var ids: [UUID] = []

        for _ in 0..<20 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Remove 15 of 20 (leaves 5/20 = 25%).
        for id in ids.prefix(15) { _ = await index.remove(id: id) }
        let totalCount = await index.count
        let liveCount = await index.liveCount
        XCTAssertEqual(liveCount, 5)
        XCTAssertEqual(totalCount, 20, "With auto-compact disabled, tombstones persist")

        // Manual compact still works.
        try await index.compact()
        let compactedCount = await index.count
        XCTAssertEqual(compactedCount, 5, "Manual compact should still reclaim tombstones")
    }

    func testAutoCompactionSearchStillWorksAfter() async throws {
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<20 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        let removedSet = Set(ids.prefix(7))
        // Remove 7 of 20 — triggers auto-compact (13/20 = 65% < 70%).
        for id in ids.prefix(7) { _ = await index.remove(id: id) }

        // After auto-compaction, search should return only live results.
        let query = Vector((0..<4).map { _ in Float.random(in: -1...1) })
        let results = await index.search(query: query, k: 5)
        XCTAssertEqual(results.count, 5)
        for result in results {
            XCTAssertFalse(removedSet.contains(result.id), "Search after auto-compact returned removed node")
        }
    }

    // ── nonisolated properties ────────────────────────────────────────────────────

    func testNonisolatedDimension() {
        let index = HNSWIndex(dimension: 384)
        let dim = index.dimension   // no await — nonisolated let
        XCTAssertEqual(dim, 384)
    }

    func testNonisolatedConfiguration() {
        let config = HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 30)
        let index = HNSWIndex(dimension: 64, config: config)
        let m = index.configuration.m           // no await — nonisolated var
        let ef = index.configuration.efSearch
        XCTAssertEqual(m, 8)
        XCTAssertEqual(ef, 30)
    }
}
