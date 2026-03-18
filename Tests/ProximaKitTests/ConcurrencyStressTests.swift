import XCTest
@testable import ProximaKit

/// Concurrent write stress tests for HNSWIndex and BruteForceIndex.
///
/// Validates actor isolation under mixed read/write contention:
/// - 10+ simultaneous writers (add, remove, search)
/// - No deadlocks, no priority inversions, no data races
/// - TSan-clean execution
final class ConcurrencyStressTests: XCTestCase {

    // ── HNSW: Concurrent Adds ───────────────────────────────────────────

    /// 50 concurrent writers adding vectors simultaneously.
    func testHNSW_ConcurrentAdds() async throws {
        let index = HNSWIndex(dimension: 16, metric: EuclideanDistance())
        let totalVectors = 200

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<totalVectors {
                group.addTask {
                    let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }
            try await group.waitForAll()
        }

        let count = await index.liveCount
        XCTAssertEqual(count, totalVectors, "All vectors should be indexed")
    }

    /// 50 concurrent writers adding vectors to BruteForceIndex simultaneously.
    func testBruteForce_ConcurrentAdds() async throws {
        let index = BruteForceIndex(dimension: 16, metric: EuclideanDistance())
        let totalVectors = 200

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<totalVectors {
                group.addTask {
                    let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }
            try await group.waitForAll()
        }

        let count = await index.count
        XCTAssertEqual(count, totalVectors, "All vectors should be indexed")
    }

    // ── HNSW: Concurrent Add + Remove ───────────────────────────────────

    /// 20 writers adding while 10 removers delete previously-added vectors.
    func testHNSW_ConcurrentAddRemove() async throws {
        let index = HNSWIndex(dimension: 16, metric: EuclideanDistance())

        // Pre-populate with vectors we'll remove
        var seedIDs: [UUID] = []
        for _ in 0..<50 {
            let id = UUID()
            let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: id)
            seedIDs.append(id)
        }

        // Concurrent adds and removes
        try await withThrowingTaskGroup(of: Void.self) { group in
            // 20 concurrent writers
            for _ in 0..<100 {
                group.addTask {
                    let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }

            // 10 concurrent removers
            for id in seedIDs.prefix(30) {
                group.addTask {
                    _ = await index.remove(id: id)
                }
            }

            try await group.waitForAll()
        }

        // 50 seed + 100 new - 30 removed = 120 live
        let live = await index.liveCount
        XCTAssertEqual(live, 120, "Expected 120 live vectors after add+remove")
    }

    // ── HNSW: Concurrent Add + Search ───────────────────────────────────

    /// 15 writers adding while 15 readers search simultaneously.
    func testHNSW_ConcurrentAddSearch() async throws {
        let index = HNSWIndex(dimension: 16, metric: EuclideanDistance())

        // Pre-populate so searches always find results
        for _ in 0..<50 {
            let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: UUID())
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            // 15 writers
            for _ in 0..<50 {
                group.addTask {
                    let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }

            // 15 concurrent readers
            for _ in 0..<50 {
                group.addTask {
                    let q = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    let results = await index.search(query: q, k: 5)
                    // At minimum, should find some results from the pre-populated set
                    XCTAssertFalse(results.isEmpty, "Search should return results")
                }
            }

            try await group.waitForAll()
        }

        let count = await index.liveCount
        XCTAssertEqual(count, 100, "50 seed + 50 concurrent adds")
    }

    // ── HNSW: Full Mixed Workload ───────────────────────────────────────

    /// Maximal contention: 1000+ concurrent mixed add/remove/search operations.
    func testHNSW_MixedWorkload_1000ops() async throws {
        let index = HNSWIndex(
            dimension: 32,
            metric: CosineDistance(),
            config: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 30)
        )

        // Pre-populate
        var seedIDs: [UUID] = []
        for _ in 0..<100 {
            let id = UUID()
            let v = Vector((0..<32).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: id)
            seedIDs.append(id)
        }

        let addCount = 400
        let removeCount = 50
        let searchCount = 600

        try await withThrowingTaskGroup(of: Void.self) { group in
            // Adds
            for _ in 0..<addCount {
                group.addTask {
                    let v = Vector((0..<32).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }

            // Removes
            for id in seedIDs.prefix(removeCount) {
                group.addTask {
                    _ = await index.remove(id: id)
                }
            }

            // Searches
            for _ in 0..<searchCount {
                group.addTask {
                    let q = Vector((0..<32).map { _ in Float.random(in: -1...1) })
                    let results = await index.search(query: q, k: 10)
                    // Should not crash or deadlock
                    XCTAssertTrue(results.count <= 10)
                }
            }

            try await group.waitForAll()
        }

        let live = await index.liveCount
        let expected = 100 + addCount - removeCount  // 450
        XCTAssertEqual(live, expected, "Live count after mixed workload")
    }

    // ── BruteForce: Full Mixed Workload ─────────────────────────────────

    /// BruteForceIndex under 1000+ concurrent mixed operations.
    func testBruteForce_MixedWorkload_1000ops() async throws {
        let index = BruteForceIndex(dimension: 32, metric: CosineDistance())

        // Pre-populate
        var seedIDs: [UUID] = []
        for _ in 0..<100 {
            let id = UUID()
            let v = Vector((0..<32).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: id)
            seedIDs.append(id)
        }

        let addCount = 400
        let removeCount = 50
        let searchCount = 600

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<addCount {
                group.addTask {
                    let v = Vector((0..<32).map { _ in Float.random(in: -1...1) })
                    try await index.add(v, id: UUID())
                }
            }

            for id in seedIDs.prefix(removeCount) {
                group.addTask {
                    _ = await index.remove(id: id)
                }
            }

            for _ in 0..<searchCount {
                group.addTask {
                    let q = Vector((0..<32).map { _ in Float.random(in: -1...1) })
                    let results = await index.search(query: q, k: 10)
                    XCTAssertTrue(results.count <= 10)
                }
            }

            try await group.waitForAll()
        }

        let count = await index.count
        let expected = 100 + addCount - removeCount
        XCTAssertEqual(count, expected, "Count after mixed workload")
    }

    // ── HNSW: Compact Under Contention ──────────────────────────────────

    /// Compact while concurrent searches are running.
    func testHNSW_CompactDuringSearch() async throws {
        let index = HNSWIndex(dimension: 16, metric: EuclideanDistance())

        // Build and remove some vectors to create tombstones
        var removeIDs: [UUID] = []
        for i in 0..<100 {
            let id = UUID()
            let v = Vector((0..<16).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: id)
            if i < 40 { removeIDs.append(id) }
        }
        for id in removeIDs { _ = await index.remove(id: id) }

        // Compact + search concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await index.compact()
            }

            for _ in 0..<20 {
                group.addTask {
                    let q = Vector((0..<16).map { _ in Float.random(in: -1...1) })
                    let results = await index.search(query: q, k: 5)
                    // Compact may reset the index mid-search; results should not crash
                    XCTAssertTrue(results.count <= 5)
                }
            }

            try await group.waitForAll()
        }

        let live = await index.liveCount
        let total = await index.count
        XCTAssertEqual(live, 60)
        XCTAssertEqual(total, 60, "Compaction should reclaim tombstones")
    }

    // ── Throughput Scaling ───────────────────────────────────────────────

    /// Measures concurrent search throughput to confirm actor serialization
    /// doesn't cause excessive contention.
    func testHNSW_ConcurrentSearchThroughput() async throws {
        let index = HNSWIndex(
            dimension: 64,
            metric: EuclideanDistance(),
            config: HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        )

        // Build index
        for _ in 0..<500 {
            let v = Vector((0..<64).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: UUID())
        }

        let concurrentReaders = 50
        let queriesPerReader = 10

        let start = CFAbsoluteTimeGetCurrent()

        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<concurrentReaders {
                group.addTask {
                    var count = 0
                    for _ in 0..<queriesPerReader {
                        let q = Vector((0..<64).map { _ in Float.random(in: -1...1) })
                        let results = await index.search(query: q, k: 10)
                        count += results.count
                    }
                    return count
                }
            }

            var totalResults = 0
            for await count in group {
                totalResults += count
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start
            let totalQueries = concurrentReaders * queriesPerReader
            let qps = Double(totalQueries) / elapsed
            print("Concurrent search throughput: \(totalQueries) queries in \(String(format: "%.2f", elapsed))s = \(String(format: "%.0f", qps)) QPS")
            print("Total results returned: \(totalResults)")
        }
    }
}
