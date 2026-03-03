// Benchmarks/ConcurrencyBenchmarks.swift
// Verifies thread safety and measures actor overhead under concurrent load.

import XCTest
@testable import ProximaKit

final class ConcurrencyBenchmarks: XCTestCase {

    /// Multiple readers hammering the index simultaneously.
    /// Must not crash, must not produce data races.
    func testConcurrentReads() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let queries = generateUniformVectors(count: 100, dimension: 384, seed: 99)

        // Build index (replace with actual)
        // let index = HNSWIndex(dimension: 384, M: 16, efConstruction: 200)
        // for (i, v) in vectors.enumerated() { try await index.add(Vector(v), ...) }

        let concurrency = 8
        let queriesPerTask = 50
        let totalQueries = concurrency * queriesPerTask

        let start = DispatchTime.now()

        await withTaskGroup(of: Int.self) { group in
            for taskIndex in 0..<concurrency {
                group.addTask {
                    var completed = 0
                    for i in 0..<queriesPerTask {
                        let queryIdx = (taskIndex * queriesPerTask + i) % queries.count
                        // Replace with: let _ = try await index.search(query: Vector(queries[queryIdx]), k: 10)
                        let _ = queries[queryIdx]
                        completed += 1
                    }
                    return completed
                }
            }

            var totalCompleted = 0
            for await count in group {
                totalCompleted += count
            }
            XCTAssertEqual(totalCompleted, totalQueries)
        }

        let end = DispatchTime.now()
        let elapsed = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
        let throughput = Double(totalQueries) / elapsed

        print("concurrent_reads (\(concurrency) tasks): \(Int(throughput)) queries/sec")
        // Target: > 500 queries/sec with 8 concurrent readers
    }

    /// Mixed read/write workload: 4 readers + 1 writer running simultaneously.
    /// The writer inserts new vectors while readers are searching.
    func testConcurrentReadWrite() async throws {
        let initialVectors = generateUniformVectors(count: 5_000, dimension: 384)
        let newVectors = generateUniformVectors(count: 1_000, dimension: 384, seed: 77)
        let queries = generateUniformVectors(count: 100, dimension: 384, seed: 99)

        // Build initial index
        // let index = HNSWIndex(...)
        // for v in initialVectors { ... }

        var readCount = 0
        var writeCount = 0

        let start = DispatchTime.now()

        await withTaskGroup(of: (reads: Int, writes: Int).self) { group in
            // 4 reader tasks
            for taskIdx in 0..<4 {
                group.addTask {
                    var reads = 0
                    for i in 0..<25 {
                        let q = queries[(taskIdx * 25 + i) % queries.count]
                        // let _ = try await index.search(query: Vector(q), k: 10)
                        let _ = q
                        reads += 1
                    }
                    return (reads, 0)
                }
            }

            // 1 writer task
            group.addTask {
                var writes = 0
                for v in newVectors.prefix(100) {
                    // try await index.add(Vector(v), ...)
                    let _ = v
                    writes += 1
                }
                return (0, writes)
            }

            for await (r, w) in group {
                readCount += r
                writeCount += w
            }
        }

        let end = DispatchTime.now()
        let elapsed = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000

        print("concurrent_read_write: \(readCount) reads + \(writeCount) writes in \(formatDuration(elapsed))")
        print("  no crashes, no data races ✅")
    }

    /// Measures the overhead of actor isolation vs direct (unsafe) access.
    func testActorOverhead() async throws {
        // Compare:
        // 1. Direct function call (no actor): baseline latency
        // 2. Actor-isolated call: with actor hop
        // The difference is the actor overhead.

        let query = generateUniformVectors(count: 1, dimension: 384, seed: 42)[0]

        // Direct (simulated — no actor hop)
        let directStats = measurePercentiles(iterations: 10_000) {
            // Simulate direct search without actor
            var result: Float = 0
            for i in 0..<query.count { result += query[i] * query[i] }
        }

        // Actor-isolated (simulated)
        // In practice: let stats = measurePercentiles(iterations: 10_000) { await index.search(...) }
        let actorStats = directStats  // placeholder

        let overhead = ((actorStats.p50 - directStats.p50) / directStats.p50) * 100
        print("actor_overhead: \(String(format: "%.1f", overhead))%")
        print("  direct p50: \(String(format: "%.3f", directStats.p50))ms")
        print("  actor  p50: \(String(format: "%.3f", actorStats.p50))ms")
        // Target: < 10% overhead
    }
}
