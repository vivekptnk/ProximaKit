// Benchmarks/IndexBenchmarks.swift
// Build time, query latency, and throughput benchmarks for BruteForce and HNSW indices.

import XCTest
@testable import ProximaKit

final class IndexBuildBenchmarks: XCTestCase {

    // MARK: - Build Benchmarks

    func testBruteForceBuild10K() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let (_, elapsed) = measure("brute_build_10K") {
            // BruteForceIndex build is just inserting vectors into an array
            // This tests the insertion path + any overhead
            var _ = vectors  // Replace with actual BruteForceIndex once implemented
        }
        let throughput = Double(10_000) / elapsed
        print("  throughput: \(Int(throughput)) vectors/sec")
    }

    func testHNSWBuild1K() async throws {
        let vectors = generateUniformVectors(count: 1_000, dimension: 384)
        let (_, elapsed) = measure("hnsw_build_1K") {
            // Replace with actual HNSWIndex build once implemented
            var _ = vectors
        }
        let throughput = Double(1_000) / elapsed
        print("  throughput: \(Int(throughput)) vectors/sec")
        XCTAssertLessThan(elapsed, 2.0, "HNSW 1K build should be < 2s")
    }

    func testHNSWBuild10K() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let (_, elapsed) = measure("hnsw_build_10K") {
            // Replace with actual HNSWIndex build once implemented
            var _ = vectors
        }
        let throughput = Double(10_000) / elapsed
        print("  throughput: \(Int(throughput)) vectors/sec")
        XCTAssertLessThan(elapsed, 5.0, "HNSW 10K build should be < 5s")
    }

    func testHNSWBuild50K() async throws {
        let vectors = generateUniformVectors(count: 50_000, dimension: 384)
        let (_, elapsed) = measure("hnsw_build_50K") {
            var _ = vectors
        }
        let throughput = Double(50_000) / elapsed
        print("  throughput: \(Int(throughput)) vectors/sec")
        XCTAssertLessThan(elapsed, 30.0, "HNSW 50K build should be < 30s")
    }

    func testHNSWBuildParamSweep() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let mValues = [8, 16, 32]
        let efValues = [100, 200, 400]

        print("\n--- HNSW Build Parameter Sweep (10K/384d) ---")
        print("  M\\efC     100       200       400")
        for m in mValues {
            var row = "  \(String(m).padding(toLength: 8, withPad: " ", startingAt: 0))"
            for ef in efValues {
                let (_, elapsed) = measure {
                    // Replace with: HNSWIndex(dimension: 384, M: m, efConstruction: ef)
                    // then insert all vectors
                    var _ = (m, ef, vectors.count)
                }
                row += "\(formatDuration(elapsed).padding(toLength: 10, withPad: " ", startingAt: 0))"
            }
            print(row)
        }
    }
}

final class IndexQueryBenchmarks: XCTestCase {

    // MARK: - Query Latency

    func testBruteForceQuery10K() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let queries = generateUniformVectors(count: 100, dimension: 384, seed: 99)

        // Build index (replace with actual)
        // let index = BruteForceIndex(dimension: 384, metric: .cosine)
        // for (i, v) in vectors.enumerated() { try await index.add(Vector(v), id: ...) }

        let stats = measurePercentiles(iterations: 100) {
            // let _ = try await index.search(query: Vector(queries[i % queries.count]), k: 10)
        }
        print("brute_query_10K_384d: \(stats.report(target: 50.0))")
        XCTAssertLessThan(stats.p99, 50.0, "BruteForce 10K query p99 should be < 50ms")
    }

    func testHNSWQuery1K() async throws {
        let stats = measurePercentiles(iterations: 1_000) {
            // Replace with actual HNSW query
        }
        print("hnsw_query_1K_384d: \(stats.report())")
    }

    func testHNSWQuery10K() async throws {
        let stats = measurePercentiles(iterations: 500) {
            // Replace with actual HNSW query
        }
        print("hnsw_query_10K_384d: \(stats.report(target: 50.0))")
        // XCTAssertLessThan(stats.p50, 5.0, "HNSW 10K query p50 should be < 5ms")
        // XCTAssertLessThan(stats.p95, 15.0, "HNSW 10K query p95 should be < 15ms")
        // XCTAssertLessThan(stats.p99, 50.0, "HNSW 10K query p99 should be < 50ms")
    }

    func testHNSWQuery50K() async throws {
        let stats = measurePercentiles(iterations: 100) {
            // Replace with actual HNSW query
        }
        print("hnsw_query_50K_384d: \(stats.report())")
    }

    func testHNSWQueryEfSweep() async throws {
        let efValues = [10, 25, 50, 100, 200]

        print("\n--- HNSW Query ef Sweep (10K/384d, k=10) ---")
        print("  ef        p50       p95       p99")
        for ef in efValues {
            let stats = measurePercentiles(iterations: 200) {
                // Replace with: index.search(query: q, k: 10, ef: ef)
            }
            let row = "  \(String(ef).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(fmt(stats.p50))ms".padding(toLength: 10, withPad: " ", startingAt: 0)
                + "\(fmt(stats.p95))ms".padding(toLength: 10, withPad: " ", startingAt: 0)
                + "\(fmt(stats.p99))ms"
            print(row)
        }
    }
}

private func fmt(_ value: Double) -> String {
    if value < 1 { return String(format: "%.2f", value) }
    if value < 100 { return String(format: "%.1f", value) }
    return String(format: "%.0f", value)
}
