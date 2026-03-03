// Benchmarks/RecallBenchmarks.swift
// Search quality benchmarks: measures what fraction of true nearest neighbors HNSW actually finds.
// This is the most important benchmark. Fast but wrong search is useless.

import XCTest
@testable import ProximaKit

final class RecallBenchmarks: XCTestCase {

    static let queryCount = 100  // Number of queries to average recall over

    // MARK: - Recall on Uniform Data

    func testRecallUniform10K() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let queries = generateUniformVectors(count: Self.queryCount, dimension: 384, seed: 99)

        // Build brute-force ground truth
        let groundTruth = computeBruteForceTopK(
            vectors: vectors, queries: queries, k: 10
        )

        let efValues = [10, 25, 50, 100, 200]

        print("\n--- Recall on Uniform Data (10K/384d) ---")
        print("  ef        @1        @5        @10")
        for ef in efValues {
            // Build HNSW and query (replace with actual)
            let hnswResults = groundTruth // placeholder — replace with actual HNSW results

            let r1 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 1)
            let r5 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 5)
            let r10 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 10)

            let pass = r10 >= 0.95 ? "✅" : "❌"
            print("  \(String(ef).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r1).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r5).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r10)) \(pass)")
        }
    }

    // MARK: - Recall on Clustered Data (the hard case)

    func testRecallClustered10K() async throws {
        let vectors = generateClusteredVectors(
            count: 10_000, dimension: 384, clusterCount: 8, spread: 0.1
        )
        let queries = generateUniformVectors(count: Self.queryCount, dimension: 384, seed: 99)

        let groundTruth = computeBruteForceTopK(
            vectors: vectors, queries: queries, k: 10
        )

        let efValues = [10, 25, 50, 100, 200]

        print("\n--- Recall on Clustered Data (10K/384d, 8 clusters) ---")
        print("  ef        @1        @5        @10")
        for ef in efValues {
            let hnswResults = groundTruth // placeholder

            let r1 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 1)
            let r5 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 5)
            let r10 = recallAtK(groundTruth: groundTruth, approximate: hnswResults, k: 10)

            let pass = r10 >= 0.92 ? "✅" : "❌"
            print("  \(String(ef).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r1).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r5).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r10)) \(pass)")
        }
    }

    // MARK: - Recall vs Dimension

    func testRecallDimensionScaling() async throws {
        let dims = [128, 384, 768]

        print("\n--- Recall vs Dimension (10K, ef=50) ---")
        print("  dim       @10       build_time")
        for dim in dims {
            let vectors = generateUniformVectors(count: 10_000, dimension: dim)
            let queries = generateUniformVectors(count: 50, dimension: dim, seed: 99)

            let (groundTruth, buildTime) = measure {
                computeBruteForceTopK(vectors: vectors, queries: queries, k: 10)
            }

            let r10 = 1.0 // placeholder — replace with actual HNSW recall
            print("  \(String(dim).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.3f", r10).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(formatDuration(buildTime))")
        }
    }

    // MARK: - Parameter Sweep: M × efConstruction × efSearch

    func testRecallParamSweep() async throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let queries = generateUniformVectors(count: 50, dimension: 384, seed: 99)
        let groundTruth = computeBruteForceTopK(vectors: vectors, queries: queries, k: 10)

        let mValues = [8, 16, 32]
        let efcValues = [100, 200]
        let efsValues = [25, 50, 100]

        print("\n--- Recall Parameter Sweep (10K/384d) ---")
        print("  M     efC    efS    recall@10   query_ms")
        for m in mValues {
            for efc in efcValues {
                for efs in efsValues {
                    // Build with (m, efc), query with (efs)
                    // Replace with actual HNSW results
                    let r10 = 1.0 // placeholder
                    let queryMs = 0.0 // placeholder
                    print("  \(String(m).padding(toLength: 6, withPad: " ", startingAt: 0))"
                        + "\(String(efc).padding(toLength: 7, withPad: " ", startingAt: 0))"
                        + "\(String(efs).padding(toLength: 7, withPad: " ", startingAt: 0))"
                        + "\(String(format: "%.3f", r10).padding(toLength: 12, withPad: " ", startingAt: 0))"
                        + "\(String(format: "%.1f", queryMs))")
                }
            }
        }
    }
}

// MARK: - Brute-Force Ground Truth

/// Computes exact top-K for each query using brute-force cosine similarity.
/// Returns array of arrays of vector indices.
func computeBruteForceTopK(
    vectors: [[Float]],
    queries: [[Float]],
    k: Int
) -> [[Int]] {
    return queries.map { query in
        let distances = vectors.enumerated().map { (index, vector) -> (Int, Float) in
            let sim = cosineSim(query, vector)
            return (index, sim)
        }
        let topK = distances.sorted { $0.1 > $1.1 }.prefix(k).map(\.0)
        return Array(topK)
    }
}

private func cosineSim(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    var aSq: Float = 0
    var bSq: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        aSq += a[i] * a[i]
        bSq += b[i] * b[i]
    }
    let denom = sqrt(aSq) * sqrt(bSq)
    return denom > 0 ? dot / denom : 0
}
