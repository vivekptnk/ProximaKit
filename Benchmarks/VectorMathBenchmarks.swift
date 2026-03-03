// Benchmarks/VectorMathBenchmarks.swift
// Micro-benchmarks for distance computation — the inner loop of search.

import XCTest
import Accelerate
@testable import ProximaKit

final class VectorMathBenchmarks: XCTestCase {

    // MARK: - Datasets (generated once per class)

    static let dims = [128, 384, 768, 1536]
    static var vectorPairs: [Int: ([Float], [Float])] = [:]
    static var batchVectors: [Int: [[Float]]] = [:]

    override class func setUp() {
        super.setUp()
        for d in dims {
            let vecs = generateUniformVectors(count: 2, dimension: d, seed: 42)
            vectorPairs[d] = (vecs[0], vecs[1])
        }
        // Batch datasets for matrix multiply benchmarks
        for count in [1_000, 10_000, 100_000] {
            batchVectors[count] = generateUniformVectors(count: count, dimension: 384, seed: 42)
        }
    }

    // MARK: - Single Pair Operations

    func testDotProduct384() throws {
        let (a, b) = Self.vectorPairs[384]!
        let stats = measurePercentiles(iterations: 100_000) {
            var result: Float = 0
            a.withUnsafeBufferPointer { aPtr in
                b.withUnsafeBufferPointer { bPtr in
                    vDSP_dotpr(aPtr.baseAddress!, 1,
                               bPtr.baseAddress!, 1,
                               &result, vDSP_Length(a.count))
                }
            }
        }
        print("dot_product_384d: \(stats.report())")
        XCTAssertLessThan(stats.p99, 0.01, "Dot product 384d should be < 10µs at p99")
    }

    func testCosineSimilarity384() throws {
        let (a, b) = Self.vectorPairs[384]!
        let stats = measurePercentiles(iterations: 100_000) {
            _ = cosineSimilarityVDSP(a, b)
        }
        print("cosine_similarity_384d: \(stats.report())")
        XCTAssertLessThan(stats.p99, 0.02, "Cosine similarity 384d should be < 20µs at p99")
    }

    func testL2Distance384() throws {
        let (a, b) = Self.vectorPairs[384]!
        let stats = measurePercentiles(iterations: 100_000) {
            _ = l2DistanceVDSP(a, b)
        }
        print("l2_distance_384d: \(stats.report())")
        XCTAssertLessThan(stats.p99, 0.02, "L2 distance 384d should be < 20µs at p99")
    }

    func testDimensionScaling() throws {
        print("\n--- Dimension Scaling (dot product) ---")
        for dim in Self.dims {
            let (a, b) = Self.vectorPairs[dim]!
            let stats = measurePercentiles(iterations: 50_000) {
                var result: Float = 0
                a.withUnsafeBufferPointer { aPtr in
                    b.withUnsafeBufferPointer { bPtr in
                        vDSP_dotpr(aPtr.baseAddress!, 1,
                                   bPtr.baseAddress!, 1,
                                   &result, vDSP_Length(dim))
                    }
                }
            }
            print("  dot_product_\(dim)d: \(stats.report())")
        }
    }

    // MARK: - Batch Operations (Matrix Multiply)

    func testBatchDot1K() throws {
        let vectors = Self.batchVectors[1_000]!
        let query = Self.vectorPairs[384]!.0
        let dim = 384
        let count = vectors.count

        // Flatten vectors into contiguous matrix
        let matrix = vectors.flatMap { $0 }
        var results = [Float](repeating: 0, count: count)

        let stats = measurePercentiles(iterations: 1_000) {
            matrix.withUnsafeBufferPointer { mPtr in
                query.withUnsafeBufferPointer { qPtr in
                    vDSP_mmul(mPtr.baseAddress!, 1,
                              qPtr.baseAddress!, 1,
                              &results, 1,
                              vDSP_Length(count), 1, vDSP_Length(dim))
                }
            }
        }
        print("batch_dot_1K_384d: \(stats.report())")
        XCTAssertLessThan(stats.p99, 1.0, "Batch dot 1K should be < 1ms")
    }

    func testBatchDot10K() throws {
        let vectors = Self.batchVectors[10_000]!
        let query = Self.vectorPairs[384]!.0
        let dim = 384
        let count = vectors.count

        let matrix = vectors.flatMap { $0 }
        var results = [Float](repeating: 0, count: count)

        let stats = measurePercentiles(iterations: 100) {
            matrix.withUnsafeBufferPointer { mPtr in
                query.withUnsafeBufferPointer { qPtr in
                    vDSP_mmul(mPtr.baseAddress!, 1,
                              qPtr.baseAddress!, 1,
                              &results, 1,
                              vDSP_Length(count), 1, vDSP_Length(dim))
                }
            }
        }
        print("batch_dot_10K_384d: \(stats.report(target: 5.0))")
        XCTAssertLessThan(stats.p99, 5.0, "Batch dot 10K should be < 5ms")
    }

    func testBatchDot100K() throws {
        let vectors = Self.batchVectors[100_000]!
        let query = Self.vectorPairs[384]!.0
        let dim = 384
        let count = vectors.count

        let matrix = vectors.flatMap { $0 }
        var results = [Float](repeating: 0, count: count)

        let stats = measurePercentiles(iterations: 10) {
            matrix.withUnsafeBufferPointer { mPtr in
                query.withUnsafeBufferPointer { qPtr in
                    vDSP_mmul(mPtr.baseAddress!, 1,
                              qPtr.baseAddress!, 1,
                              &results, 1,
                              vDSP_Length(count), 1, vDSP_Length(dim))
                }
            }
        }
        print("batch_dot_100K_384d: \(stats.report(target: 50.0))")
        XCTAssertLessThan(stats.p99, 50.0, "Batch dot 100K should be < 50ms")
    }
}

// MARK: - vDSP Helpers (used by benchmarks before ProximaKit types exist)

private func cosineSimilarityVDSP(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    var aSq: Float = 0
    var bSq: Float = 0
    vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
    vDSP_svesq(a, 1, &aSq, vDSP_Length(a.count))
    vDSP_svesq(b, 1, &bSq, vDSP_Length(b.count))
    let denom = sqrt(aSq) * sqrt(bSq)
    return denom > 0 ? dot / denom : 0
}

private func l2DistanceVDSP(_ a: [Float], _ b: [Float]) -> Float {
    var diff = [Float](repeating: 0, count: a.count)
    vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
    var sumSq: Float = 0
    vDSP_svesq(diff, 1, &sumSq, vDSP_Length(diff.count))
    return sqrt(sumSq)
}
