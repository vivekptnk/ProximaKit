import XCTest
@testable import ProximaKit
import Accelerate

/// Benchmarks comparing vDSP/SIMD-accelerated vector operations against
/// naive (manual loop) implementations.
///
/// These tests validate that ProximaKit's Accelerate-backed operations
/// produce correct results AND measure the speedup over naive loops.
///
/// Run with: `swift test --filter SIMDBenchmarkTests`
final class SIMDBenchmarkTests: XCTestCase {

    // ── Naive Baselines ────────────────────────────────────────────────
    // These are the "before" implementations — pure Swift loops, no Accelerate.

    /// Naive dot product: manual loop over elements.
    private func naiveDot(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count { sum += a[i] * b[i] }
        return sum
    }

    /// Naive L2 distance: manual subtraction + sum of squares + sqrt.
    private func naiveL2(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let d = a[i] - b[i]
            sum += d * d
        }
        return sqrt(sum)
    }

    /// Naive magnitude: manual sum of squares + sqrt.
    private func naiveMagnitude(_ a: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count { sum += a[i] * a[i] }
        return sqrt(sum)
    }

    /// Naive batch dot products: N individual dot products via loops.
    private func naiveBatchDots(query: [Float], matrix: [Float], count: Int, dim: Int) -> [Float] {
        var results = [Float](repeating: 0, count: count)
        for i in 0..<count {
            var sum: Float = 0
            let offset = i * dim
            for j in 0..<dim {
                sum += query[j] * matrix[offset + j]
            }
            results[i] = sum
        }
        return results
    }

    /// Naive batch L2 distances: N individual L2 computations via loops.
    private func naiveBatchL2(query: [Float], matrix: [Float], count: Int, dim: Int) -> [Float] {
        var results = [Float](repeating: 0, count: count)
        for i in 0..<count {
            var sum: Float = 0
            let offset = i * dim
            for j in 0..<dim {
                let d = query[j] - matrix[offset + j]
                sum += d * d
            }
            results[i] = sqrt(sum)
        }
        return results
    }

    // ── Correctness: vDSP matches naive ────────────────────────────────

    func testDotProductMatchesNaive() {
        let dim = 384
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let va = Vector(a)
        let vb = Vector(b)

        let naiveResult = naiveDot(a, b)
        let simdResult = va.dot(vb)

        XCTAssertEqual(simdResult, naiveResult, accuracy: 1e-3,
                       "vDSP dot product must match naive loop")
    }

    func testL2DistanceMatchesNaive() {
        let dim = 384
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }
        let b = (0..<dim).map { _ in Float.random(in: -1...1) }

        let naiveResult = naiveL2(a, b)
        let simdResult = Vector(a).l2Distance(Vector(b))

        XCTAssertEqual(simdResult, naiveResult, accuracy: 1e-3,
                       "vDSP L2 distance must match naive loop")
    }

    func testMagnitudeMatchesNaive() {
        let dim = 384
        let a = (0..<dim).map { _ in Float.random(in: -1...1) }

        let naiveResult = naiveMagnitude(a)
        let simdResult = Vector(a).magnitude

        XCTAssertEqual(simdResult, naiveResult, accuracy: 1e-3,
                       "vDSP magnitude must match naive loop")
    }

    func testBatchDotProductsMatchNaive() {
        let dim = 384
        let count = 1000
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let matrix = (0..<count * dim).map { _ in Float.random(in: -1...1) }

        let naiveResults = naiveBatchDots(query: query, matrix: matrix, count: count, dim: dim)
        let simdResults = batchDotProducts(
            query: Vector(query), matrix: matrix,
            vectorCount: count, dimension: dim
        )

        for i in 0..<count {
            XCTAssertEqual(simdResults[i], naiveResults[i], accuracy: 1e-2,
                           "Batch dot mismatch at index \(i)")
        }
    }

    func testBatchL2DistancesMatchNaive() {
        let dim = 384
        let count = 1000
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let matrix = (0..<count * dim).map { _ in Float.random(in: -1...1) }

        let naiveResults = naiveBatchL2(query: query, matrix: matrix, count: count, dim: dim)
        let simdResults = batchL2Distances(
            query: Vector(query), matrix: matrix,
            vectorCount: count, dimension: dim
        )

        for i in 0..<count {
            XCTAssertEqual(simdResults[i], naiveResults[i], accuracy: 1e-2,
                           "Batch L2 mismatch at index \(i)")
        }
    }

    func testBatchEuclideanDistanceFastPath() {
        // Verify batchDistances uses the optimized path for EuclideanDistance
        let metric = EuclideanDistance()
        let query = Vector([1.0, 2.0, 3.0])
        let vectors = [
            Vector([4.0, 5.0, 6.0]),
            Vector([0.0, 0.0, 0.0]),
            Vector([1.0, 2.0, 3.0]),
        ]

        let batch = batchDistances(query: query, vectors: vectors, metric: metric)

        // Verify against individual computation
        for (i, v) in vectors.enumerated() {
            let individual = metric.distance(query, v)
            XCTAssertEqual(batch[i], individual, accuracy: 1e-4,
                           "Batch Euclidean mismatch at index \(i)")
        }
    }

    // ── Benchmarks: SIMD vs Naive ──────────────────────────────────────

    func testBenchmarkDotProduct_384d() {
        let dim = 384
        let a = Vector((0..<dim).map { _ in Float.random(in: -1...1) })
        let b = Vector((0..<dim).map { _ in Float.random(in: -1...1) })
        let aArr = Array(a.components)
        let bArr = Array(b.components)

        let iterations = 10_000

        // Naive timing
        let naiveStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = naiveDot(aArr, bArr) }
        let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStart

        // vDSP timing
        let simdStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { _ = a.dot(b) }
        let simdTime = CFAbsoluteTimeGetCurrent() - simdStart

        let speedup = naiveTime / simdTime
        print("\n[Benchmark] Dot product (384d, \(iterations) iterations)")
        print("  Naive: \(String(format: "%.3f", naiveTime * 1000))ms")
        print("  vDSP:  \(String(format: "%.3f", simdTime * 1000))ms")
        print("  Speedup: \(String(format: "%.1f", speedup))x")

        // vDSP should be at least competitive (on small dims the overhead
        // can reduce the gap, but at 384d we expect meaningful speedup)
        XCTAssertGreaterThan(speedup, 0.5, "vDSP should not be significantly slower than naive")
    }

    func testBenchmarkBatchDotProducts_1K_384d() {
        let dim = 384
        let count = 1_000
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let matrix = (0..<count * dim).map { _ in Float.random(in: -1...1) }
        let queryVec = Vector(query)

        let iterations = 100

        // Naive timing
        let naiveStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = naiveBatchDots(query: query, matrix: matrix, count: count, dim: dim)
        }
        let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStart

        // vDSP_mmul timing
        let simdStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = batchDotProducts(query: queryVec, matrix: matrix,
                                vectorCount: count, dimension: dim)
        }
        let simdTime = CFAbsoluteTimeGetCurrent() - simdStart

        let speedup = naiveTime / simdTime
        print("\n[Benchmark] Batch dot products (1K×384d, \(iterations) iterations)")
        print("  Naive:     \(String(format: "%.3f", naiveTime * 1000))ms")
        print("  vDSP_mmul: \(String(format: "%.3f", simdTime * 1000))ms")
        print("  Speedup:   \(String(format: "%.1f", speedup))x")

        XCTAssertGreaterThan(speedup, 1.0, "Batch vDSP_mmul should outperform naive loops")
    }

    func testBenchmarkBatchL2_1K_384d() {
        let dim = 384
        let count = 1_000
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let matrix = (0..<count * dim).map { _ in Float.random(in: -1...1) }
        let queryVec = Vector(query)

        let iterations = 100

        // Naive timing
        let naiveStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = naiveBatchL2(query: query, matrix: matrix, count: count, dim: dim)
        }
        let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStart

        // vDSP batch L2 timing
        let simdStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = batchL2Distances(query: queryVec, matrix: matrix,
                                vectorCount: count, dimension: dim)
        }
        let simdTime = CFAbsoluteTimeGetCurrent() - simdStart

        let speedup = naiveTime / simdTime
        print("\n[Benchmark] Batch L2 distance (1K×384d, \(iterations) iterations)")
        print("  Naive:  \(String(format: "%.3f", naiveTime * 1000))ms")
        print("  vDSP:   \(String(format: "%.3f", simdTime * 1000))ms")
        print("  Speedup: \(String(format: "%.1f", speedup))x")

        XCTAssertGreaterThan(speedup, 1.0, "Batch vDSP L2 should outperform naive loops")
    }

    func testBenchmarkBatchL2_10K_384d() {
        let dim = 384
        let count = 10_000
        let query = (0..<dim).map { _ in Float.random(in: -1...1) }
        let matrix = (0..<count * dim).map { _ in Float.random(in: -1...1) }
        let queryVec = Vector(query)

        let iterations = 10

        // Naive timing
        let naiveStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = naiveBatchL2(query: query, matrix: matrix, count: count, dim: dim)
        }
        let naiveTime = CFAbsoluteTimeGetCurrent() - naiveStart

        // vDSP batch L2 timing
        let simdStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = batchL2Distances(query: queryVec, matrix: matrix,
                                vectorCount: count, dimension: dim)
        }
        let simdTime = CFAbsoluteTimeGetCurrent() - simdStart

        let speedup = naiveTime / simdTime
        print("\n[Benchmark] Batch L2 distance (10K×384d, \(iterations) iterations)")
        print("  Naive:  \(String(format: "%.3f", naiveTime * 1000))ms")
        print("  vDSP:   \(String(format: "%.3f", simdTime * 1000))ms")
        print("  Speedup: \(String(format: "%.1f", speedup))x")

        XCTAssertGreaterThan(speedup, 1.0, "Batch vDSP L2 at 10K scale should outperform naive")
    }
}
