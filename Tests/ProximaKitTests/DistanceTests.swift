import XCTest
@testable import ProximaKit

final class DistanceTests: XCTestCase {

    // ── CosineDistance ────────────────────────────────────────────────

    func testCosineDistanceIdenticalVectors() {
        let metric = CosineDistance()
        let v = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(metric.distance(v, v), 0.0, accuracy: 1e-6)
    }

    func testCosineDistanceOppositeVectors() {
        let metric = CosineDistance()
        let a = Vector([1.0, 0.0])
        let b = Vector([-1.0, 0.0])
        XCTAssertEqual(metric.distance(a, b), 2.0, accuracy: 1e-6)
    }

    func testCosineDistanceOrthogonal() {
        let metric = CosineDistance()
        let a = Vector([1.0, 0.0])
        let b = Vector([0.0, 1.0])
        XCTAssertEqual(metric.distance(a, b), 1.0, accuracy: 1e-6)
    }

    func testCosineDistanceIgnoresMagnitude() {
        let metric = CosineDistance()
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([10.0, 20.0, 30.0])
        XCTAssertEqual(metric.distance(a, b), 0.0, accuracy: 1e-5)
    }

    // ── EuclideanDistance ─────────────────────────────────────────────

    func testEuclideanDistanceSameVector() {
        let metric = EuclideanDistance()
        let v = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(metric.distance(v, v), 0.0, accuracy: 1e-6)
    }

    func testEuclideanDistanceKnown() {
        let metric = EuclideanDistance()
        let a = Vector([0.0, 0.0])
        let b = Vector([3.0, 4.0])
        XCTAssertEqual(metric.distance(a, b), 5.0, accuracy: 1e-5)
    }

    func testEuclideanDistanceSymmetric() {
        let metric = EuclideanDistance()
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([4.0, 5.0, 6.0])
        XCTAssertEqual(metric.distance(a, b), metric.distance(b, a), accuracy: 1e-6)
    }

    // ── DotProductDistance ────────────────────────────────────────────

    func testDotProductDistanceParallel() {
        let metric = DotProductDistance()
        let a = Vector([1.0, 0.0]).normalized()
        let b = Vector([1.0, 0.0]).normalized()
        // Same direction unit vectors: dot = 1, distance = -1
        XCTAssertEqual(metric.distance(a, b), -1.0, accuracy: 1e-6)
    }

    func testDotProductDistanceOrthogonal() {
        let metric = DotProductDistance()
        let a = Vector([1.0, 0.0])
        let b = Vector([0.0, 1.0])
        // Perpendicular: dot = 0, distance = 0
        XCTAssertEqual(metric.distance(a, b), 0.0, accuracy: 1e-6)
    }

    func testDotProductDistanceOrderingMatchesCosine() {
        // For normalized vectors, DotProduct ordering should match Cosine ordering
        let cos = CosineDistance()
        let dot = DotProductDistance()

        let query = Vector([1.0, 2.0, 3.0]).normalized()
        let close = Vector([1.0, 2.1, 3.0]).normalized()
        let far = Vector([3.0, -1.0, 0.5]).normalized()

        let cosClose = cos.distance(query, close)
        let cosFar = cos.distance(query, far)
        let dotClose = dot.distance(query, close)
        let dotFar = dot.distance(query, far)

        // Closer vector should have smaller distance in both metrics
        XCTAssertLessThan(cosClose, cosFar)
        XCTAssertLessThan(dotClose, dotFar)
    }

    // ── Batch Dot Products ───────────────────────────────────────────

    func testBatchDotProductsBasic() {
        let query = Vector([1.0, 2.0, 3.0])
        // Matrix: 3 vectors of dimension 3
        let matrix: [Float] = [
            1.0, 0.0, 0.0,  // v0: dot = 1
            0.0, 1.0, 0.0,  // v1: dot = 2
            0.0, 0.0, 1.0,  // v2: dot = 3
        ]
        let results = batchDotProducts(query: query, matrix: matrix,
                                       vectorCount: 3, dimension: 3)
        XCTAssertEqual(results[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(results[1], 2.0, accuracy: 1e-6)
        XCTAssertEqual(results[2], 3.0, accuracy: 1e-6)
    }

    func testBatchDotProductsMatchesSequential() {
        // Batch result should match individual dot products
        let query = Vector([0.5, -0.3, 0.8, 0.1])
        let vectors = [
            Vector([1.0, 2.0, 3.0, 4.0]),
            Vector([-1.0, 0.5, 0.5, -0.5]),
            Vector([0.0, 0.0, 0.0, 0.0]),
        ]

        let matrix = vectors.flatMap { Array($0.components) }
        let batch = batchDotProducts(query: query, matrix: matrix,
                                     vectorCount: 3, dimension: 4)

        for (i, v) in vectors.enumerated() {
            XCTAssertEqual(batch[i], query.dot(v), accuracy: 1e-5,
                           "Mismatch at index \(i)")
        }
    }

    func testBatchDotProductsHighDimensional() {
        // Test with 384 dims (common embedding size)
        let dim = 384
        let query = Vector(dimension: dim, repeating: 1.0)
        let matrix = [Float](repeating: 2.0, count: dim * 100)

        let results = batchDotProducts(query: query, matrix: matrix,
                                       vectorCount: 100, dimension: dim)

        // Each dot = sum of 1.0 * 2.0 * 384 = 768
        for r in results {
            XCTAssertEqual(r, 768.0, accuracy: 1e-2)
        }
    }

    // ── Batch Distances ──────────────────────────────────────────────

    func testBatchDistancesCosine() {
        let metric = CosineDistance()
        let query = Vector([1.0, 0.0])
        let vectors = [
            Vector([1.0, 0.0]),   // identical: distance 0
            Vector([0.0, 1.0]),   // orthogonal: distance 1
            Vector([-1.0, 0.0]),  // opposite: distance 2
        ]

        let distances = batchDistances(query: query, vectors: vectors, metric: metric)

        XCTAssertEqual(distances[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1], 1.0, accuracy: 1e-5)
        XCTAssertEqual(distances[2], 2.0, accuracy: 1e-5)
    }

    func testBatchDistancesDotProduct() {
        let metric = DotProductDistance()
        let query = Vector([1.0, 2.0])
        let vectors = [
            Vector([3.0, 4.0]),   // dot = 11, distance = -11
            Vector([0.0, 0.0]),   // dot = 0, distance = 0
        ]

        let distances = batchDistances(query: query, vectors: vectors, metric: metric)

        XCTAssertEqual(distances[0], -11.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1], 0.0, accuracy: 1e-5)
    }

    func testBatchDistancesEuclidean() {
        let metric = EuclideanDistance()
        let query = Vector([0.0, 0.0])
        let vectors = [
            Vector([3.0, 4.0]),   // distance = 5
            Vector([0.0, 0.0]),   // distance = 0
        ]

        let distances = batchDistances(query: query, vectors: vectors, metric: metric)

        XCTAssertEqual(distances[0], 5.0, accuracy: 1e-5)
        XCTAssertEqual(distances[1], 0.0, accuracy: 1e-5)
    }

    func testBatchDistancesMatchesIndividual() {
        // Batch should produce identical results to computing one at a time
        let metrics: [any DistanceMetric] = [CosineDistance(), EuclideanDistance(), DotProductDistance()]
        let query = Vector([0.5, -0.3, 0.8])
        let vectors = [
            Vector([1.0, 0.0, 0.0]),
            Vector([0.0, 1.0, 0.0]),
            Vector([-0.5, 0.3, -0.8]),
        ]

        for metric in metrics {
            let batch = batchDistances(query: query, vectors: vectors, metric: metric)
            for (i, v) in vectors.enumerated() {
                let individual = metric.distance(query, v)
                XCTAssertEqual(batch[i], individual, accuracy: 1e-4,
                               "Mismatch for \(type(of: metric)) at index \(i)")
            }
        }
    }

    func testBatchDistancesEmpty() {
        let metric = CosineDistance()
        let query = Vector([1.0, 2.0])
        let distances = batchDistances(query: query, vectors: [], metric: metric)
        XCTAssertTrue(distances.isEmpty)
    }
}
