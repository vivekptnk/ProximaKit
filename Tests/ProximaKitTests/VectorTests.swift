import XCTest
@testable import ProximaKit

final class VectorTests: XCTestCase {

    // ── Initialization ────────────────────────────────────────────────

    func testInitFromArray() {
        let v = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(v.dimension, 3)
        XCTAssertEqual(v[0], 1.0)
        XCTAssertEqual(v[1], 2.0)
        XCTAssertEqual(v[2], 3.0)
    }

    func testInitFromContiguousArray() {
        let data: ContiguousArray<Float> = [4.0, 5.0]
        let v = Vector(data)
        XCTAssertEqual(v.dimension, 2)
        XCTAssertEqual(v[0], 4.0)
    }

    func testInitFromUnsafeBufferPointer() {
        let data: [Float] = [7.0, 8.0, 9.0]
        let v = data.withUnsafeBufferPointer { Vector($0) }
        XCTAssertEqual(v.dimension, 3)
        XCTAssertEqual(v[2], 9.0)
    }

    func testInitWithDimensionAndFill() {
        let v = Vector(dimension: 5, repeating: 3.0)
        XCTAssertEqual(v.dimension, 5)
        for i in 0..<5 {
            XCTAssertEqual(v[i], 3.0)
        }
    }

    func testInitWithDimensionDefaultsToZero() {
        let v = Vector(dimension: 3)
        XCTAssertEqual(v[0], 0.0)
        XCTAssertEqual(v[1], 0.0)
        XCTAssertEqual(v[2], 0.0)
    }

    // ── Properties ────────────────────────────────────────────────────

    func testMagnitudeOfUnitVector() {
        // [1, 0, 0] has magnitude 1
        let v = Vector([1.0, 0.0, 0.0])
        XCTAssertEqual(v.magnitude, 1.0, accuracy: 1e-6)
    }

    func testMagnitude345Triangle() {
        // Classic 3-4-5 right triangle: magnitude of [3, 4] = 5
        let v = Vector([3.0, 4.0])
        XCTAssertEqual(v.magnitude, 5.0, accuracy: 1e-6)
    }

    func testMagnitudeOfZeroVector() {
        let v = Vector(dimension: 3)
        XCTAssertEqual(v.magnitude, 0.0)
    }

    // ── Dot Product ───────────────────────────────────────────────────

    func testDotProductBasic() {
        // [1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([4.0, 5.0, 6.0])
        XCTAssertEqual(a.dot(b), 32.0, accuracy: 1e-6)
    }

    func testDotProductIsCommutative() {
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([4.0, 5.0, 6.0])
        XCTAssertEqual(a.dot(b), b.dot(a), accuracy: 1e-6)
    }

    func testDotProductOrthogonal() {
        // Perpendicular vectors have dot product 0
        let a = Vector([1.0, 0.0])
        let b = Vector([0.0, 1.0])
        XCTAssertEqual(a.dot(b), 0.0, accuracy: 1e-6)
    }

    func testDotProductWithSelf() {
        // v · v = |v|² (dot product with itself equals magnitude squared)
        let v = Vector([3.0, 4.0])
        XCTAssertEqual(v.dot(v), 25.0, accuracy: 1e-6) // 3² + 4² = 25
    }

    func testDotProductHighDimensional() {
        // Test with 384 dims (common embedding size) to verify vDSP at scale
        let a = Vector(dimension: 384, repeating: 1.0)
        let b = Vector(dimension: 384, repeating: 2.0)
        // Each pair contributes 1×2 = 2, so total = 384 × 2 = 768
        XCTAssertEqual(a.dot(b), 768.0, accuracy: 1e-3)
    }

    // ── Cosine Similarity ─────────────────────────────────────────────

    func testCosineSimilarityIdenticalVectors() {
        let v = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(v.cosineSimilarity(v), 1.0, accuracy: 1e-6)
    }

    func testCosineSimilarityOppositeVectors() {
        let a = Vector([1.0, 0.0])
        let b = Vector([-1.0, 0.0])
        XCTAssertEqual(a.cosineSimilarity(b), -1.0, accuracy: 1e-6)
    }

    func testCosineSimilarityOrthogonalVectors() {
        let a = Vector([1.0, 0.0])
        let b = Vector([0.0, 1.0])
        XCTAssertEqual(a.cosineSimilarity(b), 0.0, accuracy: 1e-6)
    }

    func testCosineSimilarityIgnoresMagnitude() {
        // Cosine similarity only cares about direction, not length.
        // [1,2,3] and [10,20,30] point the same way → cosine = 1.0
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([10.0, 20.0, 30.0])
        XCTAssertEqual(a.cosineSimilarity(b), 1.0, accuracy: 1e-5)
    }

    func testCosineSimilarityZeroVector() {
        let a = Vector([1.0, 2.0])
        let zero = Vector(dimension: 2)
        XCTAssertEqual(a.cosineSimilarity(zero), 0.0)
    }

    // ── L2 Distance ───────────────────────────────────────────────────

    func testL2DistanceSameVector() {
        let v = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(v.l2Distance(v), 0.0, accuracy: 1e-6)
    }

    func testL2DistanceKnownValue() {
        // Distance between [1,0] and [0,1] = sqrt(1² + 1²) = sqrt(2)
        let a = Vector([1.0, 0.0])
        let b = Vector([0.0, 1.0])
        XCTAssertEqual(a.l2Distance(b), sqrt(2.0), accuracy: 1e-5)
    }

    func testL2DistanceIsSymmetric() {
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([4.0, 5.0, 6.0])
        XCTAssertEqual(a.l2Distance(b), b.l2Distance(a), accuracy: 1e-6)
    }

    func testL2DistanceTriangleInequality() {
        // For any three points: d(a,c) <= d(a,b) + d(b,c)
        let a = Vector([0.0, 0.0])
        let b = Vector([1.0, 0.0])
        let c = Vector([0.0, 1.0])
        let ab = a.l2Distance(b)
        let bc = b.l2Distance(c)
        let ac = a.l2Distance(c)
        XCTAssertLessThanOrEqual(ac, ab + bc + 1e-6)
    }

    // ── Normalization ─────────────────────────────────────────────────

    func testNormalizedHasUnitMagnitude() {
        let v = Vector([3.0, 4.0])
        let n = v.normalized()
        XCTAssertEqual(n.magnitude, 1.0, accuracy: 1e-6)
    }

    func testNormalizedPreservesDirection() {
        // Normalized vector should have cosine similarity 1.0 with original
        let v = Vector([3.0, 4.0])
        let n = v.normalized()
        XCTAssertEqual(v.cosineSimilarity(n), 1.0, accuracy: 1e-5)
    }

    func testNormalizedValues() {
        // [3, 4] normalized = [3/5, 4/5] = [0.6, 0.8]
        let v = Vector([3.0, 4.0])
        let n = v.normalized()
        XCTAssertEqual(n[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(n[1], 0.8, accuracy: 1e-6)
    }

    func testNormalizedZeroVectorReturnsSelf() {
        let zero = Vector(dimension: 3)
        let n = zero.normalized()
        XCTAssertEqual(n[0], 0.0)
        XCTAssertEqual(n[1], 0.0)
        XCTAssertEqual(n[2], 0.0)
    }

    func testNormalizedAlreadyUnit() {
        let v = Vector([1.0, 0.0, 0.0])
        let n = v.normalized()
        XCTAssertEqual(n[0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(n[1], 0.0, accuracy: 1e-6)
    }

    func testNormalizedDotProductEqualsCosineSimilarity() {
        // Key insight: for unit vectors, dot product IS cosine similarity.
        // This is why we pre-normalize vectors in search indices.
        let a = Vector([1.0, 2.0, 3.0]).normalized()
        let b = Vector([4.0, 5.0, 6.0]).normalized()
        let dotResult = a.dot(b)
        let cosResult = a.cosineSimilarity(b)
        XCTAssertEqual(dotResult, cosResult, accuracy: 1e-5)
    }

    // ── Equatable / Hashable ──────────────────────────────────────────

    func testEquality() {
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([1.0, 2.0, 3.0])
        XCTAssertEqual(a, b)
    }

    func testInequality() {
        let a = Vector([1.0, 2.0, 3.0])
        let b = Vector([1.0, 2.0, 4.0])
        XCTAssertNotEqual(a, b)
    }

    func testHashConsistency() {
        let a = Vector([1.0, 2.0])
        let b = Vector([1.0, 2.0])
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    func testUsableInSet() {
        let a = Vector([1.0, 2.0])
        let b = Vector([1.0, 2.0])
        let c = Vector([3.0, 4.0])
        let set: Set<Vector> = [a, b, c]
        XCTAssertEqual(set.count, 2) // a and b are equal
    }

    // ── Codable ───────────────────────────────────────────────────────

    func testCodableRoundTrip() throws {
        let original = Vector([1.5, 2.5, 3.5])
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(Vector.self, from: data)
        XCTAssertEqual(original, decoded)
    }
}
