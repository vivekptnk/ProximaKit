// ScalarQuantizerTests.swift
// ProximaKitTests
//
// Tests for ScalarQuantizer: round-trip error bounds, zero-vector handling,
// extreme values, batch parity, and the memory math from ADR-007.

import XCTest
@testable import ProximaKit

final class ScalarQuantizerTests: XCTestCase {

    // ── Seeded RNG (deterministic vector generation) ─────────────────


    private func randomVector(dimension: Int, range: ClosedRange<Float>, rng: inout SeededRandom) -> Vector {
        Vector((0..<dimension).map { _ in Float.random(in: range, using: &rng) })
    }

    // ── Round-Trip Error Bound ───────────────────────────────────────

    /// The ADR-007 contract: |x - dequant(quant(x))| <= scale/2 per component
    /// (round-to-nearest with step size `scale`).
    func testRoundTripErrorBoundedByHalfScale() {
        let dim = 384
        let sq = ScalarQuantizer(dimension: dim)
        var rng = SeededRandom(seed: 0xC0FFEE)

        for _ in 0..<20 {
            let vector = randomVector(dimension: dim, range: -10...10, rng: &rng)
            let (codes, scale) = sq.encode(vector)
            let decoded = sq.decode(codes, scale: scale)

            XCTAssertGreaterThan(scale, 0)
            XCTAssertEqual(codes.count, dim)
            // Tolerance: scale/2 plus a sliver for the Float multiply in decode.
            let bound = scale / 2 + scale * 1e-3
            for d in 0..<dim {
                XCTAssertLessThanOrEqual(
                    abs(vector[d] - decoded[d]), bound,
                    "component \(d): |\(vector[d]) - \(decoded[d])| exceeds scale/2 = \(scale / 2)")
            }
        }
    }

    /// The largest-magnitude component must map to code ±127 and reconstruct
    /// (essentially) exactly, with its sign preserved.
    func testMaxMagnitudeComponentReconstructsExactly() {
        let sq = ScalarQuantizer(dimension: 4)
        let (codes, scale) = sq.encode([0.5, -3.0, 1.5, 2.0])

        XCTAssertEqual(scale, 3.0 / 127, accuracy: 1e-9)
        XCTAssertEqual(codes[1], -127, "largest-magnitude component must hit the code extreme")

        let decoded = sq.decode(codes, scale: scale)
        XCTAssertEqual(decoded[1], -3.0, accuracy: 3.0 * 1e-6)
        // Signs are preserved everywhere.
        XCTAssertGreaterThan(decoded[0], 0)
        XCTAssertGreaterThan(decoded[2], 0)
        XCTAssertGreaterThan(decoded[3], 0)
    }

    // ── Zero Vector ──────────────────────────────────────────────────

    /// Zero vector → scale 0, all-zero codes, exact zero reconstruction.
    /// No division by zero may occur.
    func testZeroVectorEncodesWithZeroScale() {
        let dim = 32
        let sq = ScalarQuantizer(dimension: dim)
        let (codes, scale) = sq.encode([Float](repeating: 0, count: dim))

        XCTAssertEqual(scale, 0)
        XCTAssertTrue(codes.allSatisfy { $0 == 0 })

        let decoded = sq.decode(codes, scale: scale)
        XCTAssertTrue(decoded.allSatisfy { $0 == 0 }, "zero vector must round-trip exactly")
    }

    // ── Extreme Values ───────────────────────────────────────────────

    /// Components at the Float32 extremes must not overflow the codec.
    func testExtremeMagnitudesStayWithinErrorBound() {
        let sq = ScalarQuantizer(dimension: 6)
        let vector: [Float] = [
            Float.greatestFiniteMagnitude,
            -Float.greatestFiniteMagnitude,
            1e30, -1e30, 1.0, 0,
        ]
        let (codes, scale) = sq.encode(vector)
        let decoded = sq.decode(codes, scale: scale)

        XCTAssertEqual(codes[0], 127)
        XCTAssertEqual(codes[1], -127)
        XCTAssertTrue(scale.isFinite)
        let bound = scale / 2 + scale * 1e-3
        for d in 0..<vector.count {
            XCTAssertTrue(decoded[d].isFinite)
            XCTAssertLessThanOrEqual(abs(vector[d] - decoded[d]), bound, "component \(d)")
        }
    }

    /// Tiny magnitudes: the per-vector scale adapts, so relative accuracy holds
    /// even when every component is far below 1.
    func testTinyMagnitudesQuantizeWithAdaptiveScale() {
        let sq = ScalarQuantizer(dimension: 3)
        let vector: [Float] = [1e-30, -5e-31, 2.5e-31]
        let (codes, scale) = sq.encode(vector)
        let decoded = sq.decode(codes, scale: scale)

        XCTAssertEqual(codes[0], 127)
        let bound = scale / 2 + scale * 1e-3
        for d in 0..<vector.count {
            XCTAssertLessThanOrEqual(abs(vector[d] - decoded[d]), bound, "component \(d)")
        }
    }

    // ── Batch Variants ───────────────────────────────────────────────

    /// Batch encode/decode must agree exactly with the per-vector variants.
    func testBatchMatchesSingleVectorEncoding() {
        let dim = 64
        let sq = ScalarQuantizer(dimension: dim)
        var rng = SeededRandom(seed: 0xBADD_CAFE)
        var vectors = (0..<10).map { _ in randomVector(dimension: dim, range: -2...2, rng: &rng) }
        vectors.append(Vector(dimension: dim))  // include a zero vector

        let (batchCodes, batchScales) = sq.encodeBatch(vectors)
        XCTAssertEqual(batchCodes.count, vectors.count)
        XCTAssertEqual(batchScales.count, vectors.count)

        for (i, vector) in vectors.enumerated() {
            let (codes, scale) = sq.encode(vector)
            XCTAssertEqual(batchCodes[i], codes, "row \(i): batch codes must match single encode")
            XCTAssertEqual(batchScales[i], scale, "row \(i): batch scale must match single encode")
        }

        let decoded = sq.decodeBatch(codes: batchCodes, scales: batchScales)
        for (i, vector) in vectors.enumerated() {
            let single = sq.decodeToVector(batchCodes[i], scale: batchScales[i])
            XCTAssertEqual(decoded[i], single, "row \(i): batch decode must match single decode")
            XCTAssertEqual(decoded[i].dimension, vector.dimension)
        }
    }

    // ── Memory Math ──────────────────────────────────────────────────

    /// ADR-007 memory math: d + 4 bytes vs 4d bytes ≈ 3.96× at d = 384.
    func testCompressionRatioMatchesMemoryMath() {
        let sq = ScalarQuantizer(dimension: 384)
        XCTAssertEqual(sq.bytesPerEncodedVector, 388)       // 384 codes + 4-byte scale
        XCTAssertEqual(sq.bytesPerOriginalVector, 1536)     // 384 * 4
        XCTAssertEqual(sq.compressionRatio, 1536.0 / 388.0, accuracy: 1e-4)
        XCTAssertGreaterThan(sq.compressionRatio, 3.9, "≈4x memory claim (ADR-007)")
    }

    /// Subnormal inputs: maxAbs/127 underflows to exactly 0 for
    /// 0 < maxAbs < ~1.6e-43. The encoder must fall back to the zero-vector
    /// encoding (scale 0, all-zero codes) rather than divide by zero and emit
    /// garbage codes. CHA-201 wave-2 judge finding.
    func testSubnormalMagnitudesEncodeAsZeroVector() {
        let sq = ScalarQuantizer(dimension: 4)
        let subnormal: [Float] = [1e-44, -1e-44, 0, 5e-45]
        let (codes, scale) = sq.encode(subnormal)

        XCTAssertEqual(scale, 0, "underflowed scale must collapse to the zero encoding")
        XCTAssertTrue(codes.allSatisfy { $0 == 0 },
                      "scale 0 must never pair with nonzero codes")
        let decoded = sq.decode(codes, scale: scale)
        XCTAssertTrue(decoded.allSatisfy { $0 == 0 && $0.isFinite })
    }

    /// Non-finite inputs: a ±infinite component has no representable scale, so
    /// `maxAbs/127` stays +inf. The encoder must fall back to the zero-vector
    /// encoding immediately rather than step `scale.nextDown` tens of millions
    /// of times toward a finite reconstruction — the overflow-guard loop's
    /// "at most two steps" bound only holds for finite maxAbs. CHA-201
    /// mission-3 judge finding.
    func testInfiniteMagnitudeEncodesAsZeroVector() {
        let sq = ScalarQuantizer(dimension: 4)
        let (codes, scale) = sq.encode([.infinity, 0, -1.0, 2.0])

        XCTAssertEqual(scale, 0, "an infinite component must collapse to the zero encoding")
        XCTAssertTrue(codes.allSatisfy { $0 == 0 },
                      "scale 0 must never pair with nonzero codes")
        let decoded = sq.decode(codes, scale: scale)
        XCTAssertTrue(decoded.allSatisfy { $0 == 0 && $0.isFinite },
                      "zero encoding must decode to a finite zero vector")
    }

}
