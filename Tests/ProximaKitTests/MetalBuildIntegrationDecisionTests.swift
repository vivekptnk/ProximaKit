// MetalBuildIntegrationDecisionTests.swift
// ProximaKitTests
//
// Portable acceptance assertion backing the ADR-009 insert-loop
// integration NO-GO decision (docs/adr/ADR-009-metal-backend.md addendum).
//
// The DECISION rests on latency (GPU never beats vDSP at build shapes),
// but latency is hardware-dependent and MUST NOT be asserted in CI
// (ADR-009). What IS stable and machine-portable is *correctness at
// build scale*: the GPU kernel must still match the vDSP path at the
// largest realistic build slice (N = 100K), and must actually run on the
// GPU there — a silent CPU fallback would make the whole GO/NO-GO
// measurement compare vDSP against vDSP. That is what this file asserts.
//
// Gating (two independent skips, so this never burdens normal CI):
//   1. No Metal device (CI runners) → XCTSkip, like MetalBatchDistanceTests.
//   2. Not opted in (PROXIMA_GPU_BENCH != "1") → XCTSkip. The N=100K
//      matrices are ~150–300 MB and take a beat to allocate; this is a
//      benchmark-scale check, run on demand, not every `swift test`.
//
// Run with:
//   PROXIMA_GPU_BENCH=1 swift test --filter MetalBuildIntegrationDecisionTests

import XCTest
@testable import ProximaKit

final class MetalBuildIntegrationDecisionTests: XCTestCase {

    /// The largest N the ADR-009 sweep treats as a realistic build slice.
    /// Parity here is hardware-independent; only latency (asserted nowhere
    /// in CI) varies by machine.
    private static let buildScaleN = 102_400

    /// Skips unless a Metal device exists AND the opt-in env var is set.
    private func requireGPUBench() throws -> MetalBatchDistance {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["PROXIMA_GPU_BENCH"] == "1",
            "Set PROXIMA_GPU_BENCH=1 to run the build-scale (N=100K) GPU parity check"
        )
        let metal = MetalBatchDistance()
        try XCTSkipIf(
            metal == nil,
            "No Metal device on this machine (e.g. CI runner) — skipping build-scale GPU parity"
        )
        try metal!.compilePipelines()
        return metal!
    }

    private func seededMatrix(
        vectorCount: Int, dimension: Int, rng: inout SeededRandom
    ) -> [Float] {
        var matrix = [Float]()
        matrix.reserveCapacity(vectorCount * dimension)
        for _ in 0..<(vectorCount * dimension) {
            matrix.append(Float.random(in: -1...1, using: &rng))
        }
        return matrix
    }

    private func seededQuery(dimension: Int, rng: inout SeededRandom) -> Vector {
        Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    /// Relative-tolerance parity with a small absolute floor (identical to
    /// MetalBatchDistanceTests): relative tolerance is meaningless where the
    /// reference is ~0 (near-identical vectors → squared L2 ≈ 0).
    private func assertParity(
        _ actual: [Float], _ expected: [Float],
        relativeTolerance: Float, absoluteFloor: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "count mismatch", file: file, line: line)
        for i in 0..<min(actual.count, expected.count) {
            let tol = max(relativeTolerance * max(abs(expected[i]), abs(actual[i])), absoluteFloor)
            XCTAssertEqual(actual[i], expected[i], accuracy: tol,
                           "mismatch at index \(i)", file: file, line: line)
        }
    }

    // ── The portable acceptance assertion ────────────────────────────

    /// GPU squared-L2 matches (vDSP Euclidean)² at the build-scale N=100K
    /// slice, and the GPU path actually ran. This is the correctness half
    /// of the ADR-009 NO-GO: the utility is right at build scale; it simply
    /// has no latency win there (latency asserted nowhere — see the ADR).
    func testGPUSquaredL2ParityHoldsAtBuildScaleN100K() throws {
        let metal = try requireGPUBench()
        let dimension = 384
        var rng = SeededRandom(seed: 0x0009_100C)
        let matrix = seededMatrix(vectorCount: Self.buildScaleN, dimension: dimension, rng: &rng)
        let query = seededQuery(dimension: dimension, rng: &rng)

        let result = metal.batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: Self.buildScaleN, dimension: dimension, allowGPU: true
        )
        XCTAssertEqual(result.path, .gpu,
                       "GPU path must run at N=100K, or the NO-GO benchmark compared vDSP to vDSP")

        let expected = batchL2Distances(
            query: query, matrix: matrix,
            vectorCount: Self.buildScaleN, dimension: dimension
        ).map { $0 * $0 }
        // Squared L2 of unit-range vectors reaches ~1e3, so 1e-4 relative
        // tolerance leaves an absolute band ~1e-1 — matches the sweep's
        // observed max-abs-diff ≤ 1.0e-3 with comfortable margin.
        assertParity(result.distances, expected, relativeTolerance: 1e-4, absoluteFloor: 1e-2)
    }

    /// GPU cosine matches the vDSP batch path at N=100K, GPU path actually
    /// ran. Cosine is bounded in [0, 2], so a tight absolute floor applies.
    func testGPUCosineParityHoldsAtBuildScaleN100K() throws {
        let metal = try requireGPUBench()
        let dimension = 384
        var rng = SeededRandom(seed: 0x0009_C051)
        let matrix = seededMatrix(vectorCount: Self.buildScaleN, dimension: dimension, rng: &rng)
        let query = seededQuery(dimension: dimension, rng: &rng)

        let result = metal.batchCosineDistances(
            query: query, matrix: matrix,
            vectorCount: Self.buildScaleN, dimension: dimension, allowGPU: true
        )
        XCTAssertEqual(result.path, .gpu,
                       "GPU path must run at N=100K, or the NO-GO benchmark compared vDSP to vDSP")

        let expected = batchDistances(
            query: query, matrix: matrix,
            vectorCount: Self.buildScaleN, dimension: dimension,
            metric: CosineDistance()
        )
        assertParity(result.distances, expected, relativeTolerance: 1e-4, absoluteFloor: 1e-5)
    }
}
