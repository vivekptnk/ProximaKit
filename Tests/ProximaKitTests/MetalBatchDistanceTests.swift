// MetalBatchDistanceTests.swift
// ProximaKitTests
//
// Parity tests for the v1 Metal batch distance backend (ADR-009).
//
// GPU tests are gated with XCTSkipIf(MetalBatchDistance() == nil, ...)
// so CI runners without a Metal device skip cleanly. Pipeline
// compilation failure is NOT a skip: the MSL source is a fixed string,
// so a compile error is our bug and must fail the test.
//
// All parity assertions compare against the existing vDSP batch path
// (BatchDistance.swift) on SeededRandom data, and assert that the GPU
// path actually ran — otherwise a fallback would silently compare vDSP
// against vDSP and prove nothing.
//
// Run with: `swift test --filter MetalBatchDistanceTests`

import XCTest
@testable import ProximaKit

final class MetalBatchDistanceTests: XCTestCase {

    // ── Helpers ──────────────────────────────────────────────────────

    /// Returns a ready-to-dispatch instance, or skips when no Metal
    /// device exists (CI runners, simulators). Compilation failures
    /// throw (test fails) rather than skip — see file header.
    private func requireMetal() throws -> MetalBatchDistance {
        let metal = MetalBatchDistance()
        try XCTSkipIf(
            metal == nil,
            "No Metal device on this machine (e.g. CI runner) — skipping GPU parity tests"
        )
        try metal!.compilePipelines()
        return metal!
    }

    /// Generates a deterministic flat row-major matrix.
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

    /// Generates a deterministic query vector.
    private func seededQuery(dimension: Int, rng: inout SeededRandom) -> Vector {
        Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    /// Asserts element-wise parity to a relative tolerance (1e-4 per
    /// ADR-009), with a small absolute floor: relative tolerance is
    /// meaningless when the reference is ~0 (e.g. squared L2 of
    /// near-identical vectors).
    private func assertParity(
        _ actual: [Float],
        _ expected: [Float],
        relativeTolerance: Float = 1e-4,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, "count mismatch",
                       file: file, line: line)
        for i in 0..<min(actual.count, expected.count) {
            let tolerance = max(
                relativeTolerance * max(abs(expected[i]), abs(actual[i])),
                1e-5
            )
            XCTAssertEqual(
                actual[i], expected[i], accuracy: tolerance,
                "mismatch at index \(i)", file: file, line: line
            )
        }
    }

    // ── Availability (runs everywhere, no skip) ──────────────────────

    /// The availability probe must be stable within a process: both nil
    /// (no device / non-Metal platform) or both non-nil. This test runs
    /// unconditionally, so the suite exercises the no-GPU code path —
    /// and proves the skip gating compiles — on every machine.
    func testAvailabilityProbeIsDeterministic() {
        let first = MetalBatchDistance() != nil
        let second = MetalBatchDistance() != nil
        XCTAssertEqual(first, second,
                       "Metal availability must not flap between probes")
    }

    /// The forced-CPU path must agree with the vDSP reference exactly
    /// (it IS the vDSP path). Skips on machines without a Metal device —
    /// constructing MetalBatchDistance requires one even to exercise its
    /// forced-CPU mode; no-GPU platforms take the public-API fallback
    /// covered by testAvailabilityProbeIsDeterministic.
    func testForcedCPUFallbackMatchesVDSPReference() throws {
        guard let metal = MetalBatchDistance() else {
            throw XCTSkip("No Metal device — fallback wiring is exercised by testAvailabilityProbeIsDeterministic's platform")
        }
        var rng = SeededRandom(seed: 0xFA11_BACC)
        let dimension = 96
        let vectorCount = 257
        let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
        let query = seededQuery(dimension: dimension, rng: &rng)

        let l2 = metal.batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: false
        )
        XCTAssertEqual(l2.path, .cpuFallback)
        let expectedL2 = batchL2Distances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension
        ).map { $0 * $0 }
        assertParity(l2.distances, expectedL2)

        let cos = metal.batchCosineDistances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: false
        )
        XCTAssertEqual(cos.path, .cpuFallback)
        let expectedCos = batchDistances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension,
            metric: CosineDistance()
        )
        assertParity(cos.distances, expectedCos)
    }

    // ── Squared-L2 Parity ────────────────────────────────────────────

    /// GPU squared L2 matches (vDSP Euclidean)² to 1e-4 relative
    /// tolerance across dimensions, including non-multiples of 4.
    func testSquaredL2ParityAcrossDimensions() throws {
        let metal = try requireMetal()
        for dimension in [3, 7, 33, 128, 257, 384] {
            var rng = SeededRandom(seed: 0x0102_0000 &+ UInt64(dimension))
            let vectorCount = 300
            let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
            let query = seededQuery(dimension: dimension, rng: &rng)

            let result = metal.batchSquaredL2(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension, allowGPU: true
            )
            XCTAssertEqual(result.path, .gpu,
                           "GPU path must run at dimension \(dimension), or this test compares vDSP to vDSP")

            let expected = batchL2Distances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ).map { $0 * $0 }
            assertParity(result.distances, expected)
        }
    }

    /// Vector counts around the 256-thread threadgroup boundary exercise
    /// the kernel's gid bounds check (padded dispatch).
    func testSquaredL2ParityAcrossThreadgroupBoundaries() throws {
        let metal = try requireMetal()
        let dimension = 64
        for vectorCount in [1, 2, 255, 256, 257, 1000] {
            var rng = SeededRandom(seed: 0xB0DD_5EED &+ UInt64(vectorCount))
            let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
            let query = seededQuery(dimension: dimension, rng: &rng)

            let result = metal.batchSquaredL2(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension, allowGPU: true
            )
            XCTAssertEqual(result.path, .gpu, "vectorCount \(vectorCount)")

            let expected = batchL2Distances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ).map { $0 * $0 }
            assertParity(result.distances, expected)
        }
    }

    /// Rows bit-identical to the query must come back (near-)zero: the
    /// kernel subtracts directly, so no cancellation artifacts.
    func testSquaredL2IdenticalVectorsAreZero() throws {
        let metal = try requireMetal()
        var rng = SeededRandom(seed: 0x5E1F_5E1F)
        let dimension = 33
        let query = seededQuery(dimension: dimension, rng: &rng)

        // Rows 0 and 2 are copies of the query; row 1 is offset by +1 in
        // every component, so its squared L2 is exactly `dimension`.
        var matrix = [Float]()
        matrix.append(contentsOf: query.components)
        matrix.append(contentsOf: query.components.map { $0 + 1.0 })
        matrix.append(contentsOf: query.components)

        let result = metal.batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: 3, dimension: dimension, allowGPU: true
        )
        XCTAssertEqual(result.path, .gpu)
        XCTAssertEqual(result.distances[0], 0, accuracy: 1e-6)
        XCTAssertEqual(result.distances[1], Float(dimension), accuracy: 1e-3)
        XCTAssertEqual(result.distances[2], 0, accuracy: 1e-6)
    }

    // ── Cosine Parity ────────────────────────────────────────────────

    /// GPU cosine distance matches the vDSP batch path to 1e-4 relative
    /// tolerance across dimensions, including non-multiples of 4.
    func testCosineParityAcrossDimensions() throws {
        let metal = try requireMetal()
        for dimension in [3, 7, 33, 128, 257, 384] {
            var rng = SeededRandom(seed: 0xC051_0000 &+ UInt64(dimension))
            let vectorCount = 300
            let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
            let query = seededQuery(dimension: dimension, rng: &rng)

            let result = metal.batchCosineDistances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension, allowGPU: true
            )
            XCTAssertEqual(result.path, .gpu,
                           "GPU path must run at dimension \(dimension)")

            let expected = batchDistances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension,
                metric: CosineDistance()
            )
            assertParity(result.distances, expected)
        }
    }

    /// Zero-vector semantics must match the vDSP path (BatchDistance.swift):
    /// a zero query yields the neutral distance 1.0 for every row, never
    /// a perfect match. The kernel handles this through its denominator
    /// guard — no CPU-side special case — so this must run on the GPU.
    func testCosineZeroQueryYieldsNeutralDistanceOnGPU() throws {
        let metal = try requireMetal()
        var rng = SeededRandom(seed: 0x0000_C051)
        let dimension = 16
        let vectorCount = 10
        let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
        let zeroQuery = Vector(dimension: dimension)

        let result = metal.batchCosineDistances(
            query: zeroQuery, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: true
        )
        XCTAssertEqual(result.path, .gpu)
        for (i, d) in result.distances.enumerated() {
            XCTAssertEqual(d, 1.0, accuracy: 1e-6, "row \(i)")
        }

        let expected = batchDistances(
            query: zeroQuery, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension,
            metric: CosineDistance()
        )
        assertParity(result.distances, expected)
    }

    /// A zero ROW (zero database vector) must also yield 1.0, matching
    /// the vDSP path's per-row denominator guard.
    func testCosineZeroRowYieldsNeutralDistanceOnGPU() throws {
        let metal = try requireMetal()
        var rng = SeededRandom(seed: 0xDEAD_0051)
        let dimension = 24
        let query = seededQuery(dimension: dimension, rng: &rng)

        // Row 1 of 3 is all zeros.
        var matrix = [Float]()
        matrix.append(contentsOf: seededQuery(dimension: dimension, rng: &rng).components)
        matrix.append(contentsOf: [Float](repeating: 0, count: dimension))
        matrix.append(contentsOf: seededQuery(dimension: dimension, rng: &rng).components)

        let result = metal.batchCosineDistances(
            query: query, matrix: matrix,
            vectorCount: 3, dimension: dimension, allowGPU: true
        )
        XCTAssertEqual(result.path, .gpu)
        XCTAssertEqual(result.distances[1], 1.0, accuracy: 1e-6,
                       "zero row must be neutral (1.0), not a perfect match")

        let expected = batchDistances(
            query: query, matrix: matrix,
            vectorCount: 3, dimension: dimension,
            metric: CosineDistance()
        )
        assertParity(result.distances, expected)
    }

    // ── Edge Cases ───────────────────────────────────────────────────

    /// An empty matrix returns an empty result from both methods, like
    /// the vDSP batch path.
    func testEmptyMatrixReturnsEmpty() throws {
        let metal = try requireMetal()
        let query = Vector([1, 2, 3])
        XCTAssertEqual(
            metal.batchSquaredL2(query: query, matrix: [], vectorCount: 0, dimension: 3),
            []
        )
        XCTAssertEqual(
            metal.batchCosineDistances(query: query, matrix: [], vectorCount: 0, dimension: 3),
            []
        )
    }

    // ── Timing (prints only — never asserts on time, per ADR-009) ────

    /// Local-only timing comparison: GPU vs vDSP at a build-phase batch
    /// size. Timing is hardware-dependent, so this PRINTS and asserts
    /// only correctness (parity), never speed.
    func testTimingComparisonPrintsButDoesNotAssert() throws {
        let metal = try requireMetal()
        var rng = SeededRandom(seed: 0x71D1_0000)
        let dimension = 256
        let vectorCount = 16_384
        let matrix = seededMatrix(vectorCount: vectorCount, dimension: dimension, rng: &rng)
        let query = seededQuery(dimension: dimension, rng: &rng)

        // Warm up the pipeline cache so the GPU timing below measures
        // dispatch, not the one-time MSL compile.
        _ = metal.batchSquaredL2(
            query: query, matrix: matrix, vectorCount: vectorCount, dimension: dimension
        )

        let gpuStart = Date()
        let gpu = metal.batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: true
        )
        let gpuSeconds = Date().timeIntervalSince(gpuStart)

        let cpuStart = Date()
        let cpu = batchL2Distances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension
        ).map { $0 * $0 }
        let cpuSeconds = Date().timeIntervalSince(cpuStart)

        XCTAssertEqual(gpu.path, .gpu)
        assertParity(gpu.distances, cpu)

        print("""
        [MetalBatchDistanceTests] timing (local only, NOT asserted): \
        N=\(vectorCount) dim=\(dimension) — \
        GPU \(String(format: "%.3f", gpuSeconds * 1000)) ms, \
        vDSP \(String(format: "%.3f", cpuSeconds * 1000)) ms
        """)
    }
}
