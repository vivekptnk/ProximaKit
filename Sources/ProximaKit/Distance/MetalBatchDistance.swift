// MetalBatchDistance.swift
// ProximaKit
//
// GPU batch distance computation for large index BUILDS (ADR-009).
// Computes one-query-to-N-vectors squared-L2 / cosine distances with a
// Metal compute kernel — one thread per database vector — over the same
// flat row-major matrix layout as `batchDistances` (BatchDistance.swift).
//
// Scope (v1, per ADR-009): a standalone utility. NOT integrated into the
// HNSWIndex insert loop yet, and NOT for per-query search latency — at
// search-time candidate counts, kernel launch overhead dominates and the
// vDSP path wins (ADR-001).
//
// The MSL source is an inline string compiled at first use via
// `device.makeLibrary(source:options:)` and cached: `.metal` files in a
// Swift package require Package.swift resource handling and compile
// inconsistently between the Xcode build system and the `swift build`
// CLI, so runtime compilation is the portable route (ADR-009).
//
// Every failure mode (no device, compile failure, allocation failure,
// command-buffer error) degrades to the existing vDSP batch path, so
// results are always correct and the public methods never throw.

import Accelerate
import Foundation

// ── Execution Path (shared across Metal / non-Metal builds) ──────────

/// Which backend actually produced a batch result. Internal: parity tests
/// assert `.gpu` so they never silently compare vDSP against vDSP.
enum MetalBatchDistanceExecutionPath: Equatable {
    /// The Metal compute kernel ran.
    case gpu
    /// The vDSP fallback ran (no pipelines, a Metal runtime failure,
    /// an empty input, or GPU execution explicitly disallowed).
    case cpuFallback
}

/// Errors raised while building the compute pipelines. Internal: the
/// public API falls back to vDSP instead of throwing; tests use these to
/// fail loudly (not skip) when the fixed MSL source is broken.
enum MetalBatchDistanceError: Error, Equatable {
    /// `makeLibrary(source:options:)` rejected the MSL source.
    case shaderCompilationFailed(String)
    /// The compiled library is missing an expected kernel function.
    case kernelFunctionMissing(String)
    /// `makeComputePipelineState(function:)` failed.
    case pipelineCreationFailed(String)
}

// ── vDSP Fallback (shared across Metal / non-Metal builds) ───────────

/// Squared-L2 fallback: the existing Euclidean batch path, squared.
/// Squaring the (already clamped, sqrt'd) vDSP result keeps the fallback
/// semantics bit-for-bit aligned with what callers of `batchL2Distances`
/// would compute themselves.
private func fallbackSquaredL2(
    query: Vector, matrix: [Float], vectorCount: Int, dimension: Int
) -> [Float] {
    let l2 = batchL2Distances(
        query: query, matrix: matrix,
        vectorCount: vectorCount, dimension: dimension
    )
    var squared = [Float](repeating: 0, count: vectorCount)
    l2.withUnsafeBufferPointer { ptr in
        vDSP_vsq(ptr.baseAddress!, 1, &squared, 1, vDSP_Length(vectorCount))
    }
    return squared
}

/// Cosine fallback: the existing vDSP batch path, including its
/// zero-vector neutral-distance (1.0) semantics.
private func fallbackCosineDistances(
    query: Vector, matrix: [Float], vectorCount: Int, dimension: Int
) -> [Float] {
    batchDistances(
        query: query, matrix: matrix,
        vectorCount: vectorCount, dimension: dimension,
        metric: CosineDistance()
    )
}

#if canImport(Metal)

import Metal

// ── MSL Kernel Source ────────────────────────────────────────────────
// One thread per database vector; each thread loops over the dimension
// with fma accumulation. `gid` is bounds-checked so the dispatch can pad
// to whole threadgroups without requiring non-uniform threadgroup
// support. Row offsets are computed in ulong so the addressable matrix
// size is limited by maxBufferLength, not 32-bit index arithmetic.
//
// Squared L2 subtracts directly (no |a|²+|b|²−2·dot identity), so there
// is no catastrophic cancellation for near-identical vectors.
//
// Cosine matches the vDSP path's zero-vector semantics: a zero
// denominator yields the neutral distance 1.0, never a perfect match.
private let distanceKernelsSource = """
#include <metal_stdlib>
using namespace metal;

struct BatchDistanceParams {
    uint  vectorCount;
    uint  dimension;
    float queryMagnitude;
};

kernel void batch_squared_l2(
    device const float        *query     [[buffer(0)]],
    device const float        *matrix    [[buffer(1)]],
    device float              *distances [[buffer(2)]],
    constant BatchDistanceParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.vectorCount) { return; }
    device const float *row = matrix + (ulong)gid * (ulong)params.dimension;
    float sum = 0.0f;
    for (uint j = 0; j < params.dimension; ++j) {
        float d = query[j] - row[j];
        sum = fma(d, d, sum);
    }
    distances[gid] = sum;
}

kernel void batch_cosine_distance(
    device const float        *query     [[buffer(0)]],
    device const float        *matrix    [[buffer(1)]],
    device float              *distances [[buffer(2)]],
    constant BatchDistanceParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.vectorCount) { return; }
    device const float *row = matrix + (ulong)gid * (ulong)params.dimension;
    float dot = 0.0f;
    float normSq = 0.0f;
    for (uint j = 0; j < params.dimension; ++j) {
        float v = row[j];
        dot = fma(query[j], v, dot);
        normSq = fma(v, v, normSq);
    }
    float denominator = params.queryMagnitude * sqrt(normSq);
    distances[gid] = denominator > 0.0f ? 1.0f - dot / denominator : 1.0f;
}
"""

/// GPU batch distance computation for large index builds (ADR-009).
///
/// Computes one-query-to-N-vectors distances with a Metal compute kernel,
/// over the same flat row-major matrix layout as ``batchDistances(query:matrix:vectorCount:dimension:metric:)``.
/// Intended for build-phase workloads where N is large enough to amortize
/// kernel launch overhead; for search-time candidate counts the vDSP path
/// remains the right tool (ADR-001).
///
/// ## Availability
///
/// `init?` returns `nil` when no Metal device exists (CI runners, some
/// simulators, older hardware). Any runtime Metal failure falls back
/// automatically to the existing vDSP batch path, so results are always
/// correct and these methods never throw.
///
/// ```swift
/// guard let metal = MetalBatchDistance() else {
///     // No GPU — use batchDistances(...) directly.
/// }
/// let d2 = metal.batchSquaredL2(query: query, matrix: flat,
///                               vectorCount: 100_000, dimension: 384)
/// // d2[i] = |query − vectors[i]|²  (squared: ranking-equivalent, no sqrt)
/// ```
///
/// ## Semantics
///
/// - ``batchSquaredL2(query:matrix:vectorCount:dimension:)`` returns
///   **squared** Euclidean distances. Ranking is invariant under `sqrt`,
///   which is all the build phase needs.
/// - ``batchCosineDistances(query:matrix:vectorCount:dimension:)`` returns
///   `1 − dot/(|q|·|v|)` with the same zero-vector neutral-distance
///   semantics (`1.0`) as the vDSP path.
public final class MetalBatchDistance: @unchecked Sendable {

    // `@unchecked Sendable` justification (ADR-009): MTLDevice,
    // MTLCommandQueue, and MTLComputePipelineState are documented
    // thread-safe by Metal; each dispatch uses its own command buffer
    // and encoder. The only mutable state is the lazily compiled
    // pipeline cache below, which is protected by `pipelineLock`.

    // ── State ────────────────────────────────────────────────────────

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    /// The two compiled compute pipelines, built together on first use.
    private struct Pipelines {
        let squaredL2: MTLComputePipelineState
        let cosine: MTLComputePipelineState
    }

    private let pipelineLock = NSLock()
    private var cachedPipelines: Pipelines?
    /// Set after a compile failure so every subsequent batch call does
    /// not retry compilation (the source is fixed; retrying cannot help).
    private var compileFailed = false

    /// Matches the MSL `BatchDistanceParams` layout: three 4-byte,
    /// 4-byte-aligned fields, passed via `setBytes`.
    private struct KernelParams {
        var vectorCount: UInt32
        var dimension: UInt32
        var queryMagnitude: Float
    }

    // ── Initialization ───────────────────────────────────────────────

    /// Creates a GPU batch distance computer, or returns `nil` when
    /// Metal is unavailable (no device, or no command queue).
    ///
    /// Initialization is cheap: the MSL source is compiled lazily on the
    /// first batch call and cached, so availability probes (e.g.
    /// `XCTSkipIf(MetalBatchDistance() == nil, ...)`) pay no shader
    /// compilation cost.
    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { return nil }
        self.device = device
        self.commandQueue = queue
    }

    // ── Pipeline Compilation (lazy, cached, lock-protected) ──────────

    /// Compiles both kernels from the inline MSL source and caches the
    /// pipelines. Internal so tests can fail loudly — not skip — when
    /// the fixed source no longer compiles.
    ///
    /// - Throws: ``MetalBatchDistanceError`` on compile/pipeline failure.
    func compilePipelines() throws {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        _ = try compilePipelinesLocked()
    }

    /// Returns the cached pipelines, compiling on first use. Returns
    /// `nil` (caller falls back to vDSP) after a recorded failure.
    private func pipelines() -> Pipelines? {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        if let cached = cachedPipelines { return cached }
        if compileFailed { return nil }
        do {
            return try compilePipelinesLocked()
        } catch {
            compileFailed = true
            return nil
        }
    }

    /// Must be called with `pipelineLock` held.
    private func compilePipelinesLocked() throws -> Pipelines {
        if let cached = cachedPipelines { return cached }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(
                source: distanceKernelsSource, options: nil
            )
        } catch {
            throw MetalBatchDistanceError.shaderCompilationFailed(
                String(describing: error)
            )
        }

        func pipeline(for name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw MetalBatchDistanceError.kernelFunctionMissing(name)
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw MetalBatchDistanceError.pipelineCreationFailed(
                    "\(name): \(String(describing: error))"
                )
            }
        }

        let compiled = Pipelines(
            squaredL2: try pipeline(for: "batch_squared_l2"),
            cosine: try pipeline(for: "batch_cosine_distance")
        )
        cachedPipelines = compiled
        return compiled
    }

    // ── Public API ───────────────────────────────────────────────────

    /// Computes **squared** L2 distances from a query to all vectors in
    /// a flat row-major matrix: `result[i] = |query − vec_i|²`.
    ///
    /// Squared (not Euclidean) because build-phase consumers only rank
    /// candidates, and ranking is invariant under `sqrt` (ADR-009).
    /// Falls back to the vDSP path (squared ``batchL2Distances``) on any
    /// Metal failure.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - matrix: A flat row-major matrix of `vectorCount × dimension` floats.
    ///   - vectorCount: Number of vectors in the matrix.
    ///   - dimension: Dimension of each vector.
    /// - Returns: An array of `vectorCount` squared Euclidean distances.
    public func batchSquaredL2(
        query: Vector,
        matrix: [Float],
        vectorCount: Int,
        dimension: Int
    ) -> [Float] {
        batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension,
            allowGPU: true
        ).distances
    }

    /// Computes cosine distances (`1 − dot/(|q|·|v|)`) from a query to
    /// all vectors in a flat row-major matrix.
    ///
    /// Zero-vector semantics match the vDSP path: a zero query or a zero
    /// row yields the neutral distance `1.0`, never a perfect match.
    /// Falls back to the vDSP path on any Metal failure.
    ///
    /// - Parameters:
    ///   - query: The query vector.
    ///   - matrix: A flat row-major matrix of `vectorCount × dimension` floats.
    ///   - vectorCount: Number of vectors in the matrix.
    ///   - dimension: Dimension of each vector.
    /// - Returns: An array of `vectorCount` cosine distances.
    public func batchCosineDistances(
        query: Vector,
        matrix: [Float],
        vectorCount: Int,
        dimension: Int
    ) -> [Float] {
        batchCosineDistances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension,
            allowGPU: true
        ).distances
    }

    // ── Internal API (execution-path observable, for tests) ──────────

    /// Squared-L2 with an observable execution path. `allowGPU: false`
    /// exercises the fallback wiring deterministically in tests.
    func batchSquaredL2(
        query: Vector,
        matrix: [Float],
        vectorCount: Int,
        dimension: Int,
        allowGPU: Bool
    ) -> (distances: [Float], path: MetalBatchDistanceExecutionPath) {
        guard vectorCount > 0 else { return ([], .cpuFallback) }
        precondition(query.dimension == dimension, "Query dimension mismatch")
        precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")

        if allowGPU, let pipelines = pipelines(),
           let result = runKernel(
               pipelines.squaredL2,
               query: query, matrix: matrix,
               vectorCount: vectorCount, dimension: dimension,
               queryMagnitude: 0 // unused by batch_squared_l2
           ) {
            return (result, .gpu)
        }

        return (
            fallbackSquaredL2(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ),
            .cpuFallback
        )
    }

    /// Cosine with an observable execution path.
    func batchCosineDistances(
        query: Vector,
        matrix: [Float],
        vectorCount: Int,
        dimension: Int,
        allowGPU: Bool
    ) -> (distances: [Float], path: MetalBatchDistanceExecutionPath) {
        guard vectorCount > 0 else { return ([], .cpuFallback) }
        precondition(query.dimension == dimension, "Query dimension mismatch")
        precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")

        if allowGPU, let pipelines = pipelines(),
           let result = runKernel(
               pipelines.cosine,
               query: query, matrix: matrix,
               vectorCount: vectorCount, dimension: dimension,
               queryMagnitude: query.magnitude
           ) {
            return (result, .gpu)
        }

        return (
            fallbackCosineDistances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ),
            .cpuFallback
        )
    }

    // ── Kernel Dispatch ──────────────────────────────────────────────

    /// Runs one kernel over the matrix; returns `nil` on any Metal
    /// failure so the caller can fall back to vDSP.
    private func runKernel(
        _ pipeline: MTLComputePipelineState,
        query: Vector,
        matrix: [Float],
        vectorCount: Int,
        dimension: Int,
        queryMagnitude: Float
    ) -> [Float]? {
        let matrixLength = matrix.count * MemoryLayout<Float>.stride
        guard matrixLength <= device.maxBufferLength else { return nil }

        // makeBuffer(bytes:) copies; zero-copy persistent buffers are a
        // follow-up for the insert-loop integration (ADR-009).
        guard
            let queryBuffer = query.components.withUnsafeBufferPointer({ ptr in
                device.makeBuffer(
                    bytes: ptr.baseAddress!,
                    length: dimension * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
            }),
            let matrixBuffer = matrix.withUnsafeBufferPointer({ ptr in
                device.makeBuffer(
                    bytes: ptr.baseAddress!,
                    length: matrixLength,
                    options: .storageModeShared
                )
            }),
            let outputBuffer = device.makeBuffer(
                length: vectorCount * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ),
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }

        var params = KernelParams(
            vectorCount: UInt32(vectorCount),
            dimension: UInt32(dimension),
            queryMagnitude: queryMagnitude
        )

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<KernelParams>.stride, index: 3)

        // Pad to whole threadgroups; the kernel bounds-checks `gid`, so
        // this works on every device without non-uniform threadgroup
        // support. 256 is a multiple of the thread execution width on
        // all Apple GPUs and well under every maxTotalThreadsPerThreadgroup.
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerGroup = MTLSize(width: width, height: 1, depth: 1)
        let groups = MTLSize(
            width: (vectorCount + width - 1) / width, height: 1, depth: 1
        )
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed, commandBuffer.error == nil else {
            return nil
        }

        let resultPointer = outputBuffer.contents()
            .bindMemory(to: Float.self, capacity: vectorCount)
        return Array(UnsafeBufferPointer(start: resultPointer, count: vectorCount))
    }
}

#else

// ── Non-Metal Platforms: API-Compatible Stub ─────────────────────────
// Keeps call sites and tests compiling unconditionally. `init?` always
// returns nil (tests gate on this with XCTSkipIf); the batch methods
// delegate to the vDSP fallback for API symmetry, though they are
// unreachable through a normal `init?` flow.

/// Stub for platforms without Metal. ``init()`` always returns `nil`;
/// see the Metal-backed documentation in this file for the real API.
public final class MetalBatchDistance: Sendable {

    /// Always `nil`: Metal cannot be imported on this platform.
    public init?() { return nil }

    /// Unreachable through `init?`; delegates to the vDSP path.
    public func batchSquaredL2(
        query: Vector, matrix: [Float], vectorCount: Int, dimension: Int
    ) -> [Float] {
        batchSquaredL2(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: true
        ).distances
    }

    /// Unreachable through `init?`; delegates to the vDSP path.
    public func batchCosineDistances(
        query: Vector, matrix: [Float], vectorCount: Int, dimension: Int
    ) -> [Float] {
        batchCosineDistances(
            query: query, matrix: matrix,
            vectorCount: vectorCount, dimension: dimension, allowGPU: true
        ).distances
    }

    /// No-op: there is no MSL to compile on this platform.
    func compilePipelines() throws {}

    func batchSquaredL2(
        query: Vector, matrix: [Float], vectorCount: Int, dimension: Int,
        allowGPU: Bool
    ) -> (distances: [Float], path: MetalBatchDistanceExecutionPath) {
        guard vectorCount > 0 else { return ([], .cpuFallback) }
        precondition(query.dimension == dimension, "Query dimension mismatch")
        precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")
        return (
            fallbackSquaredL2(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ),
            .cpuFallback
        )
    }

    func batchCosineDistances(
        query: Vector, matrix: [Float], vectorCount: Int, dimension: Int,
        allowGPU: Bool
    ) -> (distances: [Float], path: MetalBatchDistanceExecutionPath) {
        guard vectorCount > 0 else { return ([], .cpuFallback) }
        precondition(query.dimension == dimension, "Query dimension mismatch")
        precondition(matrix.count == vectorCount * dimension, "Matrix size mismatch")
        return (
            fallbackCosineDistances(
                query: query, matrix: matrix,
                vectorCount: vectorCount, dimension: dimension
            ),
            .cpuFallback
        )
    }
}

#endif
