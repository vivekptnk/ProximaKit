// DistanceKernelBench.swift
// ProximaBench
//
// Build-phase distance-kernel benchmark for the ADR-009 GO/NO-GO decision:
// GPU `MetalBatchDistance` vs the vDSP flat batch path (`BatchDistance.swift`)
// at a sweep of one-query-to-N shapes, cosine + euclidean, seeded data.
//
// It also characterizes the ACTUAL distance-computation shapes the HNSW
// insert loop produces (`insert-shape` mode), because the whole decision
// turns on whether the "one-query-to-N" batch ADR-009 assumed even exists
// in the real build loop.
//
// Two subcommands, wired in ProximaBenchCLI.swift:
//   distance-kernel  — GPU-vs-vDSP N×d sweep, emits a sweep JSON
//   insert-shape     — instruments HNSWIndex.add()'s distance evals
//
// Latency lives here (and in the emitted JSON) and is NEVER asserted in CI
// (ADR-009): timings are hardware-dependent. The portable parity/path
// assertion lives in Tests/ (MetalBuildIntegrationDecisionTests).

import Foundation
import ProximaKit

// ── Deterministic RNG ────────────────────────────────────────────────
// SplitMix64 — identical algorithm to the library's test `SeededRandom`
// and `SplitMix64` (both internal, so not visible to this standalone
// package). No system RNG anywhere in the harness.

struct BenchSeededRandom: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

// ── Result schema (distinct from the ANN BenchResult schema) ─────────
// The ANN v1 schema (JSON_SCHEMA.md) models an index run (recall@k,
// efSearch, build time). A distance-kernel crossover sweep is a different
// experiment, so it uses its own document shape — but reuses the same
// Platform block, seed, and reproducibility fields (house style), and is
// still one logical experiment per file.

struct DistanceKernelSweep: Codable {
    let schemaVersion: Int          // 1
    let kind: String                // "distance-kernel-sweep"
    let library: String             // "ProximaKit"
    let libraryVersion: String
    let metrics: [String]
    let metalDeviceAvailable: Bool  // MTLCreateSystemDefaultDevice() != nil
    let platform: Platform
    let seed: Int
    let reps: Int
    let warmupReps: Int
    let runStartedAt: String
    let runDurationSeconds: Double
    let notes: String
    let measurements: [DistanceKernelMeasurement]
}

struct DistanceKernelMeasurement: Codable {
    let metric: String              // "euclidean" | "cosine"
    let vectorCount: Int
    let dimension: Int
    // Whole-array wall-clock of the PUBLIC API, median + spread over reps.
    let gpuMedianMs: Double
    let gpuMinMs: Double
    let gpuMaxMs: Double
    let vdspMedianMs: Double
    let vdspMinMs: Double
    let vdspMaxMs: Double
    /// vdspMedian / gpuMedian. > 1 ⇒ GPU faster; < 1 ⇒ vDSP faster.
    let gpuSpeedupOverVDSP: Double
    /// Max abs diff GPU-vs-vDSP over the array (fallback would read ~0).
    let parityMaxAbsDiff: Double
}

struct InsertShapeSweep: Codable {
    let schemaVersion: Int          // 1
    let kind: String                // "hnsw-insert-shape"
    let library: String
    let libraryVersion: String
    let platform: Platform
    let seed: Int
    let runStartedAt: String
    let notes: String
    let rows: [InsertShapeRow]
}

struct InsertShapeRow: Codable {
    let metric: String
    let dimension: Int
    let vectorCount: Int
    let m: Int
    let mMax0: Int
    let efConstruction: Int
    let totalDistanceEvals: Int
    let meanEvalsPerInsert: Double
    /// Distance evals whose first arg is the vector being inserted — the
    /// `searchLayer` graph-traversal evals (query-to-one-candidate).
    let traversalEvals: Int
    /// The remaining evals: pairwise heuristic-selection / pruning evals.
    let pairwiseEvals: Int
    /// Structural ceiling on the largest possible one-query-to-N batch a
    /// single node expansion could dispatch: a layer-0 adjacency list is
    /// pruned to ≤ mMax0, so no batchable unit exceeds this.
    let maxOneQueryToNBatchCeiling: Int
}

// ── Data generation ──────────────────────────────────────────────────

private func seededMatrix(vectorCount: Int, dimension: Int, rng: inout BenchSeededRandom) -> [Float] {
    var matrix = [Float]()
    matrix.reserveCapacity(vectorCount * dimension)
    for _ in 0..<(vectorCount * dimension) {
        matrix.append(Float.random(in: -1...1, using: &rng))
    }
    return matrix
}

private func seededQuery(dimension: Int, rng: inout BenchSeededRandom) -> Vector {
    Vector((0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) })
}

private func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    let n = s.count
    return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
}

// ── distance-kernel subcommand ───────────────────────────────────────

enum DistanceKernelBench {

    struct Options {
        let dimensions: [Int]
        let counts: [Int]
        let metrics: [String]        // "euclidean" | "cosine"
        let reps: Int
        let warmup: Int
        let seed: UInt64
        let libraryVersion: String
        let notes: String
        let outputPath: String
    }

    static func run(_ opts: Options) throws {
        let started = Date()
        let clock = ContinuousClock()
        let runStart = clock.now

        let metal = MetalBatchDistance()
        let deviceAvailable = metal != nil

        var measurements: [DistanceKernelMeasurement] = []

        for metricName in opts.metrics {
            for dim in opts.dimensions {
                for count in opts.counts {
                    // Fresh seed per shape so each cell is independent yet
                    // fully reproducible from (seed, metric, dim, count).
                    var rng = BenchSeededRandom(
                        seed: opts.seed
                            &+ UInt64(dim) &* 0x1_0000
                            &+ UInt64(count)
                            &+ (metricName == "cosine" ? 0xC05 : 0x1_2)
                    )
                    let matrix = seededMatrix(vectorCount: count, dimension: dim, rng: &rng)
                    let query = seededQuery(dimension: dim, rng: &rng)

                    // vDSP reference (also the parity target).
                    let vdspRef = vdspBatch(
                        metric: metricName, query: query, matrix: matrix,
                        vectorCount: count, dimension: dim
                    )

                    // ── GPU timing ─────────────────────────────────────
                    var gpuMs: [Double] = []
                    var parityMaxAbs = Double.nan
                    if let metal = metal {
                        // Warmup: compiles pipelines + primes buffers.
                        for _ in 0..<opts.warmup {
                            _ = gpuBatch(metal, metric: metricName, query: query,
                                         matrix: matrix, vectorCount: count, dimension: dim)
                        }
                        for _ in 0..<opts.reps {
                            let t0 = clock.now
                            let out = gpuBatch(metal, metric: metricName, query: query,
                                               matrix: matrix, vectorCount: count, dimension: dim)
                            gpuMs.append((clock.now - t0).ms)
                            if parityMaxAbs.isNaN {
                                parityMaxAbs = maxAbsDiff(out, vdspRef)
                            }
                        }
                    }

                    // ── vDSP timing ────────────────────────────────────
                    var vdspMs: [Double] = []
                    for _ in 0..<opts.warmup {
                        _ = vdspBatch(metric: metricName, query: query, matrix: matrix,
                                      vectorCount: count, dimension: dim)
                    }
                    for _ in 0..<opts.reps {
                        let t0 = clock.now
                        _ = vdspBatch(metric: metricName, query: query, matrix: matrix,
                                      vectorCount: count, dimension: dim)
                        vdspMs.append((clock.now - t0).ms)
                    }

                    let gpuMed = median(gpuMs)
                    let vdspMed = median(vdspMs)
                    let speedup = (deviceAvailable && gpuMed > 0) ? vdspMed / gpuMed : 0
                    measurements.append(DistanceKernelMeasurement(
                        metric: metricName,
                        vectorCount: count,
                        dimension: dim,
                        gpuMedianMs: gpuMed,
                        gpuMinMs: gpuMs.min() ?? 0,
                        gpuMaxMs: gpuMs.max() ?? 0,
                        vdspMedianMs: vdspMed,
                        vdspMinMs: vdspMs.min() ?? 0,
                        vdspMaxMs: vdspMs.max() ?? 0,
                        gpuSpeedupOverVDSP: speedup,
                        parityMaxAbsDiff: parityMaxAbs.isNaN ? -1 : parityMaxAbs
                    ))

                    let tag = deviceAvailable
                        ? String(format: "GPU %.3fms  vDSP %.3fms  x%.2f", gpuMed, vdspMed, speedup)
                        : String(format: "vDSP %.3fms  (no Metal device)", vdspMed)
                    FileHandle.standardError.write(Data(
                        "[distance-kernel] \(metricName) d=\(dim) N=\(count)  \(tag)\n".utf8))
                }
            }
        }

        let sweep = DistanceKernelSweep(
            schemaVersion: 1,
            kind: "distance-kernel-sweep",
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            metrics: opts.metrics,
            metalDeviceAvailable: deviceAvailable,
            platform: PlatformProbe.current(),
            seed: Int(bitPattern: UInt(truncatingIfNeeded: opts.seed)),
            reps: opts.reps,
            warmupReps: opts.warmup,
            runStartedAt: ISO8601DateFormatter().string(from: started),
            runDurationSeconds: (clock.now - runStart).seconds,
            notes: opts.notes,
            measurements: measurements
        )

        try writeJSON(sweep, to: opts.outputPath)
        FileHandle.standardError.write(Data(
            "[distance-kernel] wrote \(opts.outputPath) (\(measurements.count) cells, deviceAvailable=\(deviceAvailable))\n".utf8))
    }

    // ── metric dispatch ──────────────────────────────────────────────

    private static func gpuBatch(
        _ metal: MetalBatchDistance, metric: String, query: Vector,
        matrix: [Float], vectorCount: Int, dimension: Int
    ) -> [Float] {
        switch metric {
        case "cosine":
            return metal.batchCosineDistances(query: query, matrix: matrix,
                                              vectorCount: vectorCount, dimension: dimension)
        default: // euclidean → squared L2 (build-phase ranking uses squared)
            return metal.batchSquaredL2(query: query, matrix: matrix,
                                        vectorCount: vectorCount, dimension: dimension)
        }
    }

    private static func vdspBatch(
        metric: String, query: Vector, matrix: [Float],
        vectorCount: Int, dimension: Int
    ) -> [Float] {
        switch metric {
        case "cosine":
            return batchDistances(query: query, matrix: matrix,
                                  vectorCount: vectorCount, dimension: dimension,
                                  metric: CosineDistance())
        default:
            // Squared, to match the GPU squared-L2 kernel (ranking-equivalent).
            return batchL2Distances(query: query, matrix: matrix,
                                    vectorCount: vectorCount, dimension: dimension).map { $0 * $0 }
        }
    }

    private static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count else { return .infinity }
        var m: Float = 0
        for i in a.indices { m = max(m, abs(a[i] - b[i])) }
        return Double(m)
    }
}

// ── insert-shape subcommand ──────────────────────────────────────────

enum InsertShapeBench {

    struct Options {
        let dimensions: [Int]
        let counts: [Int]
        let metrics: [String]
        let m: Int
        let efConstruction: Int
        let seed: UInt64
        let libraryVersion: String
        let notes: String
        let outputPath: String
    }

    static func run(_ opts: Options) async throws {
        let started = Date()
        var rows: [InsertShapeRow] = []

        for metricName in opts.metrics {
            for dim in opts.dimensions {
                for count in opts.counts {
                    var rng = BenchSeededRandom(
                        seed: opts.seed &+ UInt64(dim) &* 0x1_0000 &+ UInt64(count))
                    let matrix = seededMatrix(vectorCount: count, dimension: dim, rng: &rng)

                    // Deterministic build: fixed levelSeed so the graph
                    // topology (hence the eval count) is reproducible.
                    let config = HNSWConfiguration(
                        m: opts.m, efConstruction: opts.efConstruction,
                        efSearch: 50, autoCompactionThreshold: nil, levelSeed: 0xB111_5EED)

                    // Build the index once with a single counting metric
                    // whose "traversal target" is reset to the current insert
                    // vector before each add, so we can attribute
                    // graph-traversal evals (a == insert vector, i.e.
                    // `searchLayer` query-to-candidate) vs the pairwise
                    // heuristic-selection / pruning evals.
                    let liveCounter = MutableTargetCountingMetric(baseMetric(metricName))
                    let index = HNSWIndex(dimension: dim, metric: liveCounter, config: config)
                    for i in 0..<count {
                        let start = i * dim
                        let vec = Vector(Array(matrix[start..<start + dim]))
                        liveCounter.setTarget(vec)
                        try await index.add(vec, id: UUID())
                    }
                    let total = liveCounter.total
                    let traversal = liveCounter.traversal

                    let mMax0 = 2 * opts.m
                    rows.append(InsertShapeRow(
                        metric: metricName,
                        dimension: dim,
                        vectorCount: count,
                        m: opts.m,
                        mMax0: mMax0,
                        efConstruction: opts.efConstruction,
                        totalDistanceEvals: total,
                        meanEvalsPerInsert: count > 0 ? Double(total) / Double(count) : 0,
                        traversalEvals: traversal,
                        pairwiseEvals: total - traversal,
                        maxOneQueryToNBatchCeiling: mMax0
                    ))

                    let perInsert = String(format: "%.1f", Double(total) / Double(max(count, 1)))
                    let line = "[insert-shape] \(metricName) d=\(dim) N=\(count)  "
                        + "evals=\(total) (\(perInsert)/insert)  "
                        + "traversal=\(traversal) pairwise=\(total - traversal)  "
                        + "batch-ceiling=mMax0=\(mMax0)\n"
                    FileHandle.standardError.write(Data(line.utf8))
                }
            }
        }

        let sweep = InsertShapeSweep(
            schemaVersion: 1,
            kind: "hnsw-insert-shape",
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            platform: PlatformProbe.current(),
            seed: Int(bitPattern: UInt(truncatingIfNeeded: opts.seed)),
            runStartedAt: ISO8601DateFormatter().string(from: started),
            notes: opts.notes,
            rows: rows
        )
        try writeJSON(sweep, to: opts.outputPath)
        FileHandle.standardError.write(Data(
            "[insert-shape] wrote \(opts.outputPath) (\(rows.count) rows)\n".utf8))
    }

    private static func baseMetric(_ name: String) -> any DistanceMetric {
        name == "cosine" ? CosineDistance() : EuclideanDistance()
    }
}

/// Counting metric whose "traversal" target can be reset before each
/// insert, so a single instance can attribute graph-traversal evals
/// (a == current insert vector) across the whole build.
final class MutableTargetCountingMetric: DistanceMetric, @unchecked Sendable {
    private let inner: any DistanceMetric
    private let lock = NSLock()
    private var target: Vector?
    private(set) var total = 0
    private(set) var traversal = 0

    init(_ inner: any DistanceMetric) { self.inner = inner }

    func setTarget(_ v: Vector) {
        lock.lock(); target = v; lock.unlock()
    }

    func distance(_ a: Vector, _ b: Vector) -> Float {
        lock.lock()
        total += 1
        if let t = target, a == t { traversal += 1 }
        lock.unlock()
        return inner.distance(a, b)
    }
}

// ── shared JSON writer ───────────────────────────────────────────────

private func writeJSON<T: Encodable>(_ value: T, to path: String) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(value)
    try data.write(to: URL(fileURLWithPath: path))
}

// ── Duration helpers (mirrors BenchRunner's private extension) ───────

private extension Duration {
    var seconds: Double {
        let p = components
        return Double(p.seconds) + Double(p.attoseconds) * 1e-18
    }
    var ms: Double { seconds * 1000.0 }
}
