// Benchmarks/BenchmarkKit.swift
// Core infrastructure for ProximaKit benchmarks.
// Not part of the library — lives in the Benchmarks/ target.

import Foundation

// MARK: - Timing

/// Measures execution time of a closure with nanosecond precision.
/// Returns (result, elapsed seconds).
@discardableResult
public func measure<T>(_ label: String = "", _ block: () throws -> T) rethrows -> (T, TimeInterval) {
    let start = DispatchTime.now()
    let result = try block()
    let end = DispatchTime.now()
    let nanos = end.uptimeNanoseconds - start.uptimeNanoseconds
    let seconds = Double(nanos) / 1_000_000_000
    if !label.isEmpty {
        print("  \(label): \(formatDuration(seconds))")
    }
    return (result, seconds)
}

/// Runs a closure `iterations` times and returns percentile latencies.
public func measurePercentiles(
    iterations: Int,
    warmup: Int = 10,
    _ block: () -> Void
) -> LatencyStats {
    // Warmup
    for _ in 0..<warmup { block() }

    var times: [Double] = []
    times.reserveCapacity(iterations)

    for _ in 0..<iterations {
        let start = DispatchTime.now()
        block()
        let end = DispatchTime.now()
        let nanos = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
        times.append(nanos / 1_000_000) // milliseconds
    }

    times.sort()
    return LatencyStats(
        p50: percentile(times, 0.50),
        p95: percentile(times, 0.95),
        p99: percentile(times, 0.99),
        mean: times.reduce(0, +) / Double(times.count),
        min: times.first ?? 0,
        max: times.last ?? 0,
        count: iterations
    )
}

public struct LatencyStats {
    public let p50: Double   // ms
    public let p95: Double   // ms
    public let p99: Double   // ms
    public let mean: Double  // ms
    public let min: Double   // ms
    public let max: Double   // ms
    public let count: Int

    public func report(target p99Target: Double? = nil) -> String {
        let pass = p99Target.map { p99 <= $0 } ?? true
        let marker = pass ? "✅" : "❌"
        var s = "p50=\(fmt(p50))ms  p95=\(fmt(p95))ms  p99=\(fmt(p99))ms"
        if let target = p99Target {
            s += "  (target: <\(fmt(target))ms) \(marker)"
        }
        return s
    }
}

// MARK: - Data Generation

/// Generates uniform random vectors with a fixed seed for reproducibility.
public func generateUniformVectors(
    count: Int,
    dimension: Int,
    seed: UInt64 = 42
) -> [[Float]] {
    var rng = SeededRNG(seed: seed)
    return (0..<count).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
    }
}

/// Generates clustered vectors that simulate real embedding distributions.
/// Creates `clusterCount` centers, then generates points around each with Gaussian noise.
public func generateClusteredVectors(
    count: Int,
    dimension: Int,
    clusterCount: Int = 8,
    spread: Float = 0.1,
    seed: UInt64 = 42
) -> [[Float]] {
    var rng = SeededRNG(seed: seed)

    // Generate cluster centers
    let centers: [[Float]] = (0..<clusterCount).map { _ in
        (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    // Generate points around centers
    let perCluster = count / clusterCount
    var vectors: [[Float]] = []
    vectors.reserveCapacity(count)

    for (i, center) in centers.enumerated() {
        let n = (i == clusterCount - 1) ? (count - vectors.count) : perCluster
        for _ in 0..<n {
            let point = center.map { $0 + gaussianNoise(spread: spread, rng: &rng) }
            vectors.append(point)
        }
    }

    return vectors
}

/// Generates low-rank vectors (high embedding dim, low intrinsic dim).
/// First `rank` dimensions carry signal, rest is low-amplitude noise.
public func generateCorrelatedVectors(
    count: Int,
    dimension: Int,
    rank: Int = 50,
    seed: UInt64 = 42
) -> [[Float]] {
    var rng = SeededRNG(seed: seed)
    return (0..<count).map { _ in
        (0..<dimension).map { d in
            if d < rank {
                return Float.random(in: -1...1, using: &rng)
            } else {
                return Float.random(in: -0.01...0.01, using: &rng)
            }
        }
    }
}

// MARK: - Recall Calculation

/// Computes recall@K: what fraction of the true top-K were found by approximate search.
/// `groundTruth` and `approximate` are arrays of result IDs (or indices), one per query.
public func recallAtK(
    groundTruth: [[Int]],
    approximate: [[Int]],
    k: Int
) -> Double {
    precondition(groundTruth.count == approximate.count, "Query count mismatch")
    var totalRecall: Double = 0

    for (truth, approx) in zip(groundTruth, approximate) {
        let truthSet = Set(truth.prefix(k))
        let approxSet = Set(approx.prefix(k))
        let hits = truthSet.intersection(approxSet).count
        totalRecall += Double(hits) / Double(min(k, truthSet.count))
    }

    return totalRecall / Double(groundTruth.count)
}

// MARK: - Baseline Comparison

public struct BenchmarkBaseline: Codable {
    public let device: String
    public let swiftVersion: String
    public let date: String
    public let results: [String: BaselineEntry]
}

public struct BaselineEntry: Codable {
    public let value: Double
    public let unit: String         // "ms", "%", "MB", "KB", "s"
    public let threshold: Double?   // absolute threshold to pass
}

/// Loads baseline from Benchmarks/baselines.json
public func loadBaseline(from path: String = "Benchmarks/baselines.json") -> BenchmarkBaseline? {
    guard let data = FileManager.default.contents(atPath: path) else { return nil }
    return try? JSONDecoder().decode(BenchmarkBaseline.self, from: data)
}

/// Compares a result against baseline. Returns (pass, delta%).
public func compareToBaseline(
    key: String,
    value: Double,
    baseline: BenchmarkBaseline?,
    regressionThreshold: Double = 0.10  // 10% regression = fail
) -> (pass: Bool, delta: Double?) {
    guard let entry = baseline?.results[key] else { return (true, nil) }

    // For latency/time/memory: lower is better, regression = value increased
    // For recall: higher is better, regression = value decreased
    let isHigherBetter = entry.unit == "%"

    let delta: Double
    if isHigherBetter {
        delta = (value - entry.value) / entry.value  // positive = improvement
    } else {
        delta = (entry.value - value) / entry.value   // positive = improvement
    }

    // Check absolute threshold if specified
    if let threshold = entry.threshold {
        if isHigherBetter {
            return (value >= threshold, delta)
        } else {
            return (value <= threshold, delta)
        }
    }

    // Otherwise check regression against baseline
    return (delta > -regressionThreshold, delta)
}

// MARK: - Report Generation

public struct BenchmarkReport {
    public var sections: [ReportSection] = []

    public struct ReportSection {
        public let title: String
        public var rows: [ReportRow]
    }

    public struct ReportRow {
        public let name: String
        public let value: String
        public let pass: Bool
    }

    public mutating func section(_ title: String, _ builder: (inout ReportSection) -> Void) {
        var s = ReportSection(title: title, rows: [])
        builder(&s)
        sections.append(s)
    }

    public func render() -> String {
        var lines: [String] = []
        lines.append("ProximaKit Benchmark Report")
        lines.append("Date: \(ISO8601DateFormatter().string(from: Date()))")
        lines.append(String(repeating: "═", count: 60))

        for section in sections {
            lines.append("")
            lines.append(section.title.uppercased())
            for row in section.rows {
                let marker = row.pass ? "✅" : "❌"
                lines.append("  \(row.name.padding(toLength: 28, withPad: " ", startingAt: 0)) \(row.value) \(marker)")
            }
        }

        let failCount = sections.flatMap(\.rows).filter { !$0.pass }.count
        lines.append("")
        lines.append(String(repeating: "═", count: 60))
        if failCount == 0 {
            lines.append("ALL BENCHMARKS PASSED ✅")
        } else {
            lines.append("\(failCount) BENCHMARK(S) FAILED ❌")
        }

        return lines.joined(separator: "\n")
    }

    /// Exports results to JSON for baseline storage
    public func toJSON() -> [String: BaselineEntry] {
        // Override in specific benchmark implementations
        return [:]
    }
}

// MARK: - Helpers

private func percentile(_ sorted: [Double], _ p: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let index = p * Double(sorted.count - 1)
    let lower = Int(index)
    let upper = min(lower + 1, sorted.count - 1)
    let weight = index - Double(lower)
    return sorted[lower] * (1 - weight) + sorted[upper] * weight
}

private func fmt(_ value: Double) -> String {
    if value < 0.01 { return String(format: "%.3f", value) }
    if value < 1 { return String(format: "%.2f", value) }
    if value < 100 { return String(format: "%.1f", value) }
    return String(format: "%.0f", value)
}

func formatDuration(_ seconds: TimeInterval) -> String {
    if seconds < 0.001 { return String(format: "%.1fµs", seconds * 1_000_000) }
    if seconds < 1 { return String(format: "%.1fms", seconds * 1_000) }
    return String(format: "%.2fs", seconds)
}

/// Simple seeded RNG for reproducible benchmarks.
struct SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        // xorshift64
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

private func gaussianNoise(spread: Float, rng: inout SeededRNG) -> Float {
    // Box-Muller transform
    let u1 = max(Float.random(in: 0..<1, using: &rng), 1e-10)
    let u2 = Float.random(in: 0..<1, using: &rng)
    return spread * sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
}
