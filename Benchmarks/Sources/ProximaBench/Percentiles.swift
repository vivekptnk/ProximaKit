// Percentiles.swift
// Latency statistics: mean + p50 + p95, computed from sorted doubles.

import Foundation

enum Percentiles {
    struct Stats {
        let meanMs: Double
        let p50Ms: Double
        let p95Ms: Double
    }

    /// `latenciesMs` is modified in place (sorted). Passed `inout` to avoid a copy.
    static func compute(_ latenciesMs: inout [Double]) -> Stats {
        guard !latenciesMs.isEmpty else {
            return Stats(meanMs: 0, p50Ms: 0, p95Ms: 0)
        }
        latenciesMs.sort()
        let mean = latenciesMs.reduce(0, +) / Double(latenciesMs.count)
        return Stats(
            meanMs: mean,
            p50Ms: percentile(sorted: latenciesMs, q: 0.50),
            p95Ms: percentile(sorted: latenciesMs, q: 0.95)
        )
    }

    /// Nearest-rank percentile on a pre-sorted array.
    private static func percentile(sorted: [Double], q: Double) -> Double {
        let rank = Int((q * Double(sorted.count)).rounded(.up))
        let idx = max(0, min(sorted.count - 1, rank - 1))
        return sorted[idx]
    }
}
