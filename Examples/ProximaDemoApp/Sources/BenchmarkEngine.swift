// BenchmarkEngine.swift
// ProximaDemoApp
//
// v1.6.0 Benchmark tab (ROADMAP "in-app recall/latency sweep"). Runs an
// efSearch sweep over a reproducible synthetic corpus and measures HNSW
// recall@10 against a BruteForceIndex ground truth, plus live query latency.
//
// Reproducible by construction: the corpus and queries come from
// `SyntheticCorpus` (fixed seeds) and the HNSW graph is built with a fixed
// `levelSeed`, so recall@10 is identical on every run. Latency is measured
// live per query (it naturally varies), so the table reports median and p90.

import Foundation
import ProximaKit

/// One point on the recall-vs-latency curve: the result of the sweep at a
/// single `efSearch` value.
struct BenchmarkPoint: Identifiable, Sendable {
    var id: Int { efSearch }
    let efSearch: Int
    let recallAt10: Double     // 0...1, vs BruteForce top-10
    let medianMs: Double
    let p90Ms: Double
}

@Observable
@MainActor
final class BenchmarkEngine {

    // ── Fixed, reproducible benchmark parameters (shown in the UI) ─────
    let corpusSize = 3_000
    let dimension = 128
    let queryCount = 100
    let k = 10
    let efValues = [16, 32, 64, 128, 256]
    /// Tuned so the recall curve spans a visible range (≈0.46 → 0.96) rather
    /// than saturating at 1.0 — a low `m` and moderate `efConstruction` make
    /// low-ef queries genuinely miss, which is the whole point of the chart.
    private let graphM = 8
    private let graphEfConstruction = 64
    private let levelSeed: UInt64 = 0xBEEF

    enum Phase: Equatable { case idle, running, done }

    var phase: Phase = .idle
    var progress: Double = 0
    var statusLabel = ""
    var results: [BenchmarkPoint] = []
    var errorMessage: String?
    /// Wall-clock time of the last completed run (build + truth + sweep).
    var lastRunSeconds: Double = 0

    func run() async {
        guard phase != .running else { return }
        phase = .running
        progress = 0
        results = []
        errorMessage = nil
        statusLabel = "Building index…"
        let started = DispatchTime.now().uptimeNanoseconds

        let brute = BruteForceIndex(dimension: dimension, metric: CosineDistance())
        let hnsw = HNSWIndex(
            dimension: dimension,
            metric: CosineDistance(),
            config: HNSWConfiguration(
                m: graphM, efConstruction: graphEfConstruction, efSearch: 50,
                autoCompactionThreshold: nil, levelSeed: levelSeed))

        do {
            // ── Build: exact baseline + approximate index over the same corpus ─
            for i in 0..<corpusSize {
                let v = SyntheticCorpus.vector(i, dimension: dimension)
                try await brute.add(v, id: SyntheticCorpus.id(i))
                try await hnsw.add(v, id: SyntheticCorpus.id(i))
                if i % 200 == 0 {
                    progress = 0.6 * Double(i) / Double(corpusSize)
                }
            }

            // ── Ground truth from the exact index (query ids offset far past
            //    the corpus so they are genuine held-out points) ──────────────
            statusLabel = "Computing ground truth…"
            var truth: [Set<UUID>] = []
            truth.reserveCapacity(queryCount)
            for q in 0..<queryCount {
                let query = SyntheticCorpus.vector(1_000_000 + q, dimension: dimension)
                let exact = await brute.search(query: query, k: k)
                truth.append(Set(exact.map(\.id)))
            }
            progress = 0.7

            // ── Sweep efSearch, measuring recall@10 and live latency ─────────
            statusLabel = "Sweeping efSearch…"
            var points: [BenchmarkPoint] = []
            for (idx, ef) in efValues.enumerated() {
                var totalHits = 0
                var latencies: [Double] = []
                latencies.reserveCapacity(queryCount)
                for q in 0..<queryCount {
                    let query = SyntheticCorpus.vector(1_000_000 + q, dimension: dimension)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let got = await hnsw.search(query: query, k: k, efSearch: ef)
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    latencies.append(Double(t1 - t0) / 1_000_000)
                    totalHits += Set(got.map(\.id)).intersection(truth[q]).count
                }
                latencies.sort()
                let median = latencies[latencies.count / 2]
                let p90 = latencies[min(latencies.count - 1, Int(Double(latencies.count) * 0.9))]
                points.append(BenchmarkPoint(
                    efSearch: ef,
                    recallAt10: Double(totalHits) / Double(queryCount * k),
                    medianMs: median,
                    p90Ms: p90))
                progress = 0.7 + 0.3 * Double(idx + 1) / Double(efValues.count)
            }

            results = points
            lastRunSeconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            statusLabel = ""
            phase = .done
        } catch {
            errorMessage = error.localizedDescription
            phase = .idle
        }
    }
}
