// SearchProviderBench.swift
// ProximaBench
//
// Resident-mode search-latency regression harness for ADR-013 Stage 2
// (the vector-provider abstraction). Builds a seeded HNSW fixture, warms up,
// then times `search` across many seeded queries over several reps and reports
// the median + spread. Run it on the code BEFORE and AFTER the provider change
// (same seed / same flags) to prove the resident path is free of regression.
//
// Public API only — it links the same way against pre- and post-change
// ProximaKit, so the two medians are directly comparable.

import Foundation
import ProximaKit

enum SearchProviderBench {

    struct Options {
        let count: Int
        let dimension: Int
        let m: Int
        let efConstruction: Int
        let efSearch: Int
        let k: Int
        let queryCount: Int
        let reps: Int
        let warmup: Int
        let seed: UInt64
    }

    /// SplitMix64 — deterministic fixture/query generation (matches the
    /// library's own `levelSeed` generator so runs are reproducible).
    private struct SMix: RandomNumberGenerator {
        private var s: UInt64
        init(seed: UInt64) { s = seed }
        mutating func next() -> UInt64 {
            s &+= 0x9E37_79B9_7F4A_7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private static func randomVector(_ rng: inout SMix, dim: Int) -> Vector {
        Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    private static func deterministicUUID(_ i: Int) -> UUID {
        var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                     UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &bytes) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: bytes)
    }

    static func run(_ opts: Options) async throws {
        let banner = "[search-provider-bench] building \(opts.count)×\(opts.dimension)d fixture "
            + "(m=\(opts.m), efc=\(opts.efConstruction), seed=\(opts.seed))…\n"
        FileHandle.standardError.write(Data(banner.utf8))

        let config = HNSWConfiguration(
            m: opts.m, efConstruction: opts.efConstruction, efSearch: opts.efSearch,
            autoCompactionThreshold: nil, levelSeed: opts.seed)
        let index = HNSWIndex(dimension: opts.dimension, metric: EuclideanDistance(), config: config)

        var buildRNG = SMix(seed: opts.seed &* 2654435761)
        let buildStart = DispatchTime.now()
        for i in 0..<opts.count {
            try await index.add(randomVector(&buildRNG, dim: opts.dimension), id: deterministicUUID(i))
            if i % 10_000 == 0 && i > 0 {
                FileHandle.standardError.write(Data("  … \(i) inserted\n".utf8))
            }
        }
        let buildMs = Double(DispatchTime.now().uptimeNanoseconds - buildStart.uptimeNanoseconds) / 1_000_000

        // Seeded query set (disjoint seed from the fixture).
        var queryRNG = SMix(seed: opts.seed ^ 0xD1B5_4A32_D192_ED03)
        let queries = (0..<opts.queryCount).map { _ in randomVector(&queryRNG, dim: opts.dimension) }

        // Warmup — fault caches, JIT actor machinery, stabilize clocks.
        for _ in 0..<opts.warmup {
            for q in queries { _ = await index.search(query: q, k: opts.k) }
        }

        // Timed reps: total wall time for the whole query set, per rep.
        var perQueryNs: [Double] = []
        var checksum = 0
        for _ in 0..<opts.reps {
            let start = DispatchTime.now()
            for q in queries {
                let r = await index.search(query: q, k: opts.k)
                checksum &+= r.count
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
            perQueryNs.append(elapsed / Double(opts.queryCount))
        }
        perQueryNs.sort()

        let median = perQueryNs[perQueryNs.count / 2]
        let minV = perQueryNs.first!
        let maxV = perQueryNs.last!
        let spread = (maxV - minV) / median * 100

        print("""
        ── search-provider-bench (resident mode) ─────────────────────────
        fixture      : \(opts.count) × \(opts.dimension)d, m=\(opts.m), efc=\(opts.efConstruction)
        query        : k=\(opts.k), efSearch=\(opts.efSearch), queries=\(opts.queryCount), reps=\(opts.reps), warmup=\(opts.warmup)
        seed         : \(opts.seed)
        build        : \(String(format: "%.0f", buildMs)) ms
        per-query ns : median=\(String(format: "%.0f", median))  min=\(String(format: "%.0f", minV))  max=\(String(format: "%.0f", maxV))  spread=\(String(format: "%.1f", spread))%
        all reps ns  : \(perQueryNs.map { String(format: "%.0f", $0) }.joined(separator: ", "))
        checksum     : \(checksum)
        ──────────────────────────────────────────────────────────────────
        """)
    }
}
