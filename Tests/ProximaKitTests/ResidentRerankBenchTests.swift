// ResidentRerankBenchTests.swift
// ProximaKitTests
//
// ADR-014 Stage 2 resident-regression proof (ADR-005 discipline): times the
// RESIDENT rerank hot path so the before/after of threading the two-case
// `OriginalsStore` through the single rerank read site can be shown unregressed.
// Uses ONLY the public `search(query:k:rerankDepth:)` API, so the SAME file
// compiles and runs on the pre-change tree (HEAD) and the post-change tree —
// enabling a same-machine A/B in release mode.
//
// Env-gated (PROXIMA_RESIDENT_BENCH=1), CI-excluded. Run:
//   PROXIMA_RESIDENT_BENCH=1 swift test -c release \
//     --filter ResidentRerankBenchTests

import XCTest
@testable import ProximaKit

final class ResidentRerankBenchTests: XCTestCase {

    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0xBEEF &* UInt64(i + 1))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 65_536) / 65_536.0 })
    }

    func testResidentRerankThroughput() async throws {
        guard ProcessInfo.processInfo.environment["PROXIMA_RESIDENT_BENCH"] == "1" else {
            throw XCTSkip("set PROXIMA_RESIDENT_BENCH=1 to run the resident-rerank regression bench")
        }
        let n = 8_000, dim = 128, k = 10, depth = 40
        let queries = 400, reps = 9

        let vectors = (0..<n).map { vec($0, dim: dim) }
        let ids = (0..<n).map { _ in UUID() }
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: HNSWConfiguration(m: 16, efConstruction: 64, efSearch: 64, levelSeed: 0xB0),
            pqConfig: PQConfiguration(subspaceCount: 16, trainingIterations: 8, seed: 0xB0),
            retainOriginals: true)                       // RESIDENT originals

        let qs = (0..<queries).map { vec(500_000 + $0, dim: dim) }

        // Warmup.
        for q in qs { _ = try await index.search(query: q, k: k, rerankDepth: depth) }

        var perRep: [Double] = []
        for _ in 0..<reps {
            let t0 = DispatchTime.now().uptimeNanoseconds
            for q in qs { _ = try await index.search(query: q, k: k, rerankDepth: depth) }
            let t1 = DispatchTime.now().uptimeNanoseconds
            perRep.append(Double(t1 - t0) / 1_000_000.0 / Double(queries))   // ms/query
        }
        perRep.sort()
        let median = perRep[perRep.count / 2]
        let p10 = perRep[max(0, perRep.count / 10)]
        let p90 = perRep[min(perRep.count - 1, perRep.count * 9 / 10)]
        print("""
        ── resident rerank throughput (release) ──────────────────────────
        fixture   : \(n) × \(dim)d retained, k=\(k), rerankDepth=\(depth)
        reps      : \(reps) × \(queries) queries
        ms/query  : median \(String(format: "%.4f", median))  \
        (p10 \(String(format: "%.4f", p10)), p90 \(String(format: "%.4f", p90)))
        ──────────────────────────────────────────────────────────────────
        """)
        XCTAssertGreaterThan(median, 0)
    }
}
