// BenchRunner.swift
// Builds the HNSW index over the base set, measures recall against a
// pre-computed ground truth file, and emits a BenchResult JSON document.

import Foundation
import ProximaKit

struct BenchOptions {
    let baseVectorsPath: String
    let queryVectorsPath: String
    let groundTruthPath: String
    let dataset: String
    let datasetSize: Int?          // cap on base set; nil = all
    let queryCount: Int?           // cap on queries; nil = all
    let k: Int
    let M: Int
    let efConstruction: Int
    let efSearch: Int
    let metric: String             // "l2" or "cosine"
    let seed: Int
    let notes: String
    let libraryVersion: String
    let outputPath: String
}

enum BenchRunner {
    static func run(_ opts: BenchOptions) async throws {
        let started = Date()

        // ── Load data ─────────────────────────────────────────────────
        let base = try FVecLoader.loadFvecs(path: opts.baseVectorsPath, limit: opts.datasetSize)
        let queries = try FVecLoader.loadFvecs(path: opts.queryVectorsPath, limit: opts.queryCount)
        precondition(base.dimension == queries.dimension,
                     "base dim \(base.dimension) != query dim \(queries.dimension)")

        let gt = try loadGroundTruth(path: opts.groundTruthPath)
        precondition(gt.queryCount == queries.count,
                     "GT queryCount \(gt.queryCount) != loaded queries \(queries.count)")
        precondition(gt.k >= opts.k,
                     "GT was computed with k=\(gt.k), cannot evaluate recall at k=\(opts.k)")

        // ── Build index ───────────────────────────────────────────────
        let metric = makeMetric(opts.metric)
        let config = HNSWConfiguration(m: opts.M, efConstruction: opts.efConstruction, efSearch: opts.efSearch)
        let index = HNSWIndex(dimension: base.dimension, metric: metric, config: config)

        // Map our integer IDs [0, datasetSize) to UUIDs. We keep the reverse
        // map so that when HNSW returns UUIDs we can compute recall against
        // the integer-indexed ground truth.
        var uuidByIndex: [UUID] = []
        uuidByIndex.reserveCapacity(base.count)
        var indexByUuid: [UUID: Int] = [:]
        indexByUuid.reserveCapacity(base.count)
        for i in 0..<base.count {
            let id = UUID()
            uuidByIndex.append(id)
            indexByUuid[id] = i
        }

        let buildStart = Date()
        for i in 0..<base.count {
            let start = i * base.dimension
            let end = start + base.dimension
            let slice = Array(base.data[start..<end])
            try await index.add(Vector(slice), id: uuidByIndex[i])
        }
        let buildTime = Date().timeIntervalSince(buildStart)
        let rssBytes = PlatformProbe.residentMemoryBytes()

        // ── Search + measure ──────────────────────────────────────────
        var latenciesMs: [Double] = []
        latenciesMs.reserveCapacity(queries.count)
        var totalRecall: Double = 0

        for q in 0..<queries.count {
            let qs = q * queries.dimension
            let qe = qs + queries.dimension
            let queryVec = Vector(Array(queries.data[qs..<qe]))

            let t0 = Date()
            let results = await index.search(query: queryVec, k: opts.k)
            let elapsed = Date().timeIntervalSince(t0) * 1000.0
            latenciesMs.append(elapsed)

            let gtStart = q * gt.k
            let gtNeighbors = Set(gt.neighbors[gtStart..<gtStart + opts.k].map { Int($0) })

            var hits = 0
            for r in results {
                if let idx = indexByUuid[r.id], gtNeighbors.contains(idx) {
                    hits += 1
                }
            }
            totalRecall += Double(hits) / Double(opts.k)
        }

        let stats = Percentiles.compute(&latenciesMs)
        let qps = stats.meanMs > 0 ? 1000.0 / stats.meanMs : 0
        let recall = totalRecall / Double(queries.count)

        // ── Emit JSON ─────────────────────────────────────────────────
        let result = BenchResult(
            schemaVersion: 1,
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            dataset: opts.dataset,
            datasetSize: base.count,
            dimension: base.dimension,
            metric: opts.metric,
            indexParams: IndexParams(
                type: "hnsw",
                M: opts.M,
                efConstruction: opts.efConstruction,
                efSearch: opts.efSearch
            ),
            k: opts.k,
            queryCount: queries.count,
            buildTimeSeconds: buildTime,
            searchLatencyMeanMs: stats.meanMs,
            searchLatencyP50Ms: stats.p50Ms,
            searchLatencyP95Ms: stats.p95Ms,
            queriesPerSecond: qps,
            recallAt10: recall,
            residentMemoryMb: Double(rssBytes) / (1024.0 * 1024.0),
            platform: PlatformProbe.current(),
            seed: opts.seed,
            runStartedAt: ISO8601DateFormatter().string(from: started),
            runDurationSeconds: Date().timeIntervalSince(started),
            notes: opts.notes
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)
        try data.write(to: URL(fileURLWithPath: opts.outputPath))

        FileHandle.standardError.write(Data("""
        [ProximaBench] wrote \(opts.outputPath)
          dataset=\(opts.dataset) size=\(base.count) dim=\(base.dimension)
          build=\(String(format: "%.2fs", buildTime))  p50=\(String(format: "%.2fms", stats.p50Ms))  p95=\(String(format: "%.2fms", stats.p95Ms))
          recall@\(opts.k)=\(String(format: "%.3f", recall))  rss=\(String(format: "%.1fMB", Double(rssBytes) / (1024 * 1024)))

        """.utf8))
    }

    private static func loadGroundTruth(path: String) throws -> GroundTruthFile {
        let data = try Data(contentsOf: URL(fileURLWithPath: path), options: [.mappedIfSafe])
        return try JSONDecoder().decode(GroundTruthFile.self, from: data)
    }

    private static func makeMetric(_ name: String) -> any DistanceMetric {
        switch name.lowercased() {
        case "cosine": return CosineDistance()
        case "l2", "euclidean": return EuclideanDistance()
        default:
            FileHandle.standardError.write(Data("[ProximaBench] unknown metric '\(name)', defaulting to l2\n".utf8))
            return EuclideanDistance()
        }
    }
}
