// GroundTruthBuilder.swift
// Exact k-nearest ground truth via BruteForceIndex. Used when the dataset
// does not ship a .ivecs groundtruth file (e.g. MS MARCO) or when the
// caller wants GT on a subset of the base set.

import Foundation
import ProximaKit

enum GroundTruthBuilder {
    struct Options {
        let baseVectorsPath: String
        let queryVectorsPath: String
        let dataset: String
        let datasetSize: Int?
        let queryCount: Int?
        let k: Int
        let metric: String
        let outputPath: String
    }

    static func run(_ opts: Options) async throws {
        let base = try FVecLoader.loadFvecs(path: opts.baseVectorsPath, limit: opts.datasetSize)
        let queries = try FVecLoader.loadFvecs(path: opts.queryVectorsPath, limit: opts.queryCount)
        precondition(base.dimension == queries.dimension)

        let metric = makeMetric(opts.metric)
        let brute = BruteForceIndex(dimension: base.dimension, metric: metric)

        // Build the brute-force index with integer-indexed UUIDs so we can
        // round-trip result ids back to dataset row indices.
        var uuidByIndex: [UUID] = []
        uuidByIndex.reserveCapacity(base.count)
        var indexByUuid: [UUID: Int] = [:]
        indexByUuid.reserveCapacity(base.count)
        for i in 0..<base.count {
            let id = UUID()
            uuidByIndex.append(id)
            indexByUuid[id] = i
            let start = i * base.dimension
            let end = start + base.dimension
            try await brute.add(Vector(Array(base.data[start..<end])), id: id)
        }

        var flatNeighbors: [Int] = []
        flatNeighbors.reserveCapacity(queries.count * opts.k)

        for q in 0..<queries.count {
            let qs = q * queries.dimension
            let qe = qs + queries.dimension
            let queryVec = Vector(Array(queries.data[qs..<qe]))
            let results = await brute.search(query: queryVec, k: opts.k)
            precondition(results.count == opts.k,
                         "brute-force returned \(results.count) results for k=\(opts.k) on query \(q)")
            for r in results {
                guard let idx = indexByUuid[r.id] else {
                    fatalError("BruteForce returned unknown UUID")
                }
                flatNeighbors.append(idx)
            }
        }

        let gt = GroundTruthFile(
            schemaVersion: 1,
            dataset: opts.dataset,
            datasetSize: base.count,
            dimension: base.dimension,
            metric: opts.metric,
            k: opts.k,
            queryCount: queries.count,
            neighbors: flatNeighbors
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]  // no prettyPrinted — GT is big
        let data = try encoder.encode(gt)
        try data.write(to: URL(fileURLWithPath: opts.outputPath))

        FileHandle.standardError.write(Data("""
        [ProximaBench/GT] wrote \(opts.outputPath)
          dataset=\(opts.dataset) size=\(base.count) queries=\(queries.count) k=\(opts.k)

        """.utf8))
    }

    private static func makeMetric(_ name: String) -> any DistanceMetric {
        switch name.lowercased() {
        case "cosine": return CosineDistance()
        case "l2", "euclidean": return EuclideanDistance()
        default: return EuclideanDistance()
        }
    }
}
