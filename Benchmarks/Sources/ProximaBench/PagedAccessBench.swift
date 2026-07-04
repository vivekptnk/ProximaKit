// PagedAccessBench.swift
// ProximaBench
//
// ADR-013 / ADR-014 zero-copy decision bench. Quantifies the warm paged
// overhead in vector reads without changing library sources: the mmap/raw
// pointer prototype below is local to this harness.

import Darwin
import Foundation
import ProximaKit

private struct PagedAccessBenchError: Error, CustomStringConvertible {
    let description: String
}

private func benchFail(_ message: String) -> PagedAccessBenchError {
    PagedAccessBenchError(description: message)
}

private final class BenchFloatSink: @unchecked Sendable {
    var value: Float = 0
}

private let pagedAccessSink = BenchFloatSink()

@inline(never)
private func consumeBenchFloat(_ value: Float) {
    pagedAccessSink.value += value
}

private struct TimingDistribution: Codable {
    let medianNs: Double
    let minNs: Double
    let maxNs: Double
    let spreadPercent: Double
    let repsNs: [Double]
}

private struct CountDistribution: Codable {
    let mean: Double
    let median: Double
    let min: Int
    let max: Int
}

private struct CopyIsolationMeasurement: Codable {
    let dimension: Int
    let vectorCount: Int
    let accessesPerRep: Int
    let copy: TimingDistribution
    let rawUnsafeRead: TimingDistribution
    let copyOverRawMedianNs: Double
    let copyOverRawSpreadPercent: Double
    let checksum: Double
}

private struct HNSWImpactMeasurement: Codable {
    let vectorCount: Int
    let dimension: Int
    let m: Int
    let efConstruction: Int
    let efSearch: Int
    let k: Int
    let queryCount: Int
    let resident: TimingDistribution
    let paged: TimingDistribution
    let observedPagedMinusResidentNs: Double
    let observedPagedOverheadFractionPercent: Double
    let vectorReadsPerQuery: CountDistribution
    let isolationExtrapolationNsPerQueryNonTransferable: Double
    let isolationExtrapolationPercentNonTransferable: Double
    let checksum: Int

    private enum CodingKeys: String, CodingKey {
        case vectorCount
        case dimension
        case m
        case efConstruction
        case efSearch
        case k
        case queryCount
        case resident
        case paged
        case observedPagedMinusResidentNs
        case observedPagedOverheadFractionPercent
        case vectorReadsPerQuery
        case isolationExtrapolationNsPerQueryNonTransferable = "isolationExtrapolationNsPerQuery_nonTransferable"
        case isolationExtrapolationPercentNonTransferable = "isolationExtrapolationPercent_nonTransferable"
        case checksum
    }
}

private struct RerankImpactMeasurement: Codable {
    let vectorCount: Int
    let dimension: Int
    let m: Int
    let efConstruction: Int
    let efSearch: Int
    let k: Int
    let queryCount: Int
    let rerankDepth: Int
    let pqSubspaces: Int
    let pqTrainingIterations: Int
    let resident: TimingDistribution
    let paged: TimingDistribution
    let observedPagedMinusResidentNs: Double
    let observedPagedOverheadFractionPercent: Double
    let isolationExtrapolationNsPerQueryNonTransferable: Double
    let isolationExtrapolationPercentNonTransferable: Double
    let checksum: Int

    private enum CodingKeys: String, CodingKey {
        case vectorCount
        case dimension
        case m
        case efConstruction
        case efSearch
        case k
        case queryCount
        case rerankDepth
        case pqSubspaces
        case pqTrainingIterations
        case resident
        case paged
        case observedPagedMinusResidentNs
        case observedPagedOverheadFractionPercent
        case isolationExtrapolationNsPerQueryNonTransferable = "isolationExtrapolationNsPerQuery_nonTransferable"
        case isolationExtrapolationPercentNonTransferable = "isolationExtrapolationPercent_nonTransferable"
        case checksum
    }
}

private struct PagedAccessDecision: Codable {
    let thresholdDeclared: String
    let thresholdPercent: Double
    let gateMetric: String
    let gateObservedPagedOverheadFractionPercent: Double
    let decisionScope: String
    let decision: String
    let reasoning: String
    let reopenConditions: [String]
}

private struct PagedAccessBenchReport: Codable {
    let schemaVersion: Int
    let kind: String
    let library: String
    let libraryVersion: String
    let platform: Platform
    let seed: Int
    let reps: Int
    let warmupReps: Int
    let runStartedAt: String
    let runDurationSeconds: Double
    let notes: String
    let copyIsolation: [CopyIsolationMeasurement]
    let hnsw: HNSWImpactMeasurement
    let rerank: RerankImpactMeasurement
    let decision: PagedAccessDecision
}

private struct CachedVectorReadCounts: Codable {
    let seed: UInt64
    let vectorCount: Int
    let dimension: Int
    let m: Int
    let efConstruction: Int
    let efSearch: Int
    let k: Int
    let queryCount: Int
    let counts: [Int]
}

private final class LocalMappedFloatVectors {
    let count: Int
    let dimension: Int

    private let fd: Int32
    private let mapping: UnsafeMutableRawPointer
    private let mappingLength: Int

    init(url: URL, count: Int, dimension: Int) throws {
        self.count = count
        self.dimension = dimension
        self.mappingLength = count * dimension * MemoryLayout<Float>.size

        let opened = url.path.withCString { Darwin.open($0, O_RDONLY) }
        guard opened >= 0 else {
            throw benchFail("open failed for \(url.path) (errno \(errno))")
        }
        self.fd = opened

        let mapped = mmap(nil, mappingLength, PROT_READ, MAP_PRIVATE, opened, 0)
        guard let mapped, mapped != MAP_FAILED else {
            Darwin.close(opened)
            throw benchFail("mmap failed for \(url.path) (errno \(errno))")
        }
        self.mapping = mapped
    }

    @inline(__always)
    func vector(at node: Int) -> Vector {
        let ptr = mapping
            .advanced(by: node * dimension * MemoryLayout<Float>.size)
            .assumingMemoryBound(to: Float.self)
        return Vector(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    @inline(__always)
    func withUnsafeVector<R>(at node: Int, _ body: (UnsafeBufferPointer<Float>) -> R) -> R {
        let ptr = mapping
            .advanced(by: node * dimension * MemoryLayout<Float>.size)
            .assumingMemoryBound(to: Float.self)
        return body(UnsafeBufferPointer(start: ptr, count: dimension))
    }

    deinit {
        munmap(mapping, mappingLength)
        Darwin.close(fd)
    }
}

private final class CountingEuclideanMetric: DistanceMetric, @unchecked Sendable {
    private let lock = NSLock()
    private var total = 0

    func reset() {
        lock.lock()
        total = 0
        lock.unlock()
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return total
    }

    func distance(_ a: Vector, _ b: Vector) -> Float {
        let d = a.l2Distance(b)
        lock.lock()
        total += 1
        lock.unlock()
        return d
    }
}

enum PagedAccessBench {
    struct Options {
        let count: Int
        let dimension: Int
        let isolationDimensions: [Int]
        let accessesPerRep: Int
        let queryCount: Int
        let reps: Int
        let warmup: Int
        let m: Int
        let efConstruction: Int
        let efSearch: Int
        let k: Int
        let rerankDepth: Int
        let pqSubspaces: Int
        let pqTrainingIterations: Int
        let seed: UInt64
        let thresholdPercent: Double
        let reuseFixtures: Bool
        let libraryVersion: String
        let notes: String
        let workDir: String
        let outputPath: String
    }

    static func run(_ opts: Options) async throws {
        let started = Date()
        let runStart = DispatchTime.now().uptimeNanoseconds
        let fileManager = FileManager.default
        let work = URL(fileURLWithPath: opts.workDir)
        try fileManager.createDirectory(at: work, withIntermediateDirectories: true)

        let thresholdText = String(
            format: "GO only if observed HNSW paged-overhead fraction exceeds %.1f%% of a warm per-query median; otherwise NO-GO.",
            opts.thresholdPercent
        )
        log("[paged-access-bench] threshold: \(thresholdText)")

        let copyRows = try runCopyIsolation(opts: opts, workDir: work)
        guard let copy384 = copyRows.first(where: { $0.dimension == opts.dimension }) else {
            throw benchFail("copy isolation did not include dimension \(opts.dimension)")
        }

        let fixture = fixturePaths(opts: opts, workDir: work)
        var vectors: [Vector] = []
        if !opts.reuseFixtures || !fixturesExist(fixture) || !fileManager.fileExists(atPath: fixture.counts.path) {
            vectors = makeVectors(count: opts.count, dimension: opts.dimension, seed: opts.seed &+ 0x5EED_1000)
        }

        let queries = makeVectors(
            count: opts.queryCount, dimension: opts.dimension,
            seed: opts.seed ^ 0xD1B5_4A32_D192_ED03
        )

        if !opts.reuseFixtures || !fileManager.fileExists(atPath: fixture.hnswBase.path) {
            if vectors.isEmpty {
                vectors = makeVectors(count: opts.count, dimension: opts.dimension, seed: opts.seed &+ 0x5EED_1000)
            }
            try await buildHNSWFixture(vectors: vectors, opts: opts, base: fixture.hnswBase, wal: fixture.hnswWal)
        } else {
            log("[paged-access-bench] reusing HNSW fixture \(fixture.hnswBase.path)")
        }

        let hnswCounts = try await loadOrBuildVectorReadCounts(
            vectors: vectors, queries: queries, opts: opts, path: fixture.counts
        )

        if !opts.reuseFixtures || !fileManager.fileExists(atPath: fixture.pq.path) {
            if vectors.isEmpty {
                vectors = makeVectors(count: opts.count, dimension: opts.dimension, seed: opts.seed &+ 0x5EED_1000)
            }
            try await buildPQFixture(vectors: vectors, opts: opts, url: fixture.pq)
        } else {
            log("[paged-access-bench] reusing PQ fixture \(fixture.pq.path)")
        }

        let hnsw = try await runHNSWImpact(
            opts: opts, queries: queries, base: fixture.hnswBase, wal: fixture.hnswWal,
            vectorReadCounts: hnswCounts, copyOverRawNs: copy384.copyOverRawMedianNs
        )
        let rerank = try await runRerankImpact(
            opts: opts, queries: queries, pqURL: fixture.pq,
            copyOverRawNs: copy384.copyOverRawMedianNs
        )

        let gateObservedFraction = hnsw.observedPagedOverheadFractionPercent
        let isGo = gateObservedFraction > opts.thresholdPercent
        let decision = PagedAccessDecision(
            thresholdDeclared: thresholdText,
            thresholdPercent: opts.thresholdPercent,
            gateMetric: "hnsw.observedPagedOverheadFractionPercent",
            gateObservedPagedOverheadFractionPercent: gateObservedFraction,
            decisionScope: "paged HNSW search only; retained-originals rerank remains copy-on-access",
            decision: isGo ? "GO" : "NO-GO",
            reasoning: isGo
                ? "The observed HNSW paged-overhead fraction exceeds the pre-declared threshold; design work for scoped zero-copy is justified for paged HNSW search only."
                : "The observed HNSW paged-overhead fraction stays at or below the pre-declared threshold; zero-copy does not clear the complexity bar.",
            reopenConditions: [
                "A future warm benchmark on target hardware shows observed HNSW paged-overhead below the threshold for production query mixes.",
                "Re-measure on target consumer hardware before any implementation.",
                "A new in-search measurement shows retained-originals rerank paged-overhead above the threshold.",
                "A scoped pointer API can prove actor-remap safety and bit-identical parity with lower complexity than ADR-013/014 assumed."
            ]
        )

        let report = PagedAccessBenchReport(
            schemaVersion: 1,
            kind: "paged-access-zero-copy-decision",
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            platform: PlatformProbe.current(),
            seed: Int(bitPattern: UInt(truncatingIfNeeded: opts.seed)),
            reps: opts.reps,
            warmupReps: opts.warmup,
            runStartedAt: ISO8601DateFormatter().string(from: started),
            runDurationSeconds: secondsSince(runStart),
            notes: opts.notes,
            copyIsolation: copyRows,
            hnsw: hnsw,
            rerank: rerank,
            decision: decision
        )

        try writeJSON(report, to: URL(fileURLWithPath: opts.outputPath))
        log("[paged-access-bench] wrote \(opts.outputPath)")
        printSummary(report)
    }

    private static func runCopyIsolation(opts: Options, workDir: URL) throws -> [CopyIsolationMeasurement] {
        var rows: [CopyIsolationMeasurement] = []
        for dim in opts.isolationDimensions {
            let url = workDir.appendingPathComponent("copy-isolation-\(opts.count)x\(dim)-seed\(opts.seed).bin")
            try writeFloatFixture(url: url, count: opts.count, dimension: dim, seed: opts.seed &+ UInt64(dim))
            let mapped = try LocalMappedFloatVectors(url: url, count: opts.count, dimension: dim)
            let indices = makeAccessIndices(count: opts.accessesPerRep, modulo: opts.count, seed: opts.seed ^ UInt64(dim))

            for _ in 0..<opts.warmup {
                _ = timeCopyAccess(mapped, indices: indices)
                _ = timeRawAccess(mapped, indices: indices)
            }

            var copyReps: [Double] = []
            var rawReps: [Double] = []
            var checksum: Float = 0
            for rep in 0..<opts.reps {
                if rep.isMultiple(of: 2) {
                    let raw = timeRawAccess(mapped, indices: indices)
                    let copy = timeCopyAccess(mapped, indices: indices)
                    rawReps.append(raw.nsPerAccess)
                    copyReps.append(copy.nsPerAccess)
                    checksum += raw.checksum + copy.checksum
                } else {
                    let copy = timeCopyAccess(mapped, indices: indices)
                    let raw = timeRawAccess(mapped, indices: indices)
                    rawReps.append(raw.nsPerAccess)
                    copyReps.append(copy.nsPerAccess)
                    checksum += raw.checksum + copy.checksum
                }
            }

            let copy = distribution(copyReps)
            let raw = distribution(rawReps)
            let delta = max(0, copy.medianNs - raw.medianNs)
            let spread = copy.medianNs > 0 ? (copy.spreadPercent + raw.spreadPercent) / 2 : 0
            rows.append(CopyIsolationMeasurement(
                dimension: dim,
                vectorCount: opts.count,
                accessesPerRep: opts.accessesPerRep,
                copy: copy,
                rawUnsafeRead: raw,
                copyOverRawMedianNs: delta,
                copyOverRawSpreadPercent: spread,
                checksum: Double(checksum)
            ))
            log(String(
                format: "[paged-access-bench] copy d=%d copy=%.1fns raw=%.1fns delta=%.1fns",
                dim, copy.medianNs, raw.medianNs, delta
            ))
        }
        return rows
    }

    private static func buildHNSWFixture(
        vectors: [Vector], opts: Options, base: URL, wal: URL
    ) async throws {
        log("[paged-access-bench] building HNSW fixture \(opts.count)x\(opts.dimension)")
        let config = HNSWConfiguration(
            m: opts.m, efConstruction: opts.efConstruction, efSearch: opts.efSearch,
            autoCompactionThreshold: nil, levelSeed: opts.seed
        )
        let index = HNSWIndex(dimension: opts.dimension, metric: EuclideanDistance(), config: config)
        for (i, vector) in vectors.enumerated() {
            try await index.add(vector, id: deterministicUUID(i))
            if i > 0 && i.isMultiple(of: 10_000) {
                log("[paged-access-bench]   HNSW inserted \(i)")
            }
        }
        try await index.checkpoint(baseURL: base, walURL: wal)
        await index.closeJournal()
    }

    private static func loadOrBuildVectorReadCounts(
        vectors: [Vector], queries: [Vector], opts: Options, path: URL
    ) async throws -> [Int] {
        if opts.reuseFixtures,
           let cached = try? readJSON(CachedVectorReadCounts.self, from: path),
           cached.seed == opts.seed,
           cached.vectorCount == opts.count,
           cached.dimension == opts.dimension,
           cached.m == opts.m,
           cached.efConstruction == opts.efConstruction,
           cached.efSearch == opts.efSearch,
           cached.k == opts.k,
           cached.queryCount == opts.queryCount {
            log("[paged-access-bench] reusing vector-read counts \(path.path)")
            return cached.counts
        }

        var source = vectors
        if source.isEmpty {
            source = makeVectors(count: opts.count, dimension: opts.dimension, seed: opts.seed &+ 0x5EED_1000)
        }
        log("[paged-access-bench] building counting HNSW fixture for vector-read estimates")
        let counter = CountingEuclideanMetric()
        let config = HNSWConfiguration(
            m: opts.m, efConstruction: opts.efConstruction, efSearch: opts.efSearch,
            autoCompactionThreshold: nil, levelSeed: opts.seed
        )
        let index = HNSWIndex(dimension: opts.dimension, metric: counter, config: config)
        for (i, vector) in source.enumerated() {
            try await index.add(vector, id: deterministicUUID(i))
            if i > 0 && i.isMultiple(of: 10_000) {
                log("[paged-access-bench]   counting HNSW inserted \(i)")
            }
        }

        var counts: [Int] = []
        counts.reserveCapacity(queries.count)
        for query in queries {
            counter.reset()
            _ = await index.search(query: query, k: opts.k)
            counts.append(counter.count)
        }
        let cached = CachedVectorReadCounts(
            seed: opts.seed,
            vectorCount: opts.count,
            dimension: opts.dimension,
            m: opts.m,
            efConstruction: opts.efConstruction,
            efSearch: opts.efSearch,
            k: opts.k,
            queryCount: opts.queryCount,
            counts: counts
        )
        try writeJSON(cached, to: path)
        return counts
    }

    private static func buildPQFixture(vectors: [Vector], opts: Options, url: URL) async throws {
        log("[paged-access-bench] building retained PQ fixture \(opts.count)x\(opts.dimension)")
        let ids = (0..<vectors.count).map { deterministicUUID($0) }
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: opts.dimension,
            hnswConfig: HNSWConfiguration(
                m: opts.m, efConstruction: opts.efConstruction, efSearch: opts.efSearch,
                autoCompactionThreshold: nil, levelSeed: opts.seed
            ),
            pqConfig: PQConfiguration(
                subspaceCount: opts.pqSubspaces,
                trainingIterations: opts.pqTrainingIterations,
                seed: opts.seed
            ),
            retainOriginals: true
        )
        try await index.save(to: url, layout: .pagedV3)
    }

    private static func runHNSWImpact(
        opts: Options,
        queries: [Vector],
        base: URL,
        wal: URL,
        vectorReadCounts: [Int],
        copyOverRawNs: Double
    ) async throws -> HNSWImpactMeasurement {
        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)

        for _ in 0..<opts.warmup {
            _ = await timeHNSWSearch(index: resident, queries: queries, k: opts.k)
            _ = await timeHNSWSearch(index: paged, queries: queries, k: opts.k)
        }

        var residentReps: [Double] = []
        var pagedReps: [Double] = []
        var checksum = 0
        for rep in 0..<opts.reps {
            if rep.isMultiple(of: 2) {
                let r = await timeHNSWSearch(index: resident, queries: queries, k: opts.k)
                let p = await timeHNSWSearch(index: paged, queries: queries, k: opts.k)
                residentReps.append(r.nsPerQuery)
                pagedReps.append(p.nsPerQuery)
                checksum &+= r.checksum &+ p.checksum
            } else {
                let p = await timeHNSWSearch(index: paged, queries: queries, k: opts.k)
                let r = await timeHNSWSearch(index: resident, queries: queries, k: opts.k)
                residentReps.append(r.nsPerQuery)
                pagedReps.append(p.nsPerQuery)
                checksum &+= r.checksum &+ p.checksum
            }
        }

        await resident.closeJournal()
        await paged.closeJournal()

        let residentDist = distribution(residentReps)
        let pagedDist = distribution(pagedReps)
        let observedDelta = max(0, pagedDist.medianNs - residentDist.medianNs)
        let observedFraction = pagedDist.medianNs > 0 ? observedDelta / pagedDist.medianNs * 100 : 0
        let counts = countDistribution(vectorReadCounts)
        let isolationExtrapolationNs = counts.mean * copyOverRawNs
        let isolationExtrapolationPercent = pagedDist.medianNs > 0
            ? isolationExtrapolationNs / pagedDist.medianNs * 100
            : 0

        log(String(
            format: "[paged-access-bench] HNSW resident=%.0fns paged=%.0fns observed=%.2f%% isolation-diagnostic=%.2f%%",
            residentDist.medianNs, pagedDist.medianNs, observedFraction, isolationExtrapolationPercent
        ))

        return HNSWImpactMeasurement(
            vectorCount: opts.count,
            dimension: opts.dimension,
            m: opts.m,
            efConstruction: opts.efConstruction,
            efSearch: opts.efSearch,
            k: opts.k,
            queryCount: opts.queryCount,
            resident: residentDist,
            paged: pagedDist,
            observedPagedMinusResidentNs: observedDelta,
            observedPagedOverheadFractionPercent: observedFraction,
            vectorReadsPerQuery: counts,
            isolationExtrapolationNsPerQueryNonTransferable: isolationExtrapolationNs,
            isolationExtrapolationPercentNonTransferable: isolationExtrapolationPercent,
            checksum: checksum
        )
    }

    private static func runRerankImpact(
        opts: Options,
        queries: [Vector],
        pqURL: URL,
        copyOverRawNs: Double
    ) async throws -> RerankImpactMeasurement {
        let resident = try QuantizedHNSWIndex.load(from: pqURL, mode: .resident)
        let paged = try QuantizedHNSWIndex.load(from: pqURL, mode: .paged)

        for _ in 0..<opts.warmup {
            _ = try await timeRerankSearch(index: resident, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth)
            _ = try await timeRerankSearch(index: paged, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth)
        }

        var residentReps: [Double] = []
        var pagedReps: [Double] = []
        var checksum = 0
        for rep in 0..<opts.reps {
            if rep.isMultiple(of: 2) {
                let r = try await timeRerankSearch(
                    index: resident, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth
                )
                let p = try await timeRerankSearch(
                    index: paged, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth
                )
                residentReps.append(r.nsPerQuery)
                pagedReps.append(p.nsPerQuery)
                checksum &+= r.checksum &+ p.checksum
            } else {
                let p = try await timeRerankSearch(
                    index: paged, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth
                )
                let r = try await timeRerankSearch(
                    index: resident, queries: queries, k: opts.k, rerankDepth: opts.rerankDepth
                )
                residentReps.append(r.nsPerQuery)
                pagedReps.append(p.nsPerQuery)
                checksum &+= r.checksum &+ p.checksum
            }
        }

        let residentDist = distribution(residentReps)
        let pagedDist = distribution(pagedReps)
        let observedDelta = max(0, pagedDist.medianNs - residentDist.medianNs)
        let observedFraction = pagedDist.medianNs > 0 ? observedDelta / pagedDist.medianNs * 100 : 0
        let isolationExtrapolationNs = Double(opts.rerankDepth) * copyOverRawNs
        let isolationExtrapolationPercent = pagedDist.medianNs > 0
            ? isolationExtrapolationNs / pagedDist.medianNs * 100
            : 0

        log(String(
            format: "[paged-access-bench] rerank resident=%.0fns paged=%.0fns observed=%.2f%% isolation-diagnostic=%.2f%%",
            residentDist.medianNs, pagedDist.medianNs, observedFraction, isolationExtrapolationPercent
        ))

        return RerankImpactMeasurement(
            vectorCount: opts.count,
            dimension: opts.dimension,
            m: opts.m,
            efConstruction: opts.efConstruction,
            efSearch: opts.efSearch,
            k: opts.k,
            queryCount: opts.queryCount,
            rerankDepth: opts.rerankDepth,
            pqSubspaces: opts.pqSubspaces,
            pqTrainingIterations: opts.pqTrainingIterations,
            resident: residentDist,
            paged: pagedDist,
            observedPagedMinusResidentNs: observedDelta,
            observedPagedOverheadFractionPercent: observedFraction,
            isolationExtrapolationNsPerQueryNonTransferable: isolationExtrapolationNs,
            isolationExtrapolationPercentNonTransferable: isolationExtrapolationPercent,
            checksum: checksum
        )
    }

    @inline(never)
    private static func timeCopyAccess(
        _ mapped: LocalMappedFloatVectors, indices: [Int]
    ) -> (nsPerAccess: Double, checksum: Float) {
        let start = DispatchTime.now().uptimeNanoseconds
        var checksum: Float = 0
        for node in indices {
            let vector = mapped.vector(at: node)
            checksum += vector[0]
            checksum += vector[mapped.dimension - 1]
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
        consumeBenchFloat(checksum)
        return (elapsed / Double(indices.count), checksum)
    }

    @inline(never)
    private static func timeRawAccess(
        _ mapped: LocalMappedFloatVectors, indices: [Int]
    ) -> (nsPerAccess: Double, checksum: Float) {
        let start = DispatchTime.now().uptimeNanoseconds
        var checksum: Float = 0
        for node in indices {
            mapped.withUnsafeVector(at: node) { buffer in
                checksum += buffer[0]
                checksum += buffer[mapped.dimension - 1]
            }
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
        consumeBenchFloat(checksum)
        return (elapsed / Double(indices.count), checksum)
    }

    @inline(never)
    private static func timeHNSWSearch(
        index: HNSWIndex, queries: [Vector], k: Int
    ) async -> (nsPerQuery: Double, checksum: Int) {
        let start = DispatchTime.now().uptimeNanoseconds
        var checksum = 0
        for query in queries {
            let results = await index.search(query: query, k: k)
            checksum &+= resultChecksum(results)
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
        return (elapsed / Double(queries.count), checksum)
    }

    @inline(never)
    private static func timeRerankSearch(
        index: QuantizedHNSWIndex, queries: [Vector], k: Int, rerankDepth: Int
    ) async throws -> (nsPerQuery: Double, checksum: Int) {
        let start = DispatchTime.now().uptimeNanoseconds
        var checksum = 0
        for query in queries {
            let results = try await index.search(query: query, k: k, rerankDepth: rerankDepth)
            checksum &+= resultChecksum(results)
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start)
        return (elapsed / Double(queries.count), checksum)
    }

    private static func resultChecksum(_ results: [SearchResult]) -> Int {
        var checksum = results.count
        for result in results {
            checksum &+= Int(result.distance.bitPattern & 0xFFFF)
        }
        return checksum
    }

    private static func writeFloatFixture(url: URL, count: Int, dimension: Int, seed: UInt64) throws {
        var rng = BenchSeededRandom(seed: seed)
        let total = count * dimension
        var data = Data(count: total * MemoryLayout<Float>.size)
        try data.withUnsafeMutableBytes { rawBuffer in
            let floats = rawBuffer.bindMemory(to: Float.self)
            guard floats.count == total else {
                throw benchFail("could not bind float fixture buffer")
            }
            for i in 0..<total {
                floats[i] = Float.random(in: -1...1, using: &rng)
            }
        }
        try data.write(to: url, options: .atomic)
    }

    private static func makeVectors(count: Int, dimension: Int, seed: UInt64) -> [Vector] {
        var rng = BenchSeededRandom(seed: seed)
        var vectors: [Vector] = []
        vectors.reserveCapacity(count)
        for _ in 0..<count {
            var components: [Float] = []
            components.reserveCapacity(dimension)
            for _ in 0..<dimension {
                components.append(Float.random(in: -1...1, using: &rng))
            }
            vectors.append(Vector(components))
        }
        return vectors
    }

    private static func makeAccessIndices(count: Int, modulo: Int, seed: UInt64) -> [Int] {
        var rng = BenchSeededRandom(seed: seed)
        var indices: [Int] = []
        indices.reserveCapacity(count)
        for _ in 0..<count {
            indices.append(Int(rng.next() % UInt64(modulo)))
        }
        return indices
    }

    private static func deterministicUUID(_ i: Int) -> UUID {
        var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                     UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        withUnsafeMutableBytes(of: &bytes) { raw in
            raw.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self)
        }
        return UUID(uuid: bytes)
    }

    private static func distribution(_ values: [Double]) -> TimingDistribution {
        let sorted = values.sorted()
        guard let minValue = sorted.first, let maxValue = sorted.last else {
            return TimingDistribution(medianNs: 0, minNs: 0, maxNs: 0, spreadPercent: 0, repsNs: [])
        }
        let medianValue = median(sorted)
        let spread = medianValue > 0 ? (maxValue - minValue) / medianValue * 100 : 0
        return TimingDistribution(
            medianNs: medianValue,
            minNs: minValue,
            maxNs: maxValue,
            spreadPercent: spread,
            repsNs: values
        )
    }

    private static func countDistribution(_ values: [Int]) -> CountDistribution {
        let sorted = values.sorted()
        guard let minValue = sorted.first, let maxValue = sorted.last else {
            return CountDistribution(mean: 0, median: 0, min: 0, max: 0)
        }
        let mean = Double(values.reduce(0, +)) / Double(values.count)
        let med: Double
        if sorted.count.isMultiple(of: 2) {
            med = Double(sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        } else {
            med = Double(sorted[sorted.count / 2])
        }
        return CountDistribution(mean: mean, median: med, min: minValue, max: maxValue)
    }

    private static func median(_ sorted: [Double]) -> Double {
        if sorted.count.isMultiple(of: 2) {
            return (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        }
        return sorted[sorted.count / 2]
    }

    private static func fixturesExist(_ paths: FixturePaths) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: paths.hnswBase.path)
            && fm.fileExists(atPath: paths.hnswWal.path)
            && fm.fileExists(atPath: paths.pq.path)
    }

    private struct FixturePaths {
        let hnswBase: URL
        let hnswWal: URL
        let pq: URL
        let counts: URL
    }

    private static func fixturePaths(opts: Options, workDir: URL) -> FixturePaths {
        let stem = "fixture-\(opts.count)x\(opts.dimension)-m\(opts.m)-efc\(opts.efConstruction)-ef\(opts.efSearch)-seed\(opts.seed)"
        return FixturePaths(
            hnswBase: workDir.appendingPathComponent("\(stem).pxkt"),
            hnswWal: workDir.appendingPathComponent("\(stem).pxwal"),
            pq: workDir.appendingPathComponent("\(stem)-pq\(opts.pqSubspaces)-it\(opts.pqTrainingIterations).pqhw"),
            counts: workDir.appendingPathComponent("\(stem)-vector-read-counts.json")
        )
    }

    private static func writeJSON<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        try data.write(to: url, options: .atomic)
    }

    private static func readJSON<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        return try JSONDecoder().decode(type, from: data)
    }

    private static func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }

    private static func printSummary(_ report: PagedAccessBenchReport) {
        let copy384 = report.copyIsolation.first { $0.dimension == report.hnsw.dimension }
        let copyText = copy384.map {
            String(format: "%.1fns", $0.copyOverRawMedianNs)
        } ?? "n/a"
        print("""
        -- paged-access-bench ------------------------------------------------
        threshold       : \(report.decision.thresholdDeclared)
        copy d=\(report.hnsw.dimension)      : \(copyText) copy-over-raw median
        HNSW observed   : \(String(format: "%.2f", report.hnsw.observedPagedOverheadFractionPercent))%  isolation diagnostic: \(String(format: "%.2f", report.hnsw.isolationExtrapolationPercentNonTransferable))%
        rerank observed : \(String(format: "%.2f", report.rerank.observedPagedOverheadFractionPercent))%  isolation diagnostic: \(String(format: "%.2f", report.rerank.isolationExtrapolationPercentNonTransferable))%
        decision gate   : \(report.decision.gateMetric)=\(String(format: "%.2f", report.decision.gateObservedPagedOverheadFractionPercent))% > \(String(format: "%.2f", report.decision.thresholdPercent))%
        decision        : \(report.decision.decision)
        ---------------------------------------------------------------------
        """)
    }
}
