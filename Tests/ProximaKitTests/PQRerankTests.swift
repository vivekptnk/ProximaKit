// PQRerankTests.swift
// ProximaKitTests
//
// Full-precision reranking for QuantizedHNSWIndex (ADR-012):
// - recall recovery on the CHA-91-style clustered fixture (asserted)
// - rerank disabled == pure-ADC behavior, byte-identical results
// - rerankDepth without retained originals throws (typed, fail-fast)
// - PQHW v2 persistence: roundtrip with/without originals, v1 (N-1)
//   backward read, corruption-matrix additions for the new section
// - remove -> save -> load keeps originals slot-aligned through compaction
//
// All data-dependent assertions use SeededRandom + levelSeed; the residual
// run-to-run variance is PQ k-means centroid initialization (same situation
// as PQBenchmarkTests), so thresholds are pinned with margin below the
// observed band.

import XCTest
@testable import ProximaKit

final class PQRerankTests: XCTestCase {

    // ── Helpers ──────────────────────────────────────────────────────

    private func tempURL(_ ext: String = "qhnsw") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("pqrerank-\(UUID().uuidString)")
            .appendingPathExtension(ext)
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    /// Overwrites 4 bytes at `offset` with a little-endian UInt32.
    private func patch(_ url: URL, at offset: Int, with value: UInt32) throws {
        var data = try Data(contentsOf: url)
        withUnsafeBytes(of: value.littleEndian) { bytes in
            for (i, byte) in bytes.enumerated() {
                data[offset + i] = byte
            }
        }
        try data.write(to: url)
    }

    /// Reads the little-endian UInt32 at `offset`.
    private func readField(_ url: URL, at offset: Int) throws -> UInt32 {
        try Data(contentsOf: url).loadLE(UInt32.self, at: offset)
    }

    private func assertThrowsPersistenceError<T>(
        _ expression: @autoclosure () throws -> T,
        _ message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try expression(), message, file: file, line: line) { error in
            XCTAssertTrue(error is PersistenceError,
                          "\(message): expected PersistenceError, got \(error)",
                          file: file, line: line)
        }
    }

    /// Seeded clustered fixture, same shape as the CHA-91 fixture in
    /// `PQBenchmarkTests.testQuantizedHNSWMemoryVsRecall`.
    private func clusteredFixture(
        count: Int, dimension: Int, clusters: Int, seed: UInt64
    ) -> (vectors: [Vector], ids: [UUID]) {
        var rng = SeededRandom(seed: seed)
        let clusterSize = count / clusters
        var vectors = [Vector]()
        vectors.reserveCapacity(count)
        for _ in 0..<clusters {
            let center = (0..<dimension).map { _ in Float.random(in: -5...5, using: &rng) }
            for _ in 0..<clusterSize {
                vectors.append(Vector((0..<dimension).map { d in
                    center[d] + Float.random(in: -0.5...0.5, using: &rng)
                }))
            }
        }
        let ids = (0..<vectors.count).map { _ in UUID() }
        return (vectors, ids)
    }

    /// Identical-results check: same ids in the same order, bit-equal distances.
    private func assertByteIdentical(
        _ a: [SearchResult], _ b: [SearchResult],
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.map(\.id), b.map(\.id), message, file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.distance, y.distance,
                           "\(message): distances must be bit-identical",
                           file: file, line: line)
        }
    }

    /// A small deterministic retained index built from explicit components
    /// (no k-means), for persistence/corruption tests with exact offsets.
    /// dim 4, M 2 => codebooks are 2 * 256 * 2 floats; codes 2 bytes/node.
    private func makeTinyRetainedIndex() -> QuantizedHNSWIndex {
        let config = PQConfiguration(subspaceCount: 2)
        let codebook = [Float](repeating: 0.5, count: 256 * 2)
        let quantizer = ProductQuantizer(
            dimension: 4, config: config, codebooks: [codebook, codebook]
        )
        let ids = [UUID(), UUID()]
        return QuantizedHNSWIndex(
            dimension: 4,
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            quantizer: quantizer,
            layers: [[[1], [0]]],
            nodeLevels: [0, 0],
            entryPointNode: 0,
            maxLevel: 0,
            codes: [[1, 2], [3, 4]],
            nodeToUUID: ids,
            uuidToNode: [ids[0]: 0, ids[1]: 1],
            metadata: [nil, nil],
            originals: [Vector([1, 2, 3, 4]), Vector([5, 6, 7, 8])]
        )
    }

    // ── (a) Recall Recovery ──────────────────────────────────────────

    func testRerankRecallRecoveryOnClusteredFixture() async throws {
        let dim = 64
        let n = 1000
        let k = 10
        let numQueries = 30

        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 10, seed: 0xCA11_AB1E_5EED_0012
        )

        // levelSeed pins the graph; PQ k-means init is the residual variance.
        let hnswConfig = HNSWConfiguration(
            m: 16, efConstruction: 200, efSearch: 100, levelSeed: 0x5EED_0012
        )

        let qIndex = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: hnswConfig,
            pqConfig: PQConfiguration(subspaceCount: 16, trainingIterations: 15),
            retainOriginals: true
        )

        let bfIndex = BruteForceIndex(dimension: dim, metric: EuclideanDistance())
        for i in 0..<n {
            try await bfIndex.add(vectors[i], id: ids[i])
        }

        var plainRecall: Float = 0
        var rerankedRecall: Float = 0
        for q in 0..<numQueries {
            let query = vectors[q]
            let exact = await bfIndex.search(query: query, k: k)
            let gt = Set(exact.map(\.id))

            let plain = try await qIndex.search(
                query: query, k: k, efSearch: 200, rerankDepth: 0)
            plainRecall += Float(gt.intersection(Set(plain.map(\.id))).count) / Float(k)

            let reranked = try await qIndex.search(
                query: query, k: k, efSearch: 200, rerankDepth: 4 * k)
            rerankedRecall += Float(gt.intersection(Set(reranked.map(\.id))).count) / Float(k)
        }
        plainRecall /= Float(numQueries)
        rerankedRecall /= Float(numQueries)

        print("\n=== PQ Rerank Recall (CHA-91-style fixture, ADR-012) ===")
        print("Plain ADC recall@\(k):  \(String(format: "%.3f", plainRecall))")
        print("Reranked recall@\(k):   \(String(format: "%.3f", rerankedRecall)) (depth \(4 * k))")

        // Plain-ADC recall on this fixture mirrors the CHA-91 band measured
        // in PQBenchmarkTests (0.667-0.717 there); reranking at depth 4k
        // recovers essentially all of the ADC loss. Observed over 5 local
        // runs of THIS fixture: plain 0.667-0.730, reranked 0.990-1.000
        // (within-run margin 0.27-0.33). Thresholds sit with margin below
        // the observed band; the data and graph are pinned (SeededRandom +
        // levelSeed), so PQ k-means init is the only variance source.
        XCTAssertGreaterThanOrEqual(rerankedRecall, plainRecall + 0.15,
            "reranked recall@10 (\(rerankedRecall)) must beat plain ADC "
            + "(\(plainRecall)) by a clear margin")
        XCTAssertGreaterThanOrEqual(rerankedRecall, 0.90,
            "reranked recall@10 must be >= 0.90 absolute (got \(rerankedRecall))")
    }

    // ── (b) Rerank Disabled == Pure ADC, Byte-Identical ──────────────

    func testRerankDisabledIsByteIdenticalToPureADC() async throws {
        let dim = 32
        let n = 300
        let k = 10

        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 6, seed: 0xCA11_AB1E_5EED_0013
        )
        let retained = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50,
                                          levelSeed: 0x5EED_0013),
            pqConfig: PQConfiguration(subspaceCount: 8, trainingIterations: 5),
            retainOriginals: true
        )

        // Twin with identical graph/codes but no originals: this IS the
        // previous (v1.5.0) behavior — the pure-ADC search path over the
        // exact same state.
        let twin = await QuantizedHNSWIndex(
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50),
            quantizer: retained.quantizer,
            layers: retained.layers,
            nodeLevels: retained.nodeLevels,
            entryPointNode: retained.entryPointNode,
            maxLevel: retained.maxLevel,
            codes: retained.codes,
            nodeToUUID: retained.nodeToUUID,
            uuidToNode: retained.uuidToNode,
            metadata: retained.metadata,
            originals: nil
        )

        for q in 0..<10 {
            let query = vectors[q * 7]
            let previous = await twin.search(query: query, k: k, efSearch: 80)

            // rerankDepth: 0 and nil both disable reranking exactly.
            let depthZero = try await retained.search(
                query: query, k: k, efSearch: 80, rerankDepth: 0)
            assertByteIdentical(depthZero, previous,
                "rerankDepth: 0 must reproduce previous pure-ADC behavior")

            let depthNil = try await retained.search(
                query: query, k: k, efSearch: 80, rerankDepth: nil)
            assertByteIdentical(depthNil, previous,
                "rerankDepth: nil must reproduce previous pure-ADC behavior")

            // The legacy entry point auto-reranks at 4*k when originals are
            // retained (ADR-012), and matches the explicit depth.
            let auto = await retained.search(query: query, k: k, efSearch: 80)
            let explicit = try await retained.search(
                query: query, k: k, efSearch: 80, rerankDepth: 4 * k)
            assertByteIdentical(auto, explicit,
                "legacy search on a retained index must auto-rerank at 4*k")
        }

        // Non-retained build: legacy and rerankDepth-0 paths agree trivially.
        let plainBuilt = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50,
                                          levelSeed: 0x5EED_0013),
            pqConfig: PQConfiguration(subspaceCount: 8, trainingIterations: 5)
        )
        let legacy = await plainBuilt.search(query: vectors[0], k: k)
        let zero = try await plainBuilt.search(query: vectors[0], k: k, rerankDepth: 0)
        assertByteIdentical(zero, legacy,
            "rerankDepth: 0 on a non-retained index must equal legacy search")
    }

    // ── Rerank Without Retention: Typed Throw ────────────────────────

    func testRerankWithoutRetentionThrowsTypedError() async throws {
        let (vectors, ids) = clusteredFixture(
            count: 100, dimension: 16, clusters: 4, seed: 0xCA11_AB1E_5EED_0014
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30,
                                          levelSeed: 0x5EED_0014),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        do {
            _ = try await index.search(query: vectors[0], k: 5, rerankDepth: 20)
            XCTFail("rerankDepth > 0 without retained originals must throw")
        } catch let error as QuantizedIndexError {
            XCTAssertEqual(error, .originalsNotRetained)
        }

        // nil / 0 / negative depths are "off", never a throw.
        for depth in [nil, 0, -3] as [Int?] {
            let results = try await index.search(
                query: vectors[0], k: 5, rerankDepth: depth)
            XCTAssertFalse(results.isEmpty,
                "rerankDepth \(String(describing: depth)) must fall back to pure ADC")
        }
    }

    // ── Memory Accounting (ADR-012 trade stated in numbers) ──────────

    func testRetentionMemoryAccounting() async throws {
        let dim = 64
        let n = 200
        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 4, seed: 0xCA11_AB1E_5EED_0015
        )
        let config = HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30,
                                       levelSeed: 0x5EED_0015)

        let retained = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: config,
            pqConfig: PQConfiguration(subspaceCount: 16, trainingIterations: 5),
            retainOriginals: true
        )
        let retainedFlag = await retained.retainsOriginals
        let originalBytes = await retained.originalStorageBytes
        let ratio = await retained.memorySavingsRatio
        XCTAssertTrue(retainedFlag)
        XCTAssertEqual(originalBytes, n * dim * 4,
            "retention costs the full 4d bytes/vector again")
        XCTAssertLessThan(ratio, 1.0,
            "with originals retained the memory story is gone (ADR-012)")

        let plain = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: config,
            pqConfig: PQConfiguration(subspaceCount: 16, trainingIterations: 5)
        )
        let plainFlag = await plain.retainsOriginals
        let plainOriginalBytes = await plain.originalStorageBytes
        let plainRatio = await plain.memorySavingsRatio
        XCTAssertFalse(plainFlag)
        XCTAssertEqual(plainOriginalBytes, 0)
        XCTAssertEqual(plainRatio, 16.0, accuracy: 0.01,
            "non-retained ratio is unchanged: dim*4 / M = 64*4/16")
    }

    // ── Reranked Distances Are Exact (and filter-safe) ───────────────

    func testRerankedDistancesAreExactAndRespectFilter() async throws {
        let dim = 16
        let n = 150
        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 3, seed: 0xCA11_AB1E_5EED_0016
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50,
                                          levelSeed: 0x5EED_0016),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5),
            retainOriginals: true
        )

        let vectorByID = Dictionary(uniqueKeysWithValues: zip(ids, vectors))
        let allowed = Set(ids.prefix(60))
        let metric = EuclideanDistance()
        let query = vectors[5]

        let results = try await index.search(
            query: query, k: 10, efSearch: 100, rerankDepth: 40
        ) { allowed.contains($0) }

        XCTAssertFalse(results.isEmpty)
        for result in results {
            XCTAssertTrue(allowed.contains(result.id),
                "filter must hold on the rerank path")
            // Reranked distances are exact Euclidean against the original.
            let expected = metric.distance(query, vectorByID[result.id]!)
            XCTAssertEqual(result.distance, expected,
                "reranked distance must be the exact L2 distance")
        }
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i - 1].distance, results[i].distance)
        }
    }

    // ── (c) Persistence v2 Roundtrip ─────────────────────────────────

    func testPersistenceV2RoundtripWithOriginals() async throws {
        let dim = 16
        let n = 120
        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 4, seed: 0xCA11_AB1E_5EED_0017
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30,
                                          levelSeed: 0x5EED_0017),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5),
            retainOriginals: true
        )

        let url = tempURL()
        defer { cleanup(url) }
        try await index.save(to: url)

        // v2 on disk, originals flag set.
        XCTAssertEqual(try readField(url, at: 4), 2, "writers always write v2")
        XCTAssertEqual(try readField(url, at: 48), 1, "originals flag must be set")

        let loaded = try QuantizedHNSWIndex.load(from: url)
        let loadedRetains = await loaded.retainsOriginals
        XCTAssertTrue(loadedRetains, "originals must survive the roundtrip")

        let count = await loaded.count
        XCTAssertEqual(count, n)

        let savedOriginals = await index.originals
        let loadedOriginals = await loaded.originals
        XCTAssertEqual(loadedOriginals, savedOriginals,
            "loaded originals must be bit-identical and in slot order")

        // Reranked search must behave identically before and after.
        for q in [0, 17, 64] {
            let before = try await index.search(
                query: vectors[q], k: 10, efSearch: 60, rerankDepth: 40)
            let after = try await loaded.search(
                query: vectors[q], k: 10, efSearch: 60, rerankDepth: 40)
            assertByteIdentical(after, before,
                "reranked search must survive the v2 roundtrip")
        }
    }

    func testPersistenceV2RoundtripWithoutOriginals() async throws {
        let dim = 16
        let n = 120
        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 4, seed: 0xCA11_AB1E_5EED_0018
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30,
                                          levelSeed: 0x5EED_0018),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let url = tempURL()
        defer { cleanup(url) }
        try await index.save(to: url)

        XCTAssertEqual(try readField(url, at: 4), 2, "writers always write v2")
        XCTAssertEqual(try readField(url, at: 48), 0, "originals flag must be clear")

        let loaded = try QuantizedHNSWIndex.load(from: url)
        let loadedRetains = await loaded.retainsOriginals
        XCTAssertFalse(loadedRetains)

        // Rerank against the loaded no-originals index still fails fast.
        do {
            _ = try await loaded.search(query: vectors[0], k: 5, rerankDepth: 20)
            XCTFail("loaded no-originals index must throw on rerank request")
        } catch let error as QuantizedIndexError {
            XCTAssertEqual(error, .originalsNotRetained)
        }

        let before = await index.search(query: vectors[3], k: 10, efSearch: 60)
        let after = await loaded.search(query: vectors[3], k: 10, efSearch: 60)
        assertByteIdentical(after, before, "pure-ADC search must survive the roundtrip")
    }

    func testOriginalsSectionSizeIsExactlyTheTrailer() async throws {
        // Same state saved with and without originals must differ by exactly
        // nodeCount * dim * 4 trailing bytes (the v2 originals section).
        let retained = makeTinyRetainedIndex()
        let withURL = tempURL()
        defer { cleanup(withURL) }
        try await retained.save(to: withURL)

        let twin = await QuantizedHNSWIndex(
            dimension: 4,
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            quantizer: retained.quantizer,
            layers: retained.layers,
            nodeLevels: retained.nodeLevels,
            entryPointNode: retained.entryPointNode,
            maxLevel: retained.maxLevel,
            codes: retained.codes,
            nodeToUUID: retained.nodeToUUID,
            uuidToNode: retained.uuidToNode,
            metadata: retained.metadata,
            originals: nil
        )
        let withoutURL = tempURL()
        defer { cleanup(withoutURL) }
        try await twin.save(to: withoutURL)

        let withSize = try Data(contentsOf: withURL).count
        let withoutSize = try Data(contentsOf: withoutURL).count
        XCTAssertEqual(withSize, withoutSize + 2 * 4 * 4,
            "originals section is nodeCount(2) * dim(4) * 4 bytes, nothing else")
    }

    // ── (d) v1 (N-1) Backward Read ───────────────────────────────────

    func testVersion1FileStillLoads() async throws {
        let dim = 16
        let n = 80
        let (vectors, ids) = clusteredFixture(
            count: n, dimension: dim, clusters: 4, seed: 0xCA11_AB1E_5EED_0019
        )
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: dim,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30,
                                          levelSeed: 0x5EED_0019),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5)
        )

        let url = tempURL()
        defer { cleanup(url) }
        try await index.save(to: url)

        // A v2 file saved without originals has zeros in the flag bytes, so
        // patching the version down yields a byte-faithful v1 file.
        XCTAssertEqual(try readField(url, at: 48), 0)
        let v2Loaded = try QuantizedHNSWIndex.load(from: url)
        try patch(url, at: 4, with: 1)

        let v1Loaded = try QuantizedHNSWIndex.load(from: url)
        let retains = await v1Loaded.retainsOriginals
        XCTAssertFalse(retains, "v1 files load with retainOriginals == false")

        let count = await v1Loaded.count
        let live = await v1Loaded.liveCount
        XCTAssertEqual(count, n)
        XCTAssertEqual(live, n)

        let fromV2 = await v2Loaded.search(query: vectors[7], k: 10, efSearch: 60)
        let fromV1 = await v1Loaded.search(query: vectors[7], k: 10, efSearch: 60)
        assertByteIdentical(fromV1, fromV2,
            "a v1 file must load to the same searchable state as its v2 twin")

        // Over-version still throws the typed error.
        try patch(url, at: 4, with: 3)
        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url)) { error in
            guard case PersistenceError.unsupportedVersion(3)? = error as? PersistenceError else {
                XCTFail("Expected unsupportedVersion(3), got \(error)"); return
            }
        }
    }

    // ── (e) Corruption Matrix Additions for v2 ───────────────────────

    func testCorruptOriginalsFlagThrows() async throws {
        let url = tempURL()
        defer { cleanup(url) }
        try await makeTinyRetainedIndex().save(to: url)

        try patch(url, at: 48, with: 7)  // flag must be 0 or 1
        assertThrowsPersistenceError(
            try QuantizedHNSWIndex.load(from: url), "originals flag == 7")
    }

    func testOriginalsFlagSetWithoutSectionThrows() async throws {
        // Flag claims originals, but the file ends after the metadata
        // section — the originals read must throw, not run off the end.
        let url = tempURL()
        defer { cleanup(url) }
        let twin = makeTinyRetainedIndex()
        let noOriginals = await QuantizedHNSWIndex(
            dimension: 4,
            hnswConfig: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            quantizer: twin.quantizer,
            layers: twin.layers,
            nodeLevels: twin.nodeLevels,
            entryPointNode: twin.entryPointNode,
            maxLevel: twin.maxLevel,
            codes: twin.codes,
            nodeToUUID: twin.nodeToUUID,
            uuidToNode: twin.uuidToNode,
            metadata: twin.metadata,
            originals: nil
        )
        try await noOriginals.save(to: url)
        try patch(url, at: 48, with: 1)
        assertThrowsPersistenceError(
            try QuantizedHNSWIndex.load(from: url), "flag set but no originals bytes")
    }

    func testTruncationInsideOriginalsSectionThrows() async throws {
        let url = tempURL()
        defer { cleanup(url) }
        try await makeTinyRetainedIndex().save(to: url)
        let fullSize = try Data(contentsOf: url).count
        let originalsBytes = 2 * 4 * 4  // nodeCount * dim * Float32

        // Cuts at the section start, mid-section, and 2 bytes short.
        let cuts = [fullSize - originalsBytes + 1,
                    fullSize - originalsBytes / 2,
                    fullSize - 2]
        for cut in cuts {
            let cutURL = tempURL()
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try QuantizedHNSWIndex.load(from: cutURL),
                "retained file truncated to \(cut)/\(fullSize) bytes must throw")
        }

        // Sanity: the untruncated file still loads with originals intact.
        let loaded = try QuantizedHNSWIndex.load(from: url)
        let originals = await loaded.originals
        XCTAssertEqual(originals, [Vector([1, 2, 3, 4]), Vector([5, 6, 7, 8])])
    }

    // ── (f) remove -> save -> load Keeps Originals Aligned ───────────

    func testRemoveSaveLoadKeepsOriginalsAligned() async throws {
        var rng = SeededRandom(seed: 0x0DEAD_0012)
        var vectors: [Vector] = []
        var ids: [UUID] = []
        for _ in 0..<64 {
            vectors.append(Vector((0..<8).map { _ in Float.random(in: -1...1, using: &rng) }))
            ids.append(UUID())
        }
        let index = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 8,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 100, efSearch: 50,
                                          levelSeed: 0x0012_5EED),
            pqConfig: PQConfiguration(subspaceCount: 2, trainingIterations: 5),
            retainOriginals: true
        )

        let removed = Array(ids.prefix(16))
        for id in removed {
            let ok = await index.remove(id: id)
            XCTAssertTrue(ok)
        }

        let url = tempURL()
        defer { cleanup(url) }
        try await index.save(to: url)
        let loaded = try QuantizedHNSWIndex.load(from: url)

        let count = await loaded.count
        let live = await loaded.liveCount
        XCTAssertEqual(count, 48, "dead slots must be compacted out at save")
        XCTAssertEqual(live, 48)
        let retains = await loaded.retainsOriginals
        XCTAssertTrue(retains)

        // CRITICAL: compaction renumbers slots; every surviving slot's
        // original must still be the vector that id was built with. A save
        // path that wrote originals uncompacted (or in a different order)
        // fails here.
        let vectorByID = Dictionary(uniqueKeysWithValues: zip(ids, vectors))
        let maybeOriginals = await loaded.originals
        let loadedOriginals = try XCTUnwrap(maybeOriginals)
        let loadedNodeToUUID = await loaded.nodeToUUID
        XCTAssertEqual(loadedOriginals.count, 48)
        for (node, uuid) in loadedNodeToUUID.enumerated() {
            XCTAssertEqual(loadedOriginals[node], vectorByID[uuid],
                "slot \(node): original must stay aligned with its id through compaction")
        }

        // Behavioral check: querying a survivor's exact vector with full-depth
        // rerank must put that id first at exact distance zero, and removed
        // ids must never surface.
        let removedSet = Set(removed)
        for probe in [20, 35, 63] {
            let results = try await loaded.search(
                query: vectors[probe], k: 5, efSearch: 64, rerankDepth: 48)
            XCTAssertEqual(results.first?.id, ids[probe],
                "exact-match query must rerank its own id to the top")
            XCTAssertEqual(results.first?.distance, 0,
                "exact rerank distance of the identical vector is 0")
            XCTAssertTrue(results.allSatisfy { !removedSet.contains($0.id) },
                "tombstoned ids must not resurface after save/load")
        }
    }
}
