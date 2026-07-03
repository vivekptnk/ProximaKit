// PersistenceCorruptionTests.swift
// ProximaKitTests
//
// Corruption-handling matrix for every binary codec (HNSW `.pxkt`,
// sparse `.pxbm`, quantized `.qhnsw`, PQ `.pqtt`): truncated files,
// bad magic/version, out-of-bounds graph indices, negative counts.
// Every case must throw `PersistenceError` — never crash.
//
// Also covers the v2 format round-trip of `autoCompactionThreshold`
// (including the `nil` = disabled case) and v1 backward compatibility.

import XCTest
@testable import ProximaKit

final class PersistenceCorruptionTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    private func tempURL(_ ext: String = "pxkt") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
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

    /// Overwrites 8 bytes at `offset` with a little-endian UInt64.
    private func patch64(_ url: URL, at offset: Int, with value: UInt64) throws {
        var data = try Data(contentsOf: url)
        withUnsafeBytes(of: value.littleEndian) { bytes in
            for (i, byte) in bytes.enumerated() {
                data[offset + i] = byte
            }
        }
        try data.write(to: url)
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

    /// Saves a small valid HNSW index and returns its file URL.
    private func savedHNSWFile(
        config: HNSWConfiguration = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10)
    ) async throws -> URL {
        let url = tempURL()
        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
        for i in 0..<20 {
            try await index.add(Vector([Float(i), 1, 2, 3]), id: UUID())
        }
        try await index.save(to: url)
        return url
    }

    // ══════════════════════════════════════════════════════════════════
    // HNSW (.pxkt) — header layout:
    //   0 magic | 4 version | 8 indexType | 12 dimension | 16 count
    //   20 metric | 24 m | 28 mMax0 | 32 efC | 36 efS | 40 entryPoint
    //   44 maxLevel | 48 layerCount | 52 metadataOffset
    //   56 thresholdFlag | 60 thresholdBits
    // ══════════════════════════════════════════════════════════════════

    func testHNSWTruncatedFileThrowsAtEverySection() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        let fullSize = try Data(contentsOf: url).count

        // Header, mid-UUID, mid-vector, mid-graph, and end-of-file cuts.
        let cuts = [10, 63, 64, 70, 64 + 20 * 16 + 5, fullSize / 2, fullSize - 2]
        for cut in cuts where cut < fullSize {
            let cutURL = tempURL()
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try HNSWIndex.load(from: cutURL),
                "HNSW file truncated to \(cut)/\(fullSize) bytes must throw")
        }
    }

    func testHNSWBadMagicThrows() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 0, with: 0xDEAD_BEEF)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "bad magic")
    }

    func testHNSWBadVersionThrows() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        for badVersion: UInt32 in [0, 99] {
            try patch(url, at: 4, with: badVersion)
            XCTAssertThrowsError(try HNSWIndex.load(from: url)) { error in
                guard case PersistenceError.unsupportedVersion(let v)? = error as? PersistenceError else {
                    XCTFail("Expected unsupportedVersion, got \(error)"); return
                }
                XCTAssertEqual(v, badVersion)
            }
        }
    }

    func testHNSWNegativeCountThrows() async throws {
        // 0xFFFFFFFF is Int32(-1) reinterpreted; must be caught as truncation,
        // not crash with an overflow or runaway allocation.
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 16, with: 0xFFFF_FFFF)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "negative count")
    }

    func testHNSWOutOfBoundsEntryPointThrows() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 40, with: 9_999)  // index has only 20 nodes
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "out-of-bounds entry point")
    }

    func testHNSWOutOfBoundsMaxLevelThrows() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 44, with: 50)  // >= layerCount
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "out-of-bounds maxLevel")
    }

    func testHNSWZeroMThrows() async throws {
        // m == 0 would trap HNSWConfiguration's precondition (and produce a
        // degenerate levelMultiplier) if it reached the initializer.
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 24, with: 0)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "m == 0")
    }

    func testHNSWMOfOneThrows() async throws {
        // m == 1 passes a naive `> 0` check but traps HNSWConfiguration's
        // m >= 2 precondition (1/log(1) is infinite). The loader must throw
        // PersistenceError instead of crashing the process.
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 24, with: 1)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "m == 1")
    }

    func testHNSWInconsistentMMax0Throws() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 28, with: 3)  // m is 4, so mMax0 must be 8
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "mMax0 != 2 * m")
    }

    func testHNSWUnknownMetricThrows() async throws {
        let url = try await savedHNSWFile()
        defer { cleanup(url) }
        try patch(url, at: 20, with: 99)
        XCTAssertThrowsError(try HNSWIndex.load(from: url)) { error in
            guard case PersistenceError.unknownMetricType(99)? = error as? PersistenceError else {
                XCTFail("Expected unknownMetricType(99), got \(error)"); return
            }
        }
    }

    func testHNSWBadThresholdEncodingThrows() async throws {
        // Out of (0, 1).
        let url1 = try await savedHNSWFile()
        defer { cleanup(url1) }
        try patch64(url1, at: 56, with: Double(42.0).bitPattern)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url1), "threshold out of range")

        // Non-finite.
        let url2 = try await savedHNSWFile()
        defer { cleanup(url2) }
        try patch64(url2, at: 56, with: Double.nan.bitPattern)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url2), "NaN threshold")
    }

    func testLoadingBruteForceFileAsHNSWThrowsUnknownIndexType() async throws {
        let url = tempURL()
        defer { cleanup(url) }
        let bf = BruteForceIndex(dimension: 4, metric: EuclideanDistance())
        try await bf.add(Vector([1, 2, 3, 4]), id: UUID())
        try await bf.save(to: url)

        XCTAssertThrowsError(try HNSWIndex.load(from: url)) { error in
            guard case PersistenceError.unknownIndexType? = error as? PersistenceError else {
                XCTFail("Expected unknownIndexType, got \(error)"); return
            }
        }
        // And the reverse direction.
        let hnswURL = try await savedHNSWFile()
        defer { cleanup(hnswURL) }
        XCTAssertThrowsError(try BruteForceIndex.load(from: hnswURL)) { error in
            guard case PersistenceError.unknownIndexType? = error as? PersistenceError else {
                XCTFail("Expected unknownIndexType, got \(error)"); return
            }
        }
    }

    func testHNSWOutOfBoundsNeighborThrows() throws {
        // Craft a structurally valid 2-node snapshot whose layer-0 adjacency
        // references node 5 — loading it must throw, not crash on first search.
        let url = tempURL()
        defer { cleanup(url) }

        let snapshot = HNSWSnapshot(
            dimension: 2,
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            metricType: .euclidean,
            vectors: [Vector([0, 0]), Vector([1, 1])],
            metadata: [nil, nil],
            nodeToUUID: [UUID(), UUID()],
            layers: [[[5], [0]]],
            nodeLevels: [0, 0],
            entryPointNode: 0,
            maxLevel: 0
        )
        try PersistenceEngine.save(snapshot, to: url)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "out-of-bounds neighbor")
    }

    func testHNSWOutOfBoundsNodeLevelThrows() throws {
        let url = tempURL()
        defer { cleanup(url) }

        let snapshot = HNSWSnapshot(
            dimension: 2,
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10),
            metricType: .euclidean,
            vectors: [Vector([0, 0]), Vector([1, 1])],
            metadata: [nil, nil],
            nodeToUUID: [UUID(), UUID()],
            layers: [[[1], [0]]],
            nodeLevels: [99, 0],  // only 1 layer exists
            entryPointNode: 0,
            maxLevel: 0
        )
        try PersistenceEngine.save(snapshot, to: url)
        assertThrowsPersistenceError(try HNSWIndex.load(from: url), "out-of-bounds node level")
    }

    // ── HNSW Config Round-Trip (autoCompactionThreshold) ──────────────

    func testHNSWAutoCompactionThresholdRoundTrip() async throws {
        let thresholds: [Double?] = [nil, 0.45, 0.7]
        for threshold in thresholds {
            let url = tempURL()
            defer { cleanup(url) }

            let config = HNSWConfiguration(
                m: 4, efConstruction: 20, efSearch: 10,
                autoCompactionThreshold: threshold
            )
            let index = HNSWIndex(dimension: 4, metric: EuclideanDistance(), config: config)
            for i in 0..<10 {
                try await index.add(Vector([Float(i), 0, 0, 0]), id: UUID())
            }
            try await index.save(to: url)

            let loaded = try HNSWIndex.load(from: url)
            let loadedConfig = loaded.configuration
            XCTAssertEqual(loadedConfig.autoCompactionThreshold, threshold,
                           "autoCompactionThreshold \(String(describing: threshold)) must survive save/load")
            XCTAssertEqual(loadedConfig.m, 4)
            XCTAssertEqual(loadedConfig.mMax0, 8)
            XCTAssertEqual(loadedConfig.efConstruction, 20)
            XCTAssertEqual(loadedConfig.efSearch, 10)
        }
    }

    func testHNSWVersion1FileLoadsWithDefaultThreshold() async throws {
        // v1 files predate threshold serialization: downgrade a v2 file by
        // rewriting the version and zeroing the (then reserved) bytes 56..64.
        // Loading must succeed and apply the documented default (0.7).
        let url = try await savedHNSWFile(
            config: HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10,
                                      autoCompactionThreshold: nil)
        )
        defer { cleanup(url) }
        try patch(url, at: 4, with: 1)
        try patch(url, at: 56, with: 0)
        try patch(url, at: 60, with: 0)

        let loaded = try HNSWIndex.load(from: url)
        XCTAssertEqual(loaded.configuration.autoCompactionThreshold, 0.7,
                       "v1 files must load with the documented default threshold")
        let count = await loaded.count
        XCTAssertEqual(count, 20)
    }

    // ══════════════════════════════════════════════════════════════════
    // Sparse (.pxbm) — header layout:
    //   0 magic | 4 version | 8 indexType | 12 count | 16 k1 | 20 b
    //   24 totalLiveTokens (8) | 32 postingsOffset | 36 metadataOffset
    //   40 thresholdFlag | 44 thresholdBits
    // ══════════════════════════════════════════════════════════════════

    private func savedSparseFile(
        configuration: BM25Configuration = BM25Configuration()
    ) async throws -> URL {
        let url = tempURL("pxbm")
        let index = SparseIndex(configuration: configuration)
        try await index.add(text: "the quick brown fox", id: UUID())
        try await index.add(text: "jumps over the lazy dog", id: UUID())
        try await index.add(text: "pack my box with five dozen jugs", id: UUID())
        try await index.save(to: url)
        return url
    }

    func testSparseTruncatedFileThrowsAtEverySection() async throws {
        let url = try await savedSparseFile()
        defer { cleanup(url) }
        let fullSize = try Data(contentsOf: url).count

        let cuts = [10, 63, 64, 70, fullSize / 2, fullSize - 2]
        for cut in cuts where cut < fullSize {
            let cutURL = tempURL("pxbm")
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try SparseIndex.load(from: cutURL),
                "Sparse file truncated to \(cut)/\(fullSize) bytes must throw")
        }
    }

    func testSparseBadMagicThrows() async throws {
        let url = try await savedSparseFile()
        defer { cleanup(url) }
        try patch(url, at: 0, with: 0xDEAD_BEEF)
        assertThrowsPersistenceError(try SparseIndex.load(from: url), "bad magic")
    }

    func testSparseBadVersionThrows() async throws {
        let url = try await savedSparseFile()
        defer { cleanup(url) }
        for badVersion: UInt32 in [0, 99] {
            try patch(url, at: 4, with: badVersion)
            XCTAssertThrowsError(try SparseIndex.load(from: url)) { error in
                guard case PersistenceError.unsupportedVersion(let v)? = error as? PersistenceError else {
                    XCTFail("Expected unsupportedVersion, got \(error)"); return
                }
                XCTAssertEqual(v, badVersion)
            }
        }
    }

    func testSparseBadIndexTypeThrows() async throws {
        let url = try await savedSparseFile()
        defer { cleanup(url) }
        try patch(url, at: 8, with: 5)
        XCTAssertThrowsError(try SparseIndex.load(from: url)) { error in
            guard case PersistenceError.unknownIndexType(5)? = error as? PersistenceError else {
                XCTFail("Expected unknownIndexType(5), got \(error)"); return
            }
        }
    }

    func testSparseNegativeCountThrows() async throws {
        let url = try await savedSparseFile()
        defer { cleanup(url) }
        try patch(url, at: 12, with: 0xFFFF_FFFF)
        assertThrowsPersistenceError(try SparseIndex.load(from: url), "negative count")
    }

    func testSparseInvalidK1AndBThrow() async throws {
        // NaN / out-of-range BM25 parameters would trap BM25Configuration's
        // preconditions if they reached the initializer.
        let url1 = try await savedSparseFile()
        defer { cleanup(url1) }
        try patch(url1, at: 16, with: Float.nan.bitPattern)
        assertThrowsPersistenceError(try SparseIndex.load(from: url1), "NaN k1")

        let url2 = try await savedSparseFile()
        defer { cleanup(url2) }
        try patch(url2, at: 20, with: Float(7.5).bitPattern)
        assertThrowsPersistenceError(try SparseIndex.load(from: url2), "b > 1")
    }

    func testSparseBadThresholdEncodingThrows() async throws {
        let url1 = try await savedSparseFile()
        defer { cleanup(url1) }
        try patch64(url1, at: 40, with: Double(-3.0).bitPattern)
        assertThrowsPersistenceError(try SparseIndex.load(from: url1), "threshold out of range")

        let url2 = try await savedSparseFile()
        defer { cleanup(url2) }
        try patch64(url2, at: 40, with: Double.infinity.bitPattern)
        assertThrowsPersistenceError(try SparseIndex.load(from: url2), "non-finite threshold")
    }

    func testSparseOutOfBoundsPostingNodeThrows() throws {
        // A posting that references node 9 in a 2-document index.
        let url = tempURL("pxbm")
        defer { cleanup(url) }

        let snapshot = SparseIndexSnapshot(
            configuration: BM25Configuration(),
            tokenCounts: [["fox": 1], ["dog": 1]],
            docLengths: [1, 1],
            metadataStore: [nil, nil],
            nodeToUUID: [UUID(), UUID()],
            postings: ["fox": [(node: 9, tf: 1)], "dog": [(node: 1, tf: 1)]]
        )
        try PersistenceEngine.save(snapshot, to: url)
        assertThrowsPersistenceError(try SparseIndex.load(from: url), "out-of-bounds posting node")
    }

    // ── Sparse Config Round-Trip (autoCompactionThreshold) ────────────

    func testSparseAutoCompactionThresholdRoundTrip() async throws {
        let thresholds: [Double?] = [nil, 0.35, 0.7]
        for threshold in thresholds {
            let config = BM25Configuration(k1: 1.5, b: 0.6, autoCompactionThreshold: threshold)
            let url = try await savedSparseFile(configuration: config)
            defer { cleanup(url) }

            let loaded = try SparseIndex.load(from: url)
            XCTAssertEqual(loaded.configuration.autoCompactionThreshold, threshold,
                           "autoCompactionThreshold \(String(describing: threshold)) must survive save/load")
            XCTAssertEqual(loaded.configuration.k1, 1.5, accuracy: 1e-6)
            XCTAssertEqual(loaded.configuration.b, 0.6, accuracy: 1e-6)
        }
    }

    func testSparseVersion1FileLoadsWithDefaultThreshold() async throws {
        let url = try await savedSparseFile(
            configuration: BM25Configuration(autoCompactionThreshold: nil)
        )
        defer { cleanup(url) }
        try patch(url, at: 4, with: 1)
        try patch(url, at: 40, with: 0)
        try patch(url, at: 44, with: 0)

        let loaded = try SparseIndex.load(from: url)
        XCTAssertEqual(loaded.configuration.autoCompactionThreshold, 0.7,
                       "v1 files must load with the documented default threshold")
    }

    // ══════════════════════════════════════════════════════════════════
    // Quantized HNSW (.qhnsw) — header layout:
    //   0 magic | 4 version | 8 dim | 12 nodeCount | 16 subspaceCount
    //   20 m | 24 efC | 28 efS | 32 maxLevel | 36 entryPoint
    //   40 layerCount | 44 trainIters
    // ══════════════════════════════════════════════════════════════════

    /// Builds a tiny, fully deterministic quantized index (dim 4, M 2).
    private func makeQuantizedIndex(
        layers: [[[Int]]] = [[[1], [0]]],
        nodeLevels: [Int] = [0, 0],
        entryPointNode: Int? = 0,
        maxLevel: Int = 0
    ) -> QuantizedHNSWIndex {
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
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPointNode,
            maxLevel: maxLevel,
            codes: [[1, 2], [3, 4]],
            nodeToUUID: ids,
            uuidToNode: [ids[0]: 0, ids[1]: 1],
            metadata: [nil, nil]
        )
    }

    private func savedQuantizedFile() async throws -> URL {
        let url = tempURL("qhnsw")
        try await makeQuantizedIndex().save(to: url)
        return url
    }

    func testQuantizedTruncatedFileThrowsAtEverySection() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        let fullSize = try Data(contentsOf: url).count

        // Mid-header, mid-codebook, mid-codes/UUIDs/graph, mid-metadata.
        let cuts = [10, 55, 56, 60, 56 + 2048, 56 + 4096 + 2, fullSize / 2, fullSize - 2]
        for cut in cuts where cut < fullSize {
            let cutURL = tempURL("qhnsw")
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try QuantizedHNSWIndex.load(from: cutURL),
                "Quantized file truncated to \(cut)/\(fullSize) bytes must throw")
        }
    }

    func testQuantizedBadMagicThrows() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 0, with: 0xDEAD_BEEF)
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "bad magic")
    }

    func testQuantizedBadVersionThrows() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 4, with: 99)
        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url)) { error in
            guard case PersistenceError.unsupportedVersion(99)? = error as? PersistenceError else {
                XCTFail("Expected unsupportedVersion(99), got \(error)"); return
            }
        }
    }

    func testQuantizedZeroSubspaceCountThrows() async throws {
        // subspaceCount == 0 used to crash with a division by zero.
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 16, with: 0)
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "subspaceCount == 0")
    }

    func testQuantizedNegativeNodeCountThrows() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 12, with: 0xFFFF_FFFF)
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "negative node count")
    }

    func testQuantizedOutOfBoundsEntryPointThrows() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 36, with: 99)  // only 2 nodes
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "out-of-bounds entry point")
    }

    func testQuantizedOutOfBoundsNeighborThrows() async throws {
        let url = tempURL("qhnsw")
        defer { cleanup(url) }
        // Layer-0 adjacency references node 9 in a 2-node index.
        let index = makeQuantizedIndex(layers: [[[9], [0]]])
        try await index.save(to: url)
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "out-of-bounds neighbor")
    }

    func testQuantizedOutOfBoundsNodeLevelThrows() async throws {
        let url = tempURL("qhnsw")
        defer { cleanup(url) }
        let index = makeQuantizedIndex(nodeLevels: [99, 0])  // only 1 layer exists
        try await index.save(to: url)
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "out-of-bounds node level")
    }

    func testQuantizedZeroHNSWConfigFieldsThrow() async throws {
        let url = try await savedQuantizedFile()
        defer { cleanup(url) }
        try patch(url, at: 20, with: 0)  // m == 0
        assertThrowsPersistenceError(try QuantizedHNSWIndex.load(from: url), "m == 0")
    }

    func testQuantizedMetadataCountMismatchThrows() async throws {
        // The metadata section is the one variable-length (JSON) section whose
        // element count is independent of nodeCount. A file whose metadata
        // array is shorter than nodeCount must be rejected at load — the
        // loader indexes metadata[node] unconditionally for every live node
        // during search, so accepting it defers a fatal index-out-of-range to
        // the first query instead of throwing (the sibling scalar codec
        // already guards this exact case). load must throw a PersistenceError,
        // never crash.
        let url = try await savedQuantizedFile()  // nodeCount 2, metadata [nil, nil]
        defer { cleanup(url) }

        // Metadata is the trailing section (no retained originals): a UInt32
        // byte-length prefix followed by the JSON payload `[null,null]`.
        // Replace the whole section with the empty array `[]` (decoded count 0
        // != nodeCount 2), keeping the length prefix consistent so the payload
        // still parses as valid JSON and only the count guard can reject it.
        let data = try Data(contentsOf: url)
        guard let payloadRange = data.range(
            of: Data("[null,null]".utf8), options: .backwards) else {
            return XCTFail("metadata payload not found in saved PQHW file")
        }
        let sectionStart = payloadRange.lowerBound - 4  // 4-byte length prefix
        let shortPayload = Data("[]".utf8)
        var rebuilt = Data(data[..<sectionStart])
        withUnsafeBytes(of: UInt32(shortPayload.count).littleEndian) {
            rebuilt.append(contentsOf: $0)
        }
        rebuilt.append(shortPayload)
        try rebuilt.write(to: url)

        assertThrowsPersistenceError(
            try QuantizedHNSWIndex.load(from: url),
            "PQHW metadata count != nodeCount must throw, never defer a trap to search")
    }

    // ══════════════════════════════════════════════════════════════════
    // Product Quantizer (.pqtt) — header layout:
    //   0 magic | 4 version | 8 dim | 12 M | 16 K | 20 trainIters
    // ══════════════════════════════════════════════════════════════════

    private func savedPQFile() throws -> URL {
        let url = tempURL("pqtt")
        let config = PQConfiguration(subspaceCount: 2)
        let codebook = [Float](repeating: 0.25, count: 256 * 2)
        let pq = ProductQuantizer(dimension: 4, config: config, codebooks: [codebook, codebook])
        try pq.save(to: url)
        return url
    }

    func testPQTruncatedFileThrows() throws {
        let url = try savedPQFile()
        defer { cleanup(url) }
        let fullSize = try Data(contentsOf: url).count

        let cuts = [10, 23, 24, 100, fullSize - 2]
        for cut in cuts where cut < fullSize {
            let cutURL = tempURL("pqtt")
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try ProductQuantizer.load(from: cutURL),
                "PQ file truncated to \(cut)/\(fullSize) bytes must throw")
        }
    }

    func testPQBadMagicThrows() throws {
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 0, with: 0xDEAD_BEEF)
        assertThrowsPersistenceError(try ProductQuantizer.load(from: url), "bad magic")
    }

    func testPQBadVersionThrows() throws {
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 4, with: 99)
        XCTAssertThrowsError(try ProductQuantizer.load(from: url)) { error in
            guard case PersistenceError.unsupportedVersion(99)? = error as? PersistenceError else {
                XCTFail("Expected unsupportedVersion(99), got \(error)"); return
            }
        }
    }

    func testPQZeroSubspaceCountThrows() throws {
        // M == 0 used to crash with a division by zero.
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 12, with: 0)
        assertThrowsPersistenceError(try ProductQuantizer.load(from: url), "M == 0")
    }

    func testPQIndivisibleDimensionThrows() throws {
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 8, with: 5)  // 5 % 2 != 0
        assertThrowsPersistenceError(try ProductQuantizer.load(from: url), "dim % M != 0")
    }

    func testPQInvalidCentroidCountThrows() throws {
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 16, with: 128)  // format fixes K at 256
        assertThrowsPersistenceError(try ProductQuantizer.load(from: url), "K != 256")
    }

    func testPQZeroTrainingIterationsThrows() throws {
        // trainingIterations == 0 would trap PQConfiguration's precondition.
        let url = try savedPQFile()
        defer { cleanup(url) }
        try patch(url, at: 20, with: 0)
        assertThrowsPersistenceError(try ProductQuantizer.load(from: url), "trainIters == 0")
    }
}
