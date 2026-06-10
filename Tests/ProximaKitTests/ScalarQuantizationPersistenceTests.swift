// ScalarQuantizationPersistenceTests.swift
// ProximaKitTests
//
// Persistence tests for the `SQHW` codec (ADR-007 / ADR-010): round-trips
// (search results, metric type, autoCompactionThreshold) plus the corruption
// matrix — truncated files, bad magic/version, out-of-bounds graph indices,
// m == 1, bad metric, invalid scales and threshold bits. Every corruption
// case must throw `PersistenceError` — never crash.

import XCTest
@testable import ProximaKit

final class ScalarQuantizationPersistenceTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    private func tempURL(_ ext: String = "sqhw") -> URL {
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

    /// Builds a small valid scalar-quantized index and returns it with its
    /// build inputs (for search comparison after a round-trip).
    private func makeIndex(
        metric: DistanceMetricType = .euclidean,
        config: HNSWConfiguration = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10)
    ) async throws -> (index: ScalarQuantizedHNSWIndex, vectors: [Vector], ids: [UUID]) {
        let vectors = (0..<20).map { Vector([Float($0), 1, 2, 3]) }
        let ids = (0..<20).map { _ in UUID() }
        let metadata: [Data?] = (0..<20).map { i in i % 3 == 0 ? Data("meta\(i)".utf8) : nil }
        let index = try await ScalarQuantizedHNSWIndex.build(
            vectors: vectors,
            ids: ids,
            metadata: metadata,
            dimension: 4,
            hnswConfig: config,
            metric: metric
        )
        return (index, vectors, ids)
    }

    /// Saves a small valid scalar-quantized index and returns its file URL.
    private func savedIndexFile(
        metric: DistanceMetricType = .euclidean,
        config: HNSWConfiguration = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 10)
    ) async throws -> URL {
        let url = tempURL()
        let (index, _, _) = try await makeIndex(metric: metric, config: config)
        try await index.save(to: url)
        return url
    }

    // ══════════════════════════════════════════════════════════════════
    // SQHW — header layout:
    //   0 magic | 4 version | 8 dimension | 12 nodeCount | 16 metric
    //   20 m | 24 efConstruction | 28 efSearch | 32 maxLevel
    //   36 entryPoint | 40 layerCount | 44 thresholdBits (8 bytes)
    //   52 reserved (12 bytes)
    // Sections (dim 4, n 20): scales @64 (80B) | codes @144 (80B)
    //   UUIDs @224 (320B) | levels @544 (80B) | graph @624
    // ══════════════════════════════════════════════════════════════════

    // ── Round-Trips ───────────────────────────────────────────────────

    func testRoundTripPreservesSearchResults() async throws {
        let (index, vectors, _) = try await makeIndex()
        let url = tempURL()
        defer { cleanup(url) }

        try await index.save(to: url)
        let loaded = try ScalarQuantizedHNSWIndex.load(from: url)

        let origCount = await index.count
        let loadedCount = await loaded.count
        XCTAssertEqual(origCount, loadedCount)

        for query in vectors.prefix(5) {
            let origResults = await index.search(query: query, k: 5)
            let loadedResults = await loaded.search(query: query, k: 5)

            XCTAssertEqual(origResults.map(\.id), loadedResults.map(\.id))
            XCTAssertEqual(origResults.map(\.metadata), loadedResults.map(\.metadata))
            for (a, b) in zip(origResults, loadedResults) {
                XCTAssertEqual(a.distance, b.distance, accuracy: 1e-6)
            }
        }
    }

    func testRoundTripPreservesMetricType() async throws {
        for metric in DistanceMetricType.allCases {
            let url = tempURL()
            defer { cleanup(url) }
            let (index, _, _) = try await makeIndex(metric: metric)
            try await index.save(to: url)
            let loaded = try ScalarQuantizedHNSWIndex.load(from: url)
            XCTAssertEqual(loaded.metricType, metric, "metric \(metric) must round-trip")
        }
    }

    /// `autoCompactionThreshold` is persisted from day one (.pxkt v2 precedent
    /// — the gap ADR-011 noted for PQHW). Both a custom value and the
    /// `nil` = disabled case must round-trip.
    func testRoundTripPreservesAutoCompactionThreshold() async throws {
        let cases: [Double?] = [0.5, nil]
        for threshold in cases {
            let url = tempURL()
            defer { cleanup(url) }
            let (index, _, _) = try await makeIndex(
                config: HNSWConfiguration(
                    m: 4, efConstruction: 20, efSearch: 10,
                    autoCompactionThreshold: threshold
                )
            )
            try await index.save(to: url)
            let loaded = try ScalarQuantizedHNSWIndex.load(from: url)
            XCTAssertEqual(loaded.configuration.autoCompactionThreshold, threshold,
                           "threshold \(String(describing: threshold)) must round-trip")
            XCTAssertEqual(loaded.configuration.m, 4)
            XCTAssertEqual(loaded.configuration.efConstruction, 20)
            XCTAssertEqual(loaded.configuration.efSearch, 10)
        }
    }

    // ── Corruption Matrix ─────────────────────────────────────────────

    func testTruncatedFileThrowsAtEverySection() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        let fullSize = try Data(contentsOf: url).count

        // Header, mid-scales, mid-codes, mid-UUID, mid-levels, mid-graph,
        // and end-of-file cuts.
        let cuts = [10, 63, 64, 70, 150, 230, 550, 630, fullSize / 2, fullSize - 2]
        for cut in cuts where cut < fullSize {
            let cutURL = tempURL()
            defer { cleanup(cutURL) }
            try Data(contentsOf: url).prefix(cut).write(to: cutURL)
            assertThrowsPersistenceError(
                try ScalarQuantizedHNSWIndex.load(from: cutURL),
                "SQHW file truncated to \(cut)/\(fullSize) bytes must throw")
        }
    }

    func testBadMagicThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 0, with: 0xDEAD_BEEF)
        XCTAssertThrowsError(try ScalarQuantizedHNSWIndex.load(from: url)) { error in
            guard case PersistenceError.invalidMagic? = error as? PersistenceError else {
                XCTFail("Expected invalidMagic, got \(error)"); return
            }
        }
    }

    func testBadVersionThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        for badVersion: UInt32 in [0, 99] {
            try patch(url, at: 4, with: badVersion)
            XCTAssertThrowsError(try ScalarQuantizedHNSWIndex.load(from: url)) { error in
                guard case PersistenceError.unsupportedVersion(let v)? = error as? PersistenceError else {
                    XCTFail("Expected unsupportedVersion, got \(error)"); return
                }
                XCTAssertEqual(v, badVersion)
            }
        }
    }

    func testZeroDimensionThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 8, with: 0)
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "zero dimension")
    }

    func testNegativeNodeCountThrows() async throws {
        // 0xFFFFFFFF reinterpreted; must be caught as truncation, not crash
        // with an overflow or runaway allocation.
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 12, with: 0xFFFF_FFFF)
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "negative node count")
    }

    func testUnknownMetricThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 16, with: 99)
        XCTAssertThrowsError(try ScalarQuantizedHNSWIndex.load(from: url)) { error in
            guard case PersistenceError.unknownMetricType(let t)? = error as? PersistenceError else {
                XCTFail("Expected unknownMetricType, got \(error)"); return
            }
            XCTAssertEqual(t, 99)
        }
    }

    func testMEqualsOneThrows() async throws {
        // m == 1 would trap HNSWConfiguration's precondition (1/log(1) is
        // infinite) if it reached the initializer; the loader must reject it.
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        for badM: UInt32 in [0, 1] {
            try patch(url, at: 20, with: badM)
            assertThrowsPersistenceError(
                try ScalarQuantizedHNSWIndex.load(from: url), "m == \(badM)")
        }
    }

    func testOutOfBoundsMaxLevelThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 32, with: 50)  // >= layerCount
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "out-of-bounds maxLevel")
    }

    func testOutOfBoundsEntryPointThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 36, with: 9_999)  // index has only 20 nodes
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "out-of-bounds entry point")
    }

    func testBadThresholdBitsThrow() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        for badThreshold in [1.0, 1.5, -0.5] {
            try patch64(url, at: 44, with: badThreshold.bitPattern)
            assertThrowsPersistenceError(
                try ScalarQuantizedHNSWIndex.load(from: url),
                "threshold \(badThreshold) outside (0, 1)")
        }
    }

    func testInvalidScaleThrows() async throws {
        // Scales must be finite and non-negative — a NaN or negative scale
        // would silently poison every distance computed against that node.
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        for badScale: Float in [.nan, .infinity, -1.0] {
            try patch(url, at: 64, with: badScale.bitPattern)  // scales[0]
            assertThrowsPersistenceError(
                try ScalarQuantizedHNSWIndex.load(from: url), "scale \(badScale)")
        }
    }

    func testOutOfBoundsNodeLevelThrows() async throws {
        // Levels section starts at 64 + 20*4 + 20*4 + 20*16 = 544 (dim 4, n 20).
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 544, with: 50)  // >= layerCount
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "out-of-bounds node level")
    }

    func testOutOfBoundsNeighborThrows() async throws {
        // Graph section starts at 64 + 80 (scales) + 80 (codes) + 320 (UUIDs)
        // + 80 (levels) = 624. The first UInt32 is node 0's layer-0 neighbor
        // count (>= 1 in a 20-node connected graph); the next is its first
        // neighbor index — patch that to an impossible node id.
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        try patch(url, at: 628, with: 9_999)
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "out-of-bounds neighbor")
    }

    func testGarbageMetadataThrows() async throws {
        let url = try await savedIndexFile()
        defer { cleanup(url) }
        // Corrupt the metadata payload (the final section) in place: flip
        // bytes after the metadata length prefix into invalid JSON.
        var data = try Data(contentsOf: url)
        data[data.count - 1] = 0x7B  // '{' — guaranteed malformed terminator
        data[data.count - 2] = 0x7B
        try data.write(to: url)
        assertThrowsPersistenceError(
            try ScalarQuantizedHNSWIndex.load(from: url), "garbage metadata JSON")
    }

    /// An empty index (zero vectors) must survive a save/load round trip.
    func testEmptyIndexRoundTrip() async throws {
        let index = try await ScalarQuantizedHNSWIndex.build(
            vectors: [], ids: [], dimension: 8, metric: .euclidean
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sq-empty-\(UUID().uuidString).sqhw")
        defer { try? FileManager.default.removeItem(at: url) }

        try await index.save(to: url)
        let loaded = try ScalarQuantizedHNSWIndex.load(from: url)
        let count = await loaded.count
        let live = await loaded.liveCount
        XCTAssertEqual(count, 0)
        XCTAssertEqual(live, 0)
        let results = await loaded.search(query: Vector([0, 0, 0, 0, 0, 0, 0, 0]), k: 5)
        XCTAssertTrue(results.isEmpty)
    }

}
