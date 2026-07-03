// MigrationRewriterTests.swift
// ProximaKitTests
//
// ADR-014 Stage 1 — the offline section-copy upgraders (M-A) for BOTH format
// families: `.pxkt` v2/unpadded-v3 → padded v3, and PQHW v2 → padded v3.
// Fidelity gates (ADR-014 acceptance 3): section payloads bit-identical, output
// parses + is paged-capable (16 KiB alignment), a resident load of the output
// equals a resident load of the input (structural + search parity), v1 / flag-0
// inputs upgrade to legal v3, and migration is idempotent.
// Also covers the `.pxkt` metadataOffset trailer-sourcing + sentinel handling.

import XCTest
@testable import ProximaKit

final class MigrationRewriterTests: XCTestCase {

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("migrate-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    private func clustered(count: Int, dim: Int, clusters: Int, seed: UInt64)
        -> (vectors: [Vector], ids: [UUID]) {
        var rng = SeededRandom(seed: seed)
        var vectors = [Vector]()
        for _ in 0..<clusters {
            let center = (0..<dim).map { _ in Float.random(in: -5...5, using: &rng) }
            for _ in 0..<(count / clusters) {
                vectors.append(Vector((0..<dim).map { center[$0] + Float.random(in: -0.5...0.5, using: &rng) }))
            }
        }
        return (vectors, (0..<vectors.count).map { _ in UUID() })
    }

    // ══════════════════════════════════════════════════════════════════
    // PQHW upgraders
    // ══════════════════════════════════════════════════════════════════

    private func buildRetainedPQHW(seed: UInt64) async throws -> (QuantizedHNSWIndex, [Vector]) {
        let (vectors, ids) = clustered(count: 160, dim: 16, clusters: 4, seed: seed)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40, levelSeed: seed),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: seed),
            retainOriginals: true)
        return (idx, vectors)
    }

    func testPQHWUpgradeRetainedRoundTrip() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (src, vectors) = try await buildRetainedPQHW(seed: 0xA11CE_0001)
        try await src.save(to: url)                                     // v2 base

        let before = try QuantizedHNSWIndex.load(from: url)
        try QuantizedHNSWIndex.upgradeToV3(at: url)                     // M-A
        let after = try QuantizedHNSWIndex.load(from: url)

        // Output is v3, originals 16 KiB-aligned.
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3)
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16)) % 16_384, 0)

        // Structural + payload parity.
        let bfp = await before.graphFingerprint; let afp = await after.graphFingerprint
        XCTAssertEqual(bfp, afp, "graph identical")
        let bc = await before.codes; let ac = await after.codes
        XCTAssertEqual(bc, ac, "codes identical")
        let bo = await before.originals; let ao = await after.originals
        XCTAssertEqual(bo?.map { $0.components }, ao?.map { $0.components }, "originals identical")

        // Search parity (with rerank) across seeded queries.
        for q in stride(from: 0, to: 160, by: 19) {
            let a = try await before.search(query: vectors[q], k: 10, rerankDepth: 40)
            let b = try await after.search(query: vectors[q], k: 10, rerankDepth: 40)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "query \(q): ids identical after upgrade")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    func testPQHWUpgradeSectionsBitIdentical() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (src, _) = try await buildRetainedPQHW(seed: 0xA11CE_0002)
        try await src.save(to: url)
        let source = try Data(contentsOf: url)

        // The image builder's own full-checksum verification is the gate.
        let image = try XCTUnwrap(QuantizedHNSWIndex.buildPaddedV3Image(from: source))
        let tmp = dir.appendingPathComponent("img.qhnsw")
        try image.write(to: tmp)
        XCTAssertNoThrow(try QuantizedHNSWIndex.verifyPaddedV3Upgrade(source: source, upgradedURL: tmp),
                         "every section payload must be bit-identical")
    }

    func testPQHWUpgradeFlag0IsLegalV3() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (vectors, ids) = clustered(count: 80, dim: 8, clusters: 2, seed: 0xA11CE_0003)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 8,
            hnswConfig: HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 30, levelSeed: 0xA11CE_0003),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: 0xA11CE_0003),
            retainOriginals: false)                     // NO originals
        try await idx.save(to: url)                     // v2, flag 0
        try QuantizedHNSWIndex.upgradeToV3(at: url)      // legal flag-0 v3

        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3)
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16)), 0, "originals entry (0,0)")
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16 + 8)), 0)
        let loaded = try QuantizedHNSWIndex.load(from: url)
        let retains = await loaded.retainsOriginals
        let cnt = await loaded.count
        XCTAssertFalse(retains)
        XCTAssertEqual(cnt, ids.count)
    }

    func testPQHWUpgradeIdempotent() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (src, _) = try await buildRetainedPQHW(seed: 0xA11CE_0004)
        try await src.save(to: url)
        try QuantizedHNSWIndex.upgradeToV3(at: url)
        let once = try Data(contentsOf: url)
        try QuantizedHNSWIndex.upgradeToV3(at: url)      // no-op
        let twice = try Data(contentsOf: url)
        XCTAssertEqual(once, twice, "re-upgrading a padded v3 is a byte-for-byte no-op")
    }

    // ══════════════════════════════════════════════════════════════════
    // .pxkt upgraders
    // ══════════════════════════════════════════════════════════════════

    private func buildHNSW(seed: UInt64, dim: Int = 12, count: Int = 120, withMetadata: Bool = false)
        async throws -> (HNSWIndex, [Vector]) {
        let (vectors, ids) = clustered(count: count, dim: dim, clusters: 4, seed: seed)
        let idx = HNSWIndex(dimension: dim, metric: EuclideanDistance(),
                            config: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40,
                                                      autoCompactionThreshold: nil, levelSeed: seed))
        for i in 0..<vectors.count {
            let md = withMetadata ? Data("m\(i)".utf8) : nil
            try await idx.add(vectors[i], id: ids[i], metadata: md)
        }
        return (idx, vectors)
    }

    func testPXKTUpgradeRoundTrip() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (src, vectors) = try await buildHNSW(seed: 0xB0B_0001)
        try await src.save(to: url)                     // v2 base

        let before = try HNSWIndex.load(from: url)
        try PersistenceEngine.upgradeToV3(at: url)       // M-A
        let after = try HNSWIndex.load(from: url)

        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3)
        // Paged-capable: vector section resolvable + 16 KiB-aligned.
        let layout = try PersistenceEngine.pagedVectorLayout(of: url)
        XCTAssertEqual(layout.vectorOffset % 16_384, 0)

        let bfp = await before.structuralFingerprint; let afp = await after.structuralFingerprint
        XCTAssertEqual(bfp, afp)
        for q in stride(from: 0, to: 120, by: 17) {
            let a = await before.search(query: vectors[q], k: 10)
            let b = await after.search(query: vectors[q], k: 10)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "query \(q): ids identical after .pxkt upgrade")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    func testPXKTUpgradeSectionsBitIdentical() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (src, _) = try await buildHNSW(seed: 0xB0B_0002)
        try await src.save(to: url)
        let source = try Data(contentsOf: url)
        let image = try XCTUnwrap(PersistenceEngine.buildPaddedV3Image(from: source))
        let tmp = dir.appendingPathComponent("img.pxkt")
        try image.write(to: tmp)
        XCTAssertNoThrow(try PersistenceEngine.verifyPaddedV3Upgrade(source: source, upgradedURL: tmp))
    }

    func testPXKTUpgradeIdempotent() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (src, _) = try await buildHNSW(seed: 0xB0B_0003)
        try await src.save(to: url)
        try PersistenceEngine.upgradeToV3(at: url)
        let once = try Data(contentsOf: url)
        try PersistenceEngine.upgradeToV3(at: url)
        let twice = try Data(contentsOf: url)
        XCTAssertEqual(once, twice, "re-upgrading a padded v3 .pxkt is a no-op")
    }

    func testPXKTUpgradeBruteForceRejected() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("bf.pxkt")
        let bf = BruteForceIndex(dimension: 8, metric: EuclideanDistance())
        for i in 0..<10 { try await bf.add(Vector((0..<8).map { Float($0 + i) }), id: UUID()) }
        try await bf.save(to: url)
        XCTAssertThrowsError(try PersistenceEngine.upgradeToV3(at: url)) { e in
            XCTAssertTrue(e is PersistenceError, "BruteForce base has nothing to page")
        }
    }

    // ── metadataOffset: trailer-sourcing + sentinel handling ─────────

    func testPXKTMetadataOffsetSentinelUsesTrailer() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("meta.pxkt")
        let (src, _) = try await buildHNSW(seed: 0xB0B_0004, dim: 8, count: 40, withMetadata: true)
        try await src.save(to: url)
        try PersistenceEngine.upgradeToV3(at: url)               // v3 with real metadata

        let truth = try HNSWIndex.load(from: url)
        // Force the sentinel into the legacy UInt32 header field @52; the v3
        // reader must ignore it and source the metadata offset from the trailer.
        var data = try Data(contentsOf: url)
        withUnsafeBytes(of: UInt32(0xFFFF_FFFF).littleEndian) { data.replaceSubrange(52..<56, with: $0) }
        try data.write(to: url)

        let viaTrailer = try HNSWIndex.load(from: url)
        // Same searchable state AND same metadata ⇒ the metadata offset was
        // resolved from the trailer, not the sentinel-poisoned header field.
        let q = Vector((0..<8).map { Float($0) })
        let a = await truth.search(query: q, k: 8)
        let b = await viaTrailer.search(query: q, k: 8)
        XCTAssertEqual(a.map(\.id), b.map(\.id))
        XCTAssertEqual(a.map(\.metadata), b.map(\.metadata),
                       "metadata resolved via trailer under the sentinel")
        XCTAssertTrue(a.contains { $0.metadata != nil }, "fixture actually carries metadata")
    }
}
