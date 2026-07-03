// PagedOriginalsMigrationTests.swift
// ProximaKitTests
//
// ADR-014 Stage-2 ride-along followups from the Stage-1 judges:
//
//  (a) The v2→v3 upgraders wrap filesystem failures (disk-full / permission)
//      in a typed `PersistenceError.migrationFailed`, preserving the underlying
//      error in the message, and leave the source untouched.
//  (b) Migration fixtures for the paths Stage 1 did not exercise directly:
//      a v1 `.pxkt` and a v1 `PQHW` base, plus an unpadded-v3 → padded-v3 rewrite
//      for BOTH families.
//
// The fidelity gates match the Stage-1 suite: legal padded v3 out, paged-capable
// (16 KiB alignment), section payloads bit-identical, and a resident/paged load
// of the output parity-matches the source.

import XCTest
@testable import ProximaKit

final class PagedOriginalsMigrationTests: XCTestCase {

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("paged-mig-\(UUID().uuidString)")
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

    private func patchVersion(_ url: URL, to v: UInt32) throws {
        var data = try Data(contentsOf: url)
        withUnsafeBytes(of: v.littleEndian) { data.replaceSubrange(4..<8, with: $0) }
        try data.write(to: url)
    }

    private func isAlignedPQHWOriginals(_ url: URL) throws -> Bool {
        let data = try Data(contentsOf: url)
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        return Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16)) % 16_384 == 0
    }

    // ══════════════════════════════════════════════════════════════════
    // (b) v1 base upgrades
    // ══════════════════════════════════════════════════════════════════

    func testPQHWV1UpgradesToLegalFlag0V3() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (vectors, ids) = clustered(count: 100, dim: 8, clusters: 4, seed: 0xD1_0001)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 8,
            hnswConfig: HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 30, levelSeed: 0xD1_0001),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: 0xD1_0001),
            retainOriginals: false)
        try await idx.save(to: url)                       // v2, flag 0
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 48), 0)
        try patchVersion(url, to: 1)                       // byte-faithful v1

        let before = try QuantizedHNSWIndex.load(from: url)
        let beforeRetains = await before.retainsOriginals
        XCTAssertFalse(beforeRetains, "v1 loads with retainOriginals == false")

        try QuantizedHNSWIndex.upgradeToV3(at: url)         // M-A
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3, "output is v3")
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16)), 0, "flag-0 originals entry (0,0)")

        let after = try QuantizedHNSWIndex.load(from: url)
        let bfp = await before.graphFingerprint; let afp = await after.graphFingerprint
        XCTAssertEqual(bfp, afp, "graph identical across v1→v3")
        for q in stride(from: 0, to: 100, by: 13) {
            let a = await before.search(query: vectors[q], k: 8)
            let b = await after.search(query: vectors[q], k: 8)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "q\(q): v1→v3 ids")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    func testPXKTV1UpgradesToPagedCapableV3() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (vectors, ids) = clustered(count: 100, dim: 10, clusters: 4, seed: 0xD1_0002)
        let idx = HNSWIndex(dimension: 10, metric: EuclideanDistance(),
                            config: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40,
                                                      autoCompactionThreshold: nil, levelSeed: 0xD1_0002))
        for i in 0..<vectors.count { try await idx.add(vectors[i], id: ids[i]) }
        try await idx.save(to: url)                        // v2
        try patchVersion(url, to: 1)                       // v1

        let source = try Data(contentsOf: url)
        let before = try HNSWIndex.load(from: url)

        // Section payloads bit-identical (the M-A fidelity gate) via the verifier.
        let image = try XCTUnwrap(PersistenceEngine.buildPaddedV3Image(from: source))
        let tmp = dir.appendingPathComponent("img.pxkt"); try image.write(to: tmp)
        XCTAssertNoThrow(try PersistenceEngine.verifyPaddedV3Upgrade(source: source, upgradedURL: tmp),
                         "v1 .pxkt sections must be bit-identical after upgrade")

        try PersistenceEngine.upgradeToV3(at: url)
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 4), 3)
        let layout = try PersistenceEngine.pagedVectorLayout(of: url)
        XCTAssertEqual(layout.vectorOffset % 16_384, 0, "output is paged-capable")

        let after = try HNSWIndex.load(from: url)
        for q in stride(from: 0, to: 100, by: 13) {
            let a = await before.search(query: vectors[q], k: 8)
            let b = await after.search(query: vectors[q], k: 8)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "q\(q): v1 .pxkt→v3 ids")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // (b) unpadded-v3 → padded-v3 (both families)
    // ══════════════════════════════════════════════════════════════════

    func testPQHWUnpaddedV3UpgradesToPadded() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (vectors, ids) = clustered(count: 160, dim: 16, clusters: 4, seed: 0xD1_0003)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40, levelSeed: 0xD1_0003),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 6, seed: 0xD1_0003),
            retainOriginals: true)
        try await idx.encodedV3(padOriginals: false).write(to: url)   // unpadded v3
        XCTAssertFalse(try isAlignedPQHWOriginals(url), "fixture originals start unaligned")

        // Paged open of the unpadded base is rejected up front...
        XCTAssertThrowsError(try QuantizedHNSWIndex.load(from: url, mode: .paged))

        // ...upgrade pads it, and now paged open succeeds + parity-matches resident.
        try QuantizedHNSWIndex.upgradeToV3(at: url)
        XCTAssertTrue(try isAlignedPQHWOriginals(url), "upgrade aligns the originals")
        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        let paged = try QuantizedHNSWIndex.load(from: url, mode: .paged)
        for q in stride(from: 0, to: 160, by: 11) {
            let a = try await resident.search(query: vectors[q], k: 8, rerankDepth: 32)
            let b = try await paged.search(query: vectors[q], k: 8, rerankDepth: 32)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "q\(q): unpadded→padded paged parity")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }

        // Idempotent: re-upgrading the now-padded base is a byte no-op.
        let once = try Data(contentsOf: url)
        try QuantizedHNSWIndex.upgradeToV3(at: url)
        XCTAssertEqual(try Data(contentsOf: url), once)
    }

    func testPXKTUnpaddedV3UpgradesToPadded() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (vectors, ids) = clustered(count: 120, dim: 20, clusters: 4, seed: 0xD1_0004)
        let builder = HNSWIndex(dimension: 20, metric: EuclideanDistance(),
                                config: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40,
                                                          autoCompactionThreshold: nil, levelSeed: 0xD1_0004))
        for i in 0..<vectors.count { try await builder.add(vectors[i], id: ids[i]) }
        let snapshot = try await builder.persistenceSnapshot()
        try PersistenceEngine.saveHNSW(snapshot, generation: 3, to: url, padVectorSection: false)  // unpadded v3
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 4), 3)

        try PersistenceEngine.upgradeToV3(at: url)
        let layout = try PersistenceEngine.pagedVectorLayout(of: url)
        XCTAssertEqual(layout.vectorOffset % 16_384, 0, "upgrade aligns the vector section")
        // Generation is preserved across the unpadded→padded rewrite.
        XCTAssertEqual(try PersistenceEngine.readGeneration(from: url), 3)

        let before = try HNSWIndex.load(from: url)
        for q in stride(from: 0, to: 120, by: 17) {
            let r = await before.search(query: vectors[q], k: 8)
            XCTAssertEqual(r.count, 8, "q\(q): output searchable")
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // (a) filesystem-failure wrapping (typed, underlying preserved, source safe)
    // ══════════════════════════════════════════════════════════════════

    /// Makes `dir` non-writable for the duration of `body` (so temp writes fail),
    /// restoring 0o755 afterward. Skips if the process can still write (e.g. root).
    private func withReadOnlyDir(_ dir: URL, _ body: () throws -> Void) throws {
        let fm = FileManager.default
        try fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer { try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path) }
        // Verify the directory is actually non-writable in this environment.
        let probe = dir.appendingPathComponent("probe-\(UUID().uuidString)")
        if fm.createFile(atPath: probe.path, contents: Data([0])) {
            try? fm.removeItem(at: probe)
            throw XCTSkip("directory remains writable (likely running as root); cannot exercise the failure path")
        }
        try body()
    }

    func testPQHWUpgradeWrapsWriteFailureTyped() async throws {
        let dir = tempDir(); defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
            cleanup(dir)
        }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (vectors, ids) = clustered(count: 80, dim: 8, clusters: 4, seed: 0xD1_0005)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 8,
            hnswConfig: HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 30, levelSeed: 0xD1_0005),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: 0xD1_0005),
            retainOriginals: true)
        try await idx.save(to: url)
        let sourceBytes = try Data(contentsOf: url)

        try withReadOnlyDir(dir) {
            XCTAssertThrowsError(try QuantizedHNSWIndex.upgradeToV3(at: url)) { error in
                guard case PersistenceError.migrationFailed(let detail)? = error as? PersistenceError else {
                    return XCTFail("expected .migrationFailed, got \(error)")
                }
                XCTAssertFalse(detail.isEmpty, "underlying error must be preserved in the message")
            }
        }
        // Source untouched — the temp/rename discipline held under the failure.
        XCTAssertEqual(try Data(contentsOf: url), sourceBytes, "source must survive a failed upgrade")
    }

    func testPXKTUpgradeWrapsWriteFailureTyped() async throws {
        let dir = tempDir(); defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
            cleanup(dir)
        }
        let url = dir.appendingPathComponent("index.pxkt")
        let (vectors, ids) = clustered(count: 80, dim: 10, clusters: 4, seed: 0xD1_0006)
        let idx = HNSWIndex(dimension: 10, metric: EuclideanDistance(),
                            config: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40,
                                                      autoCompactionThreshold: nil, levelSeed: 0xD1_0006))
        for i in 0..<vectors.count { try await idx.add(vectors[i], id: ids[i]) }
        try await idx.save(to: url)
        let sourceBytes = try Data(contentsOf: url)

        try withReadOnlyDir(dir) {
            XCTAssertThrowsError(try PersistenceEngine.upgradeToV3(at: url)) { error in
                guard case PersistenceError.migrationFailed(let detail)? = error as? PersistenceError else {
                    return XCTFail("expected .migrationFailed, got \(error)")
                }
                XCTAssertFalse(detail.isEmpty, "underlying error must be preserved in the message")
            }
        }
        XCTAssertEqual(try Data(contentsOf: url), sourceBytes, "source must survive a failed upgrade")
    }
}
