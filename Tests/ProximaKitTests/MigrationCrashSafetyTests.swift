// MigrationCrashSafetyTests.swift
// ProximaKitTests
//
// ADR-014 Stage 1 — crash safety for the section-copy upgraders (M-A), mirroring
// the WAL truncation-sweep discipline (WALTruncationSweepTests): a torn or
// interrupted upgrade must never corrupt the source and must never leave a torn
// output that a subsequent open would trust.
//
// The upgrader's contract is temp-write → full-checksum verify → atomic replace.
// This suite proves:
//   • building + writing the temp never mutates the source;
//   • the full (untruncated) image verifies OK;
//   • the image truncated at EVERY 32-byte boundary (and across the trailer)
//     is rejected by the verification gate — so the atomic replace never fires
//     and the source stays intact;
//   • a completed upgrade replaces the source and loads to the same state.

import XCTest
@testable import ProximaKit

final class MigrationCrashSafetyTests: XCTestCase {

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("migcrash-\(UUID().uuidString)")
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

    /// Sweeps every truncation of `image` and asserts the verifier rejects each
    /// torn prefix while the full image passes. `sourceBytes` must be unchanged
    /// throughout (the temp is a sibling; the source is never opened for write).
    private func assertTornOutputsRejected(
        source: Data, image: Data, dir: URL,
        verify: (Data, URL) throws -> Void
    ) throws {
        // Full image verifies.
        let full = dir.appendingPathComponent("full.bin")
        defer { cleanup(full) }
        try image.write(to: full)
        XCTAssertNoThrow(try verify(source, full), "the complete upgrade image must verify")

        // Every torn prefix is rejected (typed throw), never trusted.
        var cuts = Set<Int>()
        for c in stride(from: 0, to: image.count, by: 32) { cuts.insert(c) }
        for c in (image.count - 200)...image.count where c >= 0 { cuts.insert(c) }
        for cut in cuts.sorted() where cut >= 0 && cut < image.count {
            let torn = dir.appendingPathComponent("torn-\(cut).bin")
            try image.prefix(cut).write(to: torn)
            XCTAssertThrowsError(try verify(source, torn), "torn output @\(cut) must be rejected") { e in
                XCTAssertTrue(e is PersistenceError, "torn output @\(cut): typed error, got \(e)")
            }
            cleanup(torn)
        }
    }

    // ── PQHW ─────────────────────────────────────────────────────────

    func testPQHWUpgradeCrashSafety() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (vectors, ids) = clustered(count: 120, dim: 16, clusters: 4, seed: 0xC0DE_0001)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40, levelSeed: 0xC0DE_0001),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: 0xC0DE_0001),
            retainOriginals: true)
        try await idx.save(to: url)
        let sourceBytes = try Data(contentsOf: url)

        let image = try XCTUnwrap(QuantizedHNSWIndex.buildPaddedV3Image(from: sourceBytes))
        try assertTornOutputsRejected(source: sourceBytes, image: image, dir: dir,
                                      verify: QuantizedHNSWIndex.verifyPaddedV3Upgrade)
        // The source was never touched by temp writes.
        XCTAssertEqual(try Data(contentsOf: url), sourceBytes, "source untouched during upgrade")

        // A completed upgrade replaces the source and loads to the same state.
        try QuantizedHNSWIndex.upgradeToV3(at: url)
        let after = try QuantizedHNSWIndex.load(from: url)
        let before = try QuantizedHNSWIndex.load(from: {
            let u = dir.appendingPathComponent("orig.qhnsw"); try? sourceBytes.write(to: u); return u
        }())
        for q in stride(from: 0, to: 120, by: 29) {
            let a = try await before.search(query: vectors[q], k: 8, rerankDepth: 32)
            let b = try await after.search(query: vectors[q], k: 8, rerankDepth: 32)
            XCTAssertEqual(a.map(\.id), b.map(\.id))
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    // ── .pxkt ────────────────────────────────────────────────────────

    func testPXKTUpgradeCrashSafety() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let url = dir.appendingPathComponent("index.pxkt")
        let (vectors, ids) = clustered(count: 120, dim: 12, clusters: 4, seed: 0xC0DE_0002)
        let idx = HNSWIndex(dimension: 12, metric: EuclideanDistance(),
                            config: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40,
                                                      autoCompactionThreshold: nil, levelSeed: 0xC0DE_0002))
        for i in 0..<vectors.count { try await idx.add(vectors[i], id: ids[i]) }
        try await idx.save(to: url)
        let sourceBytes = try Data(contentsOf: url)

        let image = try XCTUnwrap(PersistenceEngine.buildPaddedV3Image(from: sourceBytes))
        try assertTornOutputsRejected(source: sourceBytes, image: image, dir: dir,
                                      verify: PersistenceEngine.verifyPaddedV3Upgrade)
        XCTAssertEqual(try Data(contentsOf: url), sourceBytes, "source untouched during upgrade")

        try PersistenceEngine.upgradeToV3(at: url)
        let after = try HNSWIndex.load(from: url)
        for q in stride(from: 0, to: 120, by: 29) {
            let b = await after.search(query: vectors[q], k: 8)
            XCTAssertEqual(b.count, 8)
        }
    }
}
