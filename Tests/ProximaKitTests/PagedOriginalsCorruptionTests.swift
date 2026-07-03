// PagedOriginalsCorruptionTests.swift
// ProximaKitTests
//
// ADR-014 corruption-matrix items deferred from Stage 1 to Stage 2 (they need
// the paged-open entry point Stage 2 introduces) — the ADR "corruption matrix"
// items 5 (second half) and 10:
//
//  • A `.paged` open of a v2 base → typed error naming the upgrade path.
//  • A `.paged` open of a flag-0 v3 base → typed error ("nothing to page").
//  • A `.paged` open of an UNALIGNED v3 base → typed error, while the SAME file
//    still loads `.resident` unaffected (mirrors the HNSW
//    `testUnpaddedV3PagedRejectedButResidentLoads` precedent).
//
// Recovery is always a typed `PersistenceError`, never a trap (ADR-010 rule 5).

import XCTest
@testable import ProximaKit

final class PagedOriginalsCorruptionTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-corrupt-\(UUID().uuidString)")
            .appendingPathExtension("qhnsw")
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

    private func buildRetained(seed: UInt64) async throws -> (QuantizedHNSWIndex, [Vector]) {
        let (vectors, ids) = clustered(count: 120, dim: 12, clusters: 4, seed: seed)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 12,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 40, levelSeed: seed),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 6, seed: seed),
            retainOriginals: true)
        return (idx, vectors)
    }

    private func assertPersistenceError<T>(
        _ expr: @autoclosure () throws -> T, contains needle: String,
        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try expr(), file: file, line: line) { error in
            guard let pe = error as? PersistenceError else {
                return XCTFail("expected PersistenceError, got \(error)", file: file, line: line)
            }
            XCTAssertTrue((pe.errorDescription ?? "").lowercased().contains(needle.lowercased()),
                          "message '\(pe.errorDescription ?? "")' should mention '\(needle)'",
                          file: file, line: line)
        }
    }

    // ── (10a) v2 base → paged open rejected, upgrade path named ────────
    func testPagedOpenOfV2BaseThrowsWithUpgradePath() async throws {
        let url = tempURL(); defer { cleanup(url) }
        let (idx, _) = try await buildRetained(seed: 0xC0_0001)
        try await idx.save(to: url)                       // v2 (default writer)
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 4), 2)

        assertPersistenceError(try QuantizedHNSWIndex.load(from: url, mode: .paged),
                               contains: "upgrade")
        // Resident open of the same v2 base is unaffected.
        XCTAssertNoThrow(try QuantizedHNSWIndex.load(from: url, mode: .resident))
    }

    // ── (10b) flag-0 v3 base → paged open rejected ("nothing to page") ─
    func testPagedOpenOfFlag0V3ThrowsNothingToPage() async throws {
        let url = tempURL(); defer { cleanup(url) }
        let (vectors, ids) = clustered(count: 90, dim: 8, clusters: 3, seed: 0xC0_0002)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 8,
            hnswConfig: HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 30, levelSeed: 0xC0_0002),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: 0xC0_0002),
            retainOriginals: false)                        // NO originals
        // Force a real v3 stamp with flag 0 via the test seam (save(layout:) would
        // fall back to v2 with nothing to page).
        try await idx.encodedV3(padOriginals: true).write(to: url)
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 4), 3)
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 48), 0, "flag 0")

        assertPersistenceError(try QuantizedHNSWIndex.load(from: url, mode: .paged),
                               contains: "nothing to page")
        // Resident open of the flag-0 v3 base is unaffected.
        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        let retains = await resident.retainsOriginals
        XCTAssertFalse(retains)
    }

    // ── (5b) unaligned v3 → paged rejected, resident unaffected ────────
    func testUnalignedV3PagedRejectedButResidentLoads() async throws {
        let url = tempURL(); defer { cleanup(url) }
        let (idx, vectors) = try await buildRetained(seed: 0xC0_0003)
        // Unpadded v3: valid trailer, retained originals, but the originals offset
        // is NOT 16 KiB-aligned.
        try await idx.encodedV3(padOriginals: false).write(to: url)
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3)
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        let originalsOffset = Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16))
        XCTAssertNotEqual(originalsOffset % 16_384, 0, "fixture originals must be unaligned")

        assertPersistenceError(try QuantizedHNSWIndex.load(from: url, mode: .paged),
                               contains: "aligned")

        // The SAME unaligned base still loads resident and reranks correctly.
        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        for q in stride(from: 0, to: 120, by: 17) {
            let hits = try await resident.search(query: vectors[q], k: 5, rerankDepth: 20)
            let live = try await idx.search(query: vectors[q], k: 5, rerankDepth: 20)
            XCTAssertEqual(hits.map(\.id), live.map(\.id), "resident unaligned-v3 rerank @\(q)")
        }
    }
}
