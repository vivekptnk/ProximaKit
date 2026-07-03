// PagedOriginalsParityTests.swift
// ProximaKitTests
//
// ADR-014 Stage 2, acceptance criterion 2 (rerank parity) + open question 2
// (accounting honesty). Always-on, modest-fixture functional tests:
//
//  • Opening the SAME PQHW v3 retaining base `.resident` vs `.paged` yields
//    byte-identical rerank results — same ids AND bit-equal Float32 distances —
//    across seeded queries × {rerank off, rerank on, filtered, post-remove}.
//  • The paged accounting never reports the mapped originals as resident:
//    `originalStorageBytes == 0`, `originalsArePaged == true`, and the savings
//    ratio rises back to the full PQ compression story while `retainsOriginals`
//    stays true and rerank stays exact.
//
// The heavy `phys_footprint` memory acceptance lives in PagedOriginalsMemoryTests
// (env-gated). This file is CI-safe (dim 16, ~160 vectors).

import XCTest
@testable import ProximaKit

final class PagedOriginalsParityTests: XCTestCase {

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-orig-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

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

    /// Builds a retained PQHW index and writes it as a padded v3 base.
    private func buildPagedBase(seed: UInt64, url: URL)
        async throws -> (index: QuantizedHNSWIndex, vectors: [Vector], ids: [UUID]) {
        let (vectors, ids) = clustered(count: 200, dim: 16, clusters: 5, seed: seed)
        let idx = try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 80, efSearch: 50, levelSeed: seed),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 8, seed: seed),
            retainOriginals: true)
        try await idx.save(to: url, layout: .pagedV3)
        // Confirm it really is a padded v3 base (not a v2 fallback).
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.loadLE(UInt32.self, at: 4), 3, "fixture must be v3")
        let ts = data.count - (4 + 7 * 16 + 8 + 4)
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16)) % 16_384, 0,
                       "fixture originals must be 16 KiB-aligned")
        return (idx, vectors, ids)
    }

    private func assertIdentical(
        _ a: [SearchResult], _ b: [SearchResult], _ what: String,
        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.map(\.id), b.map(\.id), "\(what): ids", file: file, line: line)
        XCTAssertEqual(a.count, b.count, "\(what): count", file: file, line: line)
        for (x, y) in zip(a, b) {
            XCTAssertEqual(x.distance, y.distance,
                           "\(what): distances must be bit-identical", file: file, line: line)
        }
    }

    // ── Parity across rerank on/off + filtered ────────────────────────
    func testPagedMatchesResidentAcrossRerankAndFilter() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (_, vectors, ids) = try await buildPagedBase(seed: 0x9A_0001, url: url)

        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        let paged = try QuantizedHNSWIndex.load(from: url, mode: .paged)
        let isPaged = await paged.originalsArePaged
        XCTAssertTrue(isPaged, "paged open must serve originals from the mapping")

        // Filter accepting a deterministic ~half of the ids.
        let accepted = Set(ids.enumerated().filter { $0.offset % 2 == 0 }.map(\.element))
        let filter: @Sendable (UUID) -> Bool = { accepted.contains($0) }

        for q in stride(from: 0, to: 200, by: 13) {
            let query = vectors[q]
            // rerank OFF (pure ADC — originals untouched, still must match)
            let rOff = try await resident.search(query: query, k: 10, rerankDepth: 0)
            let pOff = try await paged.search(query: query, k: 10, rerankDepth: 0)
            assertIdentical(rOff, pOff, "q\(q) rerank-off")
            // rerank ON (the mapped read site)
            let rOn = try await resident.search(query: query, k: 10, rerankDepth: 40)
            let pOn = try await paged.search(query: query, k: 10, rerankDepth: 40)
            assertIdentical(rOn, pOn, "q\(q) rerank-on")
            // filtered rerank (liveness+filter gates precede the mapped read)
            let rF = try await resident.search(query: query, k: 10, rerankDepth: 40, filter: filter)
            let pF = try await paged.search(query: query, k: 10, rerankDepth: 40, filter: filter)
            assertIdentical(rF, pF, "q\(q) filtered-rerank")
        }
    }

    // ── Parity after identical removes (tombstoned + compacted states) ─
    func testPagedMatchesResidentPostRemove() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        let (_, vectors, ids) = try await buildPagedBase(seed: 0x9A_0002, url: url)

        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        let paged = try QuantizedHNSWIndex.load(from: url, mode: .paged)

        // Remove the same deterministic slice from both.
        for i in stride(from: 0, to: ids.count, by: 5) {
            let rr = await resident.remove(id: ids[i])
            let pr = await paged.remove(id: ids[i])
            XCTAssertEqual(rr, pr, "remove(\(i)) must agree")
        }

        for q in stride(from: 1, to: 200, by: 11) {
            let query = vectors[q]
            let r = try await resident.search(query: query, k: 10, rerankDepth: 40)
            let p = try await paged.search(query: query, k: 10, rerankDepth: 40)
            assertIdentical(r, p, "q\(q) post-remove rerank")
        }

        // And a save of the paged (post-remove) index reads its live originals
        // back through the mapping and compacts them: the re-loaded result still
        // matches the resident twin.
        let out = dir.appendingPathComponent("compacted.qhnsw")
        try await paged.save(to: out, layout: .pagedV3)
        let reloaded = try QuantizedHNSWIndex.load(from: out, mode: .paged)
        for q in stride(from: 2, to: 200, by: 23) {
            let r = try await resident.search(query: vectors[q], k: 10, rerankDepth: 40)
            let p = try await reloaded.search(query: vectors[q], k: 10, rerankDepth: 40)
            assertIdentical(r, p, "q\(q) post-remove save/reload rerank")
        }
    }

    // ── Accounting honesty (ADR-014 open question 2) ──────────────────
    func testPagedAccountingNeverReportsOriginalsResident() async throws {
        let dir = tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("index.qhnsw")
        _ = try await buildPagedBase(seed: 0x9A_0003, url: url)

        let resident = try QuantizedHNSWIndex.load(from: url, mode: .resident)
        let paged = try QuantizedHNSWIndex.load(from: url, mode: .paged)

        let count = await paged.count
        let dim = paged.dimension
        let payload = count * dim * 4

        // Paged: retains originals, but they are NOT resident.
        let pRetains = await paged.retainsOriginals
        let pIsPaged = await paged.originalsArePaged
        let pResidentBytes = await paged.originalStorageBytes
        let pMappedBytes = await paged.mappedOriginalStorageBytes
        let pRatio = await paged.memorySavingsRatio
        let pCodeBytes = await paged.codeStorageBytes
        let pEquiv = await paged.equivalentFullPrecisionBytes
        XCTAssertTrue(pRetains, "paged index can still rerank ⇒ retainsOriginals")
        XCTAssertTrue(pIsPaged)
        XCTAssertEqual(pResidentBytes, 0, "mapped originals must NOT be counted as resident")
        XCTAssertEqual(pMappedBytes, payload, "on-flash originals reported separately")
        // With originals off the resident heap, the savings ratio is the pure PQ
        // ratio (equivalent / codes), i.e. the 32× story restored.
        XCTAssertEqual(pRatio, Float(pEquiv) / Float(pCodeBytes), accuracy: 1e-3)
        XCTAssertGreaterThan(pRatio, 1.0, "paged retention restores a compression story")

        // Resident (same file): the historical accounting — originals resident,
        // ratio below 1.0.
        let rIsPaged = await resident.originalsArePaged
        let rResidentBytes = await resident.originalStorageBytes
        let rMappedBytes = await resident.mappedOriginalStorageBytes
        let rRatio = await resident.memorySavingsRatio
        XCTAssertFalse(rIsPaged)
        XCTAssertEqual(rResidentBytes, payload, "resident originals counted as resident")
        XCTAssertEqual(rMappedBytes, 0)
        XCTAssertLessThan(rRatio, 1.0, "resident retention forfeits the compression story (ADR-012)")

        // Rerank still works in paged mode (exactness proven by the parity tests).
        let hits = try await paged.search(query: Vector((0..<dim).map { _ in Float(0.1) }),
                                          k: 5, rerankDepth: 20)
        XCTAssertFalse(hits.isEmpty)
    }
}
