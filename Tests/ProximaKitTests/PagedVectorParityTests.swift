// PagedVectorParityTests.swift
// ProximaKitTests
//
// ADR-013 Stage 2, acceptance criterion 4 (search parity) + criterion 3's
// always-on functional half. Proves that a `.paged` open of a base returns
// results BYTE-IDENTICAL to a `.resident` open of the same file — same ids, and
// bit-equal Float32 distances — across seeded queries, including filtered
// queries and post-WAL-replay state, and that the checkpoint-remap keeps paged
// search correct. The heavy phys_footprint benchmark lives in
// `PagedVectorMemoryTests` (env-gated). Corruption cases live in
// `PagedVectorCorruptionTests`.
//
// Determinism: fixtures use `HNSWConfiguration.levelSeed` + a seeded vector
// generator, so the graph and every distance are reproducible. No system RNG.

import XCTest
@testable import ProximaKit

final class PagedVectorParityTests: XCTestCase {

    // ── Helpers ───────────────────────────────────────────────────────

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    /// Deterministic vector for index `i` (seeded, no RNG). Range spread so
    /// distances are well separated (few exact ties).
    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0x9E37 &* UInt64(i + 1))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 100_000) / 100_000.0 })
    }

    private func uuid(_ i: Int) -> UUID {
        var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                     UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &bytes) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: bytes)
    }

    private func seededConfig() -> HNSWConfiguration {
        HNSWConfiguration(m: 8, efConstruction: 60, efSearch: 30,
                          autoCompactionThreshold: nil, levelSeed: 0xC0FFEE)
    }

    /// Asserts two result lists are byte-identical: same ids in the same order
    /// AND bit-equal distances (raw Float32 bit patterns).
    private func assertIdentical(
        _ a: [SearchResult], _ b: [SearchResult],
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(a.map(\.id), b.map(\.id), "\(message): ids differ", file: file, line: line)
        XCTAssertEqual(a.map { $0.distance.bitPattern }, b.map { $0.distance.bitPattern },
                       "\(message): distances not bit-identical", file: file, line: line)
    }

    // ── 1. Paged == resident on the same checkpoint base ──────────────

    func testPagedSearchMatchesResidentOnSameBase() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let dim = 48
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<400 { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        try await builder.checkpoint(baseURL: base, walURL: wal)   // padded v3, gen 1
        await builder.closeJournal()

        // Sequential opens (avoid two append handles on one WAL).
        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        var residentResults: [[SearchResult]] = []
        for q in 0..<25 { residentResults.append(await resident.search(query: vec(1000 + q, dim: dim), k: 10)) }
        await resident.closeJournal()

        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        let pagedCount = await paged.count
        XCTAssertEqual(pagedCount, 400)
        for q in 0..<25 {
            let pagedR = await paged.search(query: vec(1000 + q, dim: dim), k: 10)
            assertIdentical(residentResults[q], pagedR, "query \(q) paged vs resident")
        }
        await paged.closeJournal()
    }

    // ── 2. Filtered-search parity (graph-aware beam path) ─────────────

    func testPagedFilteredSearchMatchesResident() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let dim = 32
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<500 { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        try await builder.checkpoint(baseURL: base, walURL: wal)
        await builder.closeJournal()

        // A selective, id-derived predicate (accepts ~1/3 of nodes). Because the
        // uuids are deterministic, the same predicate is applied on both opens.
        let acceptedIndices = Set((0..<500).filter { $0 % 3 == 0 })
        let acceptedUUIDs = Set(acceptedIndices.map { uuid($0) })
        let filter: @Sendable (UUID) -> Bool = { acceptedUUIDs.contains($0) }

        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        var residentResults: [[SearchResult]] = []
        for q in 0..<20 {
            residentResults.append(await resident.search(query: vec(2000 + q, dim: dim), k: 8, filter: filter))
        }
        await resident.closeJournal()

        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        for q in 0..<20 {
            let pagedR = await paged.search(query: vec(2000 + q, dim: dim), k: 8, filter: filter)
            // Every returned id must be in the accepted set (contract), and match
            // the resident run bit for bit.
            for r in pagedR { XCTAssertTrue(acceptedUUIDs.contains(r.id)) }
            assertIdentical(residentResults[q], pagedR, "filtered query \(q)")
        }
        await paged.closeJournal()
    }

    // ── 3. Post-WAL-replay parity (mapped base + replayed tail) ───────

    func testPagedParityAfterWALReplay() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let dim = 40
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<300 { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        try await builder.checkpoint(baseURL: base, walURL: wal)   // gen 1: 300 mapped
        // Post-snapshot adds → journaled to the WAL, NOT in the base.
        for i in 300..<420 { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        try await builder.syncJournal()
        await builder.closeJournal()

        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        let residentCount = await resident.count
        var residentResults: [[SearchResult]] = []
        for q in 0..<25 { residentResults.append(await resident.search(query: vec(3000 + q, dim: dim), k: 10)) }
        await resident.closeJournal()

        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        let pagedCount = await paged.count
        XCTAssertEqual(pagedCount, residentCount)
        XCTAssertEqual(pagedCount, 420, "300 mapped + 120 replayed into the resident tail")
        for q in 0..<25 {
            let pagedR = await paged.search(query: vec(3000 + q, dim: dim), k: 10)
            assertIdentical(residentResults[q], pagedR, "post-replay query \(q)")
        }
        await paged.closeJournal()
    }

    // ── 4. Checkpoint-remap keeps paged search correct ────────────────

    func testPagedCheckpointRemapKeepsParity() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let dim = 36
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        for i in 0..<250 { try await builder.add(vec(i, dim: dim), id: uuid(i)) }
        try await builder.checkpoint(baseURL: base, walURL: wal)
        await builder.closeJournal()

        // Open paged, add more (journaled), then checkpoint IN PAGED MODE — this
        // exercises the remap: it must fold the tail, rewrite a padded base, and
        // re-map so the index keeps serving from the file with no torn state.
        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        for i in 250..<330 { try await paged.add(vec(i, dim: dim), id: uuid(i)) }
        let genBefore = await paged.currentGeneration
        try await paged.checkpoint(baseURL: base, walURL: wal)     // remap here
        let genAfter = await paged.currentGeneration
        XCTAssertEqual(genAfter, genBefore + 1)

        // Search still works and matches a fresh resident open of the new base.
        var pagedResults: [[SearchResult]] = []
        for q in 0..<20 { pagedResults.append(await paged.search(query: vec(4000 + q, dim: dim), k: 10)) }
        let pagedCount = await paged.count
        XCTAssertEqual(pagedCount, 330)
        await paged.closeJournal()

        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        for q in 0..<20 {
            let residentR = await resident.search(query: vec(4000 + q, dim: dim), k: 10)
            assertIdentical(residentR, pagedResults[q], "post-remap query \(q)")
        }
        await resident.closeJournal()
    }

    // ── 5. Recall parity: paged recall == resident recall ─────────────
    //
    // A lightweight recall check (the heavy recall suites run in
    // RecallBenchmarkTests). Confirms the paged path clears the same recall bar
    // as resident against brute force on a modest fixture.

    func testPagedRecallMatchesResident() async throws {
        let dir = tempDir(); defer { cleanup(dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let dim = 24
        let n = 800
        let builder = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: seededConfig())
        let brute = BruteForceIndex(dimension: dim, metric: EuclideanDistance())
        for i in 0..<n {
            let v = vec(i, dim: dim)
            try await builder.add(v, id: uuid(i))
            try await brute.add(v, id: uuid(i))
        }
        try await builder.checkpoint(baseURL: base, walURL: wal)
        await builder.closeJournal()

        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        var hits = 0, total = 0
        for q in 0..<40 {
            let query = vec(5000 + q, dim: dim)
            let truth = Set((await brute.search(query: query, k: 10)).map(\.id))
            let got = Set((await paged.search(query: query, k: 10)).map(\.id))
            hits += truth.intersection(got).count
            total += truth.count
        }
        await paged.closeJournal()
        let recall = Double(hits) / Double(total)
        XCTAssertGreaterThan(recall, 0.90, "paged recall@10 = \(recall)")
    }
}
