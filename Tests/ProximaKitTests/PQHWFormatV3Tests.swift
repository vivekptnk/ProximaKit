// PQHWFormatV3Tests.swift
// ProximaKitTests
//
// ADR-014 Stage 1 — PQHW v3 format (trailer + 16 KiB originals padding).
// Verifies:
//   • the 56-byte header stays byte-for-byte identical to v2;
//   • the body (codebooks…metadata) stays byte-for-byte identical to v2;
//   • `save(to:)` and `save(to:layout: .resident)` are byte-identical to before;
//   • `.pagedV3` stamps v3 with a 128-byte PQH3 trailer and a 16 KiB-aligned
//     originals section when originals are retained, and falls back to v2 when
//     there is nothing to page;
//   • v3 loads resident (Stage 1 has no paged reads) with originals + rerank
//     intact, and padded/unpadded v3 load to identical state.

import XCTest
@testable import ProximaKit

final class PQHWFormatV3Tests: XCTestCase {

    private func tempURL(_ ext: String = "qhnsw") -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("pqhwv3-\(UUID().uuidString)").appendingPathExtension(ext)
    }
    private func cleanup(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    /// Deterministic tiny retained index (dim 4, M 2, 2 nodes) — exact offsets.
    private func makeTinyRetainedIndex(retained: Bool = true) -> QuantizedHNSWIndex {
        let config = PQConfiguration(subspaceCount: 2)
        let codebook = [Float](repeating: 0.5, count: 256 * 2)
        let quantizer = ProductQuantizer(dimension: 4, config: config, codebooks: [codebook, codebook])
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
            originals: retained ? [Vector([1, 2, 3, 4]), Vector([5, 6, 7, 8])] : nil)
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

    private func buildRetained(seed: UInt64) async throws -> QuantizedHNSWIndex {
        let (vectors, ids) = clustered(count: 120, dim: 16, clusters: 4, seed: seed)
        return try await QuantizedHNSWIndex.build(
            vectors: vectors, ids: ids, dimension: 16,
            hnswConfig: HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30, levelSeed: seed),
            pqConfig: PQConfiguration(subspaceCount: 4, trainingIterations: 5, seed: seed),
            retainOriginals: true)
    }

    // ── Header + body byte-identity to v2 ────────────────────────────

    func testResidentLayoutIsByteIdenticalToSaveTo() async throws {
        let idx = makeTinyRetainedIndex()
        let a = tempURL(); let b = tempURL()
        defer { cleanup(a); cleanup(b) }
        try await idx.save(to: a)
        try await idx.save(to: b, layout: .resident)
        XCTAssertEqual(try Data(contentsOf: a), try Data(contentsOf: b),
                       ".resident must be byte-identical to save(to:)")
        // And it is a v2 file.
        XCTAssertEqual(try Data(contentsOf: a).loadLE(UInt32.self, at: 4), 2)
    }

    func testV3HeaderAndBodyByteIdenticalToV2() async throws {
        let idx = makeTinyRetainedIndex()
        let v2URL = tempURL(); let v3URL = tempURL()
        defer { cleanup(v2URL); cleanup(v3URL) }
        try await idx.save(to: v2URL)                    // v2
        try await idx.save(to: v3URL, layout: .pagedV3)  // v3
        let v2 = try Data(contentsOf: v2URL)
        let v3 = try Data(contentsOf: v3URL)

        // Header: byte-for-byte identical EXCEPT the version word @4 (2 vs 3).
        XCTAssertEqual(Array(v2[0..<4]), Array(v3[0..<4]), "magic identical")
        XCTAssertEqual(v3.loadLE(UInt32.self, at: 4), 3)
        XCTAssertEqual(Array(v2[8..<56]), Array(v3[8..<56]), "header fields 8…56 identical")

        // Body (codebooks…metadata) is byte-identical. In v2 the originals
        // section starts right after metadata; the v2 file minus its 32-byte
        // originals tail is exactly the v3 body prefix.
        let bodyLen = v2.count - 2 * 4 * 4   // v2 minus originals(32)
        XCTAssertEqual(Array(v2[56..<bodyLen]), Array(v3[56..<bodyLen]),
                       "codebooks…metadata bytes identical between v2 and v3")
    }

    // ── Trailer + padding ────────────────────────────────────────────

    func testPagedV3TrailerAndAlignment() async throws {
        let idx = makeTinyRetainedIndex()
        let url = tempURL(); defer { cleanup(url) }
        try await idx.save(to: url, layout: .pagedV3)
        let data = try Data(contentsOf: url)

        let trailerSize = 4 + 7 * 16 + 8 + 4
        let ts = data.count - trailerSize
        XCTAssertEqual(data.loadLE(UInt32.self, at: ts), 7, "sectionCount == 7")
        XCTAssertEqual(data.loadLE(UInt32.self, at: data.count - 4), 0x5051_4833, "PQH3 magic")
        XCTAssertEqual(data.loadLE(UInt64.self, at: ts + 4 + 7 * 16), 0, "generation reserved 0")

        // Originals is section 6: (offset, length). Offset 16 KiB-aligned,
        // length == nodeCount(2) * dim(4) * 4.
        let oOff = Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16))
        let oLen = Int(data.loadLE(UInt64.self, at: ts + 4 + 6 * 16 + 8))
        XCTAssertEqual(oOff % 16_384, 0, "originals section is 16 KiB-aligned")
        XCTAssertEqual(oLen, 2 * 4 * 4)
        // Codebooks (section 0) start right after the 56-byte header.
        XCTAssertEqual(Int(data.loadLE(UInt64.self, at: ts + 4)), 56)
    }

    func testPagedV3NonRetainedFallsBackToV2() async throws {
        let idx = makeTinyRetainedIndex(retained: false)
        let url = tempURL(); defer { cleanup(url) }
        try await idx.save(to: url, layout: .pagedV3)
        XCTAssertEqual(try Data(contentsOf: url).loadLE(UInt32.self, at: 4), 2,
                       "no originals to page ⇒ .pagedV3 writes v2")
    }

    // ── v3 loads resident, rerank intact ─────────────────────────────

    func testV3LoadsResidentWithOriginalsAndRerank() async throws {
        let idx = try await buildRetained(seed: 0xF00D_0001)
        let (vectors, _) = clustered(count: 120, dim: 16, clusters: 4, seed: 0xF00D_0001)
        let url = tempURL(); defer { cleanup(url) }
        try await idx.save(to: url, layout: .pagedV3)

        let loaded = try QuantizedHNSWIndex.load(from: url)
        let retains = await loaded.retainsOriginals
        let lc = await loaded.count; let ic = await idx.count
        XCTAssertTrue(retains, "v3 retained originals load resident")
        XCTAssertEqual(lc, ic)

        // Rerank parity against the in-memory source across seeded queries.
        for q in stride(from: 0, to: 120, by: 17) {
            let a = try await idx.search(query: vectors[q], k: 10, rerankDepth: 40)
            let b = try await loaded.search(query: vectors[q], k: 10, rerankDepth: 40)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "query \(q): ids identical after v3 round-trip")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance, "query \(q): bit-equal distance") }
        }
    }

    func testPaddedAndUnpaddedV3LoadIdentically() async throws {
        let idx = try await buildRetained(seed: 0xF00D_0002)
        let (vectors, _) = clustered(count: 120, dim: 16, clusters: 4, seed: 0xF00D_0002)
        let padded = tempURL(); let unpadded = tempURL()
        defer { cleanup(padded); cleanup(unpadded) }
        try await idx.encodedV3(padOriginals: true).write(to: padded)
        try await idx.encodedV3(padOriginals: false).write(to: unpadded)

        // The unpadded v3's originals offset is NOT 16 KiB-aligned, but resident
        // load reads it from the trailer regardless (alignment only matters for
        // the Stage-2 paged path).
        let trailerSize = 4 + 7 * 16 + 8 + 4
        let ud = try Data(contentsOf: unpadded)
        let uOff = Int(ud.loadLE(UInt64.self, at: ud.count - trailerSize + 4 + 6 * 16))
        XCTAssertNotEqual(uOff % 16_384, 0, "unpadded v3 originals offset is not aligned")

        let lp = try QuantizedHNSWIndex.load(from: padded)
        let lu = try QuantizedHNSWIndex.load(from: unpadded)
        for q in stride(from: 0, to: 120, by: 23) {
            let a = try await lp.search(query: vectors[q], k: 10, rerankDepth: 40)
            let b = try await lu.search(query: vectors[q], k: 10, rerankDepth: 40)
            XCTAssertEqual(a.map(\.id), b.map(\.id), "padded/unpadded v3 identical ids @\(q)")
            for (x, y) in zip(a, b) { XCTAssertEqual(x.distance, y.distance) }
        }
    }

    // ── N-1 / N-2: v2 still loads resident byte-identically ──────────

    func testV2StillLoadsUnderV3Reader() async throws {
        let idx = makeTinyRetainedIndex()
        let url = tempURL(); defer { cleanup(url) }
        try await idx.save(to: url)   // v2
        let loaded = try QuantizedHNSWIndex.load(from: url)
        let retains = await loaded.retainsOriginals
        let cnt = await loaded.count
        XCTAssertTrue(retains)
        XCTAssertEqual(cnt, 2)
    }
}
