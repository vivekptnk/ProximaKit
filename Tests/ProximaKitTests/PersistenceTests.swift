import XCTest
@testable import ProximaKit

final class PersistenceTests: XCTestCase {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("proximakit")
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    // ── BruteForce Roundtrip ──────────────────────────────────────────

    func testBruteForceRoundtrip() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        let index = BruteForceIndex(dimension: 4, metric: EuclideanDistance())
        var ids: [UUID] = []
        for _ in 0..<50 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Save
        try await index.save(to: url)

        // Load
        let loaded = try BruteForceIndex.load(from: url)

        // Verify count
        let originalCount = await index.count
        let loadedCount = await loaded.count
        XCTAssertEqual(loadedCount, originalCount)

        // Verify identical search results
        for _ in 0..<5 {
            let q = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let origResults = await index.search(query: q, k: 5)
            let loadedResults = await loaded.search(query: q, k: 5)
            XCTAssertEqual(origResults.map(\.id), loadedResults.map(\.id))
            for (o, l) in zip(origResults, loadedResults) {
                XCTAssertEqual(o.distance, l.distance, accuracy: 1e-5)
            }
        }
    }

    // ── HNSW Roundtrip ────────────────────────────────────────────────

    func testHNSWRoundtrip() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        let config = HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 30)
        let index = HNSWIndex(dimension: 8, metric: EuclideanDistance(), config: config)

        for _ in 0..<100 {
            let v = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            try await index.add(v, id: UUID())
        }

        try await index.save(to: url)
        let loaded = try HNSWIndex.load(from: url)

        let loadedCount = await loaded.count
        XCTAssertEqual(loadedCount, 100)

        // Config preserved
        let loadedConfig = loaded.configuration
        XCTAssertEqual(loadedConfig.m, 8)
        XCTAssertEqual(loadedConfig.efSearch, 30)

        // Search results match
        for _ in 0..<5 {
            let q = Vector((0..<8).map { _ in Float.random(in: -1...1) })
            let origResults = await index.search(query: q, k: 10)
            let loadedResults = await loaded.search(query: q, k: 10)
            XCTAssertEqual(origResults.map(\.id), loadedResults.map(\.id))
        }
    }

    // ── Metadata Roundtrip ────────────────────────────────────────────

    func testRoundtripWithMetadata() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        struct Tag: Codable, Equatable { let name: String; let score: Float }

        let index = HNSWIndex(dimension: 4)
        let id = UUID()
        let tag = Tag(name: "test-tag", score: 0.95)
        let encoded = try JSONEncoder().encode(tag)

        try await index.add(Vector([1, 2, 3, 4]), id: id, metadata: encoded)
        try await index.save(to: url)

        let loaded = try HNSWIndex.load(from: url)
        let results = await loaded.search(query: Vector([1, 2, 3, 4]), k: 1)
        let decoded = results.first?.decodeMetadata(as: Tag.self)
        XCTAssertEqual(decoded, tag)
    }

    // ── Roundtrip After Deletions ─────────────────────────────────────

    func testRoundtripAfterDeletions() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        let index = HNSWIndex(dimension: 4, metric: EuclideanDistance())
        var ids: [UUID] = []

        for _ in 0..<50 {
            let v = Vector((0..<4).map { _ in Float.random(in: -1...1) })
            let id = UUID()
            try await index.add(v, id: id)
            ids.append(id)
        }

        // Remove 20
        let removedIDs = Set(ids.prefix(20))
        for id in removedIDs { _ = await index.remove(id: id) }

        // Save (should compact internally)
        try await index.save(to: url)

        let loaded = try HNSWIndex.load(from: url)
        let loadedCount = await loaded.count
        XCTAssertEqual(loadedCount, 30) // no tombstones after compact + save

        // Removed nodes must not appear
        let q = Vector((0..<4).map { _ in Float.random(in: -1...1) })
        let results = await loaded.search(query: q, k: 30)
        for r in results {
            XCTAssertFalse(removedIDs.contains(r.id))
        }
    }

    // ── Empty Index Roundtrip ─────────────────────────────────────────

    func testEmptyHNSWRoundtrip() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        let index = HNSWIndex(dimension: 16)
        try await index.save(to: url)

        let loaded = try HNSWIndex.load(from: url)
        let count = await loaded.count
        XCTAssertEqual(count, 0)

        let results = await loaded.search(query: Vector(dimension: 16, repeating: 1.0), k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testEmptyBruteForceRoundtrip() async throws {
        let url = tempURL()
        defer { cleanup(url) }

        let index = BruteForceIndex(dimension: 8)
        try await index.save(to: url)

        let loaded = try BruteForceIndex.load(from: url)
        let count = await loaded.count
        XCTAssertEqual(count, 0)
    }

    // ── All Three Metrics ─────────────────────────────────────────────

    func testAllMetricsRoundtrip() async throws {
        let metrics: [(any DistanceMetric, String)] = [
            (CosineDistance(), "cosine"),
            (EuclideanDistance(), "euclidean"),
            (DotProductDistance(), "dotProduct"),
        ]

        for (metric, name) in metrics {
            let url = tempURL()
            defer { cleanup(url) }

            let index = HNSWIndex(dimension: 4, metric: metric)
            try await index.add(Vector([1, 0, 0, 0]), id: UUID())
            try await index.add(Vector([0, 1, 0, 0]), id: UUID())

            try await index.save(to: url)
            let loaded = try HNSWIndex.load(from: url)

            let q = Vector([1, 0, 0, 0])
            let origResults = await index.search(query: q, k: 2)
            let loadedResults = await loaded.search(query: q, k: 2)
            XCTAssertEqual(origResults.map(\.id), loadedResults.map(\.id),
                           "Metric \(name) roundtrip failed")
        }
    }

    // ── Error Cases ───────────────────────────────────────────────────

    func testInvalidMagicThrows() {
        let url = tempURL()
        defer { cleanup(url) }

        // Write garbage
        try? Data([0, 1, 2, 3, 4, 5, 6, 7] + Array(repeating: UInt8(0), count: 56))
            .write(to: url)

        XCTAssertThrowsError(try HNSWIndex.load(from: url)) { error in
            XCTAssertTrue(error is PersistenceError)
        }
    }

    func testTruncatedFileThrows() {
        let url = tempURL()
        defer { cleanup(url) }

        try? Data([0x54, 0x4B, 0x58, 0x50]).write(to: url) // just 4 bytes

        XCTAssertThrowsError(try HNSWIndex.load(from: url)) { error in
            guard let pe = error as? PersistenceError else {
                XCTFail("Expected PersistenceError"); return
            }
            if case .fileTooSmall = pe { /* expected */ }
            else { XCTFail("Expected fileTooSmall, got \(pe)") }
        }
    }
}
