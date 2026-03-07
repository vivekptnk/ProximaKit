import XCTest
@testable import ProximaKit

final class HNSWTests: XCTestCase {

    // ── Basic Operations ──────────────────────────────────────────────

    func testAddAndCount() async throws {
        let index = HNSWIndex(dimension: 3)

        var count = await index.count
        XCTAssertEqual(count, 0)

        try await index.add(Vector([1, 2, 3]), id: UUID())
        count = await index.count
        XCTAssertEqual(count, 1)

        try await index.add(Vector([4, 5, 6]), id: UUID())
        count = await index.count
        XCTAssertEqual(count, 2)
    }

    func testDimensionMismatchThrows() async {
        let index = HNSWIndex(dimension: 3)
        do {
            try await index.add(Vector([1, 2]), id: UUID())
            XCTFail("Expected dimensionMismatch")
        } catch is IndexError {
            // expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testSearchEmptyIndex() async {
        let index = HNSWIndex(dimension: 3)
        let results = await index.search(query: Vector([1, 2, 3]), k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchSingleVector() async throws {
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())
        let id = UUID()
        try await index.add(Vector([1, 0]), id: id)

        let results = await index.search(query: Vector([1.1, 0.1]), k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, id)
    }

    func testSearchFindsNearestVector() async throws {
        let config = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)

        let nearID = UUID()
        let farID = UUID()

        try await index.add(Vector([1, 0]), id: nearID)
        try await index.add(Vector([10, 10]), id: farID)

        let results = await index.search(query: Vector([1.1, 0.1]), k: 1)
        XCTAssertEqual(results[0].id, nearID)
    }

    func testSearchReturnsKResults() async throws {
        let config = HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 50)
        let index = HNSWIndex(dimension: 3, config: config)

        for _ in 0..<20 {
            try await index.add(
                Vector([Float.random(in: -1...1),
                        Float.random(in: -1...1),
                        Float.random(in: -1...1)]),
                id: UUID()
            )
        }

        let results = await index.search(query: Vector([0, 0, 0]), k: 5)
        XCTAssertEqual(results.count, 5)
    }

    func testSearchKGreaterThanCount() async throws {
        let index = HNSWIndex(dimension: 2)
        try await index.add(Vector([1, 0]), id: UUID())
        try await index.add(Vector([0, 1]), id: UUID())

        let results = await index.search(query: Vector([0, 0]), k: 10)
        XCTAssertEqual(results.count, 2)
    }

    func testSearchResultsSortedByDistance() async throws {
        let config = HNSWConfiguration(m: 8, efConstruction: 50, efSearch: 50)
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)

        for _ in 0..<30 {
            try await index.add(
                Vector([Float.random(in: -10...10),
                        Float.random(in: -10...10)]),
                id: UUID()
            )
        }

        let results = await index.search(query: Vector([0, 0]), k: 10)
        for i in 0..<(results.count - 1) {
            XCTAssertLessThanOrEqual(results[i].distance, results[i + 1].distance)
        }
    }

    // ── Filter ────────────────────────────────────────────────────────

    func testSearchWithFilter() async throws {
        let config = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance(), config: config)

        let includeID = UUID()
        let excludeID = UUID()

        try await index.add(Vector([0, 0]), id: excludeID)
        try await index.add(Vector([1, 0]), id: includeID)

        let results = await index.search(
            query: Vector([0.1, 0]),
            k: 1,
            filter: { $0 == includeID }
        )
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, includeID)
    }

    // ── Remove ────────────────────────────────────────────────────────

    func testRemove() async throws {
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())

        let keepID = UUID()
        let removeID = UUID()

        try await index.add(Vector([0, 0]), id: removeID)
        try await index.add(Vector([1, 0]), id: keepID)

        let removed = await index.remove(id: removeID)
        XCTAssertTrue(removed)

        let results = await index.search(query: Vector([0, 0]), k: 10)
        // Should only find the kept vector
        XCTAssertTrue(results.allSatisfy { $0.id == keepID })
    }

    func testRemoveNonexistent() async {
        let index = HNSWIndex(dimension: 2)
        let removed = await index.remove(id: UUID())
        XCTAssertFalse(removed)
    }

    // ── Metadata ──────────────────────────────────────────────────────

    func testMetadata() async throws {
        let index = HNSWIndex(dimension: 2)
        struct Tag: Codable, Equatable { let name: String }
        let tag = Tag(name: "test")
        let data = try JSONEncoder().encode(tag)
        let id = UUID()

        try await index.add(Vector([1, 0]), id: id, metadata: data)
        let results = await index.search(query: Vector([1, 0]), k: 1)
        let decoded = results.first?.decodeMetadata(as: Tag.self)
        XCTAssertEqual(decoded, tag)
    }

    // ── Duplicate ID ──────────────────────────────────────────────────

    func testDuplicateIDReplacesVector() async throws {
        let index = HNSWIndex(dimension: 2, metric: EuclideanDistance())
        let id = UUID()

        try await index.add(Vector([10, 10]), id: id)
        try await index.add(Vector([0, 0]), id: id)

        let results = await index.search(query: Vector([0, 0]), k: 1)
        XCTAssertEqual(results.first?.id, id)
        XCTAssertEqual(results.first?.distance ?? 999, 0.0, accuracy: 1e-5)
    }

    // ── Recall vs Brute Force ─────────────────────────────────────────

    /// Tests that NSW achieves >90% recall@10 against brute-force on 100 vectors.
    func testRecallAt10_100Vectors() async throws {
        try await verifyRecall(vectorCount: 100, dimension: 16, targetRecall: 0.90)
    }

    /// Tests recall on 1000 vectors.
    func testRecallAt10_1000Vectors() async throws {
        try await verifyRecall(vectorCount: 1000, dimension: 32, targetRecall: 0.90)
    }

    // ── Recall Helper ─────────────────────────────────────────────────

    private func verifyRecall(
        vectorCount: Int,
        dimension: Int,
        targetRecall: Double,
        k: Int = 10,
        queries: Int = 20,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
        let hnsw = HNSWIndex(dimension: dimension, metric: EuclideanDistance(), config: config)
        let brute = BruteForceIndex(dimension: dimension, metric: EuclideanDistance())

        // Insert the same vectors into both indices.
        for _ in 0..<vectorCount {
            let components = (0..<dimension).map { _ in Float.random(in: -1...1) }
            let v = Vector(components)
            let id = UUID()
            try await hnsw.add(v, id: id)
            try await brute.add(v, id: id)
        }

        // Run queries and measure recall.
        var totalRecall: Double = 0

        for _ in 0..<queries {
            let qComponents = (0..<dimension).map { _ in Float.random(in: -1...1) }
            let query = Vector(qComponents)

            let bruteResults = await brute.search(query: query, k: k)
            let hnswResults = await hnsw.search(query: query, k: k)

            let bruteIDs = Set(bruteResults.map(\.id))
            let hnswIDs = Set(hnswResults.map(\.id))
            let hits = bruteIDs.intersection(hnswIDs).count
            totalRecall += Double(hits) / Double(k)
        }

        let avgRecall = totalRecall / Double(queries)
        XCTAssertGreaterThan(
            avgRecall, targetRecall,
            "Recall@\(k) = \(String(format: "%.1f%%", avgRecall * 100)) " +
            "(target: >\(String(format: "%.0f%%", targetRecall * 100))) " +
            "with \(vectorCount) vectors, dim=\(dimension)",
            file: file, line: line
        )
    }
}
