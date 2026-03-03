import XCTest
@testable import ProximaKit

final class BruteForceIndexTests: XCTestCase {

    // ── Add ───────────────────────────────────────────────────────────

    func testAddAndCount() async throws {
        let index = BruteForceIndex(dimension: 3)
        var count = await index.count
        XCTAssertEqual(count, 0)

        try await index.add(Vector([1, 2, 3]), id: UUID())
        count = await index.count
        XCTAssertEqual(count, 1)

        try await index.add(Vector([4, 5, 6]), id: UUID())
        count = await index.count
        XCTAssertEqual(count, 2)
    }

    func testAddDimensionMismatchThrows() async {
        let index = BruteForceIndex(dimension: 3)
        do {
            try await index.add(Vector([1, 2]), id: UUID())
            XCTFail("Expected dimensionMismatch error")
        } catch let error as IndexError {
            if case .dimensionMismatch(let expected, let got) = error {
                XCTAssertEqual(expected, 3)
                XCTAssertEqual(got, 2)
            } else {
                XCTFail("Wrong error case")
            }
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testAddDuplicateIDReplacesVector() async throws {
        let index = BruteForceIndex(dimension: 2, metric: EuclideanDistance())
        let id = UUID()

        try await index.add(Vector([1, 0]), id: id)
        try await index.add(Vector([0, 1]), id: id)  // replace

        let count = await index.count
        XCTAssertEqual(count, 1)

        // Search should find the new vector [0,1], not the old [1,0]
        let query = Vector([0, 1])
        let results = await index.search(query: query, k: 1)
        XCTAssertEqual(results.first?.id, id)
        XCTAssertEqual(results.first?.distance ?? 999, 0.0, accuracy: 1e-5)
    }

    // ── Search ────────────────────────────────────────────────────────

    func testSearchFindsNearestVector() async throws {
        let index = BruteForceIndex(dimension: 2, metric: EuclideanDistance())

        let nearID = UUID()
        let farID = UUID()

        try await index.add(Vector([1, 0]), id: nearID)
        try await index.add(Vector([10, 10]), id: farID)

        let results = await index.search(query: Vector([1.1, 0.1]), k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, nearID)
    }

    func testSearchReturnsKResults() async throws {
        let index = BruteForceIndex(dimension: 2)

        for _ in 0..<10 {
            try await index.add(Vector([Float.random(in: -1...1),
                                        Float.random(in: -1...1)]), id: UUID())
        }

        let results = await index.search(query: Vector([0, 0]), k: 5)
        XCTAssertEqual(results.count, 5)
    }

    func testSearchKGreaterThanCount() async throws {
        let index = BruteForceIndex(dimension: 2)
        try await index.add(Vector([1, 0]), id: UUID())
        try await index.add(Vector([0, 1]), id: UUID())

        // Asking for 10 but only 2 exist
        let results = await index.search(query: Vector([0, 0]), k: 10)
        XCTAssertEqual(results.count, 2)
    }

    func testSearchEmptyIndex() async {
        let index = BruteForceIndex(dimension: 3)
        let results = await index.search(query: Vector([1, 2, 3]), k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testSearchResultsAreSortedByDistance() async throws {
        let index = BruteForceIndex(dimension: 2, metric: EuclideanDistance())

        try await index.add(Vector([10, 0]), id: UUID())
        try await index.add(Vector([1, 0]), id: UUID())
        try await index.add(Vector([5, 0]), id: UUID())

        let results = await index.search(query: Vector([0, 0]), k: 3)
        XCTAssertEqual(results.count, 3)

        // Verify ascending distance order
        for i in 0..<(results.count - 1) {
            XCTAssertLessThanOrEqual(results[i].distance, results[i + 1].distance)
        }
    }

    func testSearchWithCosineMetric() async throws {
        let index = BruteForceIndex(dimension: 2, metric: CosineDistance())

        let sameDirectionID = UUID()
        let orthogonalID = UUID()

        try await index.add(Vector([2, 0]), id: sameDirectionID)
        try await index.add(Vector([0, 5]), id: orthogonalID)

        let results = await index.search(query: Vector([1, 0]), k: 2)
        // [2,0] is same direction as [1,0] → cosine distance ≈ 0
        XCTAssertEqual(results[0].id, sameDirectionID)
        XCTAssertEqual(results[0].distance, 0.0, accuracy: 1e-5)
    }

    // ── Search with Filter ────────────────────────────────────────────

    func testSearchWithFilter() async throws {
        let index = BruteForceIndex(dimension: 2, metric: EuclideanDistance())

        let includeID = UUID()
        let excludeID = UUID()

        // excludeID is actually closer to the query
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

    func testSearchFilterExcludesAll() async throws {
        let index = BruteForceIndex(dimension: 2)
        try await index.add(Vector([1, 0]), id: UUID())

        let results = await index.search(
            query: Vector([1, 0]),
            k: 10,
            filter: { _ in false }
        )
        XCTAssertTrue(results.isEmpty)
    }

    // ── Remove ────────────────────────────────────────────────────────

    func testRemoveExistingVector() async throws {
        let index = BruteForceIndex(dimension: 2)
        let id = UUID()

        try await index.add(Vector([1, 2]), id: id)
        var count = await index.count
        XCTAssertEqual(count, 1)

        let removed = await index.remove(id: id)
        XCTAssertTrue(removed)
        count = await index.count
        XCTAssertEqual(count, 0)
    }

    func testRemoveNonexistentID() async {
        let index = BruteForceIndex(dimension: 2)
        let removed = await index.remove(id: UUID())
        XCTAssertFalse(removed)
    }

    func testRemoveThenSearchDoesNotFindRemoved() async throws {
        let index = BruteForceIndex(dimension: 2, metric: EuclideanDistance())

        let keepID = UUID()
        let removeID = UUID()

        try await index.add(Vector([0, 0]), id: removeID)
        try await index.add(Vector([1, 0]), id: keepID)

        _ = await index.remove(id: removeID)

        let results = await index.search(query: Vector([0, 0]), k: 10)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, keepID)
    }

    // ── Metadata ──────────────────────────────────────────────────────

    func testMetadataRoundTrip() async throws {
        let index = BruteForceIndex(dimension: 2)

        struct Note: Codable, Equatable { let text: String }
        let note = Note(text: "hello")
        let encoded = try JSONEncoder().encode(note)

        let id = UUID()
        try await index.add(Vector([1, 0]), id: id, metadata: encoded)

        let results = await index.search(query: Vector([1, 0]), k: 1)
        let decoded = results.first?.decodeMetadata(as: Note.self)
        XCTAssertEqual(decoded, note)
    }

    func testNoMetadataReturnsNil() async throws {
        let index = BruteForceIndex(dimension: 2)
        try await index.add(Vector([1, 0]), id: UUID())

        let results = await index.search(query: Vector([1, 0]), k: 1)
        XCTAssertNil(results.first?.metadata)
    }

    // ── Property-Based: Correctness ──────────────────────────────────

    func testNearestNeighborIsAlwaysCorrect() async throws {
        // Property: for random data, brute-force must return the actual
        // nearest neighbor (it's exact search, so recall must be 100%).
        let dim = 16
        let n = 100
        let index = BruteForceIndex(dimension: dim, metric: EuclideanDistance())

        // Add random vectors
        var insertedVectors: [(UUID, Vector)] = []
        for _ in 0..<n {
            let components = (0..<dim).map { _ in Float.random(in: -1...1) }
            let v = Vector(components)
            let id = UUID()
            try await index.add(v, id: id)
            insertedVectors.append((id, v))
        }

        // Run 20 random queries
        let metric = EuclideanDistance()
        for _ in 0..<20 {
            let queryComponents = (0..<dim).map { _ in Float.random(in: -1...1) }
            let query = Vector(queryComponents)

            // Find the true nearest neighbor manually
            var bestDist: Float = .infinity
            var bestID: UUID?
            for (id, v) in insertedVectors {
                let d = metric.distance(query, v)
                if d < bestDist {
                    bestDist = d
                    bestID = id
                }
            }

            // BruteForceIndex must agree
            let results = await index.search(query: query, k: 1)
            XCTAssertEqual(results.first?.id, bestID,
                           "Brute force did not find the true nearest neighbor")
            XCTAssertEqual(results.first?.distance ?? 999, bestDist, accuracy: 1e-4)
        }
    }

    // ── Concurrent Access ─────────────────────────────────────────────

    func testConcurrentReads() async throws {
        let index = BruteForceIndex(dimension: 3, metric: CosineDistance())

        for _ in 0..<50 {
            try await index.add(
                Vector([Float.random(in: -1...1),
                        Float.random(in: -1...1),
                        Float.random(in: -1...1)]),
                id: UUID()
            )
        }

        // Fire off 10 concurrent searches — actor guarantees no data races
        await withTaskGroup(of: [SearchResult].self) { group in
            for _ in 0..<10 {
                group.addTask {
                    let q = Vector([Float.random(in: -1...1),
                                    Float.random(in: -1...1),
                                    Float.random(in: -1...1)])
                    return await index.search(query: q, k: 5)
                }
            }

            for await results in group {
                XCTAssertEqual(results.count, 5)
            }
        }
    }
}
