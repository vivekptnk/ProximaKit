import XCTest
@testable import ProximaEmbeddings
import ProximaKit

final class NLEmbeddingProviderTests: XCTestCase {

    private func makeProvider() throws -> NLEmbeddingProvider {
        do {
            return try NLEmbeddingProvider()
        } catch {
            throw XCTSkip("NLEmbedding not available on this system")
        }
    }

    func testEmbedSingleWord() async throws {
        let provider = try makeProvider()
        let vector = try await provider.embed("hello")

        let dim = provider.dimension
        XCTAssertEqual(vector.dimension, dim)
        XCTAssertGreaterThan(vector.magnitude, 0, "Embedding should be non-zero")
    }

    func testEmbedSentence() async throws {
        let provider = try makeProvider()
        let vector = try await provider.embed("The quick brown fox jumps over the lazy dog")

        let dim = provider.dimension
        XCTAssertEqual(vector.dimension, dim)
        XCTAssertGreaterThan(vector.magnitude, 0)
    }

    func testSimilarWordsCloserThanDissimilar() async throws {
        let provider = try makeProvider()

        let cat = try await provider.embed("cat")
        let kitten = try await provider.embed("kitten")
        let car = try await provider.embed("car")

        let catKittenSim = cat.cosineSimilarity(kitten)
        let catCarSim = cat.cosineSimilarity(car)

        // "cat" and "kitten" should be more similar than "cat" and "car"
        XCTAssertGreaterThan(catKittenSim, catCarSim,
            "Expected cat-kitten similarity (\(catKittenSim)) > cat-car similarity (\(catCarSim))")
    }

    func testBatchEmbedding() async throws {
        let provider = try makeProvider()
        let texts = ["hello", "world", "swift", "code", "vector"]

        let vectors = try await provider.embedBatch(texts)
        XCTAssertEqual(vectors.count, 5)

        for vector in vectors {
            XCTAssertEqual(vector.dimension, provider.dimension)
        }
    }

    func testEmptyStringThrows() async throws {
        let provider = try makeProvider()

        do {
            _ = try await provider.embed("")
            XCTFail("Expected error for empty string")
        } catch is EmbeddingError {
            // expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testWhitespaceOnlyThrows() async throws {
        let provider = try makeProvider()

        do {
            _ = try await provider.embed("   \n\t  ")
            XCTFail("Expected error for whitespace-only string")
        } catch is EmbeddingError {
            // expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testDimensionConsistency() async throws {
        let provider = try makeProvider()

        let v1 = try await provider.embed("first text")
        let v2 = try await provider.embed("second text")
        XCTAssertEqual(v1.dimension, v2.dimension,
                       "All vectors from same provider should have equal dimension")
    }
}
