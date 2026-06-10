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

    // ── Output Normalization Contract ─────────────────────────────────

    /// Regression test (CHA audit): both the sentence-embedding path and the
    /// word-averaging fallback must return L2-normalized (unit-length)
    /// vectors. Before the fix, the sentence path returned the raw model
    /// vector (magnitude ≈ 9 for English), while the fallback normalized —
    /// so EuclideanDistance/DotProductDistance behavior depended on which
    /// path the language happened to select.
    func testEmbedReturnsUnitVectors() async throws {
        let provider = try makeProvider()

        let inputs = [
            "hello",
            "The quick brown fox jumps over the lazy dog",
            "vector search on device",
        ]
        for text in inputs {
            let vector = try await provider.embed(text)
            XCTAssertEqual(vector.magnitude, 1.0, accuracy: 1e-4,
                "Embedding for \"\(text)\" should be unit-length, got magnitude \(vector.magnitude)")
        }
    }

    /// Normalization must not change cosine-similarity rankings.
    func testNormalizationPreservesCosineRanking() async throws {
        let provider = try makeProvider()

        let cat = try await provider.embed("cat")
        let kitten = try await provider.embed("kitten")
        let car = try await provider.embed("car")

        XCTAssertGreaterThan(cat.cosineSimilarity(kitten), cat.cosineSimilarity(car),
            "Unit-normalized vectors should preserve semantic ranking")
    }

    func testDimensionConsistency() async throws {
        let provider = try makeProvider()

        let v1 = try await provider.embed("first text")
        let v2 = try await provider.embed("second text")
        XCTAssertEqual(v1.dimension, v2.dimension,
                       "All vectors from same provider should have equal dimension")
    }
}
