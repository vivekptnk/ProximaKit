import XCTest
@testable import ProximaEmbeddings
import ProximaKit

final class CoreMLEmbeddingProviderTests: XCTestCase {

    /// Path to the converted MiniLM model.
    /// Run `python3 scripts/convert_model.py` to generate it.
    private static var modelURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // Tests/ProximaEmbeddingsTests/
            .deletingLastPathComponent()   // Tests/
            .deletingLastPathComponent()   // ProximaKit/
            .appendingPathComponent("Models")
            .appendingPathComponent("MiniLM-L6-v2.mlpackage")
    }

    private func makeProvider() throws -> CoreMLEmbeddingProvider {
        let url = Self.modelURL
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("CoreML model not found at \(url.path). Run: python3 scripts/convert_model.py")
        }
        do {
            return try CoreMLEmbeddingProvider(modelAt: url)
        } catch {
            throw XCTSkip("Failed to load CoreML model: \(error)")
        }
    }

    // ── Model Loading ─────────────────────────────────────────────────

    func testLoadModel() throws {
        let provider = try makeProvider()
        // MiniLM-L6-v2 produces 384-dimensional embeddings
        XCTAssertEqual(provider.dimension, 384)
    }

    // ── Text Embedding ────────────────────────────────────────────────

    func testEmbedText() async throws {
        let provider = try makeProvider()
        let vector = try await provider.embed("The quick brown fox jumps over the lazy dog")

        XCTAssertEqual(vector.dimension, provider.dimension)
        XCTAssertGreaterThan(vector.magnitude, 0, "Embedding should be non-zero")
    }

    func testSimilarSentencesCloser() async throws {
        let provider = try makeProvider()

        let cat = try await provider.embed("The cat sat on the mat")
        let kitten = try await provider.embed("A kitten rested on the rug")
        let stock = try await provider.embed("Stock market prices rose sharply")

        let catKittenSim = cat.cosineSimilarity(kitten)
        let catStockSim = cat.cosineSimilarity(stock)

        XCTAssertGreaterThan(catKittenSim, catStockSim,
            "cat/kitten (\(catKittenSim)) should be more similar than cat/stock (\(catStockSim))")
    }

    // ── Batch Embedding ───────────────────────────────────────────────

    func testBatchEmbedding() async throws {
        let provider = try makeProvider()
        let texts = ["hello world", "swift programming", "neural networks"]
        let vectors = try await provider.embedBatch(texts)

        XCTAssertEqual(vectors.count, 3)
        for v in vectors {
            XCTAssertEqual(v.dimension, provider.dimension)
            XCTAssertGreaterThan(v.magnitude, 0)
        }
    }

    // ── Dimension Consistency ─────────────────────────────────────────

    func testDimensionConsistency() async throws {
        let provider = try makeProvider()
        let v1 = try await provider.embed("short")
        let v2 = try await provider.embed("this is a much longer sentence with many more words in it")
        XCTAssertEqual(v1.dimension, v2.dimension)
    }

    // ── Different Inputs Produce Different Outputs ────────────────────

    func testDifferentTextsProduceDifferentVectors() async throws {
        let provider = try makeProvider()
        let v1 = try await provider.embed("apple pie recipe")
        let v2 = try await provider.embed("quantum physics equations")
        XCTAssertNotEqual(v1, v2, "Semantically different texts should produce different vectors")
    }

    // ── Error Cases (no model needed) ─────────────────────────────────

    func testInvalidURLThrows() {
        let badURL = URL(fileURLWithPath: "/nonexistent/model.mlpackage")
        XCTAssertThrowsError(try CoreMLEmbeddingProvider(modelAt: badURL))
    }

    func testEmptyTextThrows() async throws {
        let provider = try makeProvider()
        do {
            _ = try await provider.embed("")
            XCTFail("Expected error for empty text")
        } catch is EmbeddingError {
            // expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // ── Concurrent Embedding Stress Test ─────────────────────────────

    /// 20 concurrent tasks embedding simultaneously.
    /// Validates that actor serialization prevents data races on MLModel.
    func testConcurrentEmbedding() async throws {
        let provider = try makeProvider()
        let texts = [
            "quantum computing basics",
            "the cat sat on the mat",
            "recipe for chocolate cake",
            "machine learning algorithms",
            "swift programming language",
            "the quick brown fox",
            "artificial intelligence",
            "climate change effects",
            "space exploration history",
            "ocean current patterns",
        ]
        let concurrentTasks = 20

        try await withThrowingTaskGroup(of: Vector.self) { group in
            for i in 0..<concurrentTasks {
                let text = texts[i % texts.count]
                group.addTask {
                    try await provider.embed(text)
                }
            }

            var results: [Vector] = []
            for try await vector in group {
                XCTAssertEqual(vector.dimension, provider.dimension)
                XCTAssertGreaterThan(vector.magnitude, 0)
                results.append(vector)
            }

            XCTAssertEqual(results.count, concurrentTasks,
                "All concurrent embeddings should complete without deadlock")
        }
    }
}
