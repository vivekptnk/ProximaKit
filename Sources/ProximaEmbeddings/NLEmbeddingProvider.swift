// NLEmbeddingProvider.swift
// ProximaEmbeddings
//
// Text → Vector using Apple's NaturalLanguage framework.
// Uses sentence embeddings when available, falls back to word-level averaging.

import NaturalLanguage
import ProximaKit

/// Embeds text using Apple's built-in NaturalLanguage embedding models.
///
/// This is the fastest option for on-device text embeddings — no custom model needed.
/// Quality is lower than dedicated sentence-transformers but good for prototyping.
///
/// ```swift
/// let provider = try NLEmbeddingProvider(language: .english)
/// let vector = try await provider.embed("a cute kitten")
/// ```
///
/// **How it works:**
/// - iOS 17+/macOS 14+: Uses `NLEmbedding.sentenceEmbedding` for full-sentence vectors
/// - Fallback: Splits text into words, embeds each with `NLEmbedding.wordEmbedding`,
///   then averages the word vectors. Simple but effective for short texts.
///
/// **Dimension:** Depends on the language model. English word embeddings are typically 512d.
public struct NLEmbeddingProvider: EmbeddingProvider, Sendable {

    /// The dimension of vectors this provider produces.
    public let dimension: Int

    /// The language used for embeddings.
    public let language: NLLanguage

    /// Whether sentence-level embedding is available.
    private let hasSentenceEmbedding: Bool

    /// Creates an NLEmbedding provider for the given language.
    ///
    /// - Parameter language: The language to use. Defaults to `.english`.
    /// - Throws: ``EmbeddingError/modelNotAvailable(_:)`` if no embedding model
    ///   exists for the given language.
    public init(language: NLLanguage = .english) throws {
        self.language = language

        // Check sentence embedding first (better quality, available iOS 17+).
        if let sentenceEmbedding = NLEmbedding.sentenceEmbedding(for: language) {
            self.dimension = sentenceEmbedding.dimension
            self.hasSentenceEmbedding = true
        } else if let wordEmbedding = NLEmbedding.wordEmbedding(for: language) {
            // Fall back to word embeddings.
            self.dimension = wordEmbedding.dimension
            self.hasSentenceEmbedding = false
        } else {
            throw EmbeddingError.modelNotAvailable(
                "No NLEmbedding model available for language: \(language.rawValue)"
            )
        }
    }

    /// Embeds a text string into a vector.
    ///
    /// Uses sentence embedding if available, otherwise averages word embeddings.
    ///
    /// - Parameter text: The text to embed. Must not be empty.
    /// - Returns: A vector representing the text's meaning.
    public func embed(_ text: String) async throws -> Vector {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw EmbeddingError.unsupportedInput("Text is empty")
        }

        if hasSentenceEmbedding {
            return try embedSentence(trimmed)
        } else {
            return try embedByAveragingWords(trimmed)
        }
    }

    // ── Sentence Embedding ────────────────────────────────────────────

    private func embedSentence(_ text: String) throws -> Vector {
        guard let embedding = NLEmbedding.sentenceEmbedding(for: language) else {
            throw EmbeddingError.modelNotAvailable("Sentence embedding unavailable")
        }

        guard let vector = embedding.vector(for: text) else {
            throw EmbeddingError.embeddingNotFound(
                "No sentence embedding for: \"\(text.prefix(50))\""
            )
        }

        return Vector(vector.map { Float($0) })
    }

    // ── Word Averaging Fallback ───────────────────────────────────────

    private func embedByAveragingWords(_ text: String) throws -> Vector {
        guard let embedding = NLEmbedding.wordEmbedding(for: language) else {
            throw EmbeddingError.modelNotAvailable("Word embedding unavailable")
        }

        // Tokenize into words.
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        let words = tokenizer.tokens(for: text.startIndex..<text.endIndex)
            .map { String(text[$0]) }

        guard !words.isEmpty else {
            throw EmbeddingError.unsupportedInput("No embeddable words in text")
        }

        // Embed each word, collect successful embeddings.
        var wordVectors: [[Double]] = []
        for word in words {
            if let vec = embedding.vector(for: word.lowercased()) {
                wordVectors.append(vec)
            }
        }

        guard !wordVectors.isEmpty else {
            throw EmbeddingError.embeddingNotFound(
                "No word embeddings found for any word in: \"\(text.prefix(50))\""
            )
        }

        // Average the word vectors element-wise.
        let dim = wordVectors[0].count
        var sum = [Float](repeating: 0, count: dim)
        for vec in wordVectors {
            for i in 0..<dim {
                sum[i] += Float(vec[i])
            }
        }
        let count = Float(wordVectors.count)
        let averaged = sum.map { $0 / count }

        return Vector(averaged).normalized()
    }
}
