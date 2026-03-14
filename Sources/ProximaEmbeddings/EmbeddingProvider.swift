// EmbeddingProvider.swift
// ProximaEmbeddings
//
// Protocol for converting content into vectors.
// Implementations wrap Apple frameworks: NaturalLanguage, Vision, CoreML.

import Foundation
import ProximaKit

/// A provider that converts text into vectors for similarity search.
///
/// Conform to this protocol to create custom embedding backends.
/// ProximaEmbeddings ships two built-in providers:
/// - ``NLEmbeddingProvider``: Apple's NaturalLanguage framework (fast, lower quality)
/// - `CoreMLEmbeddingProvider`: Any Core ML model that outputs float arrays (future)
///
/// ```swift
/// let provider = try NLEmbeddingProvider()
/// let vector = try await provider.embed("sunset over the ocean")
/// try await index.add(vector, id: UUID())
/// ```
public protocol EmbeddingProvider: Sendable {
    /// The dimension of vectors this provider produces.
    /// All vectors from a given provider have the same dimension.
    var dimension: Int { get }

    /// Embeds a single text string into a vector.
    ///
    /// - Parameter text: The text to embed. Must not be empty.
    /// - Returns: A vector representation of the text's meaning.
    /// - Throws: ``EmbeddingError`` if the text cannot be embedded.
    func embed(_ text: String) async throws -> Vector

    /// Embeds multiple texts in one call.
    ///
    /// The default implementation calls `embed(_:)` for each text using TaskGroup.
    /// Providers can override this for batch-optimized implementations.
    ///
    /// - Parameter texts: The texts to embed.
    /// - Returns: Vectors in the same order as the input texts.
    func embedBatch(_ texts: [String]) async throws -> [Vector]
}

// Default batch implementation using TaskGroup.
extension EmbeddingProvider {
    public func embedBatch(_ texts: [String]) async throws -> [Vector] {
        // Use indexed tasks to preserve order (TaskGroup results arrive out of order).
        try await withThrowingTaskGroup(of: (Int, Vector).self) { group in
            for (i, text) in texts.enumerated() {
                group.addTask {
                    let vector = try await self.embed(text)
                    return (i, vector)
                }
            }

            var results = [(Int, Vector)]()
            results.reserveCapacity(texts.count)
            for try await result in group {
                results.append(result)
            }

            return results.sorted { $0.0 < $1.0 }.map(\.1)
        }
    }
}

/// Errors from embedding operations.
public enum EmbeddingError: Error, LocalizedError, Sendable {
    /// The input text has no embedding in the model's vocabulary.
    case embeddingNotFound(String)

    /// The embedding model or language is not available on this device.
    case modelNotAvailable(String)

    /// Image processing failed (Vision framework error).
    case imageProcessingFailed(String)

    /// The input is not supported by this provider (e.g., empty string).
    case unsupportedInput(String)

    public var errorDescription: String? {
        switch self {
        case .embeddingNotFound(let detail):
            return "Embedding not found: \(detail)"
        case .modelNotAvailable(let detail):
            return "Model not available: \(detail)"
        case .imageProcessingFailed(let detail):
            return "Image processing failed: \(detail)"
        case .unsupportedInput(let detail):
            return "Unsupported input: \(detail)"
        }
    }
}
