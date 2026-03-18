// TextEmbedder.swift
// ProximaKit
//
// Protocol for text-to-vector conversion used by VectorStore.
// Defined in ProximaKit so VectorStore doesn't depend on ProximaEmbeddings.
// ProximaEmbeddings' EmbeddingProvider conforms automatically.

import Foundation

/// A type that converts text into vectors for similarity search.
///
/// This protocol mirrors the shape of `EmbeddingProvider` from ProximaEmbeddings
/// but lives in ProximaKit so that ``VectorStore`` has no dependency on the
/// embeddings module. Any `EmbeddingProvider` automatically satisfies this protocol.
///
/// ```swift
/// struct MyEmbedder: TextEmbedder {
///     let dimension = 384
///     func embed(_ text: String) async throws -> Vector { ... }
/// }
/// let store = try VectorStore(name: "docs", embedder: MyEmbedder(), ...)
/// ```
public protocol TextEmbedder: Sendable {
    /// The dimension of vectors this embedder produces.
    var dimension: Int { get }

    /// Embeds a single text string into a vector.
    func embed(_ text: String) async throws -> Vector

    /// Embeds multiple texts in one call.
    func embedBatch(_ texts: [String]) async throws -> [Vector]
}

extension TextEmbedder {
    /// Default batch implementation: embeds each text concurrently.
    public func embedBatch(_ texts: [String]) async throws -> [Vector] {
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
