// ProximaEmbeddings — Converts content into vectors.
//
// This module provides protocol-based embedding providers:
// - NLEmbeddingProvider (Apple's NaturalLanguage framework)
// - CoreMLEmbeddingProvider (any CoreML model outputting floats)
// - VisionEmbeddingProvider (image embeddings via Vision)
//
// Depends on ProximaKit for the Vector type.
// Imports: CoreML, NaturalLanguage, Vision.

import ProximaKit

/// The ProximaEmbeddings namespace.
public enum ProximaEmbeddings {
    /// The current module version.
    public static let version = ProximaKit.version
}

