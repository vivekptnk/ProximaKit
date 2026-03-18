// ChunkMetadata.swift
// ProximaKit
//
// Metadata for document chunks stored in a VectorStore.
// Part of the ProximaKit ↔ Lumen RAG integration (ADR-006).

import Foundation

/// Metadata stored alongside each vector chunk in a ``VectorStore``.
///
/// Tracks which document a chunk came from, its position within the document,
/// and the original text for retrieval display.
///
/// ```swift
/// let meta = ChunkMetadata(
///     documentId: "doc-42",
///     chunkIndex: 3,
///     text: "Vectors are stored as contiguous Float32 values."
/// )
/// ```
public struct ChunkMetadata: Codable, Sendable, Equatable {
    /// The source document identifier (e.g., Lumen's document ID).
    public let documentId: String

    /// Zero-based chunk index within the document.
    public let chunkIndex: Int

    /// The original text of this chunk (for retrieval display).
    public let text: String

    /// Optional additional metadata (title, page number, etc).
    public var extra: [String: String]?

    public init(
        documentId: String,
        chunkIndex: Int,
        text: String,
        extra: [String: String]? = nil
    ) {
        self.documentId = documentId
        self.chunkIndex = chunkIndex
        self.text = text
        self.extra = extra
    }
}
