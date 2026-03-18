// VectorStoreError.swift
// ProximaKit
//
// Errors specific to VectorStore operations.

import Foundation

/// Errors thrown by ``VectorStore`` operations.
public enum VectorStoreError: Error, LocalizedError, Sendable {
    /// The number of chunks and metadata entries don't match.
    case chunkMetadataMismatch(chunks: Int, metadata: Int)

    /// No chunks were provided to add.
    case emptyChunks

    /// The document ID was not found in the store.
    case documentNotFound(String)

    /// Persistence failed.
    case persistenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .chunkMetadataMismatch(let chunks, let metadata):
            return "Chunk count (\(chunks)) does not match metadata count (\(metadata))"
        case .emptyChunks:
            return "No chunks provided"
        case .documentNotFound(let id):
            return "Document not found: \(id)"
        case .persistenceFailed(let detail):
            return "Persistence failed: \(detail)"
        }
    }
}
