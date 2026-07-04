// IndexResidency.swift
// ProximaKit
//
// Shared residency vocabulary for index families that can serve their large
// vector payloads either resident or from a read-only file mapping.

/// How an index materializes its large vector payload.
///
/// `.resident` preserves the historical behavior: all vector payload bytes are
/// decoded into memory. `.paged` serves the payload from a read-only file
/// mapping and faults pages on demand. The exact payload differs by index
/// family: full-precision HNSW maps the base vector section; quantized HNSW
/// maps retained originals for exact rerank.
public enum IndexResidency: Sendable {
    /// Decode vector payload bytes into resident memory.
    case resident

    /// Serve vector payload bytes from a read-only file mapping.
    case paged
}
