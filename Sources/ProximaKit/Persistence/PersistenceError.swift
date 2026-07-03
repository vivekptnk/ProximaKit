// PersistenceError.swift
// ProximaKit
//
// Errors from index save/load operations.

import Foundation

/// Errors that can occur during index persistence operations.
public enum PersistenceError: Error, LocalizedError {
    /// The file does not start with the expected magic bytes.
    case invalidMagic

    /// The file format version is newer than this library supports.
    case unsupportedVersion(UInt32)

    /// The file is too small to contain a valid header.
    case fileTooSmall

    /// The file contains an unrecognized index type.
    case unknownIndexType(UInt32)

    /// The file contains an unrecognized distance metric type.
    case unknownMetricType(UInt32)

    /// The file data is truncated or corrupt.
    case corruptedData(String)

    /// The metric type cannot be serialized (custom user-defined metric).
    case unserializableMetric

    /// A write-ahead log's recorded parent generation does not match the base
    /// snapshot it is being replayed over. The WAL is stale (its base was
    /// checkpointed past it) or mispaired. Surfaced as a typed error rather
    /// than silently discarded, since silent discard hides data loss (ADR-013).
    case walGenerationMismatch(expected: UInt64, found: UInt64)

    /// A write-ahead log's recorded vector dimension does not match the base
    /// snapshot it is being replayed over — the sidecar was written for a
    /// different index. Rejected up front by `open`: a crafted or mispaired WAL
    /// with a different dimension would otherwise replay mismatched-length
    /// vectors past the public `add(_:id:)` dimension guard (ADR-013).
    case walDimensionMismatch(expected: Int, found: Int)

    /// A write-ahead log's recorded distance metric does not match the base
    /// snapshot it is being replayed over — the sidecar was written for a
    /// different index. Surfaced as a typed error rather than replayed blindly
    /// (ADR-013).
    case walMetricMismatch(expected: UInt32, found: UInt32)

    public var errorDescription: String? {
        switch self {
        case .invalidMagic:
            return "Not a ProximaKit index file (invalid magic bytes)"
        case .unsupportedVersion(let v):
            return "Unsupported format version \(v)"
        case .fileTooSmall:
            return "File too small to contain a valid index"
        case .unknownIndexType(let t):
            return "Unknown index type: \(t)"
        case .unknownMetricType(let t):
            return "Unknown metric type: \(t)"
        case .corruptedData(let detail):
            return "Corrupted index data: \(detail)"
        case .unserializableMetric:
            return "Cannot serialize custom distance metric"
        case .walGenerationMismatch(let expected, let found):
            return "Write-ahead log parent generation \(found) does not match "
                + "base snapshot generation \(expected)"
        case .walDimensionMismatch(let expected, let found):
            return "Write-ahead log dimension \(found) does not match "
                + "base snapshot dimension \(expected)"
        case .walMetricMismatch(let expected, let found):
            return "Write-ahead log metric \(found) does not match "
                + "base snapshot metric \(expected)"
        }
    }
}
