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
        }
    }
}
