// ProductQuantizerError.swift
// ProximaKit
//
// Error types for product quantization operations.

import Foundation

/// Errors that can occur during product quantization operations.
public enum ProductQuantizerError: Error, Sendable, Equatable {
    /// The training set is empty.
    case emptyTrainingSet

    /// The vector dimension is not evenly divisible by the subspace count.
    case dimensionNotDivisible(dimension: Int, subspaceCount: Int)

    /// The flat vector data length doesn't match expected size.
    case invalidVectorData(expected: Int, got: Int)

    /// The PQ code length doesn't match the expected subspace count.
    case codeLengthMismatch(expected: Int, got: Int)

    /// The vector dimension doesn't match the quantizer's trained dimension.
    case dimensionMismatch(expected: Int, got: Int)
}

extension ProductQuantizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .emptyTrainingSet:
            return "Training set must not be empty"
        case .dimensionNotDivisible(let dim, let m):
            return "Dimension \(dim) is not divisible by subspace count \(m)"
        case .invalidVectorData(let expected, let got):
            return "Expected \(expected) floats in vector data, got \(got)"
        case .codeLengthMismatch(let expected, let got):
            return "Expected \(expected) codes, got \(got)"
        case .dimensionMismatch(let expected, let got):
            return "Expected dimension \(expected), got \(got)"
        }
    }
}
