// Vector.swift
// ProximaKit
//
// The fundamental type: a fixed-dimension vector of floats.
// All math uses Accelerate/vDSP — no manual loops over elements.

import Accelerate
import Foundation

/// A dense vector of single-precision floats, backed by `ContiguousArray`
/// for zero-copy access to Accelerate/vDSP operations.
///
/// `Vector` is a value type. Copying a vector copies its data.
/// It conforms to `Sendable`, making it safe to pass across concurrency boundaries.
///
/// ```swift
/// let v = Vector([1.0, 2.0, 3.0])
/// print(v.dimension)  // 3
/// print(v.magnitude)  // 3.7416...
/// ```
public struct Vector: Sendable, Equatable, Hashable, Codable {

    // ── Storage ───────────────────────────────────────────────────────
    // ContiguousArray guarantees a single, contiguous buffer of floats.
    // This matters because vDSP functions take raw pointers — they need
    // the data laid out flat in memory, not scattered across heap objects.

    /// The raw float components of this vector.
    public let components: ContiguousArray<Float>

    // ── Initializers ──────────────────────────────────────────────────

    /// Creates a vector from an array of floats.
    ///
    /// - Parameter components: The float values. Must not be empty.
    public init(_ components: [Float]) {
        precondition(!components.isEmpty, "Vector must have at least one dimension")
        self.components = ContiguousArray(components)
    }

    /// Creates a vector from a contiguous array (zero-copy when possible).
    ///
    /// - Parameter components: The float values. Must not be empty.
    public init(_ components: ContiguousArray<Float>) {
        precondition(!components.isEmpty, "Vector must have at least one dimension")
        self.components = components
    }

    /// Creates a vector by copying from an unsafe buffer pointer.
    ///
    /// This is useful when working with C APIs or memory-mapped data
    /// that hand you a raw pointer to float data.
    ///
    /// - Parameter buffer: A buffer pointer to float values. Must not be empty.
    public init(_ buffer: UnsafeBufferPointer<Float>) {
        precondition(buffer.count > 0, "Vector must have at least one dimension")
        self.components = ContiguousArray(buffer)
    }

    /// Creates a vector of a given dimension, filled with a constant value.
    ///
    /// - Parameters:
    ///   - dimension: The number of elements. Must be positive.
    ///   - repeating: The fill value (default: 0.0).
    public init(dimension: Int, repeating value: Float = 0.0) {
        precondition(dimension > 0, "Dimension must be positive")
        self.components = ContiguousArray(repeating: value, count: dimension)
    }

    // ── Properties ────────────────────────────────────────────────────

    /// The number of dimensions (elements) in this vector.
    @inlinable
    public var dimension: Int { components.count }

    /// The L2 (Euclidean) magnitude of this vector.
    ///
    /// Computed using vDSP: sum of squares, then square root.
    /// For a vector [a, b, c], this is sqrt(a² + b² + c²).
    public var magnitude: Float {
        // vDSP_svesq computes the sum of squares in one vectorized pass.
        // This is much faster than a manual loop for high-dimensional vectors.
        var sumOfSquares: Float = 0
        components.withUnsafeBufferPointer { buffer in
            vDSP_svesq(
                buffer.baseAddress!,  // pointer to first float
                1,                     // stride (1 = every element)
                &sumOfSquares,         // output: single float
                vDSP_Length(buffer.count)
            )
        }
        return sqrt(sumOfSquares)
    }

    // ── Subscript ─────────────────────────────────────────────────────

    /// Access a component by index.
    @inlinable
    public subscript(index: Int) -> Float {
        components[index]
    }

    // ── Vector Math (all vDSP) ────────────────────────────────────────

    /// Computes the dot product of this vector with another.
    ///
    /// The dot product measures how "aligned" two vectors are:
    /// - Parallel vectors → large positive value
    /// - Perpendicular vectors → 0
    /// - Opposite vectors → large negative value
    ///
    /// Uses `vDSP_dotpr` for SIMD-accelerated computation.
    ///
    /// - Parameter other: A vector of the same dimension.
    /// - Returns: The dot product (sum of element-wise products).
    public func dot(_ other: Vector) -> Float {
        precondition(dimension == other.dimension, "Dimension mismatch: \(dimension) vs \(other.dimension)")

        var result: Float = 0

        // withUnsafeBufferPointer gives us a raw C pointer to the
        // contiguous float data — exactly what vDSP functions expect.
        components.withUnsafeBufferPointer { a in
            other.components.withUnsafeBufferPointer { b in
                vDSP_dotpr(
                    a.baseAddress!, 1,   // pointer A, stride
                    b.baseAddress!, 1,   // pointer B, stride
                    &result,             // output
                    vDSP_Length(a.count)  // element count
                )
            }
        }

        return result
    }

    /// Computes the cosine similarity between this vector and another.
    ///
    /// Cosine similarity measures the angle between two vectors, ignoring magnitude:
    /// - 1.0 = identical direction
    /// - 0.0 = perpendicular
    /// - -1.0 = opposite direction
    ///
    /// This is the most common similarity metric for text embeddings because
    /// it focuses on "what direction the vector points" rather than "how long it is."
    ///
    /// Formula: cos(θ) = dot(a, b) / (|a| × |b|)
    ///
    /// - Parameter other: A vector of the same dimension.
    /// - Returns: A value in [-1, 1]. Returns 0 if either vector has zero magnitude.
    public func cosineSimilarity(_ other: Vector) -> Float {
        let magnitudeProduct = self.magnitude * other.magnitude

        // Guard against division by zero. A zero-magnitude vector has no
        // direction, so cosine similarity is undefined — we return 0.
        guard magnitudeProduct > 0 else { return 0 }

        return dot(other) / magnitudeProduct
    }

    /// Computes the L2 (Euclidean) distance between this vector and another.
    ///
    /// This is the straight-line distance in N-dimensional space:
    /// sqrt((a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²)
    ///
    /// Uses vDSP for both the subtraction and sum-of-squares steps.
    ///
    /// - Parameter other: A vector of the same dimension.
    /// - Returns: The Euclidean distance (always >= 0).
    public func l2Distance(_ other: Vector) -> Float {
        precondition(dimension == other.dimension, "Dimension mismatch: \(dimension) vs \(other.dimension)")

        let count = vDSP_Length(dimension)

        // Step 1: Subtract the two vectors element-wise.
        // vDSP_vsub computes B - A (note the reversed order — this is a
        // quirk of the C API). We get a "difference" vector.
        var difference = ContiguousArray<Float>(repeating: 0, count: dimension)
        components.withUnsafeBufferPointer { a in
            other.components.withUnsafeBufferPointer { b in
                difference.withUnsafeMutableBufferPointer { diff in
                    vDSP_vsub(a.baseAddress!, 1, b.baseAddress!, 1, diff.baseAddress!, 1, count)
                }
            }
        }

        // Step 2: Sum of squares of the difference vector.
        var sumOfSquares: Float = 0
        difference.withUnsafeBufferPointer { diff in
            vDSP_svesq(diff.baseAddress!, 1, &sumOfSquares, count)
        }

        return sqrt(sumOfSquares)
    }

    /// Returns a new unit vector (magnitude = 1) pointing in the same direction.
    ///
    /// Normalization is essential for cosine similarity — if vectors are
    /// pre-normalized, cosine similarity simplifies to just a dot product,
    /// which is faster.
    ///
    /// Uses `vDSP_vsdiv` to divide every element by the magnitude in one pass.
    ///
    /// - Returns: A normalized copy of this vector, or a zero vector if magnitude is 0.
    public func normalized() -> Vector {
        var mag = magnitude
        guard mag > 0 else { return self }

        var result = ContiguousArray<Float>(repeating: 0, count: dimension)
        components.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                vDSP_vsdiv(
                    input.baseAddress!, 1,   // input pointer, stride
                    &mag,                     // divisor (scalar)
                    output.baseAddress!, 1,  // output pointer, stride
                    vDSP_Length(dimension)
                )
            }
        }

        return Vector(result)
    }
}
