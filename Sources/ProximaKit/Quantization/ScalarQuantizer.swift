// ScalarQuantizer.swift
// ProximaKit
//
// INT8 symmetric scalar quantization for vector compression.
// Each vector is encoded independently: one signed byte per component
// plus a single Float32 scale (scale = maxAbs / 127, code = round(x / scale)).
//
// Memory reduction: D * 4 bytes → D + 4 bytes (e.g., 384d: 1536B → 388B ≈ 3.96×)
//
// Unlike ProductQuantizer there is NO training phase — the quantizer is
// stateless (dimension only), so any vector can be encoded immediately and
// there is no codebook to persist. See ADR-007 for the design rationale.

import Accelerate
import Foundation

/// An INT8 scalar quantizer that compresses vectors to one signed byte per
/// component plus a per-vector Float32 scale.
///
/// Quantization is symmetric and per-vector:
/// - `scale = maxAbs / 127` (largest absolute component)
/// - `code = round(component / scale)`, clamped to `[-127, 127]`
/// - dequantized `component ≈ Float(code) * scale`
///
/// The per-component reconstruction error is bounded by `scale / 2`.
/// A zero vector is handled explicitly: it encodes as `scale = 0` with
/// all-zero codes and decodes back to the exact zero vector.
///
/// ## Usage
///
/// ```swift
/// let sq = ScalarQuantizer(dimension: 384)
///
/// // Encode: 388 bytes instead of 1536
/// let (codes, scale) = sq.encode(vector)
///
/// // Decode back to an approximate Float32 vector
/// let approx = sq.decodeToVector(codes, scale: scale)
/// ```
public struct ScalarQuantizer: Sendable {

    // ── Configuration ────────────────────────────────────────────────

    /// The vector dimension this quantizer encodes.
    public let dimension: Int

    /// The maximum code magnitude. Codes live in `[-127, 127]`; -128 is
    /// never produced so that the code range stays symmetric around zero.
    public static let maxCode: Float = 127

    // ── Initialization ───────────────────────────────────────────────

    /// Creates a scalar quantizer for vectors of the given dimension.
    ///
    /// - Parameter dimension: The vector dimension. Must be positive.
    public init(dimension: Int) {
        precondition(dimension > 0, "Dimension must be positive")
        self.dimension = dimension
    }

    // ── Encoding ─────────────────────────────────────────────────────

    /// Encodes a vector into INT8 codes and a per-vector scale.
    ///
    /// - Parameter vector: A flat `[Float]` of length `dimension`.
    /// - Returns: `dimension` Int8 codes and the Float32 scale needed to decode.
    public func encode(_ vector: [Float]) -> (codes: [Int8], scale: Float) {
        precondition(vector.count == dimension, "Vector dimension mismatch")

        // Largest absolute component in one vectorized pass.
        var maxAbs: Float = 0
        vector.withUnsafeBufferPointer { buffer in
            vDSP_maxmgv(buffer.baseAddress!, 1, &maxAbs, vDSP_Length(buffer.count))
        }

        // Zero vector: there is no direction to preserve. Store scale 0 with
        // all-zero codes — decode reproduces the exact zero vector and no
        // division by zero ever occurs.
        guard maxAbs > 0 else {
            return (codes: [Int8](repeating: 0, count: dimension), scale: 0)
        }

        var scale = maxAbs / Self.maxCode
        // Float division rounds to nearest, so scale can land an ulp above
        // maxAbs/127 — near greatestFiniteMagnitude that makes the decoded
        // extreme (127 * scale) overflow to infinity. Nudge down until the
        // reconstruction is finite (at most two steps).
        while !(scale * Self.maxCode).isFinite {
            scale = scale.nextDown
        }
        // Subnormal underflow: for 0 < maxAbs < ~1.6e-43 the division
        // flushes to exactly 0, and dividing by that scale would feed
        // ±inf/NaN into the rounding stage. Encode as a zero vector — the
        // reconstruction error is bounded by maxAbs itself (< 1e-42), far
        // below quantization noise for any practical embedding.
        guard scale > 0 else {
            return (codes: [Int8](repeating: 0, count: dimension), scale: 0)
        }

        // Divide every component by the scale (vDSP_vsdiv), then clamp to the
        // code range. By construction |x| / scale <= 127, but Float division
        // can land an epsilon above — the clip makes the conversion safe.
        var divisor = scale
        var scaled = [Float](repeating: 0, count: dimension)
        vector.withUnsafeBufferPointer { input in
            scaled.withUnsafeMutableBufferPointer { output in
                vDSP_vsdiv(
                    input.baseAddress!, 1,
                    &divisor,
                    output.baseAddress!, 1,
                    vDSP_Length(dimension)
                )
            }
        }
        var lo: Float = -Self.maxCode
        var hi: Float = Self.maxCode
        scaled.withUnsafeMutableBufferPointer { buffer in
            vDSP_vclip(
                buffer.baseAddress!, 1,
                &lo, &hi,
                buffer.baseAddress!, 1,
                vDSP_Length(dimension)
            )
        }

        // Round-to-nearest Float → Int8 conversion (vDSP_vfixr8).
        var codes = [Int8](repeating: 0, count: dimension)
        scaled.withUnsafeBufferPointer { input in
            codes.withUnsafeMutableBufferPointer { output in
                vDSP_vfixr8(
                    input.baseAddress!, 1,
                    output.baseAddress!, 1,
                    vDSP_Length(dimension)
                )
            }
        }

        return (codes: codes, scale: scale)
    }

    /// Encodes a `Vector` into INT8 codes and a per-vector scale.
    public func encode(_ vector: Vector) -> (codes: [Int8], scale: Float) {
        encode(Array(vector.components))
    }

    /// Encodes a batch of vectors.
    ///
    /// - Parameter vectors: Vectors of dimension `dimension`.
    /// - Returns: Parallel arrays of codes and scales (row i encodes vector i).
    public func encodeBatch(_ vectors: [Vector]) -> (codes: [[Int8]], scales: [Float]) {
        var codes = [[Int8]]()
        codes.reserveCapacity(vectors.count)
        var scales = [Float]()
        scales.reserveCapacity(vectors.count)
        for vector in vectors {
            let (c, s) = encode(vector)
            codes.append(c)
            scales.append(s)
        }
        return (codes: codes, scales: scales)
    }

    // ── Decoding ─────────────────────────────────────────────────────

    /// Decodes INT8 codes back to an approximate Float32 vector.
    ///
    /// Each component is `Float(code) * scale`. With `scale == 0` (a
    /// quantized zero vector) the result is the exact zero vector.
    ///
    /// - Parameters:
    ///   - codes: `dimension` Int8 codes.
    ///   - scale: The scale produced by `encode`.
    /// - Returns: The reconstructed vector as `[Float]`.
    public func decode(_ codes: [Int8], scale: Float) -> [Float] {
        precondition(codes.count == dimension, "Code length mismatch")

        // Int8 → Float in one vectorized pass (vDSP_vflt8), then scale.
        var result = [Float](repeating: 0, count: dimension)
        codes.withUnsafeBufferPointer { input in
            result.withUnsafeMutableBufferPointer { output in
                vDSP_vflt8(
                    input.baseAddress!, 1,
                    output.baseAddress!, 1,
                    vDSP_Length(dimension)
                )
            }
        }
        var multiplier = scale
        result.withUnsafeMutableBufferPointer { buffer in
            vDSP_vsmul(
                buffer.baseAddress!, 1,
                &multiplier,
                buffer.baseAddress!, 1,
                vDSP_Length(dimension)
            )
        }
        return result
    }

    /// Decodes INT8 codes to a `Vector`.
    public func decodeToVector(_ codes: [Int8], scale: Float) -> Vector {
        Vector(decode(codes, scale: scale))
    }

    /// Decodes a batch of code rows back to vectors.
    ///
    /// - Parameters:
    ///   - codes: Code rows (each of length `dimension`).
    ///   - scales: One scale per row. Must have the same count as `codes`.
    /// - Returns: The reconstructed vectors.
    public func decodeBatch(codes: [[Int8]], scales: [Float]) -> [Vector] {
        precondition(codes.count == scales.count, "codes and scales must have the same count")
        var vectors = [Vector]()
        vectors.reserveCapacity(codes.count)
        for i in 0..<codes.count {
            vectors.append(decodeToVector(codes[i], scale: scales[i]))
        }
        return vectors
    }

    // ── Memory Statistics ────────────────────────────────────────────

    /// Bytes per encoded vector: `dimension` Int8 codes + one Float32 scale.
    public var bytesPerEncodedVector: Int {
        dimension + MemoryLayout<Float>.size
    }

    /// Bytes per original Float32 vector (= dimension * 4).
    public var bytesPerOriginalVector: Int {
        dimension * MemoryLayout<Float>.size
    }

    /// Compression ratio (original / compressed). ≈ 3.96 at 384 dimensions.
    public var compressionRatio: Float {
        Float(bytesPerOriginalVector) / Float(bytesPerEncodedVector)
    }
}
