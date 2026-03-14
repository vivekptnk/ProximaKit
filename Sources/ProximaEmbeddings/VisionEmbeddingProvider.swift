// VisionEmbeddingProvider.swift
// ProximaEmbeddings
//
// Image → Vector using Apple's Vision framework.
// Uses VNGenerateImageFeaturePrintRequest to extract image features.

import CoreGraphics
import ProximaKit
import Vision

/// Embeds images into vectors using Apple's Vision framework feature prints.
///
/// Feature prints are fixed-dimension representations of images that capture
/// visual content (objects, textures, scenes). Similar images produce similar
/// feature prints, enabling semantic image search.
///
/// ```swift
/// let provider = VisionEmbeddingProvider()
/// let vector = try await provider.embed(cgImage)
/// try await index.add(vector, id: photoID)
/// ```
///
/// **Note:** This provider does NOT conform to `EmbeddingProvider` (which is
/// text-focused). It has its own `embed(_ image: CGImage)` method.
public struct VisionEmbeddingProvider: Sendable {

    /// Creates a Vision embedding provider.
    public init() {}

    /// Embeds an image into a vector using Vision feature prints.
    ///
    /// - Parameter image: The image to embed.
    /// - Returns: A vector representation of the image's visual content.
    /// - Throws: ``EmbeddingError/imageProcessingFailed(_:)`` if Vision fails.
    public func embed(_ image: CGImage) async throws -> Vector {
        let request = VNGenerateImageFeaturePrintRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])

        try handler.perform([request])

        guard let observation = request.results?.first else {
            throw EmbeddingError.imageProcessingFailed("No feature print observation returned")
        }

        return try vectorFromObservation(observation)
    }

    /// Extracts a Vector from a VNFeaturePrintObservation.
    private func vectorFromObservation(_ observation: VNFeaturePrintObservation) throws -> Vector {
        let elementCount = observation.elementCount

        guard observation.elementType == .float else {
            throw EmbeddingError.imageProcessingFailed(
                "Unexpected feature print element type: \(observation.elementType.rawValue)"
            )
        }

        // Copy feature print data into a float array.
        var floats = [Float](repeating: 0, count: elementCount)
        let data = observation.data
        data.withUnsafeBytes { rawBuffer in
            guard let src = rawBuffer.baseAddress else { return }
            floats.withUnsafeMutableBufferPointer { dest in
                dest.baseAddress?.update(
                    from: src.assumingMemoryBound(to: Float.self),
                    count: elementCount
                )
            }
        }

        return Vector(floats)
    }
}
