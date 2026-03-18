// CoreMLEmbeddingProvider.swift
// ProximaEmbeddings
//
// Text → Vector using any CoreML model that outputs a float array.
// Designed for sentence-transformer models (BERT, MiniLM, etc.)
// converted to CoreML format via coremltools.

import CoreML
import Foundation
import ProximaKit

/// Embeds text using a CoreML model that outputs a multi-array of floats.
///
/// This is the most flexible embedding provider — bring any sentence-transformer
/// model converted to CoreML format. The provider handles model loading,
/// input preparation, prediction, and output conversion.
///
/// ```swift
/// let provider = try CoreMLEmbeddingProvider(modelAt: modelURL)
/// let vector = try await provider.embed("sunset over the ocean")
/// ```
///
/// **Thread Safety:**
/// `CoreMLEmbeddingProvider` is an actor, so all calls to ``embed(_:)`` and
/// ``embedBatch(_:)`` are serialized automatically. This guarantees safe
/// concurrent access — multiple tasks can call `embed` simultaneously without
/// risk of data races on the underlying `MLModel`. The ``dimension`` property
/// is `nonisolated` and safe to read from any context.
///
/// **Model requirements:**
/// - Input: must accept "input_ids" and "attention_mask" as `MLMultiArray<Int32>`
/// - Output: must produce an `MLMultiArray` of floats (the embedding vector)
///
/// **Converting a model:**
/// ```bash
/// pip install coremltools transformers torch
/// python3 scripts/convert_model.py
/// ```
public actor CoreMLEmbeddingProvider {

    /// The dimension of vectors this provider produces.
    ///
    /// This property is `nonisolated` because it is immutable after initialization.
    public nonisolated let dimension: Int

    /// The loaded CoreML model.
    private let model: MLModel

    /// The name of the output feature containing the embedding.
    private let outputFeatureName: String

    /// Maximum sequence length the model accepts.
    private let maxLength: Int

    /// WordPiece tokenizer (nil if no vocab provided — falls back to hash tokenizer).
    private let tokenizer: WordPieceTokenizer?

    // ── Initialization ────────────────────────────────────────────────

    /// Loads a compiled CoreML model (.mlmodelc directory).
    ///
    /// - Parameters:
    ///   - url: Path to the compiled model directory.
    ///   - vocabURL: Optional path to a WordPiece vocab.txt file for proper tokenization.
    /// - Throws: `EmbeddingError.modelNotAvailable` if the model can't be loaded.
    public init(compiledModelURL url: URL, vocabURL: URL? = nil) throws {
        let model = try Self.loadModel(from: url)
        self.model = model
        (self.outputFeatureName, self.dimension, self.maxLength) = try Self.discoverModelShape(model)
        self.tokenizer = try vocabURL.map { try WordPieceTokenizer(vocabURL: $0) }
    }

    /// Compiles and loads a .mlpackage or .mlmodel file.
    ///
    /// This compiles the model on first load (may take a few seconds).
    /// For production, pre-compile and use `init(compiledModelURL:)`.
    ///
    /// - Parameter url: Path to the .mlpackage or .mlmodel file.
    /// - Throws: `EmbeddingError.modelNotAvailable` if compilation or loading fails.
    /// - Parameters:
    ///   - url: Path to the .mlpackage or .mlmodel file.
    ///   - vocabURL: Optional path to a WordPiece vocab.txt for proper tokenization.
    public init(modelAt url: URL, vocabURL: URL? = nil) throws {
        let compiledURL = try MLModel.compileModel(at: url)
        let model = try Self.loadModel(from: compiledURL)
        self.model = model
        (self.outputFeatureName, self.dimension, self.maxLength) = try Self.discoverModelShape(model)
        self.tokenizer = try vocabURL.map { try WordPieceTokenizer(vocabURL: $0) }
    }

    // ── Embedding ─────────────────────────────────────────────────────

    /// Embeds a text string into a vector.
    ///
    /// Tokenizes the text into integer IDs (basic whitespace tokenizer),
    /// runs the CoreML model, and extracts the output embedding.
    ///
    /// - Parameter text: The text to embed. Must not be empty.
    /// - Returns: A vector of dimension `self.dimension`.
    public func embed(_ text: String) async throws -> Vector {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw EmbeddingError.unsupportedInput("Text is empty")
        }

        let (inputIDs, attentionMask) = try tokenize(trimmed)
        let prediction = try await model.prediction(from: CoreMLInput(
            inputIDs: inputIDs,
            attentionMask: attentionMask
        ))

        return try vectorFromPrediction(prediction)
    }

    /// Embeds multiple texts using batch prediction.
    public func embedBatch(_ texts: [String]) async throws -> [Vector] {
        var results: [Vector] = []
        results.reserveCapacity(texts.count)
        for text in texts {
            results.append(try await embed(text))
        }
        return results
    }

    // ── Tokenization ────────────────────────────────────────────────────
    //
    // Uses WordPieceTokenizer when a vocab.txt is provided (proper BERT tokenization).
    // Falls back to hash-based tokenization otherwise (lower quality but functional).

    private func tokenize(_ text: String) throws -> (MLMultiArray, MLMultiArray) {
        let tokenIDs: [Int32]
        let attentionValues: [Int32]

        if let tokenizer = tokenizer {
            // Proper WordPiece tokenization
            let result = tokenizer.tokenize(text, maxLength: maxLength)
            tokenIDs = result.inputIDs
            attentionValues = result.attentionMask
        } else {
            // Fallback: hash-based tokenization
            let words = text.lowercased().split(separator: " ")
            var ids: [Int32] = [101] // [CLS]
            for word in words.prefix(maxLength - 2) {
                let hash = abs(word.hashValue) % 29000 + 1000
                ids.append(Int32(hash))
            }
            ids.append(102) // [SEP]
            let realLen = ids.count
            while ids.count < maxLength { ids.append(0) }
            tokenIDs = ids
            attentionValues = (0..<maxLength).map { $0 < realLen ? Int32(1) : Int32(0) }
        }

        let inputIDsArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)

        for i in 0..<maxLength {
            inputIDsArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: tokenIDs[i])
            maskArray[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: attentionValues[i])
        }

        return (inputIDsArray, maskArray)
    }

    // ── Output Conversion ─────────────────────────────────────────────

    private func vectorFromPrediction(_ prediction: MLFeatureProvider) throws -> Vector {
        guard let multiArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw EmbeddingError.unsupportedInput("Model output '\(outputFeatureName)' is not a multi-array")
        }

        return try vectorFromMultiArray(multiArray)
    }

    private func vectorFromMultiArray(_ multiArray: MLMultiArray) throws -> Vector {
        // The output may be [1, dim] or [dim]. Flatten to 1D.
        let totalCount = multiArray.count
        let count = min(totalCount, dimension)
        var floats = [Float](repeating: 0, count: count)

        let ptr = multiArray.dataPointer

        switch multiArray.dataType {
        case .float32:
            let src = ptr.bindMemory(to: Float32.self, capacity: totalCount)
            for i in 0..<count { floats[i] = src[i] }
        case .float64:
            let src = ptr.bindMemory(to: Float64.self, capacity: totalCount)
            for i in 0..<count { floats[i] = Float(src[i]) }
        case .float16:
            // Float16 → Float32 via manual IEEE 754 half-precision conversion.
            let src = ptr.bindMemory(to: UInt16.self, capacity: totalCount)
            for i in 0..<count {
                floats[i] = float16ToFloat32(src[i])
            }
        default:
            throw EmbeddingError.unsupportedInput("Unsupported MLMultiArray data type: \(multiArray.dataType.rawValue)")
        }

        return Vector(floats)
    }

    // ── Model Discovery ───────────────────────────────────────────────

    private static func loadModel(from url: URL) throws -> MLModel {
        do {
            return try MLModel(contentsOf: url)
        } catch {
            throw EmbeddingError.modelNotAvailable("Failed to load CoreML model: \(error.localizedDescription)")
        }
    }

    /// Discovers the output feature name, embedding dimension, and max input length.
    private static func discoverModelShape(_ model: MLModel) throws -> (String, Int, Int) {
        let desc = model.modelDescription

        // Find the first multi-array output
        guard let output = desc.outputDescriptionsByName.first(where: {
            $0.value.type == .multiArray
        }) else {
            throw EmbeddingError.modelNotAvailable("Model has no multi-array output")
        }

        let outputName = output.key
        guard let constraint = output.value.multiArrayConstraint else {
            throw EmbeddingError.modelNotAvailable("Cannot determine output dimensions")
        }

        // Dimension is the last element of the shape (handles [1, dim] and [dim])
        let shape = constraint.shape.map { $0.intValue }
        guard let dim = shape.last, dim > 0 else {
            throw EmbeddingError.modelNotAvailable("Invalid output shape: \(shape)")
        }

        // Find max input length from input_ids shape
        var maxLen = 128 // default
        if let inputDesc = desc.inputDescriptionsByName["input_ids"],
           let inputConstraint = inputDesc.multiArrayConstraint {
            let inputShape = inputConstraint.shape.map { $0.intValue }
            if let seqLen = inputShape.last {
                maxLen = seqLen
            }
        }

        return (outputName, dim, maxLen)
    }
}

// ── Float16 Conversion ────────────────────────────────────────────────

/// Converts a UInt16 IEEE 754 half-precision float to Float32.
private func float16ToFloat32(_ half: UInt16) -> Float {
    let sign = (half >> 15) & 0x1
    let exp = (half >> 10) & 0x1F
    let frac = half & 0x3FF
    let signF: Float = sign == 1 ? -1 : 1

    if exp == 0 {
        // Subnormal or zero
        return signF * Float(frac) / 1024.0 * pow(2, -14)
    } else if exp == 31 {
        // Infinity or NaN
        return frac == 0 ? (signF * .infinity) : .nan
    } else {
        return signF * (1.0 + Float(frac) / 1024.0) * pow(2, Float(exp) - 15)
    }
}

// ── CoreML Input Feature Provider ─────────────────────────────────────

private final class CoreMLInput: MLFeatureProvider, @unchecked Sendable {
    let inputIDs: MLMultiArray
    let attentionMask: MLMultiArray

    var featureNames: Set<String> { ["input_ids", "attention_mask"] }

    init(inputIDs: MLMultiArray, attentionMask: MLMultiArray) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids": return MLFeatureValue(multiArray: inputIDs)
        case "attention_mask": return MLFeatureValue(multiArray: attentionMask)
        default: return nil
        }
    }
}
