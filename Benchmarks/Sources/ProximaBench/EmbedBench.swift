// EmbedBench.swift
// ProximaBench
//
// Core ML compute-units decision probe for the ROADMAP ANE exploration item.
// This harness is intentionally benchmark-local: it does not change
// CoreMLEmbeddingProvider's public API while measuring whether such a knob is
// justified.

@preconcurrency import CoreML
import Foundation

private struct EmbedBenchError: Error, CustomStringConvertible {
    let description: String
}

private func embedBenchFail(_ message: String) -> EmbedBenchError {
    EmbedBenchError(description: message)
}

private struct EmbedTimingDistribution: Codable {
    let medianMs: Double
    let minMs: Double
    let maxMs: Double
    let spreadPercent: Double
    let repsMs: [Double]
}

private struct EmbedThroughputDistribution: Codable {
    let medianTextsPerSecond: Double
    let minTextsPerSecond: Double
    let maxTextsPerSecond: Double
    let spreadPercent: Double
    let repsTextsPerSecond: [Double]
}

private struct EmbedComputeUnitMeasurement: Codable {
    let computeUnits: String
    let status: String
    let error: String?
    let modelSourceCompileMs: Double?
    let modelLoadMs: Double?
    let firstLoadTotalMs: Double?
    let outputFeatureName: String?
    let dimension: Int?
    let maxSequenceLength: Int?
    let batchLatency: EmbedTimingDistribution?
    let throughput: EmbedThroughputDistribution?
    let checksum: Double?
}

private struct EmbedBenchDecision: Codable {
    let thresholdDeclared: String
    let gateMetric: String
    let observedSpeedup: Double?
    let decision: String
    let reasoning: String
    let reopenConditions: [String]
}

private struct EmbedBenchReport: Codable {
    let schemaVersion: Int
    let kind: String
    let library: String
    let libraryVersion: String
    let platform: Platform
    let modelPath: String
    let modelExists: Bool
    let modelArtifactKind: String
    let batchSize: Int
    let seed: Int
    let reps: Int
    let warmupReps: Int
    let runStartedAt: String
    let runDurationSeconds: Double
    let thresholdDeclared: String
    let inputDeterminism: String
    let feasibility: String
    let gap: String?
    let notes: String
    let measurements: [EmbedComputeUnitMeasurement]
    let decision: EmbedBenchDecision
}

private enum EmbedComputeUnitCase: CaseIterable {
    case cpuOnly
    case cpuAndGPU
    case cpuAndNeuralEngine

    var label: String {
        switch self {
        case .cpuOnly:
            return "cpuOnly"
        case .cpuAndGPU:
            return "cpuAndGPU"
        case .cpuAndNeuralEngine:
            return "cpuAndNeuralEngine"
        }
    }

    var computeUnits: MLComputeUnits {
        switch self {
        case .cpuOnly:
            return .cpuOnly
        case .cpuAndGPU:
            return .cpuAndGPU
        case .cpuAndNeuralEngine:
            return .cpuAndNeuralEngine
        }
    }
}

private final class EmbedFeatureProvider: MLFeatureProvider, @unchecked Sendable {
    let inputIDs: MLMultiArray
    let attentionMask: MLMultiArray

    var featureNames: Set<String> { ["input_ids", "attention_mask"] }

    init(inputIDs: MLMultiArray, attentionMask: MLMultiArray) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "input_ids":
            return MLFeatureValue(multiArray: inputIDs)
        case "attention_mask":
            return MLFeatureValue(multiArray: attentionMask)
        default:
            return nil
        }
    }
}

private struct LoadedEmbeddingModel {
    let model: MLModel
    let outputFeatureName: String
    let dimension: Int
    let maxSequenceLength: Int
    let sourceCompileMs: Double
    let modelLoadMs: Double
    let firstLoadTotalMs: Double
}

enum EmbedBench {
    struct Options {
        let modelPath: String
        let batchSize: Int
        let reps: Int
        let warmup: Int
        let seed: UInt64
        let thresholdSpeedup: Double
        let libraryVersion: String
        let notes: String
        let outputPath: String
    }

    static func run(_ opts: Options) async throws {
        let started = Date()
        let runStart = DispatchTime.now().uptimeNanoseconds
        let modelURL = URL(fileURLWithPath: opts.modelPath)
        let thresholdText = String(
            format: "GO only if cpuAndNeuralEngine median batch throughput is >= %.2fx cpuOnly; otherwise NO-GO: keep Core ML defaults and do not add a public computeUnits knob.",
            opts.thresholdSpeedup
        )
        log("[embed-bench] threshold: \(thresholdText)")

        guard opts.batchSize > 0 else {
            throw embedBenchFail("--batch-size must be > 0")
        }
        guard opts.reps > 0 else {
            throw embedBenchFail("--reps must be > 0")
        }
        guard opts.warmup >= 0 else {
            throw embedBenchFail("--warmup must be >= 0")
        }

        let texts = makeSeededTexts(count: opts.batchSize, seed: opts.seed)
        let exists = FileManager.default.fileExists(atPath: modelURL.path)
        let artifactKind = modelArtifactKind(modelURL)

        let report: EmbedBenchReport
        if !exists {
            let gap = "No Core ML model is available at \(modelURL.path). Provide any small sentence-embedding .mlpackage, .mlmodel, or compiled .mlmodelc with input_ids and attention_mask MLMultiArray inputs and a float MLMultiArray embedding output."
            report = makeGapReport(
                opts: opts,
                started: started,
                runStart: runStart,
                modelURL: modelURL,
                modelExists: false,
                artifactKind: artifactKind,
                thresholdText: thresholdText,
                gap: gap
            )
        } else {
            var measurements: [EmbedComputeUnitMeasurement] = []
            measurements.reserveCapacity(EmbedComputeUnitCase.allCases.count)
            for unit in EmbedComputeUnitCase.allCases {
                measurements.append(runMeasurement(unit: unit, modelURL: modelURL, texts: texts, opts: opts))
            }
            report = makeMeasuredReport(
                opts: opts,
                started: started,
                runStart: runStart,
                modelURL: modelURL,
                artifactKind: artifactKind,
                thresholdText: thresholdText,
                measurements: measurements
            )
        }

        try writeReport(report, to: URL(fileURLWithPath: opts.outputPath))
        log("[embed-bench] wrote \(opts.outputPath)")
        printSummary(report)
    }

    private static func runMeasurement(
        unit: EmbedComputeUnitCase, modelURL: URL, texts: [String], opts: Options
    ) -> EmbedComputeUnitMeasurement {
        do {
            log("[embed-bench] loading \(unit.label)")
            let loaded = try loadModel(modelURL: modelURL, computeUnit: unit)
            for _ in 0..<opts.warmup {
                _ = try embedBatch(texts, loaded: loaded)
            }

            var latencyMs: [Double] = []
            var throughputs: [Double] = []
            var checksum: Double = 0
            for rep in 0..<opts.reps {
                let timed = try timeBatch(texts, loaded: loaded)
                latencyMs.append(timed.ms)
                throughputs.append(Double(texts.count) / (timed.ms / 1_000))
                checksum += timed.checksum
                log(String(
                    format: "[embed-bench] %@ rep %d/%d %.2f texts/s",
                    unit.label, rep + 1, opts.reps, throughputs[throughputs.count - 1]
                ))
            }

            return EmbedComputeUnitMeasurement(
                computeUnits: unit.label,
                status: "measured",
                error: nil,
                modelSourceCompileMs: loaded.sourceCompileMs,
                modelLoadMs: loaded.modelLoadMs,
                firstLoadTotalMs: loaded.firstLoadTotalMs,
                outputFeatureName: loaded.outputFeatureName,
                dimension: loaded.dimension,
                maxSequenceLength: loaded.maxSequenceLength,
                batchLatency: timingDistribution(latencyMs),
                throughput: throughputDistribution(throughputs),
                checksum: checksum.isFinite ? checksum : nil
            )
        } catch {
            log("[embed-bench] \(unit.label) failed: \(error)")
            return EmbedComputeUnitMeasurement(
                computeUnits: unit.label,
                status: "failed",
                error: String(describing: error),
                modelSourceCompileMs: nil,
                modelLoadMs: nil,
                firstLoadTotalMs: nil,
                outputFeatureName: nil,
                dimension: nil,
                maxSequenceLength: nil,
                batchLatency: nil,
                throughput: nil,
                checksum: nil
            )
        }
    }

    private static func loadModel(modelURL: URL, computeUnit: EmbedComputeUnitCase) throws -> LoadedEmbeddingModel {
        let firstLoadStart = DispatchTime.now().uptimeNanoseconds
        let compiledURL: URL
        let sourceCompileMs: Double
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
            sourceCompileMs = 0
        } else {
            let compileStart = DispatchTime.now().uptimeNanoseconds
            compiledURL = try MLModel.compileModel(at: modelURL)
            sourceCompileMs = millisecondsSince(compileStart)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnit.computeUnits
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        let loadMs = millisecondsSince(loadStart)
        let shape = try discoverModelShape(model)
        return LoadedEmbeddingModel(
            model: model,
            outputFeatureName: shape.outputFeatureName,
            dimension: shape.dimension,
            maxSequenceLength: shape.maxSequenceLength,
            sourceCompileMs: sourceCompileMs,
            modelLoadMs: loadMs,
            firstLoadTotalMs: millisecondsSince(firstLoadStart)
        )
    }

    private static func discoverModelShape(
        _ model: MLModel
    ) throws -> (outputFeatureName: String, dimension: Int, maxSequenceLength: Int) {
        let description = model.modelDescription
        guard description.inputDescriptionsByName["input_ids"]?.type == .multiArray else {
            throw embedBenchFail("model is missing required multi-array input 'input_ids'")
        }
        guard description.inputDescriptionsByName["attention_mask"]?.type == .multiArray else {
            throw embedBenchFail("model is missing required multi-array input 'attention_mask'")
        }
        guard let output = description.outputDescriptionsByName.first(where: { $0.value.type == .multiArray }) else {
            throw embedBenchFail("model has no multi-array embedding output")
        }
        guard let outputConstraint = output.value.multiArrayConstraint else {
            throw embedBenchFail("cannot determine output dimensions")
        }
        let outputShape = outputConstraint.shape.map { $0.intValue }
        guard let dimension = outputShape.last, dimension > 0 else {
            throw embedBenchFail("invalid output shape \(outputShape)")
        }

        var maxLength = 128
        if let inputConstraint = description.inputDescriptionsByName["input_ids"]?.multiArrayConstraint {
            let inputShape = inputConstraint.shape.map { $0.intValue }
            if let sequenceLength = inputShape.last, sequenceLength > 0 {
                maxLength = sequenceLength
            }
        }

        return (output.key, dimension, maxLength)
    }

    @inline(never)
    private static func timeBatch(
        _ texts: [String], loaded: LoadedEmbeddingModel
    ) throws -> (ms: Double, checksum: Double) {
        let start = DispatchTime.now().uptimeNanoseconds
        let checksum = try embedBatch(texts, loaded: loaded)
        return (millisecondsSince(start), checksum)
    }

    @inline(never)
    private static func embedBatch(_ texts: [String], loaded: LoadedEmbeddingModel) throws -> Double {
        var checksum: Double = 0
        for text in texts {
            let input = try makeInput(text: text, maxLength: loaded.maxSequenceLength)
            let prediction = try loaded.model.prediction(from: input)
            checksum += try checksumPrediction(
                prediction,
                outputFeatureName: loaded.outputFeatureName,
                dimension: loaded.dimension
            )
        }
        return checksum
    }

    private static func makeInput(text: String, maxLength: Int) throws -> EmbedFeatureProvider {
        let words = text.lowercased().split(separator: " ")
        var ids: [Int32] = [101]
        ids.reserveCapacity(maxLength)
        for word in words.prefix(max(0, maxLength - 2)) {
            ids.append(Int32(stableTokenID(String(word))))
        }
        ids.append(102)
        let realLength = ids.count
        while ids.count < maxLength {
            ids.append(0)
        }

        let inputIDs = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        for i in 0..<maxLength {
            inputIDs[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: ids[i])
            attentionMask[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: i < realLength ? 1 : 0)
        }
        return EmbedFeatureProvider(inputIDs: inputIDs, attentionMask: attentionMask)
    }

    private static func checksumPrediction(
        _ prediction: MLFeatureProvider,
        outputFeatureName: String,
        dimension: Int
    ) throws -> Double {
        guard let multiArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw embedBenchFail("prediction output '\(outputFeatureName)' is not a multi-array")
        }
        let count = min(multiArray.count, dimension)
        guard count > 0 else {
            return 0
        }
        let first = try floatValue(at: 0, in: multiArray)
        let middle = try floatValue(at: count / 2, in: multiArray)
        let last = try floatValue(at: count - 1, in: multiArray)
        return first + middle + last
    }

    private static func floatValue(at index: Int, in multiArray: MLMultiArray) throws -> Double {
        guard index >= 0, index < multiArray.count else {
            throw embedBenchFail("output checksum index \(index) is outside count \(multiArray.count)")
        }
        let shape = multiArray.shape.map { max(1, $0.intValue) }
        guard !shape.isEmpty else {
            throw embedBenchFail("output MLMultiArray has empty shape")
        }

        var remainder = index
        var indices = Array(repeating: NSNumber(value: 0), count: shape.count)
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            let extent = shape[axis]
            indices[axis] = NSNumber(value: remainder % extent)
            remainder /= extent
        }
        return multiArray[indices].doubleValue
    }

    private static func makeMeasuredReport(
        opts: Options,
        started: Date,
        runStart: UInt64,
        modelURL: URL,
        artifactKind: String,
        thresholdText: String,
        measurements: [EmbedComputeUnitMeasurement]
    ) -> EmbedBenchReport {
        let decision = makeDecision(
            measurements: measurements,
            thresholdText: thresholdText,
            thresholdSpeedup: opts.thresholdSpeedup
        )
        let anyMeasured = measurements.contains { $0.status == "measured" }
        let allFailed = !anyMeasured
        return EmbedBenchReport(
            schemaVersion: 1,
            kind: "coreml-embed-compute-units-decision",
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            platform: PlatformProbe.current(),
            modelPath: modelURL.path,
            modelExists: true,
            modelArtifactKind: artifactKind,
            batchSize: opts.batchSize,
            seed: Int(bitPattern: UInt(truncatingIfNeeded: opts.seed)),
            reps: opts.reps,
            warmupReps: opts.warmup,
            runStartedAt: ISO8601DateFormatter().string(from: started),
            runDurationSeconds: secondsSince(runStart),
            thresholdDeclared: thresholdText,
            inputDeterminism: "Batch texts and fallback token IDs are generated from SplitMix64 seed \(opts.seed); no system RNG is used.",
            feasibility: allFailed ? "harness-only" : "measured",
            gap: allFailed ? "A model file was present but no compute-unit configuration produced embeddings; inspect per-unit errors." : nil,
            notes: opts.notes,
            measurements: measurements,
            decision: decision
        )
    }

    private static func makeGapReport(
        opts: Options,
        started: Date,
        runStart: UInt64,
        modelURL: URL,
        modelExists: Bool,
        artifactKind: String,
        thresholdText: String,
        gap: String
    ) -> EmbedBenchReport {
        EmbedBenchReport(
            schemaVersion: 1,
            kind: "coreml-embed-compute-units-decision",
            library: "ProximaKit",
            libraryVersion: opts.libraryVersion,
            platform: PlatformProbe.current(),
            modelPath: modelURL.path,
            modelExists: modelExists,
            modelArtifactKind: artifactKind,
            batchSize: opts.batchSize,
            seed: Int(bitPattern: UInt(truncatingIfNeeded: opts.seed)),
            reps: opts.reps,
            warmupReps: opts.warmup,
            runStartedAt: ISO8601DateFormatter().string(from: started),
            runDurationSeconds: secondsSince(runStart),
            thresholdDeclared: thresholdText,
            inputDeterminism: "Batch texts and fallback token IDs are generated from SplitMix64 seed \(opts.seed); no system RNG is used.",
            feasibility: "harness-only",
            gap: gap,
            notes: opts.notes,
            measurements: [],
            decision: EmbedBenchDecision(
                thresholdDeclared: thresholdText,
                gateMetric: "cpuAndNeuralEngine.throughput.medianTextsPerSecond / cpuOnly.throughput.medianTextsPerSecond",
                observedSpeedup: nil,
                decision: "NEEDS-MODEL",
                reasoning: "No local usable Core ML sentence-embedding model was available for this run; zero fake numbers emitted.",
                reopenConditions: [
                    "Place any small sentence-embedding .mlpackage, .mlmodel, or compiled .mlmodelc on this machine.",
                    "The model must accept 'input_ids' and 'attention_mask' MLMultiArray inputs and produce a float MLMultiArray embedding output.",
                    "Run embed-bench again with --model PATH on mission 6 hardware."
                ]
            )
        )
    }

    private static func makeDecision(
        measurements: [EmbedComputeUnitMeasurement],
        thresholdText: String,
        thresholdSpeedup: Double
    ) -> EmbedBenchDecision {
        let cpu = measurements.first { $0.computeUnits == EmbedComputeUnitCase.cpuOnly.label }
        let ane = measurements.first { $0.computeUnits == EmbedComputeUnitCase.cpuAndNeuralEngine.label }
        let cpuThroughput = cpu?.throughput?.medianTextsPerSecond
        let aneThroughput = ane?.throughput?.medianTextsPerSecond

        guard let cpuThroughput, cpuThroughput > 0, let aneThroughput, aneThroughput > 0 else {
            return EmbedBenchDecision(
                thresholdDeclared: thresholdText,
                gateMetric: "cpuAndNeuralEngine.throughput.medianTextsPerSecond / cpuOnly.throughput.medianTextsPerSecond",
                observedSpeedup: nil,
                decision: "NO-GO",
                reasoning: "The benchmark could not obtain both cpuOnly and cpuAndNeuralEngine throughput measurements, so a public computeUnits knob is not justified from this run.",
                reopenConditions: [
                    "Re-run on ANE-capable Apple silicon with a compatible local sentence-embedding Core ML model.",
                    "Keep the pre-declared >=\(String(format: "%.2f", thresholdSpeedup))x threshold before measuring."
                ]
            )
        }

        let speedup = aneThroughput / cpuThroughput
        let go = speedup >= thresholdSpeedup
        return EmbedBenchDecision(
            thresholdDeclared: thresholdText,
            gateMetric: "cpuAndNeuralEngine.throughput.medianTextsPerSecond / cpuOnly.throughput.medianTextsPerSecond",
            observedSpeedup: speedup,
            decision: go ? "GO" : "NO-GO",
            reasoning: go
                ? "cpuAndNeuralEngine cleared the pre-declared batch-throughput threshold; a public computeUnits knob is justified for design work."
                : "cpuAndNeuralEngine did not clear the pre-declared batch-throughput threshold; keep Core ML defaults and do not add a public computeUnits knob.",
            reopenConditions: [
                "Measure a production embedding model and production batch shape on target hardware.",
                "Re-open only if cpuAndNeuralEngine median batch throughput is >=\(String(format: "%.2f", thresholdSpeedup))x cpuOnly with acceptable first-load latency."
            ]
        )
    }

    private static func makeSeededTexts(count: Int, seed: UInt64) -> [String] {
        let vocabulary = [
            "atlas", "beacon", "carbon", "delta", "ember", "fusion", "garden", "harbor",
            "ion", "jupiter", "kernel", "lattice", "matrix", "nebula", "orbit", "pulse",
            "quartz", "river", "signal", "tensor", "ultra", "vector", "window", "zenith"
        ]
        var rng = BenchSeededRandom(seed: seed)
        var texts: [String] = []
        texts.reserveCapacity(count)
        for i in 0..<count {
            let length = 8 + Int(rng.next() % 12)
            var words: [String] = []
            words.reserveCapacity(length + 1)
            words.append("sample\(i)")
            for _ in 0..<length {
                words.append(vocabulary[Int(rng.next() % UInt64(vocabulary.count))])
            }
            texts.append(words.joined(separator: " "))
        }
        return texts
    }

    private static func stableTokenID(_ word: String) -> Int {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in word.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01B3
        }
        return Int(hash % 29_000) + 1_000
    }

    private static func timingDistribution(_ values: [Double]) -> EmbedTimingDistribution {
        let sorted = values.sorted()
        guard let minValue = sorted.first, let maxValue = sorted.last else {
            return EmbedTimingDistribution(medianMs: 0, minMs: 0, maxMs: 0, spreadPercent: 0, repsMs: [])
        }
        let medianValue = median(sorted)
        return EmbedTimingDistribution(
            medianMs: medianValue,
            minMs: minValue,
            maxMs: maxValue,
            spreadPercent: spreadPercent(min: minValue, max: maxValue, median: medianValue),
            repsMs: values
        )
    }

    private static func throughputDistribution(_ values: [Double]) -> EmbedThroughputDistribution {
        let sorted = values.sorted()
        guard let minValue = sorted.first, let maxValue = sorted.last else {
            return EmbedThroughputDistribution(
                medianTextsPerSecond: 0,
                minTextsPerSecond: 0,
                maxTextsPerSecond: 0,
                spreadPercent: 0,
                repsTextsPerSecond: []
            )
        }
        let medianValue = median(sorted)
        return EmbedThroughputDistribution(
            medianTextsPerSecond: medianValue,
            minTextsPerSecond: minValue,
            maxTextsPerSecond: maxValue,
            spreadPercent: spreadPercent(min: minValue, max: maxValue, median: medianValue),
            repsTextsPerSecond: values
        )
    }

    private static func median(_ sorted: [Double]) -> Double {
        if sorted.count.isMultiple(of: 2) {
            return (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
        }
        return sorted[sorted.count / 2]
    }

    private static func spreadPercent(min: Double, max: Double, median: Double) -> Double {
        median > 0 ? (max - min) / median * 100 : 0
    }

    private static func modelArtifactKind(_ url: URL) -> String {
        let ext = url.pathExtension
        return ext.isEmpty ? "unknown" : ".\(ext)"
    }

    private static func writeReport(_ report: EmbedBenchReport, to url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: url, options: .atomic)
    }

    private static func secondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    }

    private static func millisecondsSince(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("\(message)\n".utf8))
    }

    private static func printSummary(_ report: EmbedBenchReport) {
        let cpu = report.measurements.first { $0.computeUnits == "cpuOnly" }?
            .throughput?.medianTextsPerSecond
        let ane = report.measurements.first { $0.computeUnits == "cpuAndNeuralEngine" }?
            .throughput?.medianTextsPerSecond
        let cpuText = cpu.map { String(format: "%.2f texts/s", $0) } ?? "n/a"
        let aneText = ane.map { String(format: "%.2f texts/s", $0) } ?? "n/a"
        let speedupText = report.decision.observedSpeedup.map {
            String(format: "%.2fx", $0)
        } ?? "n/a"
        print("""
        -- embed-bench ------------------------------------------------------
        feasibility     : \(report.feasibility)
        threshold       : \(report.thresholdDeclared)
        cpuOnly median  : \(cpuText)
        ANE median      : \(aneText)
        ANE speedup     : \(speedupText)
        decision        : \(report.decision.decision)
        output          : \(report.modelPath)
        -------------------------------------------------------------------
        """)
    }
}
