// main.swift
// ProximaBench CLI entry point.
//
// Subcommands:
//   ground-truth  — compute exact k-NN with BruteForceIndex
//   hnsw          — run HNSW benchmark against a pre-computed GT
//
// Both emit a JSON document matching Benchmarks/JSON_SCHEMA.md.

import Foundation

@main
struct ProximaBenchCLI {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            usage()
            exit(2)
        }

        do {
            switch args[1] {
            case "hnsw":
                try await runHNSW(args: Array(args.dropFirst(2)))
            case "ground-truth":
                try await runGroundTruth(args: Array(args.dropFirst(2)))
            case "distance-kernel":
                try runDistanceKernel(args: Array(args.dropFirst(2)))
            case "insert-shape":
                try await runInsertShape(args: Array(args.dropFirst(2)))
            case "-h", "--help", "help":
                usage()
            default:
                usage()
                exit(2)
            }
        } catch {
            FileHandle.standardError.write(Data("[ProximaBench] error: \(error)\n".utf8))
            exit(1)
        }
    }

    // MARK: - hnsw

    static func runHNSW(args: [String]) async throws {
        let f = Flags(args)

        let opts = BenchOptions(
            baseVectorsPath: f.required("--base"),
            queryVectorsPath: f.required("--queries"),
            groundTruthPath: f.required("--gt"),
            dataset: f.required("--dataset"),
            datasetSize: f.int("--size"),
            queryCount: f.int("--query-count"),
            k: f.int("--k") ?? 10,
            M: f.int("--m") ?? 16,
            efConstruction: f.int("--efc") ?? 200,
            efSearch: f.int("--ef") ?? 50,
            metric: f.string("--metric") ?? "l2",
            seed: f.int("--seed") ?? 42,
            notes: f.string("--notes") ?? "",
            libraryVersion: f.string("--version") ?? "1.4.0-dev",
            outputPath: f.required("--out")
        )
        try await BenchRunner.run(opts)
    }

    // MARK: - ground-truth

    static func runGroundTruth(args: [String]) async throws {
        let f = Flags(args)

        let opts = GroundTruthBuilder.Options(
            baseVectorsPath: f.required("--base"),
            queryVectorsPath: f.required("--queries"),
            dataset: f.required("--dataset"),
            datasetSize: f.int("--size"),
            queryCount: f.int("--query-count"),
            k: f.int("--k") ?? 10,
            metric: f.string("--metric") ?? "l2",
            outputPath: f.required("--out")
        )
        try await GroundTruthBuilder.run(opts)
    }

    // MARK: - distance-kernel (ADR-009 GO/NO-GO sweep)

    static func runDistanceKernel(args: [String]) throws {
        let f = Flags(args)
        let opts = DistanceKernelBench.Options(
            dimensions: intList(f.string("--dims") ?? "384,768"),
            counts: intList(f.string("--counts") ?? "32,256,1024,10240,102400"),
            metrics: strList(f.string("--metrics") ?? "euclidean,cosine"),
            reps: f.int("--reps") ?? 7,
            warmup: f.int("--warmup") ?? 3,
            seed: UInt64(f.int("--seed") ?? 42),
            libraryVersion: f.string("--version") ?? "1.5.0-dev",
            notes: f.string("--notes") ?? "",
            outputPath: f.required("--out")
        )
        try DistanceKernelBench.run(opts)
    }

    // MARK: - insert-shape (characterize the real HNSW build distance shapes)

    static func runInsertShape(args: [String]) async throws {
        let f = Flags(args)
        let opts = InsertShapeBench.Options(
            dimensions: intList(f.string("--dims") ?? "384,768"),
            counts: intList(f.string("--counts") ?? "1000,5000"),
            metrics: strList(f.string("--metrics") ?? "euclidean,cosine"),
            m: f.int("--m") ?? 16,
            efConstruction: f.int("--efc") ?? 200,
            seed: UInt64(f.int("--seed") ?? 42),
            libraryVersion: f.string("--version") ?? "1.5.0-dev",
            notes: f.string("--notes") ?? "",
            outputPath: f.required("--out")
        )
        try await InsertShapeBench.run(opts)
    }

    /// Parses a comma-separated list of ints (e.g. "384,768").
    static func intList(_ s: String) -> [Int] {
        s.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    /// Parses a comma-separated list of strings (e.g. "euclidean,cosine").
    static func strList(_ s: String) -> [String] {
        s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    static func usage() {
        FileHandle.standardError.write(Data("""
        ProximaBench — cross-library ANN benchmark harness

        USAGE:
          ProximaBench hnsw
                --base PATH --queries PATH --gt PATH
                --dataset NAME --out PATH
                [--size N] [--query-count N] [--k 10]
                [--m 16] [--efc 200] [--ef 50]
                [--metric l2|cosine] [--seed 42]
                [--version 1.4.0-dev] [--notes TEXT]

          ProximaBench ground-truth
                --base PATH --queries PATH
                --dataset NAME --out PATH
                [--size N] [--query-count N] [--k 10]
                [--metric l2|cosine]

          ProximaBench distance-kernel        (ADR-009 GO/NO-GO sweep)
                --out PATH
                [--dims 384,768] [--counts 32,256,1024,10240,102400]
                [--metrics euclidean,cosine] [--reps 7] [--warmup 3]
                [--seed 42] [--version 1.5.0-dev] [--notes TEXT]
                Emits a distance-kernel-sweep JSON (own schema; latency
                measured here is NEVER asserted in CI, per ADR-009).

          ProximaBench insert-shape           (characterize HNSW build shapes)
                --out PATH
                [--dims 384,768] [--counts 1000,5000]
                [--metrics euclidean,cosine] [--m 16] [--efc 200]
                [--seed 42] [--version 1.5.0-dev] [--notes TEXT]
                Instruments HNSWIndex.add() distance-eval counts.

        The hnsw / ground-truth subcommands emit documents following
        Benchmarks/JSON_SCHEMA.md.

        """.utf8))
    }
}

/// Minimal --key VALUE / --flag parser. No external deps.
struct Flags {
    private let map: [String: String]

    init(_ args: [String]) {
        var m: [String: String] = [:]
        var i = 0
        while i < args.count {
            let a = args[i]
            if a.hasPrefix("--") {
                if i + 1 < args.count, !args[i + 1].hasPrefix("--") {
                    m[a] = args[i + 1]
                    i += 2
                } else {
                    m[a] = "1"
                    i += 1
                }
            } else {
                i += 1
            }
        }
        self.map = m
    }

    func string(_ key: String) -> String? { map[key] }
    func int(_ key: String) -> Int? { map[key].flatMap(Int.init) }
    func required(_ key: String) -> String {
        guard let v = map[key] else {
            FileHandle.standardError.write(Data("[ProximaBench] missing required flag \(key)\n".utf8))
            exit(2)
        }
        return v
    }
}
