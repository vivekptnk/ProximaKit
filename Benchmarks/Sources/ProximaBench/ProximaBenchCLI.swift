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

        Emits a JSON document following Benchmarks/JSON_SCHEMA.md.

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
