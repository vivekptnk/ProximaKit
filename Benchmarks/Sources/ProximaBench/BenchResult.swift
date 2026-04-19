// BenchResult.swift
// Mirrors Benchmarks/JSON_SCHEMA.md v1. Keep in sync with python/common.py.

import Foundation

struct BenchResult: Codable {
    let schemaVersion: Int
    let library: String
    let libraryVersion: String
    let dataset: String
    let datasetSize: Int
    let dimension: Int
    let metric: String
    let indexParams: IndexParams
    let k: Int
    let queryCount: Int
    let buildTimeSeconds: Double
    let searchLatencyMeanMs: Double
    let searchLatencyP50Ms: Double
    let searchLatencyP95Ms: Double
    let queriesPerSecond: Double
    let recallAt10: Double
    let residentMemoryMb: Double
    let platform: Platform
    let seed: Int
    let runStartedAt: String
    let runDurationSeconds: Double
    let notes: String
}

struct IndexParams: Codable {
    let type: String
    let M: Int?
    let efConstruction: Int?
    let efSearch: Int?
}

struct Platform: Codable {
    let os: String
    let kernel: String
    let arch: String
    let cpuModel: String
    let swiftVersion: String?
    let pythonVersion: String?
}

struct GroundTruthFile: Codable {
    let schemaVersion: Int
    let dataset: String
    let datasetSize: Int
    let dimension: Int
    let metric: String
    let k: Int
    let queryCount: Int
    // Per-query integer indices into the base set (0-based), length = queryCount * k.
    let neighbors: [Int]
}
