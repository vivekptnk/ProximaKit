// Benchmarks/PersistenceBenchmarks.swift
// Save/load performance and cold start latency.

import XCTest
import Foundation
@testable import ProximaKit

final class PersistenceBenchmarks: XCTestCase {

    // MARK: - Save

    func testSave10K() throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)

        // Build index (replace with actual)
        // let index = HNSWIndex(...)
        // for v in vectors { try await index.add(Vector(v), ...) }

        let tmpPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("bench_10k.proximakit").path

        let (_, elapsed) = measure("save_10K") {
            // Replace with: try await PersistenceEngine.save(index, to: tmpPath)

            // Placeholder: write raw float data to measure I/O baseline
            let data = vectors.flatMap { $0 }
            let byteData = Data(bytes: data, count: data.count * MemoryLayout<Float>.size)
            try? byteData.write(to: URL(fileURLWithPath: tmpPath))
        }

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: tmpPath))?[.size] as? Int ?? 0
        let sizeMB = Double(fileSize) / (1024 * 1024)

        print("  file_size: \(String(format: "%.1f", sizeMB))MB")
        XCTAssertLessThan(elapsed, 0.5, "Save 10K should be < 500ms")
        XCTAssertLessThan(sizeMB, 20.0, "File size 10K/384d should be < 20MB")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }

    // MARK: - Load (Cold Start)

    func testLoad10K() throws {
        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let tmpPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("bench_load_10k.proximakit").path

        // Save first
        let data = vectors.flatMap { $0 }
        let byteData = Data(bytes: data, count: data.count * MemoryLayout<Float>.size)
        try byteData.write(to: URL(fileURLWithPath: tmpPath))

        // Flush file system caches (best effort — real cold start requires process restart)
        // In real benchmarks, save in one process and load in another.

        let (_, elapsed) = measure("load_10K") {
            // Replace with: let index = try await PersistenceEngine.load(from: tmpPath)

            // Placeholder: memory-map the file (simulates mmap-based loading)
            let url = URL(fileURLWithPath: tmpPath)
            let mapped = try? Data(contentsOf: url, options: .alwaysMapped)
            _ = mapped?.count
        }

        print("  cold_start: \(formatDuration(elapsed))")
        XCTAssertLessThan(elapsed, 0.2, "Load 10K cold start should be < 200ms")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }

    // MARK: - End-to-End: Load + First Query

    func testLoadAndQuery10K() throws {
        // This measures the user-perceived startup time:
        // open app → load index from disk → first search result

        let vectors = generateUniformVectors(count: 10_000, dimension: 384)
        let query = generateUniformVectors(count: 1, dimension: 384, seed: 99)[0]
        let tmpPath = FileManager.default.temporaryDirectory
            .appendingPathComponent("bench_e2e_10k.proximakit").path

        // Save
        let data = vectors.flatMap { $0 }
        try Data(bytes: data, count: data.count * MemoryLayout<Float>.size)
            .write(to: URL(fileURLWithPath: tmpPath))

        let (_, elapsed) = measure("load_and_query_10K") {
            // Replace with:
            // let index = try await PersistenceEngine.load(from: tmpPath)
            // let results = try await index.search(query: Vector(query), k: 10)

            let mapped = try? Data(contentsOf: URL(fileURLWithPath: tmpPath), options: .alwaysMapped)
            _ = mapped?.count
        }

        print("  end_to_end: \(formatDuration(elapsed))")
        XCTAssertLessThan(elapsed, 0.25, "Load + first query should be < 250ms")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }
}

// MARK: - Memory Benchmarks

final class MemoryBenchmarks: XCTestCase {

    /// Measures memory overhead per vector in an HNSW index.
    /// Raw float storage for 384d = 1,536 bytes.
    /// Overhead is HNSW graph adjacency lists + metadata.
    func testMemoryPerVector() throws {
        let dimension = 384
        let rawBytesPerVector = dimension * MemoryLayout<Float>.size  // 1,536 bytes
        let counts = [1_000, 5_000, 10_000, 25_000, 50_000]

        print("\n--- Memory Per Vector (384d, M=16) ---")
        print("  count     raw_KB    total_KB  overhead_KB  overhead_%")
        for count in counts {
            let vectors = generateUniformVectors(count: count, dimension: dimension)

            let beforeMem = currentRSS()

            // Build index (replace with actual)
            // let index = HNSWIndex(dimension: dimension, M: 16, efConstruction: 200)
            // for v in vectors { try await index.add(Vector(v), ...) }
            let _ = vectors.flatMap { $0 }  // placeholder: at least allocate the data

            let afterMem = currentRSS()
            let totalBytes = afterMem - beforeMem
            let totalKBPerVector = Double(totalBytes) / Double(count) / 1024
            let rawKB = Double(rawBytesPerVector) / 1024
            let overheadKB = totalKBPerVector - rawKB
            let overheadPct = overheadKB / rawKB * 100

            print("  \(String(count).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.2f", rawKB).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.2f", totalKBPerVector).padding(toLength: 10, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.2f", overheadKB).padding(toLength: 13, withPad: " ", startingAt: 0))"
                + "\(String(format: "%.1f", overheadPct))%")
        }
    }
}

// MARK: - Helpers

/// Returns current RSS (Resident Set Size) in bytes. Platform-specific.
private func currentRSS() -> Int {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Int(info.resident_size) : 0
}
