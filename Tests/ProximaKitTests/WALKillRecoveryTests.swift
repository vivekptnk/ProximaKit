// WALKillRecoveryTests.swift
// ProximaKitTests
//
// ADR-013 acceptance 1 (out-of-process): spawn `WALKillWriter` via `Process`,
// SIGKILL it at randomized-but-seeded delays while it ingests into a journaled
// index, then reopen base + WAL in this (parent) process and assert prefix
// semantics:
//   • reopen never crashes and never throws — in particular any `corruptedData`
//     on a WAL the library itself wrote is a hard failure;
//   • recovered live count ∈ [committed high-water mark, attempted], i.e. the
//     recovered state is exactly the base plus a valid record prefix;
//   • the recovered graph is internally consistent and searchable.
//
// Two classes:
//   • `WALKillRecoverySmokeTests` — 5 iterations, RUNS in CI.
//   • `WALKillRecoveryTests` — ≥100 iterations, heavy. Gated behind the
//     `PROXIMA_RUN_KILL_RIG` env var so it self-skips in the PR job (mirrors
//     the benchmarks-only `RecallBenchmarkTests`, which CI excludes via
//     `--skip`). Run locally with `PROXIMA_RUN_KILL_RIG=1 swift test
//     --filter WALKillRecoveryTests`.

import XCTest
@testable import ProximaKit

#if canImport(Darwin)
import Darwin
#endif

/// Shared kill-rig runner. Not a test case itself.
enum WALKillRig {

    static func writerBinary(for test: XCTestCase) throws -> URL {
        #if os(macOS)
        let bundle = Bundle.allBundles.first { $0.bundlePath.hasSuffix(".xctest") }
        guard let productsDir = bundle?.bundleURL.deletingLastPathComponent() else {
            throw XCTSkip("could not locate build products directory")
        }
        let binary = productsDir.appendingPathComponent("WALKillWriter")
        guard FileManager.default.isExecutableFile(atPath: binary.path) else {
            throw XCTSkip("WALKillWriter not built at \(binary.path) — build all products first")
        }
        return binary
        #else
        throw XCTSkip("kill rig requires macOS Process/SIGKILL")
        #endif
    }

    /// Runs `iterations` kill/recover cycles with seeded delays.
    static func run(iterations: Int, seed: UInt64, test: XCTestCase) async throws {
        let binary = try writerBinary(for: test)
        var rng = SplitMix64(seed: seed)
        let dim = 8
        let attempted = 400

        var landedAfterBase = 0
        for iter in 0..<iterations {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("killrig-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }

            let base = dir.appendingPathComponent("index.pxkt")
            let wal = dir.appendingPathComponent("index.pxwal")
            let committedFile = dir.appendingPathComponent("committed.txt")

            // Seeded delay in [200ms, 500ms): long enough that the empty-base
            // checkpoint has always landed, short enough to kill mid-ingest.
            let delayMicros = 200_000 + Int(rng.next() % 300_000)
            let iterSeed = 0xC0FFEE &+ UInt64(iter)

            let process = Process()
            process.executableURL = binary
            process.arguments = [dir.path, "\(attempted)", "\(dim)", "\(iterSeed)", "2000"]
            process.standardError = FileHandle.nullDevice
            try process.run()

            usleep(useconds_t(delayMicros))
            #if canImport(Darwin)
            kill(process.processIdentifier, SIGKILL)
            #endif
            process.waitUntilExit()

            // If the kill landed before the base was even written, there is
            // nothing to recover — skip (rare with the 200ms floor).
            guard FileManager.default.fileExists(atPath: base.path),
                  FileManager.default.fileExists(atPath: wal.path) else {
                continue
            }
            landedAfterBase += 1

            let committed = (try? String(contentsOf: committedFile, encoding: .utf8))
                .flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0

            // The core assertion: reopen must succeed with a typed-only failure
            // surface. Any throw here (especially corruptedData on our own file)
            // is a hard failure.
            let recovered: HNSWIndex
            do {
                recovered = try await HNSWIndex.open(baseURL: base, walURL: wal)
            } catch {
                XCTFail("iter \(iter): reopening a library-written WAL threw \(error) "
                        + "(committed=\(committed), delay=\(delayMicros)µs)")
                continue
            }

            let live = await recovered.liveCount
            let consistent = await recovered.reverseAdjacencyIsConsistent
            // Search must not crash on the recovered graph.
            let hits = await recovered.search(query: seededQuery(dim: dim), k: 5)
            await recovered.closeJournal()

            XCTAssertGreaterThanOrEqual(
                live, committed,
                "iter \(iter): recovered \(live) < durably-committed \(committed) — lost a persisted record")
            XCTAssertLessThanOrEqual(
                live, attempted,
                "iter \(iter): recovered \(live) > attempted \(attempted)")
            XCTAssertTrue(consistent, "iter \(iter): recovered graph inconsistent")
            if live > 0 {
                XCTAssertFalse(hits.isEmpty, "iter \(iter): non-empty recovered index returned no results")
            }
        }

        // Sanity: the rig must actually exercise the post-base window, or it is
        // measuring nothing.
        XCTAssertGreaterThan(landedAfterBase, 0,
                             "no kill landed after the base was written — tune delays")
    }

    private static func seededQuery(dim: Int) -> Vector {
        var g = SplitMix64(seed: 0x9)
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 1000) / 1000.0 })
    }
}

/// Smoke version — RUNS in CI.
final class WALKillRecoverySmokeTests: XCTestCase {
    func testKillRecoverySmoke() async throws {
        try await WALKillRig.run(iterations: 5, seed: 0x5A17, test: self)
    }
}

/// Heavy version — ≥100 iterations. Self-skips unless `PROXIMA_RUN_KILL_RIG`
/// is set (CI excludes it, like `RecallBenchmarkTests`).
final class WALKillRecoveryTests: XCTestCase {
    func testKillRecovery100Iterations() async throws {
        guard ProcessInfo.processInfo.environment["PROXIMA_RUN_KILL_RIG"] != nil else {
            throw XCTSkip("heavy kill rig is opt-in; set PROXIMA_RUN_KILL_RIG=1 to run "
                          + "(CI excludes it via --skip WALKillRecoveryTests)")
        }
        try await WALKillRig.run(iterations: 100, seed: 0xF00D, test: self)
    }
}
