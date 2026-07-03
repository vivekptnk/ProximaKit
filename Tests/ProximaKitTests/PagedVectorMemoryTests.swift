// PagedVectorMemoryTests.swift
// ProximaKitTests
//
// ADR-013 Stage 2, acceptance criterion 3 (paged-mode memory). Benchmark-class
// and CI-excluded: like RecallBenchmarkTests it does real work (builds a
// ≥100K×384d fixture), so it is gated behind PROXIMA_PAGED_BENCH=1 and is a
// no-op otherwise — it never runs in the `swift test --skip RecallBenchmarkTests`
// PR gate unless the env var is explicitly set.
//
// It proves the vector payload (4·d·n bytes) is NOT resident in paged mode by
// measuring `phys_footprint` (task_vm_info) deltas: opening the SAME base paged
// vs resident, the difference must recover essentially the whole vector payload
// — the resident open pays for it, the paged open does not.

import XCTest
@testable import ProximaKit

#if canImport(Darwin)
import Darwin
#endif

final class PagedVectorMemoryTests: XCTestCase {

    /// Current process physical footprint (bytes) via `task_vm_info`'s
    /// `phys_footprint` — the same accounting the OS jetsam uses.
    private func physFootprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    private func vec(_ i: Int, dim: Int) -> Vector {
        var g = SplitMix64(seed: 0x5A5A &* UInt64(i + 1))
        return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 65_536) / 65_536.0 })
    }
    private func uuid(_ i: Int) -> UUID {
        var b = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                 UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &b) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
        return UUID(uuid: b)
    }

    /// Builds the fixture to disk and returns immediately, so the (resident)
    /// builder actor is out of scope before any footprint is measured.
    private func buildBase(base: URL, wal: URL, count: Int, dim: Int) async throws {
        let config = HNSWConfiguration(m: 16, efConstruction: 64, efSearch: 50,
                                       autoCompactionThreshold: nil, levelSeed: 0xADD_1)
        let idx = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
        for i in 0..<count {
            try await idx.add(vec(i, dim: dim), id: uuid(i))
        }
        try await idx.checkpoint(baseURL: base, walURL: wal)   // padded v3
        await idx.closeJournal()
    }

    func testPagedModeDoesNotResidentTheVectorPayload() async throws {
        guard ProcessInfo.processInfo.environment["PROXIMA_PAGED_BENCH"] == "1" else {
            throw XCTSkip("set PROXIMA_PAGED_BENCH=1 to run the paged-memory benchmark")
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("paged-mem-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let base = dir.appendingPathComponent("index.pxkt")
        let wal = dir.appendingPathComponent("index.pxwal")

        let count = 100_000
        let dim = 384
        let payloadBytes = UInt64(count * dim * 4)   // 153,600,000

        try await buildBase(base: base, wal: wal, count: count, dim: dim)

        // ── Paged open FIRST (clean measurement) ──────────────────────
        let f0 = physFootprint()
        let paged = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .paged)
        let pagedCount = await paged.count
        XCTAssertEqual(pagedCount, count)
        let f1 = physFootprint()
        let pagedDelta = f1 &- f0

        // A handful of searches: the working set faults in a bounded number of
        // vector pages, nowhere near the whole payload.
        for q in 0..<50 { _ = await paged.search(query: vec(1_000_000 + q, dim: dim), k: 10) }
        let f1b = physFootprint()
        let pagedAfterSearch = f1b &- f0

        // ── Resident open SECOND (pays for the whole payload) ─────────
        let resident = try await HNSWIndex.open(baseURL: base, walURL: wal, mode: .resident)
        let residentCount = await resident.count
        XCTAssertEqual(residentCount, count)
        let f2 = physFootprint()
        let residentDelta = f2 &- f1b

        await paged.closeJournal()
        await resident.closeJournal()

        func mb(_ b: UInt64) -> String { String(format: "%.1f MB", Double(b) / 1_048_576) }
        print("""
        ── paged-memory benchmark (phys_footprint) ───────────────────────
        fixture           : \(count) × \(dim)d  (vector payload = \(mb(payloadBytes)))
        paged open delta   : \(mb(pagedDelta))
        paged +50 searches : \(mb(pagedAfterSearch))
        resident open delta: \(mb(residentDelta))
        payload recovered  : \(mb(residentDelta > pagedDelta ? residentDelta - pagedDelta : 0))
        ──────────────────────────────────────────────────────────────────
        """)

        // The paged open must NOT resident the vector payload: opening the SAME
        // base resident costs at least ~the payload MORE than opening it paged.
        // Threshold derived from the measured baseline (see ADR-013 Stage 2
        // notes) with margin: require ≥ 60% of the payload recovered.
        XCTAssertGreaterThan(
            residentDelta, pagedDelta,
            "resident open must cost more than paged open")
        XCTAssertGreaterThan(
            residentDelta &- pagedDelta, payloadBytes * 6 / 10,
            "paged mode must keep ≥60% of the \(mb(payloadBytes)) vector payload off the resident heap "
            + "(paged=\(mb(pagedDelta)), resident=\(mb(residentDelta)))")

        // Even after warm searches, the paged footprint stays well under a full
        // resident load (bounded working set, not the whole payload).
        XCTAssertLessThan(
            pagedAfterSearch, pagedDelta &+ payloadBytes / 2,
            "warm paged searches must fault only a bounded working set, not the whole payload")
    }
}
