// DemoLabSupport.swift
// ProximaDemoApp
//
// Shared, dependency-free helpers for the two v1.6.0 "lab" features
// (Persistence panel + Benchmark tab): a live process-memory probe and a
// deterministic synthetic-vector generator.
//
// Everything here is reproducible and self-contained — no embedder, no network
// — so the persistence and benchmark demos produce the SAME corpus on every
// run, which is what makes their live measurements trustworthy.

import Foundation
import ProximaKit

#if canImport(Darwin)
import Darwin
#endif

/// Live physical-memory accounting for the current process.
///
/// This is the exact probe `PagedVectorMemoryTests` uses to prove paged mode
/// keeps the vector payload off the resident heap: `task_vm_info`'s
/// `phys_footprint`, the same number the OS jetsam accounts against. Surfacing
/// it in-app lets the Persistence panel show a real "resident vs paged" delta
/// rather than a canned figure.
enum MemoryProbe {

    /// Current process physical footprint in bytes, or 0 if unavailable.
    static func physFootprint() -> UInt64 {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
        #else
        return 0
        #endif
    }

    /// Formats a byte count as megabytes, e.g. `12.4 MB`.
    static func megabytes(_ bytes: UInt64) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }

    /// Formats a byte count as MB with sign (for deltas).
    static func megabytes(_ bytes: Int) -> String {
        String(format: "%.1f MB", Double(bytes) / 1_048_576)
    }
}

/// SplitMix64 — a tiny deterministic generator so the lab corpora are byte-for-byte
/// reproducible across launches (the same seed always yields the same vectors).
/// Mirrors the generator ProximaKit uses internally for seeded index construction;
/// re-declared here because that type is library-internal.
struct DemoRNG: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Deterministic synthetic vectors + ids for the lab corpora. Component values
/// live in `[0, 1)`; a fixed per-index seed makes vector `i` identical on every
/// run so recall and memory numbers are reproducible.
enum SyntheticCorpus {

    /// The `i`-th synthetic vector of the given dimension. Deterministic in `i`.
    static func vector(_ i: Int, dimension: Int) -> Vector {
        var g = DemoRNG(seed: 0xABCD &* (UInt64(i) &+ 1))
        return Vector((0..<dimension).map { _ in
            Float(Double(g.next() % 100_000) / 100_000.0)
        })
    }

    /// A stable UUID derived from an integer index, so re-runs address the same
    /// nodes (and recall set-intersections line up).
    static func id(_ i: Int) -> UUID {
        var bytes = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
                     UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        withUnsafeMutableBytes(of: &bytes) {
            $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self)
        }
        return UUID(uuid: bytes)
    }
}
