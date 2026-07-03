// WALKillWriter — spawn target for the out-of-process WAL kill rig (ADR-013,
// acceptance 1). It checkpoints an empty journaled base, then ingests seeded
// vectors one at a time with `.everyRecord` durability, recording the count of
// records it has durably appended in a `committed.txt` sidecar after each one.
//
// The parent test SIGKILLs this process at a randomized-but-seeded delay while
// it is mid-ingest, then reopens base + WAL and asserts prefix semantics:
// recovered live count ∈ [committed, attempted], no crash, no corruptedData.
//
// Usage: WALKillWriter <dir> <count> <dim> <seed> <sleepMicros>

import Foundation
import ProximaKit

// SplitMix64 (mirrors the library's seeded generator; kept local so this
// target needs no @testable access).
struct Mix: RandomNumberGenerator {
    var s: UInt64
    init(_ seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

func vector(_ i: Int, dim: Int) -> Vector {
    var g = Mix(0xBEEF &+ UInt64(i))
    return Vector((0..<dim).map { _ in Float(UInt32(truncatingIfNeeded: g.next()) % 1000) / 1000.0 })
}

func detUUID(_ i: Int) -> UUID {
    var b = (UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8,
             UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8)(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    withUnsafeMutableBytes(of: &b) { $0.storeBytes(of: UInt64(i).littleEndian, as: UInt64.self) }
    return UUID(uuid: b)
}

let args = CommandLine.arguments
guard args.count == 6,
      let count = Int(args[2]), let dim = Int(args[3]),
      let seed = UInt64(args[4]), let sleepMicros = UInt32(args[5]) else {
    FileHandle.standardError.write(Data("usage: WALKillWriter <dir> <count> <dim> <seed> <sleepMicros>\n".utf8))
    exit(2)
}
let dir = URL(fileURLWithPath: args[1])
let base = dir.appendingPathComponent("index.pxkt")
let wal = dir.appendingPathComponent("index.pxwal")
let committed = dir.appendingPathComponent("committed.txt")

let config = HNSWConfiguration(m: 6, efConstruction: 40, efSearch: 20,
                               autoCompactionThreshold: nil, levelSeed: seed)
let index = HNSWIndex(dimension: dim, metric: EuclideanDistance(), config: config)
do {
    // Establish the base (generation 1) and a fresh WAL, fsync-per-record.
    try await index.checkpoint(baseURL: base, walURL: wal, durability: .everyRecord)
    for i in 0..<count {
        try await index.add(vector(i, dim: dim), id: detUUID(i))
        // The record is durable (everyRecord fsync). Record the high-water
        // mark; a kill after this reopens with recovered >= (i + 1).
        try Data("\(i + 1)".utf8).write(to: committed, options: .atomic)
        if sleepMicros > 0 { usleep(sleepMicros) }
    }
} catch {
    FileHandle.standardError.write(Data("writer error: \(error)\n".utf8))
    exit(1)
}
