// MigrateCommand.swift
// ProximaBench
//
// ADR-014 Stage 1 — a thin CLI wrapper over the in-library section-copy
// upgraders (`PersistenceEngine.upgradeToV3` for `.pxkt`, and
// `QuantizedHNSWIndex.upgradeToV3` for PQHW). The upgrade itself lives in the
// library because the primary consumer is an on-device app upgrading its own
// base; this subcommand exists for operators and CI-fixture generation.
//
//   ProximaBench migrate --path index.pxkt          (auto-detect family)
//   ProximaBench migrate --path index.qhnsw --family pqhw

import Foundation
import ProximaKit

enum MigrateCommand {

    private static let pxktMagic: UInt32 = 0x50584B54   // "PXKT"
    private static let pqhwMagic: UInt32 = 0x50514857   // "PQHW"

    static func run(args: [String]) throws {
        let f = Flags(args)
        let path = f.required("--path")
        let url = URL(fileURLWithPath: path)

        let data = try Data(contentsOf: url)
        guard data.count >= 8 else {
            throw MigrateError("file too small to be a ProximaKit index: \(path)")
        }
        let magic = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 0, as: UInt32.self) }.littleEndian
        let versionBefore = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }.littleEndian
        let sizeBefore = data.count

        let family = f.string("--family") ?? detectFamily(magic: magic)
        switch family {
        case "pxkt":
            guard magic == pxktMagic else { throw MigrateError("not a .pxkt file (magic mismatch): \(path)") }
            try PersistenceEngine.upgradeToV3(at: url)
        case "pqhw":
            guard magic == pqhwMagic else { throw MigrateError("not a PQHW file (magic mismatch): \(path)") }
            try QuantizedHNSWIndex.upgradeToV3(at: url)
        default:
            throw MigrateError("unknown family '\(family)' (expected pxkt or pqhw)")
        }

        let after = try Data(contentsOf: url)
        let versionAfter = after.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 4, as: UInt32.self) }.littleEndian
        let note = (versionBefore == 3 && sizeBefore == after.count) ? "  (no-op: already padded v3)" : ""
        print("""
        [ProximaBench migrate] \(path)
          family:  \(family)
          version: \(versionBefore) -> \(versionAfter)
          size:    \(sizeBefore) -> \(after.count) bytes\(note)
        """)
    }

    private static func detectFamily(magic: UInt32) -> String {
        switch magic {
        case pxktMagic: return "pxkt"
        case pqhwMagic: return "pqhw"
        default: return "unknown"
        }
    }
}

struct MigrateError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { self.description = message }
}
