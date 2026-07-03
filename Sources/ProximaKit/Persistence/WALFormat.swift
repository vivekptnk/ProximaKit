// WALFormat.swift
// ProximaKit
//
// PXWL v1 — write-ahead-log sidecar codec (ADR-013, Stage 1 / Option A).
//
// A `.pxwal` sidecar journals graph mutations (add / remove) appended after
// each mutation, replayed over the last full `.pxkt` snapshot on load. This
// file is the PURE format layer: record types, CRC32, little-endian framing,
// and a torn-tail-tolerant decoder. It performs no I/O — the file-backed
// writer and fsync policy live in `WALJournal.swift`.
//
// On-disk layout (all little-endian, per house convention):
//
//   Header (fixed, 32 bytes):
//     0  magic:            UInt32 = "PXWL"
//     4  version:          UInt32 = 1
//     8  parentGeneration: UInt64   — binds this WAL to a base snapshot's
//                                      generation; a mismatch is a typed error.
//     16 dimension:        UInt32
//     20 metricRaw:        UInt32   — DistanceMetricType raw value (sanity)
//     24 headerCRC:        UInt32   — CRC32 over bytes 0..<24
//     28 reserved:         UInt32 = 0
//
//   Records (repeated), each framed:
//     [payloadLength: UInt32][crc32: UInt32][payload: payloadLength bytes]
//
//   Payload:
//     opcode: UInt8   (0 = add, 1 = remove)
//     add:    UUID(16) + level: Int32 + vector(dimension × Float32)
//             + metadataLength: UInt32 + metadata bytes
//     remove: UUID(16)
//
// The CRC over each record's payload gives torn-tail detection: a crash mid
// append leaves a partial or bit-damaged final record whose CRC fails (or whose
// declared length runs past EOF), so replay stops at the last intact record —
// the "longest valid prefix" contract. A CRC failure or short read is NOT a
// corrupt-file error; it is expected truncation and recovers the prefix
// silently. Only a damaged *header* (bad magic / version / header CRC) or a
// stale parent generation raises a typed `PersistenceError`.

import Foundation

// MARK: - CRC32 (IEEE 802.3, reflected, poly 0xEDB88320)

/// Standard CRC-32 used for WAL record integrity. Not cryptographic; chosen
/// for cheap, deterministic torn-tail and bit-flip detection. Pure Foundation
/// so the core library keeps its zero-dependency contract.
enum CRC32 {
    private static let table: [UInt32] = {
        (0..<256).map { i -> UInt32 in
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1)
            }
            return c
        }
    }()

    /// CRC32 over a byte region. `seed` allows chaining; callers pass 0.
    static func checksum<C: Collection>(_ bytes: C, seed: UInt32 = 0) -> UInt32
    where C.Element == UInt8 {
        var crc = ~seed
        for byte in bytes {
            crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8)
        }
        return ~crc
    }
}

// MARK: - Record model

/// A decoded WAL mutation record (value-typed, `Sendable`).
enum WALRecord: Sendable, Equatable {
    case add(id: UUID, level: Int, vector: [Float], metadata: Data?)
    case remove(id: UUID)
}

// MARK: - Constants

enum WALFormat {
    static let magic: UInt32 = 0x5058_574C   // "PXWL"
    static let version: UInt32 = 1
    static let headerSize = 32
    static let opcodeAdd: UInt8 = 0
    static let opcodeRemove: UInt8 = 1

    /// Encodes the fixed 32-byte header (little-endian) with a trailing CRC
    /// over its first 24 bytes.
    static func encodeHeader(parentGeneration: UInt64, dimension: Int, metricRaw: UInt32) -> Data {
        var data = Data()
        data.reserveCapacity(headerSize)
        appendLE(&data, magic)
        appendLE(&data, version)
        appendLE(&data, parentGeneration)
        appendLE(&data, UInt32(dimension))
        appendLE(&data, metricRaw)
        let crc = CRC32.checksum(data)          // over the first 24 bytes
        appendLE(&data, crc)
        appendLE(&data, UInt32(0))              // reserved
        assert(data.count == headerSize)
        return data
    }

    /// Encodes one framed record: `[length][crc32][payload]`.
    static func encodeRecord(_ record: WALRecord) -> Data {
        var payload = Data()
        switch record {
        case let .add(id, level, vector, metadata):
            payload.append(opcodeAdd)
            appendUUID(&payload, id)
            appendLE(&payload, Int32(level))
            vector.withUnsafeBytes { payload.append(contentsOf: $0) }
            let meta = metadata ?? Data()
            appendLE(&payload, UInt32(meta.count))
            payload.append(meta)
        case let .remove(id):
            payload.append(opcodeRemove)
            appendUUID(&payload, id)
        }
        var framed = Data()
        framed.reserveCapacity(8 + payload.count)
        appendLE(&framed, UInt32(payload.count))
        appendLE(&framed, CRC32.checksum(payload))
        framed.append(payload)
        return framed
    }
}

// MARK: - Decoder (torn-tail-tolerant, prefix semantics)

/// The outcome of decoding a WAL: the longest intact record prefix plus how
/// many trailing bytes were dropped as a torn/partial tail.
struct WALReplay: Sendable {
    let parentGeneration: UInt64
    let dimension: Int
    let metricRaw: UInt32
    let records: [WALRecord]
    /// Bytes after the last intact record that were discarded (torn tail or a
    /// bit-damaged record and everything past it). Zero for a clean WAL.
    let trailingBytesDropped: Int
}

enum WALDecoder {

    /// Decodes a whole WAL image.
    ///
    /// - Throws `PersistenceError.invalidMagic` / `.unsupportedVersion` /
    ///   `.corruptedData` for a damaged header, and `.walGenerationMismatch`
    ///   when `expectedGeneration` is supplied and disagrees with the header.
    /// - Never throws for a torn record tail: it stops at the last intact
    ///   record and reports the dropped byte count. This is the crash-recovery
    ///   contract — a torn tail is expected, not corruption.
    static func decode(_ data: Data, expectedGeneration: UInt64?) throws -> WALReplay {
        guard data.count >= WALFormat.headerSize else {
            throw PersistenceError.fileTooSmall
        }
        let fileMagic = data.loadLE(UInt32.self, at: 0)
        guard fileMagic == WALFormat.magic else {
            throw PersistenceError.invalidMagic
        }
        let version = data.loadLE(UInt32.self, at: 4)
        guard version == WALFormat.version else {
            throw PersistenceError.unsupportedVersion(version)
        }
        // Header integrity: CRC over bytes 0..<24.
        let storedHeaderCRC = data.loadLE(UInt32.self, at: 24)
        let computedHeaderCRC = data.prefix(24).withUnsafeBytes {
            CRC32.checksum($0.bindMemory(to: UInt8.self))
        }
        guard storedHeaderCRC == computedHeaderCRC else {
            throw PersistenceError.corruptedData("WAL header CRC mismatch")
        }
        let parentGeneration = data.loadLE(UInt64.self, at: 8)
        let dimension = Int(data.loadLE(UInt32.self, at: 16))
        let metricRaw = data.loadLE(UInt32.self, at: 20)
        guard dimension > 0 else {
            throw PersistenceError.corruptedData("WAL dimension must be positive, got \(dimension)")
        }
        if let expected = expectedGeneration, expected != parentGeneration {
            throw PersistenceError.walGenerationMismatch(expected: expected, found: parentGeneration)
        }

        // ── Records: read framed entries, stop at the first torn/failed one ──
        var records: [WALRecord] = []
        var offset = WALFormat.headerSize
        let end = data.count
        var lastGoodOffset = offset

        while offset + 8 <= end {
            let payloadLength = Int(data.loadLE(UInt32.self, at: offset))
            let storedCRC = data.loadLE(UInt32.self, at: offset + 4)
            // A zero-length or over-long frame can only be a torn/garbled tail
            // (the writer never emits an empty payload). Stop; drop the rest.
            guard payloadLength > 0, offset + 8 + payloadLength <= end else { break }
            let payloadStart = offset + 8
            let payloadEnd = payloadStart + payloadLength
            let payload = data[payloadStart..<payloadEnd]
            let computed = payload.withUnsafeBytes { CRC32.checksum($0.bindMemory(to: UInt8.self)) }
            guard computed == storedCRC else { break }   // torn / bit-flip: stop
            // CRC is intact, so the payload is exactly what the writer wrote —
            // an undecodable payload here would be a library bug, not truncation.
            guard let record = decodePayload(payload, dimension: dimension) else { break }
            records.append(record)
            offset = payloadEnd
            lastGoodOffset = offset
        }

        return WALReplay(
            parentGeneration: parentGeneration,
            dimension: dimension,
            metricRaw: metricRaw,
            records: records,
            trailingBytesDropped: end - lastGoodOffset
        )
    }

    /// Decodes a single CRC-verified payload. Returns `nil` only if the bytes
    /// are structurally impossible (which a passing CRC makes a library bug,
    /// not a torn tail); the caller stops replay defensively in that case.
    private static func decodePayload(_ payload: Data, dimension: Int) -> WALRecord? {
        // `payload` is a Data slice; index from its own startIndex.
        let base = payload.startIndex
        guard payload.count >= 1 else { return nil }
        let opcode = payload[base]
        switch opcode {
        case WALFormat.opcodeRemove:
            guard payload.count == 1 + 16 else { return nil }
            let id = readUUID(payload, at: base + 1)
            return .remove(id: id)
        case WALFormat.opcodeAdd:
            let vectorBytes = dimension * 4
            let fixed = 1 + 16 + 4 + vectorBytes + 4
            guard payload.count >= fixed else { return nil }
            var cursor = base + 1
            let id = readUUID(payload, at: cursor); cursor += 16
            let level = Int(Int32(bitPattern: payload.loadLE(UInt32.self, at: cursor - base))); cursor += 4
            var floats = [Float](repeating: 0, count: dimension)
            payload.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: (cursor - base))
                floats.withUnsafeMutableBytes { dst in
                    dst.copyMemory(from: UnsafeRawBufferPointer(start: src, count: vectorBytes))
                }
            }
            cursor += vectorBytes
            let metaLen = Int(payload.loadLE(UInt32.self, at: cursor - base)); cursor += 4
            guard payload.count == fixed + metaLen else { return nil }
            let metadata: Data? = metaLen == 0 ? nil : Data(payload[cursor..<cursor + metaLen])
            return .add(id: id, level: level, vector: floats, metadata: metadata)
        default:
            return nil
        }
    }

    private static func readUUID(_ data: Data, at index: Data.Index) -> UUID {
        data.withUnsafeBytes { raw in
            UUID(uuid: raw.loadUnaligned(fromByteOffset: index - data.startIndex, as: uuid_t.self))
        }
    }
}

// MARK: - Little-endian write helpers (file-private mirror of PersistenceEngine's)

private func appendLE(_ data: inout Data, _ value: UInt32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
private func appendLE(_ data: inout Data, _ value: Int32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
private func appendLE(_ data: inout Data, _ value: UInt64) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
private func appendUUID(_ data: inout Data, _ uuid: UUID) {
    withUnsafeBytes(of: uuid.uuid) { data.append(contentsOf: $0) }
}
