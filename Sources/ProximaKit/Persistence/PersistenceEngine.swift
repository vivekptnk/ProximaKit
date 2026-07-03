// PersistenceEngine.swift
// ProximaKit
//
// Binary persistence for ProximaKit indices.
// Format: 64-byte header + UUIDs + raw Float32 vectors + graph + metadata.
// Files are read with `.mappedIfSafe` (the OS maps rather than copies the
// raw bytes during parsing), but every section is then decoded into Swift
// arrays — a loaded index is fully memory-resident, not paged on demand.
// See ADR-003 for design rationale.

import Foundation

// MARK: - Constants

private let magic: UInt32 = 0x50584B54  // "PXKT"

/// Current on-disk format version.
///
/// - v1: original layout; `autoCompactionThreshold` was not serialized and
///   silently reset to the `HNSWConfiguration` default (`0.7`) on load.
/// - v2: encodes `autoCompactionThreshold` in the previously reserved header
///   bytes at offset 56 (Float64 bit pattern; all-zero bits encode `nil`).
///   v1 files remain readable; they load with the documented default (`0.7`).
/// - v3: identical 64-byte legacy header and body to v2, plus a fixed trailer
///   appended after the metadata section (ADR-013): a per-section table
///   (offset + length) and a `snapshotGeneration` that binds a WAL sidecar to
///   this base. The sequential loader stops after the metadata section, so
///   v3 files load byte-for-byte like v2 in the resident path; only the WAL
///   layer reads the trailer. `minSupportedVersion` stays 1 (v1/v2 load exactly
///   as today). The legacy `save(_:to:)` writers keep stamping v2 so their
///   output is byte-identical to before; only the streaming-persistence
///   checkpoint writer (`saveHNSW(_:generation:to:)`) stamps v3.
private let formatVersion: UInt32 = 2
private let minSupportedVersion: UInt32 = 1
/// Highest version this reader understands. v3 adds the WAL-binding trailer.
private let maxReadableVersion: UInt32 = 3
private let v3FormatVersion: UInt32 = 3
private let headerSize = 64

// ── v3 trailer (ADR-013) ──────────────────────────────────────────────
// Fixed layout appended after the metadata section:
//   sectionCount: UInt32 (= 5)
//   [uuids, vectors, nodeLevels, adjacency, metadata] × (offset: UInt64, length: UInt64)
//   snapshotGeneration: UInt64
//   trailerMagic: UInt32 ("PXK3")
private let v3SectionCount = 5
private let v3TrailerMagic: UInt32 = 0x5058_4B33  // "PXK3"
private let v3TrailerSize = 4 + v3SectionCount * 16 + 8 + 4  // = 96

/// The `HNSWConfiguration` default threshold, applied when loading v1 files
/// that predate threshold serialization.
private let legacyDefaultCompactionThreshold = 0.7

private let indexTypeBruteForce: UInt32 = 0
private let indexTypeHNSW: UInt32 = 1

// MARK: - Snapshot Types

/// A snapshot of a BruteForce index's state for persistence.
public struct BruteForceSnapshot: Sendable {
    public let dimension: Int
    public let metricType: DistanceMetricType
    public let vectorData: [Float]
    public let ids: [UUID]
    public let metadataStore: [Data?]
}

/// A snapshot of an HNSW index's state for persistence.
public struct HNSWSnapshot: Sendable {
    public let dimension: Int
    public let config: HNSWConfiguration
    public let metricType: DistanceMetricType
    public let vectors: [Vector]
    public let metadata: [Data?]
    public let nodeToUUID: [UUID]
    public let layers: [[[Int]]]
    public let nodeLevels: [Int]
    public let entryPointNode: Int?
    public let maxLevel: Int
}

// MARK: - PersistenceEngine

/// Binary persistence for ProximaKit indices.
///
/// Saves indices to a compact binary format. Vectors are stored as contiguous
/// Float32 values for fast bulk decoding. Loading reads the file with
/// `.mappedIfSafe` — which avoids an up-front copy of the raw bytes — but all
/// sections are copied into Swift arrays during parsing, so the loaded index
/// is fully resident in memory (no OS paging of vector data after load).
///
/// ```swift
/// // Save
/// try await index.save(to: fileURL)
///
/// // Load
/// let loaded = try HNSWIndex.load(from: fileURL)
/// ```
public enum PersistenceEngine {

    // MARK: - Save BruteForce

    /// Saves a BruteForce index snapshot to binary format.
    public static func save(_ snapshot: BruteForceSnapshot, to url: URL) throws {
        var data = Data()
        let count = snapshot.ids.count

        // Reserve approximate capacity
        let vectorBytes = count * snapshot.dimension * 4
        data.reserveCapacity(headerSize + count * 16 + vectorBytes)

        // ── Header ────────────────────────────────────────────────────
        appendUInt32(&data, magic)
        appendUInt32(&data, formatVersion)
        appendUInt32(&data, indexTypeBruteForce)
        appendUInt32(&data, UInt32(snapshot.dimension))
        appendUInt32(&data, UInt32(count))
        appendUInt32(&data, snapshot.metricType.rawValue)
        // HNSW fields (zeroed for BruteForce)
        for _ in 0..<4 { appendUInt32(&data, 0) }   // m, mMax0, efC, efS
        appendInt32(&data, -1)  // entryPoint
        appendInt32(&data, -1)  // maxLevel
        appendUInt32(&data, 0)  // layerCount

        // Metadata offset placeholder (filled after we know the offset)
        let metadataOffsetPosition = data.count
        appendUInt32(&data, 0)

        // Reserved
        appendUInt64(&data, 0)

        assert(data.count == headerSize)

        // ── UUIDs ─────────────────────────────────────────────────────
        for id in snapshot.ids {
            appendUUID(&data, id)
        }

        // ── Vectors (raw Float32) ─────────────────────────────────────
        snapshot.vectorData.withUnsafeBytes { buffer in
            data.append(contentsOf: buffer)
        }

        // ── Metadata ──────────────────────────────────────────────────
        let metadataOffset = UInt32(data.count)
        writeMetadataOffset(&data, offset: metadataOffset, at: metadataOffsetPosition)
        appendMetadata(&data, snapshot.metadataStore)

        try data.write(to: url, options: .atomic)
    }

    // MARK: - Load BruteForce

    /// Loads a BruteForce index from a binary file.
    public static func loadBruteForce(from url: URL) throws -> BruteForceIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try readHeader(fileData)

        guard header.indexType == indexTypeBruteForce else {
            throw PersistenceError.unknownIndexType(header.indexType)
        }

        let count = Int(header.count)
        let dimension = Int(header.dimension)
        var offset = headerSize

        // ── Header sanity (prevents traps before section reads) ───────
        guard dimension > 0 else {
            throw PersistenceError.corruptedData("Dimension must be positive, got \(dimension)")
        }

        // ── UUIDs ─────────────────────────────────────────────────────
        let ids = try readUUIDs(fileData, count: count, offset: &offset)

        // ── Vectors ───────────────────────────────────────────────────
        let (floatCount, floatOverflow) = count.multipliedReportingOverflow(by: dimension)
        guard !floatOverflow else {
            throw PersistenceError.corruptedData("Vector section truncated")
        }
        let vectorData = try readFloats(fileData, count: floatCount, offset: &offset)

        // ── Metadata ──────────────────────────────────────────────────
        var metaOffset = Int(header.metadataOffset)
        let metadata = try readMetadata(fileData, count: count, offset: &metaOffset)

        let snapshot = BruteForceSnapshot(
            dimension: dimension,
            metricType: header.metricType,
            vectorData: vectorData,
            ids: ids,
            metadataStore: metadata
        )

        return BruteForceIndex(restoring: snapshot)
    }

    // MARK: - Save HNSW

    /// Saves an HNSW index snapshot to binary format.
    public static func save(_ snapshot: HNSWSnapshot, to url: URL) throws {
        var data = Data()
        let count = snapshot.nodeToUUID.count

        // ── Header ────────────────────────────────────────────────────
        appendUInt32(&data, magic)
        appendUInt32(&data, formatVersion)
        appendUInt32(&data, indexTypeHNSW)
        appendUInt32(&data, UInt32(snapshot.dimension))
        appendUInt32(&data, UInt32(count))
        appendUInt32(&data, snapshot.metricType.rawValue)
        appendUInt32(&data, UInt32(snapshot.config.m))
        appendUInt32(&data, UInt32(snapshot.config.mMax0))
        appendUInt32(&data, UInt32(snapshot.config.efConstruction))
        appendUInt32(&data, UInt32(snapshot.config.efSearch))
        appendInt32(&data, snapshot.entryPointNode.map { Int32($0) } ?? -1)
        appendInt32(&data, Int32(snapshot.maxLevel))
        appendUInt32(&data, UInt32(snapshot.layers.count))

        let metadataOffsetPosition = data.count
        appendUInt32(&data, 0) // placeholder

        // Auto-compaction threshold (v2): Float64 bit pattern.
        // `nil` (auto-compaction disabled) is encoded as all-zero bits,
        // which can never collide with a valid threshold in (0, 1).
        appendUInt64(&data, snapshot.config.autoCompactionThreshold?.bitPattern ?? 0)

        assert(data.count == headerSize)

        // ── UUIDs ─────────────────────────────────────────────────────
        for uuid in snapshot.nodeToUUID {
            appendUUID(&data, uuid)
        }

        // ── Vectors (raw Float32) ─────────────────────────────────────
        for vector in snapshot.vectors {
            vector.components.withUnsafeBufferPointer { buffer in
                buffer.withMemoryRebound(to: UInt8.self) { bytes in
                    data.append(contentsOf: bytes)
                }
            }
        }

        // ── Graph: nodeLevels ─────────────────────────────────────────
        for level in snapshot.nodeLevels {
            appendInt32(&data, Int32(level))
        }

        // ── Graph: per-layer adjacency ────────────────────────────────
        for layer in snapshot.layers {
            for neighbors in layer {
                appendUInt32(&data, UInt32(neighbors.count))
                for neighbor in neighbors {
                    appendInt32(&data, Int32(neighbor))
                }
            }
        }

        // ── Metadata ──────────────────────────────────────────────────
        let metadataOffset = UInt32(data.count)
        writeMetadataOffset(&data, offset: metadataOffset, at: metadataOffsetPosition)
        appendMetadata(&data, snapshot.metadata)

        try data.write(to: url, options: .atomic)
    }

    // MARK: - Load HNSW

    /// Loads an HNSW index from a binary file.
    public static func loadHNSW(from url: URL) throws -> HNSWIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try readHeader(fileData)

        guard header.indexType == indexTypeHNSW else {
            throw PersistenceError.unknownIndexType(header.indexType)
        }

        let count = Int(header.count)
        let dimension = Int(header.dimension)
        let layerCount = Int(header.layerCount)
        var offset = headerSize

        // ── Header sanity (prevents traps before section reads) ───────
        guard dimension > 0 else {
            throw PersistenceError.corruptedData("Dimension must be positive, got \(dimension)")
        }
        guard layerCount <= fileData.count else {
            throw PersistenceError.corruptedData(
                "Layer count \(layerCount) implausible for file of \(fileData.count) bytes")
        }
        // m >= 2 matches HNSWConfiguration's precondition; m == 1 would trap
        // in the initializer (1/log(1) is infinite), so reject it here instead.
        guard header.m >= 2, header.efConstruction > 0, header.efSearch > 0 else {
            throw PersistenceError.corruptedData(
                "HNSW configuration fields out of range "
                + "(m: \(header.m) [min 2], efConstruction: \(header.efConstruction), efSearch: \(header.efSearch))")
        }
        guard UInt64(header.mMax0) == 2 * UInt64(header.m) else {
            throw PersistenceError.corruptedData(
                "mMax0 \(header.mMax0) inconsistent with m \(header.m) (expected \(2 * UInt64(header.m)))")
        }

        // ── UUIDs ─────────────────────────────────────────────────────
        let uuids = try readUUIDs(fileData, count: count, offset: &offset)

        // ── Vectors ───────────────────────────────────────────────────
        let vectors: [Vector] = try (0..<count).map { _ in
            let floats = try readFloats(fileData, count: dimension, offset: &offset)
            return Vector(floats)
        }

        // ── Graph: nodeLevels ─────────────────────────────────────────
        let nodeLevels: [Int] = try (0..<count).map { _ in
            let v = try readInt32(fileData, offset: &offset)
            let level = Int(v)
            guard level >= 0, level < layerCount else {
                throw PersistenceError.corruptedData(
                    "Node level \(level) outside valid range 0..<\(layerCount)")
            }
            return level
        }

        // ── Graph: per-layer adjacency ────────────────────────────────
        var layers: [[[Int]]] = []
        for _ in 0..<layerCount {
            var layer: [[Int]] = []
            for _ in 0..<count {
                let neighborCount = try readUInt32(fileData, offset: &offset)
                var neighbors: [Int] = []
                for _ in 0..<neighborCount {
                    let n = try readInt32(fileData, offset: &offset)
                    guard n >= 0, n < count else {
                        throw PersistenceError.corruptedData(
                            "Neighbor index \(n) outside valid range 0..<\(count)")
                    }
                    neighbors.append(Int(n))
                }
                layer.append(neighbors)
            }
            layers.append(layer)
        }

        // ── Graph integrity: entry point and maxLevel ─────────────────
        let maxLevel = Int(header.maxLevel)
        guard maxLevel >= -1, maxLevel < layerCount else {
            throw PersistenceError.corruptedData(
                "maxLevel \(maxLevel) outside valid range -1..<\(layerCount)")
        }
        let entryPointNode: Int? = header.entryPoint >= 0 ? Int(header.entryPoint) : nil
        if let entryPoint = entryPointNode {
            guard entryPoint < count else {
                throw PersistenceError.corruptedData(
                    "Entry point \(entryPoint) outside valid range 0..<\(count)")
            }
            guard maxLevel >= 0 else {
                throw PersistenceError.corruptedData(
                    "Entry point \(entryPoint) present but maxLevel is \(maxLevel)")
            }
        }

        // ── Metadata ──────────────────────────────────────────────────
        var metaOffset = Int(header.metadataOffset)
        let metadata = try readMetadata(fileData, count: count, offset: &metaOffset)

        let config = HNSWConfiguration(
            m: Int(header.m),
            efConstruction: Int(header.efConstruction),
            efSearch: Int(header.efSearch),
            autoCompactionThreshold: header.autoCompactionThreshold
        )

        let snapshot = HNSWSnapshot(
            dimension: dimension,
            config: config,
            metricType: header.metricType,
            vectors: vectors,
            metadata: metadata,
            nodeToUUID: uuids,
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPointNode,
            maxLevel: maxLevel
        )

        return HNSWIndex(restoring: snapshot)
    }

    // MARK: - Save HNSW (v3, WAL-binding)

    /// Saves an HNSW snapshot in `.pxkt` **v3**: the v2 body plus the trailer
    /// that binds a `snapshotGeneration` to this base (ADR-013). Used by the
    /// streaming-persistence checkpoint path; the legacy `save(_:to:)` keeps
    /// writing v2 so its bytes are unchanged.
    ///
    /// The body (header through metadata) is laid out identically to v2, so
    /// `loadHNSW(from:)` — which stops after the metadata section — restores a
    /// v3 file exactly like a v2 one. The trailer is read only by
    /// ``readGeneration(from:)``.
    public static func saveHNSW(_ snapshot: HNSWSnapshot, generation: UInt64, to url: URL) throws {
        var data = Data()
        let count = snapshot.nodeToUUID.count

        // ── Header (v3) ───────────────────────────────────────────────
        appendUInt32(&data, magic)
        appendUInt32(&data, v3FormatVersion)
        appendUInt32(&data, indexTypeHNSW)
        appendUInt32(&data, UInt32(snapshot.dimension))
        appendUInt32(&data, UInt32(count))
        appendUInt32(&data, snapshot.metricType.rawValue)
        appendUInt32(&data, UInt32(snapshot.config.m))
        appendUInt32(&data, UInt32(snapshot.config.mMax0))
        appendUInt32(&data, UInt32(snapshot.config.efConstruction))
        appendUInt32(&data, UInt32(snapshot.config.efSearch))
        appendInt32(&data, snapshot.entryPointNode.map { Int32($0) } ?? -1)
        appendInt32(&data, Int32(snapshot.maxLevel))
        appendUInt32(&data, UInt32(snapshot.layers.count))
        let metadataOffsetPosition = data.count
        appendUInt32(&data, 0) // placeholder
        appendUInt64(&data, snapshot.config.autoCompactionThreshold?.bitPattern ?? 0)
        assert(data.count == headerSize)

        // Track section boundaries for the trailer's section table.
        var sections: [(offset: Int, length: Int)] = []
        func section(_ body: () -> Void) {
            let start = data.count
            body()
            sections.append((start, data.count - start))
        }

        section {  // UUIDs
            for uuid in snapshot.nodeToUUID { appendUUID(&data, uuid) }
        }
        section {  // Vectors
            for vector in snapshot.vectors {
                vector.components.withUnsafeBufferPointer { buffer in
                    buffer.withMemoryRebound(to: UInt8.self) { data.append(contentsOf: $0) }
                }
            }
        }
        section {  // nodeLevels
            for level in snapshot.nodeLevels { appendInt32(&data, Int32(level)) }
        }
        section {  // adjacency
            for layer in snapshot.layers {
                for neighbors in layer {
                    appendUInt32(&data, UInt32(neighbors.count))
                    for neighbor in neighbors { appendInt32(&data, Int32(neighbor)) }
                }
            }
        }
        let metadataOffset = UInt32(data.count)
        writeMetadataOffset(&data, offset: metadataOffset, at: metadataOffsetPosition)
        section {  // metadata
            appendMetadata(&data, snapshot.metadata)
        }

        // ── Trailer ───────────────────────────────────────────────────
        appendUInt32(&data, UInt32(v3SectionCount))
        for s in sections {
            appendUInt64(&data, UInt64(s.offset))
            appendUInt64(&data, UInt64(s.length))
        }
        appendUInt64(&data, generation)
        appendUInt32(&data, v3TrailerMagic)

        try data.write(to: url, options: .atomic)
    }

    /// Reads the `snapshotGeneration` bound to a base file. v1/v2 files (which
    /// predate the trailer) report generation 0. A v3 file whose trailer is
    /// truncated or whose trailer magic is wrong throws a typed
    /// `PersistenceError` — never traps.
    public static func readGeneration(from url: URL) throws -> UInt64 {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try readHeader(data)   // validates magic/version/metric
        guard header.version >= v3FormatVersion else { return 0 }
        guard data.count >= headerSize + v3TrailerSize else {
            throw PersistenceError.corruptedData("v3 file too small for trailer")
        }
        let trailerStart = data.count - v3TrailerSize
        let trailerMagic = data.loadLE(UInt32.self, at: data.count - 4)
        guard trailerMagic == v3TrailerMagic else {
            throw PersistenceError.corruptedData("v3 trailer magic mismatch")
        }
        let sectionCount = data.loadLE(UInt32.self, at: trailerStart)
        guard sectionCount == UInt32(v3SectionCount) else {
            throw PersistenceError.corruptedData(
                "v3 trailer section count \(sectionCount) != \(v3SectionCount)")
        }
        let generationOffset = trailerStart + 4 + v3SectionCount * 16
        return data.loadLE(UInt64.self, at: generationOffset)
    }
}

// MARK: - Header Parsing

private struct FileHeader {
    let version: UInt32
    let indexType: UInt32
    let dimension: UInt32
    let count: UInt32
    let metricType: DistanceMetricType
    let m: UInt32
    let mMax0: UInt32
    let efConstruction: UInt32
    let efSearch: UInt32
    let entryPoint: Int32
    let maxLevel: Int32
    let layerCount: UInt32
    let metadataOffset: UInt32
    let autoCompactionThreshold: Double?
}

private func readHeader(_ data: Data) throws -> FileHeader {
    guard data.count >= headerSize else {
        throw PersistenceError.fileTooSmall
    }

    let fileMagic = data.loadLE(UInt32.self, at: 0)
    guard fileMagic == magic else {
        throw PersistenceError.invalidMagic
    }

    let version = data.loadLE(UInt32.self, at: 4)
    guard version >= minSupportedVersion, version <= maxReadableVersion else {
        throw PersistenceError.unsupportedVersion(version)
    }

    let metricRaw = data.loadLE(UInt32.self, at: 20)
    guard let metricType = DistanceMetricType(rawValue: metricRaw) else {
        throw PersistenceError.unknownMetricType(metricRaw)
    }

    // ── Auto-compaction threshold (v2+) ───────────────────────────────
    // v1 files predate serialization; they get the documented default.
    let autoCompactionThreshold: Double?
    if version >= 2 {
        let bits = data.loadLE(UInt64.self, at: 56)
        if bits == 0 {
            autoCompactionThreshold = nil
        } else {
            let value = Double(bitPattern: bits)
            guard value.isFinite, value > 0, value < 1 else {
                throw PersistenceError.corruptedData(
                    "autoCompactionThreshold \(value) outside valid range (0, 1)")
            }
            autoCompactionThreshold = value
        }
    } else {
        autoCompactionThreshold = legacyDefaultCompactionThreshold
    }

    return FileHeader(
        version: version,
        indexType: data.loadLE(UInt32.self, at: 8),
        dimension: data.loadLE(UInt32.self, at: 12),
        count: data.loadLE(UInt32.self, at: 16),
        metricType: metricType,
        m: data.loadLE(UInt32.self, at: 24),
        mMax0: data.loadLE(UInt32.self, at: 28),
        efConstruction: data.loadLE(UInt32.self, at: 32),
        efSearch: data.loadLE(UInt32.self, at: 36),
        entryPoint: Int32(bitPattern: data.loadLE(UInt32.self, at: 40)),
        maxLevel: Int32(bitPattern: data.loadLE(UInt32.self, at: 44)),
        layerCount: data.loadLE(UInt32.self, at: 48),
        metadataOffset: data.loadLE(UInt32.self, at: 52),
        autoCompactionThreshold: autoCompactionThreshold
    )
}

// MARK: - Binary Write Helpers

private func appendUInt32(_ data: inout Data, _ value: UInt32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

private func appendInt32(_ data: inout Data, _ value: Int32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

private func appendUInt64(_ data: inout Data, _ value: UInt64) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

private func appendUUID(_ data: inout Data, _ uuid: UUID) {
    withUnsafeBytes(of: uuid.uuid) { data.append(contentsOf: $0) }
}

private func appendMetadata(_ data: inout Data, _ store: [Data?]) {
    for meta in store {
        if let meta = meta {
            appendUInt32(&data, UInt32(meta.count))
            data.append(meta)
        } else {
            appendUInt32(&data, 0)
        }
    }
}

private func writeMetadataOffset(_ data: inout Data, offset: UInt32, at position: Int) {
    withUnsafeBytes(of: offset.littleEndian) { bytes in
        for (i, byte) in bytes.enumerated() {
            data[position + i] = byte
        }
    }
}

// MARK: - Binary Read Helpers

extension Data {
    func loadLE<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
        self.withUnsafeBytes { buffer in
            T(littleEndian: buffer.loadUnaligned(fromByteOffset: offset, as: T.self))
        }
    }
}

private func readUUIDs(_ data: Data, count: Int, offset: inout Int) throws -> [UUID] {
    let (needed, overflow) = count.multipliedReportingOverflow(by: 16)
    guard !overflow, count >= 0, offset <= data.count, needed <= data.count - offset else {
        throw PersistenceError.corruptedData("UUID section truncated")
    }

    var uuids: [UUID] = []
    uuids.reserveCapacity(count)
    for _ in 0..<count {
        let uuid = data.withUnsafeBytes { buffer in
            buffer.loadUnaligned(fromByteOffset: offset, as: uuid_t.self)
        }
        uuids.append(UUID(uuid: uuid))
        offset += 16
    }
    return uuids
}

private func readFloats(_ data: Data, count: Int, offset: inout Int) throws -> [Float] {
    let (needed, overflow) = count.multipliedReportingOverflow(by: 4)
    guard !overflow, count >= 0, offset <= data.count, needed <= data.count - offset else {
        throw PersistenceError.corruptedData("Vector section truncated")
    }

    var floats = [Float](repeating: 0, count: count)
    data.withUnsafeBytes { buffer in
        let src = buffer.baseAddress!.advanced(by: offset)
        floats.withUnsafeMutableBytes { dest in
            dest.copyMemory(from: UnsafeRawBufferPointer(start: src, count: needed))
        }
    }
    offset += needed
    return floats
}

private func readUInt32(_ data: Data, offset: inout Int) throws -> UInt32 {
    guard offset + 4 <= data.count else {
        throw PersistenceError.corruptedData("Unexpected end of data")
    }
    let value = data.loadLE(UInt32.self, at: offset)
    offset += 4
    return value
}

private func readInt32(_ data: Data, offset: inout Int) throws -> Int32 {
    guard offset + 4 <= data.count else {
        throw PersistenceError.corruptedData("Unexpected end of data")
    }
    let value = Int32(bitPattern: data.loadLE(UInt32.self, at: offset))
    offset += 4
    return value
}

private func readMetadata(_ data: Data, count: Int, offset: inout Int) throws -> [Data?] {
    var metadata: [Data?] = []
    metadata.reserveCapacity(count)
    for _ in 0..<count {
        let length = try readUInt32(data, offset: &offset)
        if length == 0 {
            metadata.append(nil)
        } else {
            let end = offset + Int(length)
            guard end <= data.count else {
                throw PersistenceError.corruptedData("Metadata truncated")
            }
            metadata.append(data.subdata(in: offset..<end))
            offset = end
        }
    }
    return metadata
}
