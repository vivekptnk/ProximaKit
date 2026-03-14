// PersistenceEngine.swift
// ProximaKit
//
// Binary persistence for ProximaKit indices.
// Format: 64-byte header + UUIDs + raw Float32 vectors + graph + metadata.
// Vectors are memory-mapped on load for fast cold starts.
// See ADR-003 for design rationale.

import Foundation

// MARK: - Constants

private let magic: UInt32 = 0x50584B54  // "PXKT"
private let formatVersion: UInt32 = 1
private let headerSize = 64

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
/// Float32 values, enabling memory-mapped loading for fast cold starts.
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

        // ── UUIDs ─────────────────────────────────────────────────────
        let ids = try readUUIDs(fileData, count: count, offset: &offset)

        // ── Vectors ───────────────────────────────────────────────────
        let vectorData = try readFloats(fileData, count: count * dimension, offset: &offset)

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

        appendUInt64(&data, 0) // reserved

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
            return Int(v)
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
                    neighbors.append(Int(n))
                }
                layer.append(neighbors)
            }
            layers.append(layer)
        }

        // ── Metadata ──────────────────────────────────────────────────
        var metaOffset = Int(header.metadataOffset)
        let metadata = try readMetadata(fileData, count: count, offset: &metaOffset)

        let config = HNSWConfiguration(
            m: Int(header.m),
            efConstruction: Int(header.efConstruction),
            efSearch: Int(header.efSearch)
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
            entryPointNode: header.entryPoint >= 0 ? Int(header.entryPoint) : nil,
            maxLevel: Int(header.maxLevel)
        )

        return HNSWIndex(restoring: snapshot)
    }
}

// MARK: - Header Parsing

private struct FileHeader {
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
    guard version == formatVersion else {
        throw PersistenceError.unsupportedVersion(version)
    }

    let metricRaw = data.loadLE(UInt32.self, at: 20)
    guard let metricType = DistanceMetricType(rawValue: metricRaw) else {
        throw PersistenceError.unknownMetricType(metricRaw)
    }

    return FileHeader(
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
        metadataOffset: data.loadLE(UInt32.self, at: 52)
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
    let needed = count * 16
    guard offset + needed <= data.count else {
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
    let needed = count * 4
    guard offset + needed <= data.count else {
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
