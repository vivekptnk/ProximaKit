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
/// Index of the vector section within the v3 section table.
private let v3VectorSectionIndex = 1
/// Index of the node-levels section within the v3 section table (the first
/// section after the vectors — where the paged loader resumes reading).
private let v3NodeLevelsSectionIndex = 2
/// Index of the metadata section within the v3 section table. The trailer's
/// UInt64 offset is authoritative for v3 (the legacy UInt32 header field is
/// clamped to a sentinel when the true offset exceeds `UInt32.max` — ADR-014).
private let v3MetadataSectionIndex = 4
/// 16 KiB — the Apple-Silicon page size the vector section start is padded to
/// so it can be `mmap`-ed independently (ADR-013 Stage 2).
private let vectorSectionAlignment = 16_384

/// Sentinel written into the legacy UInt32 `metadataOffset` header field when
/// the true metadata offset exceeds `UInt32.max` (a >4 GiB base). For v3 the
/// trailer's UInt64 metadata offset is authoritative, so the header field is
/// advisory only; the sentinel documents "look at the trailer" without a
/// format bump (ADR-014 metadataOffset widening). Files writable today never
/// reach it — the offset stays a faithful UInt32.
private let metadataOffsetSentinel: UInt32 = 0xFFFF_FFFF

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
        // For v3 the vector section may be page-padded away from the UUID tail;
        // jump to its recorded offset. No-op for unpadded v3 (offset already
        // there) and irrelevant to v1/v2 (no table). Everything after the
        // vector section stays contiguous with it, so the sequential graph
        // reader below resumes correctly.
        var trailer: V3Trailer?
        if header.version >= v3FormatVersion {
            trailer = try readV3Trailer(fileData)
            offset = trailer!.sections[v3VectorSectionIndex].offset
        }
        let vectors: [Vector] = try (0..<count).map { _ in
            let floats = try readFloats(fileData, count: dimension, offset: &offset)
            return Vector(floats)
        }

        // ── Graph + metadata + config (shared with the paged loader) ──
        // For v3 the metadata offset comes from the trailer (UInt64,
        // sentinel-proof); v1/v2 use the legacy UInt32 header field.
        let metadataOffset = trailer?.sections[v3MetadataSectionIndex].offset
            ?? Int(header.metadataOffset)
        let graph = try readHNSWGraph(
            fileData, header: header, count: count, layerCount: layerCount,
            offset: &offset, metadataOffset: metadataOffset)

        let snapshot = HNSWSnapshot(
            dimension: dimension,
            config: graph.config,
            metricType: header.metricType,
            vectors: vectors,
            metadata: graph.metadata,
            nodeToUUID: uuids,
            layers: graph.layers,
            nodeLevels: graph.nodeLevels,
            entryPointNode: graph.entryPointNode,
            maxLevel: graph.maxLevel
        )

        return HNSWIndex(restoring: snapshot)
    }

    // MARK: - Load HNSW (paged, ADR-013 Stage 2)

    /// Loads an HNSW index in **paged** mode: the graph, UUIDs, node levels, and
    /// metadata are decoded resident exactly as `loadHNSW` does, but the vector
    /// section is served from a read-only `MappedVectorRegion` instead of being
    /// copied into `[Vector]`. Requires a padded v3 base (a Stage-2 checkpoint);
    /// a non-v3 or unpadded base throws a typed `PersistenceError` (never traps).
    internal static func loadHNSWPaged(from url: URL) throws -> HNSWIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try readHeader(fileData)

        guard header.indexType == indexTypeHNSW else {
            throw PersistenceError.unknownIndexType(header.indexType)
        }
        guard header.version >= v3FormatVersion else {
            throw PersistenceError.corruptedData("paged open requires a v3 base; checkpoint the index first")
        }

        let count = Int(header.count)
        let dimension = Int(header.dimension)
        let layerCount = Int(header.layerCount)
        var offset = headerSize

        // ── Header sanity (mirrors loadHNSW) ──────────────────────────
        guard dimension > 0 else {
            throw PersistenceError.corruptedData("Dimension must be positive, got \(dimension)")
        }
        guard layerCount <= fileData.count else {
            throw PersistenceError.corruptedData(
                "Layer count \(layerCount) implausible for file of \(fileData.count) bytes")
        }
        guard header.m >= 2, header.efConstruction > 0, header.efSearch > 0 else {
            throw PersistenceError.corruptedData(
                "HNSW configuration fields out of range "
                + "(m: \(header.m) [min 2], efConstruction: \(header.efConstruction), efSearch: \(header.efSearch))")
        }
        guard UInt64(header.mMax0) == 2 * UInt64(header.m) else {
            throw PersistenceError.corruptedData(
                "mMax0 \(header.mMax0) inconsistent with m \(header.m) (expected \(2 * UInt64(header.m)))")
        }

        let trailer = try readV3Trailer(fileData)

        // ── UUIDs (contiguous after the header) ───────────────────────
        let uuids = try readUUIDs(fileData, count: count, offset: &offset)

        // ── Skip the mapped vector section; resume at node levels ─────
        offset = trailer.sections[v3NodeLevelsSectionIndex].offset

        // ── Graph + metadata + config (shared with the resident loader) ─
        let graph = try readHNSWGraph(
            fileData, header: header, count: count, layerCount: layerCount,
            offset: &offset, metadataOffset: trailer.sections[v3MetadataSectionIndex].offset)

        // ── Map the vector section read-only ──────────────────────────
        let region = try MappedVectorRegion(baseURL: url)
        guard region.count == count, region.dimension == dimension else {
            throw PersistenceError.corruptedData(
                "mapped vector section (\(region.count)×\(region.dimension)) disagrees with header (\(count)×\(dimension))")
        }

        let snapshot = HNSWSnapshot(
            dimension: dimension,
            config: graph.config,
            metricType: header.metricType,
            vectors: [],   // served from `region`, not resident
            metadata: graph.metadata,
            nodeToUUID: uuids,
            layers: graph.layers,
            nodeLevels: graph.nodeLevels,
            entryPointNode: graph.entryPointNode,
            maxLevel: graph.maxLevel
        )

        return HNSWIndex(restoringPaged: snapshot, region: region)
    }

    /// Decodes the node-levels, adjacency, integrity, metadata, and config —
    /// everything after the vector section — from `offset` onward. Shared by the
    /// resident and paged loaders so the two never diverge. `offset` must enter
    /// positioned at the node-levels section.
    private static func readHNSWGraph(
        _ fileData: Data, header: FileHeader, count: Int, layerCount: Int,
        offset: inout Int, metadataOffset: Int
    ) throws -> (nodeLevels: [Int], layers: [[[Int]]], entryPointNode: Int?, maxLevel: Int,
                 metadata: [Data?], config: HNSWConfiguration) {

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
        var metaOffset = metadataOffset
        let metadata = try readMetadata(fileData, count: count, offset: &metaOffset)

        let config = HNSWConfiguration(
            m: Int(header.m),
            efConstruction: Int(header.efConstruction),
            efSearch: Int(header.efSearch),
            autoCompactionThreshold: header.autoCompactionThreshold
        )

        return (nodeLevels, layers, entryPointNode, maxLevel, metadata, config)
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
        try saveHNSW(snapshot, generation: generation, to: url, padVectorSection: true)
    }

    /// v3 writer with explicit control over vector-section padding.
    ///
    /// `padVectorSection` (the shipped default via the public overload) pads the
    /// vector section start to a 16 KiB boundary (ADR-013 Stage 2) so the base
    /// can be `mmap`-ed for paged open, each fault pulling a clean vector page.
    /// The section table records the padded offset, so `loadHNSW` (which jumps
    /// to the recorded offset for v3) reads padded and unpadded v3 identically.
    /// The `false` path exists only to let tests reproduce a Stage-1-shaped
    /// (unpadded) v3 file and prove it still loads.
    internal static func saveHNSW(
        _ snapshot: HNSWSnapshot, generation: UInt64, to url: URL, padVectorSection: Bool
    ) throws {
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
        if padVectorSection {
            // Pad so the vector section START lands on a 16 KiB boundary,
            // making it independently mappable (ADR-013 Stage 2). The section
            // table below records the padded offset; only the vector section is
            // padded, so the sections after it stay contiguous with it.
            padToAlignment(&data, alignment: vectorSectionAlignment)
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
        // The legacy header field is UInt32; if the true metadata offset would
        // exceed it (a >4 GiB base), write the documented sentinel and rely on
        // the trailer's UInt64 offset — no format bump (ADR-014). Every file
        // writable today stays a faithful UInt32, so its bytes are unchanged.
        let metadataStart = data.count
        let metadataOffset: UInt32 =
            metadataStart <= Int(UInt32.max) ? UInt32(metadataStart) : metadataOffsetSentinel
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
        return try readV3Trailer(data).generation
    }

    /// Physical layout of a v3 base's mapped vector section, for paged open
    /// (`MappedVectorRegion`). Reads the header + trailer only — no vector data
    /// is copied. Throws a typed `PersistenceError` (never traps) for a non-v3
    /// base, a mis-sized vector section, or a section-table entry out of bounds.
    internal static func pagedVectorLayout(of url: URL) throws -> PagedVectorLayout {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try readHeader(data)
        guard header.indexType == indexTypeHNSW else {
            throw PersistenceError.unknownIndexType(header.indexType)
        }
        guard header.version >= v3FormatVersion else {
            throw PersistenceError.corruptedData("paged open requires a v3 base; checkpoint the index first")
        }
        let dimension = Int(header.dimension)
        let count = Int(header.count)
        guard dimension > 0 else {
            throw PersistenceError.corruptedData("Dimension must be positive, got \(dimension)")
        }
        let trailer = try readV3Trailer(data)
        let vectorSection = trailer.sections[v3VectorSectionIndex]
        // The section length must be exactly count × dimension × 4; a mismatch
        // means a corrupt table or a crafted file.
        let (stride, strideOverflow) = dimension.multipliedReportingOverflow(by: 4)
        let (expected, expectedOverflow) = count.multipliedReportingOverflow(by: stride)
        guard !strideOverflow, !expectedOverflow, vectorSection.length == expected else {
            throw PersistenceError.corruptedData(
                "v3 vector section length \(vectorSection.length) != \(count) × \(dimension) × 4")
        }
        return PagedVectorLayout(
            dimension: dimension,
            count: count,
            vectorOffset: vectorSection.offset,
            vectorLength: vectorSection.length,
            fileSize: data.count
        )
    }

    // MARK: - Migration (ADR-014 M-A: offline section-copy rewrite)

    /// Upgrades a `.pxkt` HNSW v1/v2 (or unpadded-v3) base at `url` in place to a
    /// padded v3 base — so an existing base can adopt ADR-013 Stage-2 paging
    /// without a journal or a full rebuild.
    ///
    /// The rewrite is a pure section-copy: every section payload in the output
    /// is byte-identical to the input's, the vector section is pushed onto a
    /// 16 KiB boundary, the version is stamped 3, and the WAL-binding trailer is
    /// appended (generation preserved from a v3 input, 0 from a v1/v2 input).
    /// No graph is decoded and no vector is materialized. Crash safety: the
    /// image is written to a sibling temp and verified (full section checksum)
    /// before an atomic replace, so an interrupted upgrade leaves the source
    /// untouched.
    ///
    /// Migrating an already-padded v3 file is a no-op (clean success).
    ///
    /// - Throws: `PersistenceError` for a non-`.pxkt` file, a BruteForce base
    ///   (nothing to page), an inconsistent header, or a failed verification —
    ///   never traps.
    public static func upgradeToV3(at url: URL) throws {
        let source = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let image = try buildPaddedV3Image(from: source) else {
            return   // already a padded v3 base
        }
        let tmp = url.appendingPathExtension("\(ProximaKit.FileExtension.index)v3tmp")
        defer { try? FileManager.default.removeItem(at: tmp) }
        // Filesystem errors (disk full, permission denied) are wrapped in a typed
        // PersistenceError preserving the underlying cause; the source is left
        // untouched (the temp is written and replaced, never the original).
        do {
            try image.write(to: tmp, options: .atomic)
        } catch {
            throw PersistenceError.migrationFailed(
                "could not write the upgrade image to \(tmp.path): \(error)")
        }
        try verifyPaddedV3Upgrade(source: source, upgradedURL: tmp)
        do {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
        } catch {
            throw PersistenceError.migrationFailed(
                "could not atomically replace \(url.path) with the upgraded image: \(error)")
        }
    }

    /// Builds the padded-v3 image for a `.pxkt` HNSW `source`, or `nil` if it is
    /// already a padded v3 base. Section payloads are byte-identical copies.
    /// Testable seam for the migration + crash-safety suites.
    static func buildPaddedV3Image(from source: Data) throws -> Data? {
        let layout = try PXKTSectionLayout.parse(source)
        if layout.version == v3FormatVersion, layout.vectors.offset % vectorSectionAlignment == 0 {
            return nil   // already padded v3
        }

        var out = Data()
        out.reserveCapacity(source.count + vectorSectionAlignment + v3TrailerSize)

        // Header: copy the 64 bytes verbatim; version + metadataOffset patched
        // after the body is laid out.
        out.append(source[0..<headerSize])

        // UUIDs (contiguous after the header).
        let uuidsOut = out.count
        out.append(source[layout.uuids.offset..<layout.uuids.end])
        // Pad so the vector section lands on a 16 KiB boundary.
        padToAlignment(&out, alignment: vectorSectionAlignment)
        let vectorsOut = out.count
        out.append(source[layout.vectors.offset..<layout.vectors.end])
        let nodeLevelsOut = out.count
        out.append(source[layout.nodeLevels.offset..<layout.nodeLevels.end])
        let adjacencyOut = out.count
        out.append(source[layout.adjacency.offset..<layout.adjacency.end])
        let metadataOut = out.count
        out.append(source[layout.metadata.offset..<layout.metadata.end])

        // Patch header: version = 3, metadataOffset = new position (or sentinel).
        withUnsafeBytes(of: v3FormatVersion.littleEndian) { out.replaceSubrange(4..<8, with: $0) }
        let metaField: UInt32 =
            metadataOut <= Int(UInt32.max) ? UInt32(metadataOut) : metadataOffsetSentinel
        writeMetadataOffset(&out, offset: metaField, at: 52)

        // Trailer.
        let sections: [(offset: Int, length: Int)] = [
            (uuidsOut, layout.uuids.length),
            (vectorsOut, layout.vectors.length),
            (nodeLevelsOut, layout.nodeLevels.length),
            (adjacencyOut, layout.adjacency.length),
            (metadataOut, layout.metadata.length),
        ]
        appendUInt32(&out, UInt32(v3SectionCount))
        for s in sections {
            appendUInt64(&out, UInt64(s.offset))
            appendUInt64(&out, UInt64(s.length))
        }
        appendUInt64(&out, layout.generation)
        appendUInt32(&out, v3TrailerMagic)
        return out
    }

    /// Re-reads the upgraded temp and asserts every section payload is
    /// byte-identical to the source's (ADR-014 fidelity gate; full-checksum).
    static func verifyPaddedV3Upgrade(source: Data, upgradedURL: URL) throws {
        let upgraded = try Data(contentsOf: upgradedURL)
        let src = try PXKTSectionLayout.parse(source)
        let dst = try PXKTSectionLayout.parse(upgraded)
        guard dst.version == v3FormatVersion else {
            throw PersistenceError.corruptedData("upgraded .pxkt base is not v3")
        }
        guard dst.vectors.offset % vectorSectionAlignment == 0 else {
            throw PersistenceError.corruptedData("upgraded .pxkt vector section is not 16 KiB-aligned")
        }
        let pairs: [(String, PXKTSectionLayout.Section, PXKTSectionLayout.Section)] = [
            ("uuids", src.uuids, dst.uuids),
            ("vectors", src.vectors, dst.vectors),
            ("nodeLevels", src.nodeLevels, dst.nodeLevels),
            ("adjacency", src.adjacency, dst.adjacency),
            ("metadata", src.metadata, dst.metadata),
        ]
        for (name, s, d) in pairs {
            guard s.length == d.length else {
                throw PersistenceError.corruptedData("upgraded .pxkt \(name) length drifted")
            }
            guard source[s.offset..<s.end] == upgraded[d.offset..<d.end] else {
                throw PersistenceError.corruptedData("upgraded .pxkt \(name) payload not bit-identical")
            }
        }
    }
}

/// The byte ranges of every `.pxkt` HNSW section, resolved for the migration
/// rewriter without decoding the graph. v2 boundaries come from header counts
/// plus the stored `metadataOffset`; v3 boundaries come from the trailer.
struct PXKTSectionLayout {
    struct Section { let offset: Int; let length: Int; var end: Int { offset + length } }

    let version: UInt32
    let generation: UInt64
    let uuids: Section
    let vectors: Section
    let nodeLevels: Section
    let adjacency: Section
    let metadata: Section

    static func parse(_ data: Data) throws -> PXKTSectionLayout {
        let header = try readHeader(data)
        guard header.indexType == indexTypeHNSW else {
            throw PersistenceError.corruptedData(
                "only HNSW `.pxkt` bases can be upgraded to v3 (BruteForce has nothing to page)")
        }
        let count = Int(header.count)
        let dimension = Int(header.dimension)
        guard dimension > 0, count >= 0 else {
            throw PersistenceError.corruptedData(".pxkt header fields invalid for migration")
        }

        // v3 input: trust the validated trailer.
        if header.version >= v3FormatVersion {
            let trailer = try readV3Trailer(data)
            func s(_ i: Int) -> Section {
                Section(offset: trailer.sections[i].offset, length: trailer.sections[i].length)
            }
            return PXKTSectionLayout(
                version: header.version, generation: trailer.generation,
                uuids: s(0), vectors: s(1), nodeLevels: s(2), adjacency: s(3), metadata: s(4))
        }

        // v1/v2 input: fixed-size sections + stored metadataOffset.
        func region(_ start: Int, _ elems: Int, _ unit: Int, _ what: String) throws -> Section {
            let (bytes, ov) = elems.multipliedReportingOverflow(by: unit)
            guard !ov, bytes >= 0, start <= data.count, bytes <= data.count - start else {
                throw PersistenceError.corruptedData(".pxkt \(what) section out of bounds")
            }
            return Section(offset: start, length: bytes)
        }
        let uuids = try region(headerSize, count, 16, "uuids")
        let vectors = try region(uuids.end, count, dimension * 4, "vectors")
        let nodeLevels = try region(vectors.end, count, 4, "nodeLevels")
        let metaOffset = Int(header.metadataOffset)
        guard metaOffset >= nodeLevels.end, metaOffset <= data.count else {
            throw PersistenceError.corruptedData(
                ".pxkt metadataOffset \(metaOffset) inconsistent with the graph sections")
        }
        let adjacency = Section(offset: nodeLevels.end, length: metaOffset - nodeLevels.end)
        let metadata = Section(offset: metaOffset, length: data.count - metaOffset)
        return PXKTSectionLayout(
            version: header.version, generation: 0,
            uuids: uuids, vectors: vectors, nodeLevels: nodeLevels,
            adjacency: adjacency, metadata: metadata)
    }
}

/// Physical placement of a v3 base's vector section, resolved from the section
/// table for paged (`mmap`) open.
internal struct PagedVectorLayout {
    let dimension: Int
    let count: Int
    let vectorOffset: Int
    let vectorLength: Int
    let fileSize: Int
}

/// The v3 section table + snapshot generation, parsed from the fixed 96-byte
/// trailer. Every section offset/length is bounds-checked against the file so a
/// crafted or truncated table is a typed error, never a trap.
private struct V3Trailer {
    let sections: [(offset: Int, length: Int)]   // 5 entries, table order
    let generation: UInt64
}

private func readV3Trailer(_ data: Data) throws -> V3Trailer {
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
    // Sections must lie fully within the pre-trailer body: 0 ≤ offset,
    // offset + length ≤ trailerStart. This rejects a section-table offset past
    // EOF and any length that would run into the trailer.
    let bodyEnd = UInt64(trailerStart)
    var sections: [(offset: Int, length: Int)] = []
    sections.reserveCapacity(v3SectionCount)
    for i in 0..<v3SectionCount {
        let entry = trailerStart + 4 + i * 16
        let offset = data.loadLE(UInt64.self, at: entry)
        let length = data.loadLE(UInt64.self, at: entry + 8)
        let (end, overflow) = offset.addingReportingOverflow(length)
        guard !overflow, offset <= bodyEnd, end <= bodyEnd else {
            throw PersistenceError.corruptedData(
                "v3 section \(i) (offset \(offset), length \(length)) out of bounds for a \(data.count)-byte file")
        }
        sections.append((Int(offset), Int(length)))
    }
    let generation = data.loadLE(UInt64.self, at: trailerStart + 4 + v3SectionCount * 16)
    return V3Trailer(sections: sections, generation: generation)
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

/// Zero-pads `data` so its length becomes a multiple of `alignment`. Used to
/// push the vector section start onto a page boundary (ADR-013 Stage 2).
private func padToAlignment(_ data: inout Data, alignment: Int) {
    let remainder = data.count % alignment
    if remainder != 0 {
        data.append(Data(count: alignment - remainder))
    }
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
