// QuantizedHNSWIndexPersistence.swift
// ProximaKit
//
// Binary persistence for QuantizedHNSWIndex.
//
// File format (all little-endian):
//   Magic:           0x50514857 ("PQHW")
//   Version:         UInt32 (2 or 3)
//   Dimension:       UInt32
//   NodeCount:       UInt32
//   SubspaceCount:   UInt32
//   M:               UInt32
//   EfConstruction:  UInt32
//   EfSearch:        UInt32
//   MaxLevel:        Int32
//   EntryPoint:      Int32 (-1 if nil)
//   LayerCount:      UInt32
//   TrainIters:      UInt32
//   OriginalsFlag:   UInt32 (v2+; 0 or 1 — reserved/zero in v1)
//   Reserved:        4 bytes (zero)
//   --- 56-byte header ---
//   PQ codebooks:    M * 256 * ds Float32 values
//   PQ codes:        nodeCount * M UInt8 values
//   UUIDs:           nodeCount * 16 bytes
//   Node levels:     nodeCount * Int32
//   Graph layers:    per-layer adjacency lists
//   Metadata:        JSON-encoded [Data?]
//   (v3 only) Padding: zero bytes so the Originals section starts on a
//                      16 KiB boundary — present iff OriginalsFlag == 1.
//   Originals:       nodeCount * dimension Float32 (v2+, iff OriginalsFlag == 1)
//   (v3 only) Trailer: 128-byte fixed section table (see below).
//
// v3 trailer (ADR-014, mirrors the `.pxkt` PXK3 trailer):
//   sectionCount: UInt32 (= 7)
//   [codebooks, codes, uuids, nodeLevels, adjacency, metadata, originals]
//       × (offset: UInt64, length: UInt64)
//   snapshotGeneration: UInt64  (reserved; stamped 0 until a PQ remove-journal
//                                exists — ADR-014 open question 5)
//   trailerMagic: UInt32 ("PQH3" = 0x5051_4833)
//   Size = 4 + 7·16 + 8 + 4 = 128 bytes.
// The originals entry is (0, 0) when OriginalsFlag == 0.
//
// Version history (ADR-010 / ADR-012 / ADR-014):
//   v1 — initial PQHW codec (ADR-011).
//   v2 — originals flag in the previously reserved header bytes at offset 48
//        plus an optional trailing originals section. v1 files load with the
//        documented default retainOriginals == false.
//   v3 — the 56-byte header is byte-for-byte identical to v2; the body
//        (codebooks…metadata) is byte-for-byte identical to v2; a v3 file adds
//        16 KiB padding before the originals section (iff retained) so the
//        section can be memory-mapped (ADR-014 Stage 2, not yet wired) and a
//        128-byte section-table trailer. The resident loader walks the sections
//        exactly as v2 and jumps to the trailer's recorded offset only for the
//        originals section. `minSupportedVersion` stays 1 (v1/v2/v3 all load).
//
// Writer policy (ADR-014 Stage-1 implementation decision — see the ADR
// "Implementation notes (Stage 1)" addendum):
//   • `save(to:)` is UNCHANGED and byte-identical to before: it always stamps
//     v2 (the ADR-013 deviation-1 byte-identity ground rule; a retaining index
//     still writes a v2 file with a bare originals section, nothing else).
//   • The additive `save(to:layout:)` opts in to v3: `.pagedV3` stamps v3 with
//     the originals section 16 KiB-aligned WHEN originals are retained, and
//     otherwise falls back to v2 (nothing to page → no format upgrade).
//   • `upgradeToV3(at:)` rewrites an existing v1/v2 base to a padded v3 base
//     by section-copy — no decode, no re-encode, payload bytes bit-identical.

import Foundation

private let qhMagic: UInt32 = 0x50514857       // "PQHW"
/// Version stamped by the default (`save(to:)`) writer — unchanged from v2.
private let qhWriteVersionV2: UInt32 = 2
/// Version stamped by the opt-in `.pagedV3` writer and by `upgradeToV3`.
private let qhWriteVersionV3: UInt32 = 3
private let qhMinSupportedVersion: UInt32 = 1
/// Highest version this reader understands (ADR-010 rule 2 window: v1…v3).
private let qhMaxReadableVersion: UInt32 = 3
private let qhHeaderSize = 56

// ── v3 trailer (ADR-014) ──────────────────────────────────────────────
private let qhV3SectionCount = 7
private let qhV3TrailerMagic: UInt32 = 0x5051_4833   // "PQH3"
private let qhV3TrailerSize = 4 + qhV3SectionCount * 16 + 8 + 4   // = 128
/// Index of the originals section within the v3 section table.
private let qhOriginalsSectionIndex = 6
/// 16 KiB — the Apple-Silicon page size the originals section start is padded to
/// so it can be `mmap`-ed independently (ADR-014 Stage 2).
private let qhOriginalsAlignment = 16_384

/// On-disk layout chosen by the additive `save(to:layout:)` writer (ADR-014).
public enum PQHWSaveLayout: Sendable {
    /// The historical resident format — byte-identical to `save(to:)` (v2).
    case resident
    /// The paged-capable v3 format: when originals are retained, the originals
    /// section is 16 KiB-aligned and a section-table trailer is appended.
    /// When no originals are retained there is nothing to page, so this falls
    /// back to the resident (v2) format (ADR-014 Stage-1 note).
    case pagedV3
}

extension QuantizedHNSWIndex {

    /// Live-only view of the index state for serialization.
    ///
    /// `remove(id:)` tombstones a slot (deletes the `uuidToNode` entry) but
    /// leaves the UUID in `nodeToUUID`. Writing raw slots would let `load`
    /// rebuild `uuidToNode` for EVERY slot — resurrecting deleted vectors,
    /// over-reporting `liveCount`, and re-opening the entry-point-collapse
    /// failure mode the identity-liveness fix closed (CHA-201 signoff
    /// finding). The full-precision index compacts before snapshotting;
    /// quantized indexes compact equivalently here, at save time.
    private func compactedForSave() -> (
        codes: [[UInt8]], uuids: [UUID], levels: [Int],
        metadata: [Data?], layers: [[[Int]]], entryPoint: Int?, maxLevel: Int,
        originals: [Vector]?
    ) {
        // Fast path: no tombstones — serialize state as-is (byte-identical
        // to the pre-compaction format, and no array copies).
        if uuidToNode.count == nodeToUUID.count {
            return (codes, nodeToUUID, nodeLevels, metadata,
                    layers, entryPointNode, maxLevel, originals)
        }

        // Dense renumbering of live slots (identity check, not presence).
        // Originals are a per-slot section like codes/metadata: compact them
        // in the same loop so they stay slot-aligned after renumbering
        // (ADR-012 — a reranked load would otherwise score against the
        // wrong vectors).
        var oldToNew = [Int: Int]()
        var newCodes: [[UInt8]] = []
        var newUUIDs: [UUID] = []
        var newLevels: [Int] = []
        var newMetadata: [Data?] = []
        var newOriginals: [Vector]? = originals != nil ? [] : nil
        for (node, uuid) in nodeToUUID.enumerated() where uuidToNode[uuid] == node {
            oldToNew[node] = newUUIDs.count
            newCodes.append(codes[node])
            newUUIDs.append(uuid)
            newLevels.append(nodeLevels[node])
            newMetadata.append(metadata[node])
            if let originals {
                newOriginals?.append(originals[node])
            }
        }

        // Remap adjacency, dropping any edge into a tombstoned slot.
        let newMaxLevel = newLevels.max() ?? -1
        var newLayers: [[[Int]]] = []
        newLayers.reserveCapacity(newMaxLevel + 1)
        for l in 0..<(newMaxLevel + 1) {
            var layer = [[Int]](repeating: [], count: newUUIDs.count)
            if l < layers.count {
                for (old, new) in oldToNew where old < layers[l].count {
                    layer[new] = layers[l][old].compactMap { oldToNew[$0] }
                }
            }
            newLayers.append(layer)
        }

        // Entry point: remap if live; otherwise fall back to the
        // highest-level live node (defensive — remove() already keeps the
        // entry point live in memory).
        var newEntry: Int? = entryPointNode.flatMap { oldToNew[$0] }
        if newEntry == nil, let best = newLevels.indices.max(by: { newLevels[$0] < newLevels[$1] }) {
            newEntry = best
        }

        return (newCodes, newUUIDs, newLevels, newMetadata,
                newLayers, newEntry, newMaxLevel, newOriginals)
    }

    /// Saves this quantized index to a binary file (legacy v2 format).
    ///
    /// Tombstoned (removed) slots are compacted out at save time, so a
    /// loaded index contains exactly the live vectors (`count == liveCount`).
    ///
    /// This writer is unchanged by ADR-014: it always stamps v2 and its bytes
    /// are byte-identical to prior releases. Opt in to the paged-capable v3
    /// format with `save(to:layout:)`.
    public func save(to url: URL) throws {
        let data = try encode(writeVersion: qhWriteVersionV2, padOriginals: false)
        try data.write(to: url, options: .atomic)
    }

    /// Saves this quantized index in the requested on-disk `layout` (ADR-014).
    ///
    /// - `.resident` is byte-identical to ``save(to:)`` (v2).
    /// - `.pagedV3` stamps v3 with the originals section 16 KiB-aligned and a
    ///   section-table trailer WHEN originals are retained; with no originals it
    ///   falls back to v2 (nothing to page).
    public func save(to url: URL, layout: PQHWSaveLayout) throws {
        switch layout {
        case .resident:
            try save(to: url)
        case .pagedV3 where originals != nil:
            let data = try encode(writeVersion: qhWriteVersionV3, padOriginals: true)
            try data.write(to: url, options: .atomic)
        case .pagedV3:
            // No originals to page — v3 would gain nothing, so keep v2.
            try save(to: url)
        }
    }

    /// Test-only seam: serialize as v3 with explicit padding control, mirroring
    /// the `.pxkt` `saveHNSW(padVectorSection:)` precedent so the corruption
    /// suite can reproduce an unpadded (Stage-1-shaped) v3 file and prove it
    /// still loads resident.
    func encodedV3(padOriginals: Bool) throws -> Data {
        try encode(writeVersion: qhWriteVersionV3, padOriginals: padOriginals)
    }

    /// Serializes the compacted live state at the given write version.
    ///
    /// The body (header through metadata) is byte-identical across v2 and v3.
    /// For v3, the originals section is optionally 16 KiB-aligned and a fixed
    /// 128-byte trailer records every section's (offset, length).
    private func encode(writeVersion: UInt32, padOriginals: Bool) throws -> Data {
        var data = Data()

        let live = compactedForSave()
        let nodeCount = live.codes.count
        let M = quantizer.config.subspaceCount

        // Header (56 bytes) — identical layout for v2 and v3.
        qhAppendLE(&data, qhMagic)
        qhAppendLE(&data, writeVersion)
        qhAppendLE(&data, UInt32(dimension))
        qhAppendLE(&data, UInt32(nodeCount))
        qhAppendLE(&data, UInt32(M))
        qhAppendLE(&data, UInt32(hnswConfig.m))
        qhAppendLE(&data, UInt32(hnswConfig.efConstruction))
        qhAppendLE(&data, UInt32(hnswConfig.efSearch))
        qhAppendLE(&data, Int32(live.maxLevel))
        qhAppendLE(&data, Int32(live.entryPoint ?? -1))
        qhAppendLE(&data, UInt32(live.layers.count))
        qhAppendLE(&data, UInt32(quantizer.config.trainingIterations))
        // Originals flag (v2+ — previously reserved bytes, ADR-012)
        qhAppendLE(&data, UInt32(live.originals != nil ? 1 : 0))
        // Reserved
        data.append(Data(repeating: 0, count: 4))

        assert(data.count == qhHeaderSize)

        // Section-boundary tracking for the v3 trailer. For v2 the wrapper only
        // records offsets and never changes a byte, so v2 output is unchanged.
        var sections: [(offset: Int, length: Int)] = []
        func section(_ body: () throws -> Void) throws {
            let start = data.count
            try body()
            sections.append((start, data.count - start))
        }

        try section {  // PQ codebooks: M codebooks, each 256 * ds floats
            for cb in quantizer.codebooks {
                cb.withUnsafeBytes { data.append(contentsOf: $0) }
            }
        }
        try section {  // PQ codes: nodeCount * M bytes
            for code in live.codes { data.append(contentsOf: code) }
        }
        try section {  // UUIDs: nodeCount * 16 bytes
            for uuid in live.uuids {
                let (u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15) = uuid.uuid
                data.append(contentsOf: [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15])
            }
        }
        try section {  // Node levels: nodeCount * Int32
            for level in live.levels { qhAppendLE(&data, Int32(level)) }
        }
        try section {  // Graph: neighbor count + neighbors per node per layer
            for layer in live.layers {
                for neighbors in layer {
                    qhAppendLE(&data, UInt32(neighbors.count))
                    for n in neighbors { qhAppendLE(&data, UInt32(n)) }
                }
            }
        }
        try section {  // Metadata: JSON-encoded
            let metadataPayload: Data
            do {
                metadataPayload = try JSONEncoder().encode(live.metadata.map { $0.map { Array($0) } })
            } catch {
                throw PersistenceError.corruptedData("Failed to encode index metadata")
            }
            qhAppendLE(&data, UInt32(metadataPayload.count))
            data.append(metadataPayload)
        }

        // Originals (iff retained): nodeCount * dimension Float32, row-major,
        // slot-aligned with the PQ codes (ADR-012). For v3, optionally padded to
        // a 16 KiB boundary so the section can be mapped independently.
        if let originals = live.originals {
            assert(originals.count == nodeCount,
                   "compacted originals must stay slot-aligned with codes")
            if writeVersion == qhWriteVersionV3 && padOriginals {
                padToAlignment(&data, alignment: qhOriginalsAlignment)
            }
            let start = data.count
            for vector in originals {
                vector.components.withUnsafeBytes { data.append(contentsOf: $0) }
            }
            sections.append((start, data.count - start))
        } else {
            sections.append((0, 0))   // originals entry is (0, 0) when flag == 0
        }

        // Trailer (v3 only).
        if writeVersion == qhWriteVersionV3 {
            assert(sections.count == qhV3SectionCount)
            qhAppendLE(&data, UInt32(qhV3SectionCount))
            for s in sections {
                qhAppendLE(&data, UInt64(s.offset))
                qhAppendLE(&data, UInt64(s.length))
            }
            qhAppendLE(&data, UInt64(0))   // snapshotGeneration — reserved
            qhAppendLE(&data, qhV3TrailerMagic)
        }

        return data
    }

    /// Loads a quantized index from a binary file.
    ///
    /// v1/v2 files load exactly as before (sequential cursor walk). v3 files
    /// load resident too (Stage 1 has no paged reads): the body is walked
    /// identically, and the originals section is read from the trailer's
    /// recorded offset (jumping over the 16 KiB padding).
    public static func load(from url: URL) throws -> QuantizedHNSWIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        guard fileData.count >= qhHeaderSize else {
            throw PersistenceError.fileTooSmall
        }

        var offset = 0

        // Bounds-checked little-endian readers: a truncated or corrupt file
        // must throw PersistenceError, never read out of bounds.
        func readUInt32() throws -> UInt32 {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("Quantized index file truncated")
            }
            let val = fileData.loadLE(UInt32.self, at: offset)
            offset += 4
            return val
        }

        func readInt32() throws -> Int32 {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("Quantized index file truncated")
            }
            let val = fileData.loadLE(Int32.self, at: offset)
            offset += 4
            return val
        }

        /// Verifies that `byteCount` more bytes exist at the current offset.
        func requireBytes(_ byteCount: Int, _ section: String) throws {
            guard byteCount >= 0, offset <= fileData.count,
                  byteCount <= fileData.count - offset else {
                throw PersistenceError.corruptedData("Quantized index \(section) truncated")
            }
        }

        let fileMagic = try readUInt32()
        guard fileMagic == qhMagic else {
            throw PersistenceError.invalidMagic
        }

        // v1…v3 are readable (ADR-010 rule 2); anything else throws.
        let version = try readUInt32()
        guard version >= qhMinSupportedVersion, version <= qhMaxReadableVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }

        let dim = Int(try readUInt32())
        let nodeCount = Int(try readUInt32())
        let subspaceCount = Int(try readUInt32())
        let hnswM = Int(try readUInt32())
        let efConstruction = Int(try readUInt32())
        let efSearch = Int(try readUInt32())
        let maxLevel = Int(try readInt32())
        let epRaw = Int(try readInt32())
        let entryPoint: Int? = epRaw >= 0 ? epRaw : nil
        let layerCount = Int(try readUInt32())
        let trainIters = Int(try readUInt32())

        // Originals flag lives in the previously reserved bytes at offset 48.
        // It is only meaningful from v2 on; v1 wrote zeros here, and v1 files
        // load with the documented default retainOriginals == false
        // (ADR-010 rule 3 / ADR-012).
        let originalsFlag: UInt32
        if version >= 2 {
            originalsFlag = try readUInt32()
            offset += 4  // remaining reserved
        } else {
            originalsFlag = 0
            offset += 8  // v1: all 8 bytes reserved
        }
        assert(offset == qhHeaderSize)

        guard originalsFlag <= 1 else {
            throw PersistenceError.corruptedData(
                "Quantized index originals flag must be 0 or 1, got \(originalsFlag)")
        }
        let hasOriginals = originalsFlag == 1

        // ── Header sanity (prevents traps before section reads) ───────
        // ProductQuantizer / PQConfiguration / HNSWConfiguration enforce
        // these with preconditions, so a corrupt header would otherwise
        // crash the process instead of throwing.
        guard dim > 0, subspaceCount > 0, dim % subspaceCount == 0 else {
            throw PersistenceError.corruptedData(
                "Quantized index dimension \(dim) / subspaceCount \(subspaceCount) invalid")
        }
        guard trainIters > 0 else {
            throw PersistenceError.corruptedData(
                "Quantized index trainingIterations must be positive, got \(trainIters)")
        }
        // m >= 2 matches HNSWConfiguration's precondition; m == 1 would trap
        // in the initializer (1/log(1) is infinite), so reject it here instead.
        guard hnswM >= 2, efConstruction > 0, efSearch > 0 else {
            throw PersistenceError.corruptedData(
                "Quantized index HNSW configuration fields out of range "
                + "(m: \(hnswM) [min 2], efConstruction: \(efConstruction), efSearch: \(efSearch))")
        }
        guard layerCount <= fileData.count else {
            throw PersistenceError.corruptedData(
                "Quantized index layer count \(layerCount) implausible for file of \(fileData.count) bytes")
        }
        guard maxLevel >= -1, maxLevel < layerCount else {
            throw PersistenceError.corruptedData(
                "Quantized index maxLevel \(maxLevel) outside valid range -1..<\(layerCount)")
        }
        if let entryPoint {
            guard entryPoint < nodeCount else {
                throw PersistenceError.corruptedData(
                    "Quantized index entry point \(entryPoint) outside valid range 0..<\(nodeCount)")
            }
        }

        // For v3, parse and validate the trailer up front (bounds, magic,
        // section count, contiguity). The originals section is then read from
        // its recorded offset rather than the post-metadata cursor.
        let trailer: PQHWTrailer? = version >= qhWriteVersionV3
            ? try PQHWTrailer.parse(fileData)
            : nil

        let ds = dim / subspaceCount
        let K = 256

        // PQ codebooks
        var codebooks = [[Float]]()
        codebooks.reserveCapacity(subspaceCount)
        for _ in 0..<subspaceCount {
            let floatCount = K * ds
            let byteCount = floatCount * 4
            try requireBytes(byteCount, "codebook section")
            var floats = [Float](repeating: 0, count: floatCount)
            fileData.withUnsafeBytes { buffer in
                let src = buffer.baseAddress!.advanced(by: offset)
                floats.withUnsafeMutableBytes { dest in
                    dest.copyMemory(from: UnsafeRawBufferPointer(start: src, count: byteCount))
                }
            }
            codebooks.append(floats)
            offset += byteCount
        }

        let pqConfig = PQConfiguration(subspaceCount: subspaceCount, trainingIterations: trainIters)
        let quantizer = ProductQuantizer(dimension: dim, config: pqConfig, codebooks: codebooks)

        // PQ codes
        let (codeBytes, codeOverflow) = nodeCount.multipliedReportingOverflow(by: subspaceCount)
        guard !codeOverflow else {
            throw PersistenceError.corruptedData("Quantized index code section truncated")
        }
        try requireBytes(codeBytes, "code section")
        var pqCodes = [[UInt8]]()
        pqCodes.reserveCapacity(nodeCount)
        for _ in 0..<nodeCount {
            let code = Array(fileData[offset..<offset + subspaceCount])
            pqCodes.append(code)
            offset += subspaceCount
        }

        // UUIDs
        let (uuidBytes, uuidOverflow) = nodeCount.multipliedReportingOverflow(by: 16)
        guard !uuidOverflow else {
            throw PersistenceError.corruptedData("Quantized index UUID section truncated")
        }
        try requireBytes(uuidBytes, "UUID section")
        var nodeToUUID = [UUID]()
        nodeToUUID.reserveCapacity(nodeCount)
        var uuidToNode = [UUID: Int]()
        for i in 0..<nodeCount {
            let bytes = fileData[offset..<offset + 16]
            let b = Array(bytes)
            let uuid = UUID(uuid: (b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                                   b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]))
            nodeToUUID.append(uuid)
            uuidToNode[uuid] = i
            offset += 16
        }

        // Node levels
        var nodeLevels = [Int]()
        nodeLevels.reserveCapacity(nodeCount)
        for _ in 0..<nodeCount {
            let level = Int(try readInt32())
            guard level >= 0, level < layerCount else {
                throw PersistenceError.corruptedData(
                    "Quantized index node level \(level) outside valid range 0..<\(layerCount)")
            }
            nodeLevels.append(level)
        }

        // Graph layers
        var layers = [[[Int]]]()
        layers.reserveCapacity(layerCount)
        for _ in 0..<layerCount {
            var layer = [[Int]]()
            layer.reserveCapacity(nodeCount)
            for _ in 0..<nodeCount {
                let neighborCount = Int(try readUInt32())
                var neighbors = [Int]()
                for _ in 0..<neighborCount {
                    let n = Int(try readUInt32())
                    guard n < nodeCount else {
                        throw PersistenceError.corruptedData(
                            "Quantized index neighbor \(n) outside valid range 0..<\(nodeCount)")
                    }
                    neighbors.append(n)
                }
                layer.append(neighbors)
            }
            layers.append(layer)
        }

        // Metadata
        let metadataSize = Int(try readUInt32())
        try requireBytes(metadataSize, "metadata section")
        let metadataPayload = fileData[offset..<offset + metadataSize]
        let decodedMetadata: [[UInt8]?]
        do {
            decodedMetadata = try JSONDecoder().decode([[UInt8]?].self, from: metadataPayload)
        } catch {
            throw PersistenceError.corruptedData("Quantized index metadata is not valid JSON")
        }
        guard decodedMetadata.count == nodeCount else {
            throw PersistenceError.corruptedData(
                "Quantized index metadata count \(decodedMetadata.count) != node count \(nodeCount)")
        }
        let metadata: [Data?] = decodedMetadata.map { $0.map { Data($0) } }
        offset += metadataSize

        // Originals: nodeCount * dim Float32, slot-aligned with the PQ codes.
        // v1/v2 read from the post-metadata cursor; v3 reads from the trailer's
        // recorded offset (past the 16 KiB padding). Truncation, length, and
        // flag/entry-consistency violations must all throw a typed error.
        var originals: [Vector]? = nil
        let originalsReadOffset: Int
        if let trailer {
            let entry = trailer.sections[qhOriginalsSectionIndex]
            if hasOriginals {
                guard entry.length > 0 else {
                    throw PersistenceError.corruptedData(
                        "Quantized index originals flag is 1 but the trailer originals entry is empty")
                }
                originalsReadOffset = entry.offset
            } else {
                guard entry.offset == 0, entry.length == 0 else {
                    throw PersistenceError.corruptedData(
                        "Quantized index originals flag is 0 but the trailer records a nonzero originals entry")
                }
                originalsReadOffset = offset   // unused
            }
        } else {
            originalsReadOffset = offset
        }

        if hasOriginals {
            let (floatCount, floatOverflow) = nodeCount.multipliedReportingOverflow(by: dim)
            let (originalsBytes, byteOverflow) =
                floatCount.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
            guard !floatOverflow, !byteOverflow else {
                throw PersistenceError.corruptedData("Quantized index originals section truncated")
            }
            // For v3, the trailer's recorded length must match exactly.
            if let trailer {
                let entry = trailer.sections[qhOriginalsSectionIndex]
                guard entry.length == originalsBytes else {
                    throw PersistenceError.corruptedData(
                        "Quantized index originals length \(entry.length) != \(nodeCount) × \(dim) × 4")
                }
            }
            offset = originalsReadOffset
            try requireBytes(originalsBytes, "originals section")
            var loaded = [Vector]()
            loaded.reserveCapacity(nodeCount)
            let vectorBytes = dim * MemoryLayout<Float>.size
            for _ in 0..<nodeCount {
                var floats = [Float](repeating: 0, count: dim)
                fileData.withUnsafeBytes { buffer in
                    let src = buffer.baseAddress!.advanced(by: offset)
                    floats.withUnsafeMutableBytes { dest in
                        dest.copyMemory(from: UnsafeRawBufferPointer(start: src, count: vectorBytes))
                    }
                }
                loaded.append(Vector(floats))
                offset += vectorBytes
            }
            originals = loaded
        }

        let hnswConfig = HNSWConfiguration(
            m: hnswM,
            efConstruction: efConstruction,
            efSearch: efSearch,
            autoCompactionThreshold: nil
        )

        return QuantizedHNSWIndex(
            dimension: dim,
            hnswConfig: hnswConfig,
            quantizer: quantizer,
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPoint,
            maxLevel: maxLevel,
            codes: pqCodes,
            nodeToUUID: nodeToUUID,
            uuidToNode: uuidToNode,
            metadata: metadata,
            originals: originals
        )
    }

    // MARK: - Migration (ADR-014 M-A: offline section-copy rewrite)

    /// Upgrades a PQHW v1/v2 (or unpadded-v3) base file at `url` in place to a
    /// padded v3 base, without decoding the index.
    ///
    /// The rewrite is a pure section-copy: every section payload in the output
    /// is byte-identical to the input's, the originals section (iff retained) is
    /// pushed onto a 16 KiB boundary, the version is stamped 3, and a section
    /// table trailer is appended. Crash safety: the upgraded image is written to
    /// a sibling temp file and verified (full section checksum) before an atomic
    /// replace, so an interrupted upgrade leaves the source untouched.
    ///
    /// Migrating an already-padded v3 file is a no-op (clean success): the file
    /// is left byte-for-byte as it was.
    ///
    /// - Throws: `PersistenceError` for a non-PQHW file, an inconsistent header,
    ///   an adversarial section walk, or a failed verification (never traps).
    public static func upgradeToV3(at url: URL) throws {
        let source = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let image = try buildPaddedV3Image(from: source) else {
            return   // already a padded v3 base — nothing to do
        }
        let tmp = url.appendingPathExtension("pqhwv3tmp")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try image.write(to: tmp, options: .atomic)
        // Full-checksum verification BEFORE the atomic replace: if the temp is
        // torn or a section drifted, throw and leave the source in place.
        try verifyPaddedV3Upgrade(source: source, upgradedURL: tmp)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: tmp)
    }

    /// Builds the padded-v3 image for a v1/v2/unpadded-v3 PQHW `source`, or
    /// returns `nil` if `source` is already a padded v3 base (no-op upgrade).
    /// Section payloads are byte-identical copies of the source. Testable seam
    /// for the migration + crash-safety suites.
    static func buildPaddedV3Image(from source: Data) throws -> Data? {
        let layout = try PQHWSectionLayout.parse(source)
        // Idempotence: already v3 with an aligned (or absent) originals section.
        if layout.version == qhWriteVersionV3 {
            if !layout.hasOriginals || layout.originals.offset % qhOriginalsAlignment == 0 {
                return nil
            }
        }

        var out = Data()
        out.reserveCapacity(source.count + qhOriginalsAlignment + qhV3TrailerSize)

        // Header: copy the 56 bytes verbatim, then stamp version 3 at offset 4.
        out.append(source[0..<qhHeaderSize])
        withUnsafeBytes(of: qhWriteVersionV3.littleEndian) { bytes in
            out.replaceSubrange(4..<8, with: bytes)
        }

        // Body (codebooks…metadata) is contiguous and byte-identical.
        let bodyRange = layout.codebooks.offset..<layout.metadata.end
        out.append(source[bodyRange])

        // Padding + originals (iff retained).
        var originalsEntry = (offset: 0, length: 0)
        if layout.hasOriginals {
            padToAlignment(&out, alignment: qhOriginalsAlignment)
            let start = out.count
            out.append(source[layout.originals.offset..<layout.originals.end])
            originalsEntry = (offset: start, length: layout.originals.length)
        }

        // Trailer with recomputed offsets (body offsets shift by the header —
        // which is unchanged in size — plus any padding before originals).
        let shifted = layout.shiftedSections(bodyBaseInOutput: qhHeaderSize)
        var sections = shifted
        sections.append(originalsEntry)
        appendPQHWTrailer(&out, sections: sections, generation: layout.generation)
        return out
    }

    /// Re-reads the upgraded temp file and asserts every section payload is
    /// byte-identical to the corresponding source section (ADR-014 fidelity
    /// gate; full-checksum, whose O(n) cost is a documented on-device tradeoff —
    /// open question 3).
    static func verifyPaddedV3Upgrade(source: Data, upgradedURL: URL) throws {
        let upgraded = try Data(contentsOf: upgradedURL)
        let src = try PQHWSectionLayout.parse(source)
        let dst = try PQHWSectionLayout.parse(upgraded)
        guard dst.version == qhWriteVersionV3 else {
            throw PersistenceError.corruptedData("upgraded PQHW base is not v3")
        }
        guard dst.hasOriginals == src.hasOriginals else {
            throw PersistenceError.corruptedData("upgraded PQHW originals flag drifted")
        }
        if dst.hasOriginals {
            guard dst.originals.offset % qhOriginalsAlignment == 0 else {
                throw PersistenceError.corruptedData("upgraded PQHW originals section is not 16 KiB-aligned")
            }
        }
        // Section-payload equality across all present sections.
        let pairs: [(String, PQHWSectionLayout.Section, PQHWSectionLayout.Section)] = [
            ("codebooks", src.codebooks, dst.codebooks),
            ("codes", src.codes, dst.codes),
            ("uuids", src.uuids, dst.uuids),
            ("nodeLevels", src.nodeLevels, dst.nodeLevels),
            ("adjacency", src.adjacency, dst.adjacency),
            ("metadata", src.metadata, dst.metadata),
        ] + (src.hasOriginals ? [("originals", src.originals, dst.originals)] : [])
        for (name, s, d) in pairs {
            guard s.length == d.length else {
                throw PersistenceError.corruptedData("upgraded PQHW \(name) length drifted")
            }
            guard source[s.offset..<s.end] == upgraded[d.offset..<d.end] else {
                throw PersistenceError.corruptedData("upgraded PQHW \(name) payload not bit-identical")
            }
        }
    }
}

// MARK: - v3 Trailer parsing

/// The PQHW v3 section table + reserved generation, parsed from the fixed
/// 128-byte trailer. Every offset/length is bounds-checked against the file, so
/// a crafted or truncated trailer is a typed error, never a trap (ADR-010 rule 5).
struct PQHWTrailer {
    let sections: [(offset: Int, length: Int)]   // 7 entries, table order
    let generation: UInt64

    static func parse(_ data: Data) throws -> PQHWTrailer {
        guard data.count >= qhHeaderSize + qhV3TrailerSize else {
            throw PersistenceError.corruptedData("PQHW v3 file too small for trailer")
        }
        let trailerStart = data.count - qhV3TrailerSize
        let magic = data.loadLE(UInt32.self, at: data.count - 4)
        guard magic == qhV3TrailerMagic else {
            throw PersistenceError.corruptedData("PQHW v3 trailer magic mismatch")
        }
        let sectionCount = data.loadLE(UInt32.self, at: trailerStart)
        guard sectionCount == UInt32(qhV3SectionCount) else {
            throw PersistenceError.corruptedData(
                "PQHW v3 trailer section count \(sectionCount) != \(qhV3SectionCount)")
        }

        let bodyEnd = UInt64(trailerStart)
        var sections: [(offset: Int, length: Int)] = []
        sections.reserveCapacity(qhV3SectionCount)
        for i in 0..<qhV3SectionCount {
            let entry = trailerStart + 4 + i * 16
            let offset = data.loadLE(UInt64.self, at: entry)
            let length = data.loadLE(UInt64.self, at: entry + 8)
            let (end, overflow) = offset.addingReportingOverflow(length)
            guard !overflow, offset <= bodyEnd, end <= bodyEnd else {
                throw PersistenceError.corruptedData(
                    "PQHW v3 section \(i) (offset \(offset), length \(length)) out of bounds "
                    + "for a \(data.count)-byte file")
            }
            sections.append((Int(offset), Int(length)))
        }

        // Structural integrity: codebooks start right after the 56-byte header,
        // and sections 0…5 (codebooks…metadata) are tightly contiguous — this
        // rejects overlapping and non-monotonic section tables.
        guard sections[0].offset == qhHeaderSize else {
            throw PersistenceError.corruptedData(
                "PQHW v3 first section must start at \(qhHeaderSize), got \(sections[0].offset)")
        }
        for i in 1...5 {
            guard sections[i].offset == sections[i - 1].offset + sections[i - 1].length else {
                throw PersistenceError.corruptedData(
                    "PQHW v3 section \(i) is not contiguous with section \(i - 1) (overlap/gap/non-monotonic)")
            }
        }
        // Originals (index 6): either (0, 0) — no originals — or a section that
        // begins at or after the metadata end.
        let originals = sections[qhOriginalsSectionIndex]
        let metaEnd = sections[5].offset + sections[5].length
        if originals.length == 0 {
            guard originals.offset == 0 else {
                throw PersistenceError.corruptedData(
                    "PQHW v3 empty originals section must have offset 0, got \(originals.offset)")
            }
        } else {
            guard originals.offset >= metaEnd else {
                throw PersistenceError.corruptedData(
                    "PQHW v3 originals section overlaps the metadata section")
            }
        }

        let generation = data.loadLE(UInt64.self, at: trailerStart + 4 + qhV3SectionCount * 16)
        return PQHWTrailer(sections: sections, generation: generation)
    }
}

// MARK: - Section layout (migration boundary walk)

/// The byte ranges of every PQHW section, resolved for the migration rewriter
/// WITHOUT decoding the index. For v1/v2 the adjacency length is recovered by a
/// bounds-checked count-prefixed walk; for v3 the ranges come from the trailer.
struct PQHWSectionLayout {
    struct Section { let offset: Int; let length: Int; var end: Int { offset + length } }

    let version: UInt32
    let hasOriginals: Bool
    let generation: UInt64
    let codebooks: Section
    let codes: Section
    let uuids: Section
    let nodeLevels: Section
    let adjacency: Section
    let metadata: Section
    let originals: Section

    /// Body sections [codebooks…metadata] re-based so codebooks starts at
    /// `bodyBaseInOutput` (the 56-byte header end). Used to build the v3 trailer.
    func shiftedSections(bodyBaseInOutput: Int) -> [(offset: Int, length: Int)] {
        let delta = bodyBaseInOutput - codebooks.offset
        return [codebooks, codes, uuids, nodeLevels, adjacency, metadata].map {
            (offset: $0.offset + delta, length: $0.length)
        }
    }

    static func parse(_ data: Data) throws -> PQHWSectionLayout {
        guard data.count >= qhHeaderSize else { throw PersistenceError.fileTooSmall }
        guard data.loadLE(UInt32.self, at: 0) == qhMagic else { throw PersistenceError.invalidMagic }
        let version = data.loadLE(UInt32.self, at: 4)
        guard version >= qhMinSupportedVersion, version <= qhMaxReadableVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }
        let dim = Int(data.loadLE(UInt32.self, at: 8))
        let nodeCount = Int(data.loadLE(UInt32.self, at: 12))
        let subspaceCount = Int(data.loadLE(UInt32.self, at: 16))
        let layerCount = Int(data.loadLE(UInt32.self, at: 40))
        let flag = version >= 2 ? data.loadLE(UInt32.self, at: 48) : 0
        guard flag <= 1 else {
            throw PersistenceError.corruptedData("PQHW originals flag must be 0 or 1, got \(flag)")
        }
        guard dim > 0, subspaceCount > 0, dim % subspaceCount == 0, nodeCount >= 0, layerCount >= 0 else {
            throw PersistenceError.corruptedData("PQHW header fields invalid for migration")
        }
        let hasOriginals = flag == 1
        let ds = dim / subspaceCount

        // Fixed-size sections computed from header counts (overflow-checked).
        func region(_ start: Int, _ count: Int, _ unit: Int, _ what: String) throws -> Section {
            let (bytes, ov1) = count.multipliedReportingOverflow(by: unit)
            guard !ov1, bytes >= 0, start <= data.count, bytes <= data.count - start else {
                throw PersistenceError.corruptedData("PQHW \(what) section out of bounds")
            }
            return Section(offset: start, length: bytes)
        }
        let cbFloats = subspaceCount * 256 * ds
        let codebooks = try region(qhHeaderSize, cbFloats, 4, "codebooks")
        let codes = try region(codebooks.end, nodeCount, subspaceCount, "codes")
        let uuids = try region(codes.end, nodeCount, 16, "uuids")
        let nodeLevels = try region(uuids.end, nodeCount, 4, "nodeLevels")

        // The trailing body end: v3 files carry the 128-byte trailer we must
        // stop before; v1/v2 run to EOF.
        let bodyEnd = version >= qhWriteVersionV3 ? data.count - qhV3TrailerSize : data.count
        guard bodyEnd >= nodeLevels.end else {
            throw PersistenceError.corruptedData("PQHW file truncated before adjacency")
        }

        // Adjacency: bounds-checked count-prefixed walk (no graph decode).
        var cursor = nodeLevels.end
        for _ in 0..<layerCount {
            for _ in 0..<nodeCount {
                guard cursor + 4 <= bodyEnd else {
                    throw PersistenceError.corruptedData("PQHW adjacency truncated")
                }
                let neighborCount = Int(data.loadLE(UInt32.self, at: cursor))
                cursor += 4
                let (nbytes, ov) = neighborCount.multipliedReportingOverflow(by: 4)
                guard !ov, neighborCount >= 0, cursor <= bodyEnd, nbytes <= bodyEnd - cursor else {
                    throw PersistenceError.corruptedData("PQHW adjacency neighbor list truncated")
                }
                cursor += nbytes
            }
        }
        let adjacency = Section(offset: nodeLevels.end, length: cursor - nodeLevels.end)

        // Metadata: UInt32 length prefix + payload.
        guard cursor + 4 <= bodyEnd else {
            throw PersistenceError.corruptedData("PQHW metadata length prefix truncated")
        }
        let metaLen = Int(data.loadLE(UInt32.self, at: cursor))
        let metaTotal = 4 + metaLen
        guard metaLen >= 0, cursor <= bodyEnd, metaTotal <= bodyEnd - cursor else {
            throw PersistenceError.corruptedData("PQHW metadata section truncated")
        }
        let metadata = Section(offset: cursor, length: metaTotal)

        // Originals: nodeCount * dim * 4, iff the flag is set. Must consume
        // exactly the remaining body (for v1/v2/unpadded-v3 inputs; a padded v3
        // input is short-circuited as a no-op before this parse is used to
        // rebuild, so any padding here is only tolerated when re-reading a
        // padded v3 for verification — handled via the trailer path below).
        var originals = Section(offset: 0, length: 0)
        if version >= qhWriteVersionV3 {
            // Trust the validated trailer for a v3 input.
            let trailer = try PQHWTrailer.parse(data)
            let entry = trailer.sections[qhOriginalsSectionIndex]
            originals = Section(offset: entry.offset, length: entry.length)
            let generation = trailer.generation
            return PQHWSectionLayout(
                version: version, hasOriginals: hasOriginals, generation: generation,
                codebooks: codebooks, codes: codes, uuids: uuids, nodeLevels: nodeLevels,
                adjacency: adjacency, metadata: metadata, originals: originals)
        }
        if hasOriginals {
            originals = try region(metadata.end, nodeCount, dim * 4, "originals")
            guard originals.end == bodyEnd else {
                throw PersistenceError.corruptedData("PQHW originals section does not reach end of file")
            }
        } else {
            guard metadata.end == bodyEnd else {
                throw PersistenceError.corruptedData("PQHW trailing bytes after metadata (flag 0)")
            }
        }
        return PQHWSectionLayout(
            version: version, hasOriginals: hasOriginals, generation: 0,
            codebooks: codebooks, codes: codes, uuids: uuids, nodeLevels: nodeLevels,
            adjacency: adjacency, metadata: metadata, originals: originals)
    }
}

// MARK: - Binary Helpers (namespaced to avoid collisions)

private func qhAppendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

/// Zero-pads `data` so its length becomes a multiple of `alignment` (pushes the
/// originals section start onto a 16 KiB boundary for v3 — ADR-014).
private func padToAlignment(_ data: inout Data, alignment: Int) {
    let remainder = data.count % alignment
    if remainder != 0 {
        data.append(Data(count: alignment - remainder))
    }
}

/// Appends the fixed 128-byte PQHW v3 trailer.
private func appendPQHWTrailer(
    _ data: inout Data, sections: [(offset: Int, length: Int)], generation: UInt64
) {
    precondition(sections.count == qhV3SectionCount)
    qhAppendLE(&data, UInt32(qhV3SectionCount))
    for s in sections {
        qhAppendLE(&data, UInt64(s.offset))
        qhAppendLE(&data, UInt64(s.length))
    }
    qhAppendLE(&data, generation)
    qhAppendLE(&data, qhV3TrailerMagic)
}

// Uses Data.loadLE from PersistenceEngine.swift
