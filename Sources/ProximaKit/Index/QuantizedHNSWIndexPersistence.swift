// QuantizedHNSWIndexPersistence.swift
// ProximaKit
//
// Binary persistence for QuantizedHNSWIndex.
//
// File format (all little-endian):
//   Magic:           0x50514857 ("PQHW")
//   Version:         UInt32 (1)
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
//   Reserved:        8 bytes (zero)
//   --- 56-byte header ---
//   PQ codebooks:    M * 256 * ds Float32 values
//   PQ codes:        nodeCount * M UInt8 values
//   UUIDs:           nodeCount * 16 bytes
//   Node levels:     nodeCount * Int32
//   Graph layers:    per-layer adjacency lists
//   Metadata:        JSON-encoded [Data?]

import Foundation

private let qhMagic: UInt32 = 0x50514857       // "PQHW"
private let qhFormatVersion: UInt32 = 1
private let qhHeaderSize = 56

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
        metadata: [Data?], layers: [[[Int]]], entryPoint: Int?, maxLevel: Int
    ) {
        // Fast path: no tombstones — serialize state as-is (byte-identical
        // to the pre-compaction format, and no array copies).
        if uuidToNode.count == nodeToUUID.count {
            return (codes, nodeToUUID, nodeLevels, metadata,
                    layers, entryPointNode, maxLevel)
        }

        // Dense renumbering of live slots (identity check, not presence).
        var oldToNew = [Int: Int]()
        var newCodes: [[UInt8]] = []
        var newUUIDs: [UUID] = []
        var newLevels: [Int] = []
        var newMetadata: [Data?] = []
        for (node, uuid) in nodeToUUID.enumerated() where uuidToNode[uuid] == node {
            oldToNew[node] = newUUIDs.count
            newCodes.append(codes[node])
            newUUIDs.append(uuid)
            newLevels.append(nodeLevels[node])
            newMetadata.append(metadata[node])
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
                newLayers, newEntry, newMaxLevel)
    }

    /// Saves this quantized index to a binary file.
    ///
    /// Tombstoned (removed) slots are compacted out at save time, so a
    /// loaded index contains exactly the live vectors (`count == liveCount`).
    public func save(to url: URL) throws {
        var data = Data()

        let live = compactedForSave()
        let nodeCount = live.codes.count
        let M = quantizer.config.subspaceCount

        // Header (56 bytes)
        qhAppendLE(&data, qhMagic)
        qhAppendLE(&data, qhFormatVersion)
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
        // Reserved
        data.append(Data(repeating: 0, count: 8))

        assert(data.count == qhHeaderSize)

        // PQ codebooks: M codebooks, each 256 * ds floats
        for cb in quantizer.codebooks {
            cb.withUnsafeBytes { data.append(contentsOf: $0) }
        }

        // PQ codes: nodeCount * M bytes
        for code in live.codes {
            data.append(contentsOf: code)
        }

        // UUIDs: nodeCount * 16 bytes
        for uuid in live.uuids {
            let (u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15) = uuid.uuid
            data.append(contentsOf: [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15])
        }

        // Node levels: nodeCount * Int32
        for level in live.levels {
            qhAppendLE(&data, Int32(level))
        }

        // Graph: for each layer, for each node, write neighbor count + neighbors
        for layer in live.layers {
            for neighbors in layer {
                qhAppendLE(&data, UInt32(neighbors.count))
                for n in neighbors {
                    qhAppendLE(&data, UInt32(n))
                }
            }
        }

        // Metadata: JSON-encoded
        let metadataPayload = try JSONEncoder().encode(live.metadata.map { $0.map { Array($0) } })
        qhAppendLE(&data, UInt32(metadataPayload.count))
        data.append(metadataPayload)

        try data.write(to: url, options: .atomic)
    }

    /// Loads a quantized index from a binary file.
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

        let version = try readUInt32()
        guard version == qhFormatVersion else {
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

        // Skip reserved
        offset += 8
        assert(offset == qhHeaderSize)

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
        let metadata: [Data?] = decodedMetadata.map { $0.map { Data($0) } }

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
            metadata: metadata
        )
    }
}

// MARK: - Binary Helpers (namespaced to avoid collisions)

private func qhAppendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

// Uses Data.loadLE from PersistenceEngine.swift
