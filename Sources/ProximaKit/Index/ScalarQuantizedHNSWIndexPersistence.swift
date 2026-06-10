// ScalarQuantizedHNSWIndexPersistence.swift
// ProximaKit
//
// Binary persistence for ScalarQuantizedHNSWIndex.
//
// File format (all little-endian, per ADR-007 / ADR-010):
//   Magic:           0x53514857 ("SQHW")
//   Version:         UInt32 (1)
//   Dimension:       UInt32
//   NodeCount:       UInt32
//   Metric:          UInt32 (DistanceMetricType raw value)
//   M:               UInt32
//   EfConstruction:  UInt32
//   EfSearch:        UInt32
//   MaxLevel:        Int32
//   EntryPoint:      Int32 (-1 if nil)
//   LayerCount:      UInt32
//   ThresholdBits:   UInt64 (Float64 bit pattern of autoCompactionThreshold;
//                    all-zero bits encode nil, following the .pxkt v2 precedent)
//   Reserved:        12 bytes (zero)
//   --- 64-byte header ---
//   Scales:          nodeCount * Float32
//   Codes:           nodeCount * dimension Int8 values (row-major)
//   UUIDs:           nodeCount * 16 bytes
//   Node levels:     nodeCount * Int32
//   Graph layers:    per-layer adjacency lists
//   Metadata:        JSON-encoded [Data?]
//
// There is no standalone quantizer codec (no "SQTT"): ScalarQuantizer is
// stateless, so only the index format exists. See ADR-007.

import Foundation

private let sqMagic: UInt32 = 0x53514857       // "SQHW"
private let sqFormatVersion: UInt32 = 1
private let sqHeaderSize = 64

extension ScalarQuantizedHNSWIndex {

    /// Saves this scalar-quantized index to a binary file.
    public func save(to url: URL) throws {
        var data = Data()

        let nodeCount = codes.count

        // Header (64 bytes)
        sqAppendLE(&data, sqMagic)
        sqAppendLE(&data, sqFormatVersion)
        sqAppendLE(&data, UInt32(dimension))
        sqAppendLE(&data, UInt32(nodeCount))
        sqAppendLE(&data, metricType.rawValue)
        sqAppendLE(&data, UInt32(hnswConfig.m))
        sqAppendLE(&data, UInt32(hnswConfig.efConstruction))
        sqAppendLE(&data, UInt32(hnswConfig.efSearch))
        sqAppendLE(&data, Int32(maxLevel))
        sqAppendLE(&data, Int32(entryPointNode ?? -1))
        sqAppendLE(&data, UInt32(layers.count))
        // autoCompactionThreshold: Float64 bit pattern; all-zero bits = nil.
        // (A real threshold is never 0.0 — HNSWConfiguration requires (0, 1).)
        let thresholdBits = hnswConfig.autoCompactionThreshold?.bitPattern ?? 0
        sqAppendLE(&data, thresholdBits)
        // Reserved
        data.append(Data(repeating: 0, count: 12))

        assert(data.count == sqHeaderSize)

        // Scales: nodeCount * Float32
        scales.withUnsafeBytes { data.append(contentsOf: $0) }

        // Codes: nodeCount * dimension Int8
        for code in codes {
            code.withUnsafeBytes { data.append(contentsOf: $0) }
        }

        // UUIDs: nodeCount * 16 bytes
        for uuid in nodeToUUID {
            let (u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15) = uuid.uuid
            data.append(contentsOf: [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15])
        }

        // Node levels: nodeCount * Int32
        for level in nodeLevels {
            sqAppendLE(&data, Int32(level))
        }

        // Graph: for each layer, for each node, write neighbor count + neighbors
        for layer in layers {
            for neighbors in layer {
                sqAppendLE(&data, UInt32(neighbors.count))
                for n in neighbors {
                    sqAppendLE(&data, UInt32(n))
                }
            }
        }

        // Metadata: JSON-encoded
        let metadataPayload = try JSONEncoder().encode(metadata.map { $0.map { Array($0) } })
        sqAppendLE(&data, UInt32(metadataPayload.count))
        data.append(metadataPayload)

        try data.write(to: url, options: .atomic)
    }

    /// Loads a scalar-quantized index from a binary file.
    public static func load(from url: URL) throws -> ScalarQuantizedHNSWIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        guard fileData.count >= sqHeaderSize else {
            throw PersistenceError.fileTooSmall
        }

        var offset = 0

        // Bounds-checked little-endian readers: a truncated or corrupt file
        // must throw PersistenceError, never read out of bounds.
        func readUInt32() throws -> UInt32 {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("Scalar-quantized index file truncated")
            }
            let val = fileData.loadLE(UInt32.self, at: offset)
            offset += 4
            return val
        }

        func readInt32() throws -> Int32 {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("Scalar-quantized index file truncated")
            }
            let val = fileData.loadLE(Int32.self, at: offset)
            offset += 4
            return val
        }

        func readUInt64() throws -> UInt64 {
            guard offset + 8 <= fileData.count else {
                throw PersistenceError.corruptedData("Scalar-quantized index file truncated")
            }
            let val = fileData.loadLE(UInt64.self, at: offset)
            offset += 8
            return val
        }

        /// Verifies that `byteCount` more bytes exist at the current offset.
        func requireBytes(_ byteCount: Int, _ section: String) throws {
            guard byteCount >= 0, offset <= fileData.count,
                  byteCount <= fileData.count - offset else {
                throw PersistenceError.corruptedData("Scalar-quantized index \(section) truncated")
            }
        }

        let fileMagic = try readUInt32()
        guard fileMagic == sqMagic else {
            throw PersistenceError.invalidMagic
        }

        let version = try readUInt32()
        guard version == sqFormatVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }

        let dim = Int(try readUInt32())
        let nodeCount = Int(try readUInt32())
        let metricRaw = try readUInt32()
        let hnswM = Int(try readUInt32())
        let efConstruction = Int(try readUInt32())
        let efSearch = Int(try readUInt32())
        let maxLevel = Int(try readInt32())
        let epRaw = Int(try readInt32())
        let entryPoint: Int? = epRaw >= 0 ? epRaw : nil
        let layerCount = Int(try readUInt32())
        let thresholdBits = try readUInt64()

        // Skip reserved
        offset += 12
        assert(offset == sqHeaderSize)

        // ── Header sanity (prevents traps before section reads) ───────
        // ScalarQuantizer / HNSWConfiguration enforce these with
        // preconditions, so a corrupt header would otherwise crash the
        // process instead of throwing.
        guard dim > 0 else {
            throw PersistenceError.corruptedData(
                "Scalar-quantized index dimension must be positive, got \(dim)")
        }
        guard let metricType = DistanceMetricType(rawValue: metricRaw) else {
            throw PersistenceError.unknownMetricType(metricRaw)
        }
        // m >= 2 matches HNSWConfiguration's precondition; m == 1 would trap
        // in the initializer (1/log(1) is infinite), so reject it here instead.
        guard hnswM >= 2, efConstruction > 0, efSearch > 0 else {
            throw PersistenceError.corruptedData(
                "Scalar-quantized index HNSW configuration fields out of range "
                + "(m: \(hnswM) [min 2], efConstruction: \(efConstruction), efSearch: \(efSearch))")
        }
        // Threshold: all-zero bits = nil; anything else must decode into the
        // open interval (0, 1) or HNSWConfiguration's precondition would trap.
        let autoCompactionThreshold: Double?
        if thresholdBits == 0 {
            autoCompactionThreshold = nil
        } else {
            let threshold = Double(bitPattern: thresholdBits)
            guard threshold > 0, threshold < 1 else {
                throw PersistenceError.corruptedData(
                    "Scalar-quantized index autoCompactionThreshold \(threshold) outside valid range (0, 1)")
            }
            autoCompactionThreshold = threshold
        }
        guard layerCount <= fileData.count else {
            throw PersistenceError.corruptedData(
                "Scalar-quantized index layer count \(layerCount) implausible for file of \(fileData.count) bytes")
        }
        guard maxLevel >= -1, maxLevel < layerCount else {
            throw PersistenceError.corruptedData(
                "Scalar-quantized index maxLevel \(maxLevel) outside valid range -1..<\(layerCount)")
        }
        if let entryPoint {
            guard entryPoint < nodeCount else {
                throw PersistenceError.corruptedData(
                    "Scalar-quantized index entry point \(entryPoint) outside valid range 0..<\(nodeCount)")
            }
        }

        // Scales
        let (scaleBytes, scaleOverflow) = nodeCount.multipliedReportingOverflow(by: 4)
        guard !scaleOverflow, nodeCount >= 0 else {
            throw PersistenceError.corruptedData("Scalar-quantized index scale section truncated")
        }
        try requireBytes(scaleBytes, "scale section")
        var scales = [Float](repeating: 0, count: nodeCount)
        if nodeCount > 0 {
            fileData.withUnsafeBytes { buffer in
                let src = buffer.baseAddress!.advanced(by: offset)
                scales.withUnsafeMutableBytes { dest in
                    dest.copyMemory(from: UnsafeRawBufferPointer(start: src, count: scaleBytes))
                }
            }
        }
        offset += scaleBytes
        for (i, scale) in scales.enumerated() {
            guard scale.isFinite, scale >= 0 else {
                throw PersistenceError.corruptedData(
                    "Scalar-quantized index scale for node \(i) is invalid (\(scale))")
            }
        }

        // Codes
        let (codeBytes, codeOverflow) = nodeCount.multipliedReportingOverflow(by: dim)
        guard !codeOverflow else {
            throw PersistenceError.corruptedData("Scalar-quantized index code section truncated")
        }
        try requireBytes(codeBytes, "code section")
        var sqCodes = [[Int8]]()
        sqCodes.reserveCapacity(nodeCount)
        for _ in 0..<nodeCount {
            let code = fileData[offset..<offset + dim].map { Int8(bitPattern: $0) }
            sqCodes.append(code)
            offset += dim
        }

        // UUIDs
        let (uuidBytes, uuidOverflow) = nodeCount.multipliedReportingOverflow(by: 16)
        guard !uuidOverflow else {
            throw PersistenceError.corruptedData("Scalar-quantized index UUID section truncated")
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
                    "Scalar-quantized index node level \(level) outside valid range 0..<\(layerCount)")
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
                            "Scalar-quantized index neighbor \(n) outside valid range 0..<\(nodeCount)")
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
            throw PersistenceError.corruptedData("Scalar-quantized index metadata is not valid JSON")
        }
        guard decodedMetadata.count == nodeCount else {
            throw PersistenceError.corruptedData(
                "Scalar-quantized index metadata count \(decodedMetadata.count) != node count \(nodeCount)")
        }
        let metadata: [Data?] = decodedMetadata.map { $0.map { Data($0) } }

        let hnswConfig = HNSWConfiguration(
            m: hnswM,
            efConstruction: efConstruction,
            efSearch: efSearch,
            autoCompactionThreshold: autoCompactionThreshold
        )

        return ScalarQuantizedHNSWIndex(
            dimension: dim,
            hnswConfig: hnswConfig,
            metricType: metricType,
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPoint,
            maxLevel: maxLevel,
            codes: sqCodes,
            scales: scales,
            nodeToUUID: nodeToUUID,
            uuidToNode: uuidToNode,
            metadata: metadata
        )
    }
}

// MARK: - Binary Helpers (namespaced to avoid collisions)

private func sqAppendLE<T: FixedWidthInteger>(_ data: inout Data, _ value: T) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

// Uses Data.loadLE from PersistenceEngine.swift
