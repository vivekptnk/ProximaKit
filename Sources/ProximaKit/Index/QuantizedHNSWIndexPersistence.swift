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

    /// Saves this quantized index to a binary file.
    public func save(to url: URL) throws {
        var data = Data()

        let nodeCount = codes.count
        let M = quantizer.config.subspaceCount
        let ds = quantizer.subspaceDimension

        // Header (56 bytes)
        qhAppendLE(&data, qhMagic)
        qhAppendLE(&data, qhFormatVersion)
        qhAppendLE(&data, UInt32(dimension))
        qhAppendLE(&data, UInt32(nodeCount))
        qhAppendLE(&data, UInt32(M))
        qhAppendLE(&data, UInt32(hnswConfig.m))
        qhAppendLE(&data, UInt32(hnswConfig.efConstruction))
        qhAppendLE(&data, UInt32(hnswConfig.efSearch))
        qhAppendLE(&data, Int32(maxLevel))
        qhAppendLE(&data, Int32(entryPointNode ?? -1))
        qhAppendLE(&data, UInt32(layers.count))
        qhAppendLE(&data, UInt32(quantizer.config.trainingIterations))
        // Reserved
        data.append(Data(repeating: 0, count: 8))

        assert(data.count == qhHeaderSize)

        // PQ codebooks: M codebooks, each 256 * ds floats
        for cb in quantizer.codebooks {
            cb.withUnsafeBytes { data.append(contentsOf: $0) }
        }

        // PQ codes: nodeCount * M bytes
        for code in codes {
            data.append(contentsOf: code)
        }

        // UUIDs: nodeCount * 16 bytes
        for uuid in nodeToUUID {
            let (u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15) = uuid.uuid
            data.append(contentsOf: [u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15])
        }

        // Node levels: nodeCount * Int32
        for level in nodeLevels {
            qhAppendLE(&data, Int32(level))
        }

        // Graph: for each layer, for each node, write neighbor count + neighbors
        for layer in layers {
            for neighbors in layer {
                qhAppendLE(&data, UInt32(neighbors.count))
                for n in neighbors {
                    qhAppendLE(&data, UInt32(n))
                }
            }
        }

        // Metadata: JSON-encoded
        let metadataPayload = try JSONEncoder().encode(metadata.map { $0.map { Array($0) } })
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

        func readUInt32() -> UInt32 {
            let val = fileData.qhLoadLE(UInt32.self, at: offset)
            offset += 4
            return val
        }

        func readInt32() -> Int32 {
            let val = fileData.qhLoadLE(Int32.self, at: offset)
            offset += 4
            return val
        }

        let fileMagic = readUInt32()
        guard fileMagic == qhMagic else {
            throw PersistenceError.invalidMagic
        }

        let version = readUInt32()
        guard version == qhFormatVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }

        let dim = Int(readUInt32())
        let nodeCount = Int(readUInt32())
        let subspaceCount = Int(readUInt32())
        let hnswM = Int(readUInt32())
        let efConstruction = Int(readUInt32())
        let efSearch = Int(readUInt32())
        let maxLevel = Int(readInt32())
        let epRaw = Int(readInt32())
        let entryPoint: Int? = epRaw >= 0 ? epRaw : nil
        let layerCount = Int(readUInt32())
        let trainIters = Int(readUInt32())

        // Skip reserved
        offset += 8
        assert(offset == qhHeaderSize)

        let ds = dim / subspaceCount
        let K = 256

        // PQ codebooks
        var codebooks = [[Float]]()
        codebooks.reserveCapacity(subspaceCount)
        for _ in 0..<subspaceCount {
            let floatCount = K * ds
            let byteCount = floatCount * 4
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
        var pqCodes = [[UInt8]]()
        pqCodes.reserveCapacity(nodeCount)
        for _ in 0..<nodeCount {
            let code = Array(fileData[offset..<offset + subspaceCount])
            pqCodes.append(code)
            offset += subspaceCount
        }

        // UUIDs
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
            nodeLevels.append(Int(readInt32()))
        }

        // Graph layers
        var layers = [[[Int]]]()
        layers.reserveCapacity(layerCount)
        for _ in 0..<layerCount {
            var layer = [[Int]]()
            layer.reserveCapacity(nodeCount)
            for _ in 0..<nodeCount {
                let neighborCount = Int(readUInt32())
                var neighbors = [Int]()
                neighbors.reserveCapacity(neighborCount)
                for _ in 0..<neighborCount {
                    neighbors.append(Int(readUInt32()))
                }
                layer.append(neighbors)
            }
            layers.append(layer)
        }

        // Metadata
        let metadataSize = Int(readUInt32())
        let metadataPayload = fileData[offset..<offset + metadataSize]
        let decodedMetadata = try JSONDecoder().decode([[UInt8]?].self, from: metadataPayload)
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

extension Data {
    fileprivate func qhLoadLE<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
        self.withUnsafeBytes { buffer in
            var value: T = 0
            withUnsafeMutableBytes(of: &value) { dest in
                dest.copyMemory(from: UnsafeRawBufferPointer(
                    start: buffer.baseAddress!.advanced(by: offset),
                    count: MemoryLayout<T>.size
                ))
            }
            return T(littleEndian: value)
        }
    }
}
