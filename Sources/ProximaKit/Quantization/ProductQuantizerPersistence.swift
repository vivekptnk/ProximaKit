// ProductQuantizerPersistence.swift
// ProximaKit
//
// Binary persistence for trained product quantizers.
//
// Format: 16-byte header + codebook data.
//   Magic:         0x50515454 ("PQTT")
//   Version:       1
//   Dimension:     UInt32
//   SubspaceCount: UInt32
//   Centroids/Sub: UInt32 (always 256)
//   TrainIters:    UInt32
//   Codebooks:     M * K * ds Float32 values (contiguous)

import Foundation

private let pqMagic: UInt32 = 0x50515454       // "PQTT"
private let pqFormatVersion: UInt32 = 1
private let pqHeaderSize = 24

extension ProductQuantizer {

    /// Saves this trained quantizer to a binary file.
    public func save(to url: URL) throws {
        var data = Data()

        let K = config.centroidsPerSubspace
        let M = config.subspaceCount
        let ds = subspaceDimension
        let codebookBytes = M * K * ds * 4

        data.reserveCapacity(pqHeaderSize + codebookBytes)

        // Header
        appendLE(&data, pqMagic)
        appendLE(&data, pqFormatVersion)
        appendLE(&data, UInt32(dimension))
        appendLE(&data, UInt32(M))
        appendLE(&data, UInt32(K))
        appendLE(&data, UInt32(config.trainingIterations))

        assert(data.count == pqHeaderSize)

        // Codebooks: M codebooks, each K * ds floats
        for cb in codebooks {
            cb.withUnsafeBytes { buffer in
                data.append(contentsOf: buffer)
            }
        }

        try data.write(to: url, options: .atomic)
    }

    /// Loads a trained quantizer from a binary file.
    public static func load(from url: URL) throws -> ProductQuantizer {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        guard fileData.count >= pqHeaderSize else {
            throw PersistenceError.fileTooSmall
        }

        let fileMagic = fileData.loadLE(UInt32.self, at: 0)
        guard fileMagic == pqMagic else {
            throw PersistenceError.invalidMagic
        }

        let version = fileData.loadLE(UInt32.self, at: 4)
        guard version == pqFormatVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }

        let dimension = Int(fileData.loadLE(UInt32.self, at: 8))
        let M = Int(fileData.loadLE(UInt32.self, at: 12))
        let K = Int(fileData.loadLE(UInt32.self, at: 16))
        let trainIters = Int(fileData.loadLE(UInt32.self, at: 20))

        let ds = dimension / M
        let expectedSize = pqHeaderSize + M * K * ds * 4
        guard fileData.count >= expectedSize else {
            throw PersistenceError.corruptedData("PQ file truncated: expected \(expectedSize) bytes, got \(fileData.count)")
        }

        var codebooks = [[Float]]()
        codebooks.reserveCapacity(M)

        var offset = pqHeaderSize
        let codebookFloatCount = K * ds

        for _ in 0..<M {
            let byteCount = codebookFloatCount * 4
            var floats = [Float](repeating: 0, count: codebookFloatCount)
            fileData.withUnsafeBytes { buffer in
                let src = buffer.baseAddress!.advanced(by: offset)
                floats.withUnsafeMutableBytes { dest in
                    dest.copyMemory(from: UnsafeRawBufferPointer(start: src, count: byteCount))
                }
            }
            codebooks.append(floats)
            offset += byteCount
        }

        let config = PQConfiguration(
            subspaceCount: M,
            trainingIterations: trainIters
        )

        return ProductQuantizer(
            dimension: dimension,
            config: config,
            codebooks: codebooks
        )
    }
}

// MARK: - Binary Write Helpers

private func appendLE(_ data: inout Data, _ value: UInt32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}
