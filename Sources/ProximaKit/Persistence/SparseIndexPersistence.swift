// SparseIndexPersistence.swift
// ProximaKit
//
// Binary persistence for ``SparseIndex``. Mirrors the layout conventions of
// ``PersistenceEngine`` for HNSW / BruteForce: 64-byte fixed header, followed
// by fixed-size arrays, followed by variable-length sections, ending in a
// metadata block whose start offset is recorded in the header.
//
// File suffix: `.pxbm` (ProximaKit BM25).

import Foundation

// MARK: - Snapshot

/// A snapshot of a ``SparseIndex``'s state for binary persistence.
///
/// The index is expected to be compacted (no tombstones) before snapshotting;
/// `SparseIndex.persistenceSnapshot()` handles that automatically.
public struct SparseIndexSnapshot: Sendable {
    public let configuration: BM25Configuration
    public let tokenCounts: [[String: Int]?]
    public let docLengths: [Int]
    public let metadataStore: [Data?]
    public let nodeToUUID: [UUID]
    public let postings: [String: [(node: Int, tf: Int)]]
}

// MARK: - Constants

private let sparseMagic: UInt32 = 0x50584D42  // "PXBM"
private let sparseFormatVersion: UInt32 = 1
private let sparseHeaderSize = 64

// MARK: - PersistenceEngine Extension

extension PersistenceEngine {

    /// Saves a SparseIndex snapshot to a `.pxbm` binary file.
    public static func save(_ snapshot: SparseIndexSnapshot, to url: URL) throws {
        var data = Data()
        let count = snapshot.nodeToUUID.count

        // ── Header placeholder (backfilled after we know offsets) ─────
        data.append(Data(count: sparseHeaderSize))

        // ── UUIDs ─────────────────────────────────────────────────────
        for uuid in snapshot.nodeToUUID {
            sparseAppendUUID(&data, uuid)
        }

        // ── Doc lengths (UInt32 per slot) ─────────────────────────────
        for length in snapshot.docLengths {
            sparseAppendUInt32(&data, UInt32(length))
        }

        // ── Inverted index ────────────────────────────────────────────
        // Tokens are written in sorted order so that round-trips are
        // deterministic (helps hashing / golden-file tests).
        let postingsOffset = UInt32(data.count)

        let sortedTerms = snapshot.postings.keys.sorted()
        sparseAppendUInt32(&data, UInt32(sortedTerms.count))

        for term in sortedTerms {
            guard let entries = snapshot.postings[term] else { continue }
            let termBytes = Data(term.utf8)
            sparseAppendUInt32(&data, UInt32(termBytes.count))
            data.append(termBytes)
            sparseAppendUInt32(&data, UInt32(entries.count))
            for entry in entries {
                sparseAppendUInt32(&data, UInt32(entry.node))
                sparseAppendUInt32(&data, UInt32(entry.tf))
            }
        }

        // ── Metadata ──────────────────────────────────────────────────
        let metadataOffset = UInt32(data.count)
        for meta in snapshot.metadataStore {
            if let meta {
                sparseAppendUInt32(&data, UInt32(meta.count))
                data.append(meta)
            } else {
                sparseAppendUInt32(&data, 0)
            }
        }

        // ── Backfill header ───────────────────────────────────────────
        sparseWriteHeader(
            &data,
            count: UInt32(count),
            k1: Float(snapshot.configuration.k1),
            b: Float(snapshot.configuration.b),
            totalLiveTokens: UInt64(snapshot.docLengths.reduce(0, +)),
            postingsOffset: postingsOffset,
            metadataOffset: metadataOffset
        )

        try data.write(to: url, options: .atomic)
    }

    /// Loads a SparseIndex from a `.pxbm` binary file.
    public static func loadSparse(
        from url: URL,
        tokenizer: any BM25Tokenizer
    ) throws -> SparseIndex {
        let fileData = try Data(contentsOf: url, options: .mappedIfSafe)

        guard fileData.count >= sparseHeaderSize else {
            throw PersistenceError.fileTooSmall
        }

        let magic = fileData.loadLE(UInt32.self, at: 0)
        guard magic == sparseMagic else {
            throw PersistenceError.invalidMagic
        }

        let version = fileData.loadLE(UInt32.self, at: 4)
        guard version == sparseFormatVersion else {
            throw PersistenceError.unsupportedVersion(version)
        }

        let indexType = fileData.loadLE(UInt32.self, at: 8)
        guard indexType == 2 else {
            throw PersistenceError.unknownIndexType(indexType)
        }

        let count = Int(fileData.loadLE(UInt32.self, at: 12))
        let k1 = Double(Float(bitPattern: fileData.loadLE(UInt32.self, at: 16)))
        let b = Double(Float(bitPattern: fileData.loadLE(UInt32.self, at: 20)))
        // totalLiveTokens at [24, 32) is informational — we re-derive on load
        // from docLengths, so it stays authoritative even if a writer miscomputes.
        let postingsOffset = Int(fileData.loadLE(UInt32.self, at: 32))
        let metadataOffset = Int(fileData.loadLE(UInt32.self, at: 36))

        let configuration = BM25Configuration(k1: k1, b: b)

        var offset = sparseHeaderSize

        // ── UUIDs ─────────────────────────────────────────────────────
        let needUUIDs = count * 16
        guard offset + needUUIDs <= fileData.count else {
            throw PersistenceError.corruptedData("SparseIndex UUID section truncated")
        }
        var nodeToUUID: [UUID] = []
        nodeToUUID.reserveCapacity(count)
        for _ in 0..<count {
            let uuid = fileData.withUnsafeBytes { buffer in
                buffer.loadUnaligned(fromByteOffset: offset, as: uuid_t.self)
            }
            nodeToUUID.append(UUID(uuid: uuid))
            offset += 16
        }

        // ── Doc lengths ───────────────────────────────────────────────
        let needLengths = count * 4
        guard offset + needLengths <= fileData.count else {
            throw PersistenceError.corruptedData("SparseIndex docLengths section truncated")
        }
        var docLengths: [Int] = []
        docLengths.reserveCapacity(count)
        for _ in 0..<count {
            docLengths.append(Int(fileData.loadLE(UInt32.self, at: offset)))
            offset += 4
        }

        // ── Inverted index ────────────────────────────────────────────
        offset = postingsOffset
        guard offset + 4 <= fileData.count else {
            throw PersistenceError.corruptedData("SparseIndex postings header truncated")
        }
        let termCount = Int(fileData.loadLE(UInt32.self, at: offset))
        offset += 4

        var postings: [String: [(node: Int, tf: Int)]] = [:]
        postings.reserveCapacity(termCount)

        for _ in 0..<termCount {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("SparseIndex term header truncated")
            }
            let termLen = Int(fileData.loadLE(UInt32.self, at: offset))
            offset += 4
            guard offset + termLen <= fileData.count else {
                throw PersistenceError.corruptedData("SparseIndex term bytes truncated")
            }
            let termBytes = fileData.subdata(in: offset..<(offset + termLen))
            guard let term = String(data: termBytes, encoding: .utf8) else {
                throw PersistenceError.corruptedData("SparseIndex term is not valid UTF-8")
            }
            offset += termLen

            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("SparseIndex postings count truncated")
            }
            let postingCount = Int(fileData.loadLE(UInt32.self, at: offset))
            offset += 4

            let postingBytes = postingCount * 8
            guard offset + postingBytes <= fileData.count else {
                throw PersistenceError.corruptedData("SparseIndex postings payload truncated")
            }

            var entries: [(node: Int, tf: Int)] = []
            entries.reserveCapacity(postingCount)
            for _ in 0..<postingCount {
                let node = Int(fileData.loadLE(UInt32.self, at: offset))
                let tf = Int(fileData.loadLE(UInt32.self, at: offset + 4))
                entries.append((node: node, tf: tf))
                offset += 8
            }
            postings[term] = entries
        }

        // ── Rebuild per-doc token counts from postings ────────────────
        // (The postings are the source of truth; this avoids persisting a
        // separate per-doc dictionary that would duplicate the data.)
        var tokenCounts: [[String: Int]?] = Array(repeating: [:], count: count)
        for (term, entries) in postings {
            for entry in entries {
                guard entry.node < count else {
                    throw PersistenceError.corruptedData("SparseIndex posting references invalid node")
                }
                if tokenCounts[entry.node] == nil {
                    tokenCounts[entry.node] = [:]
                }
                tokenCounts[entry.node]?[term] = entry.tf
            }
        }

        // ── Metadata ──────────────────────────────────────────────────
        offset = metadataOffset
        var metadataStore: [Data?] = []
        metadataStore.reserveCapacity(count)
        for _ in 0..<count {
            guard offset + 4 <= fileData.count else {
                throw PersistenceError.corruptedData("SparseIndex metadata length truncated")
            }
            let length = Int(fileData.loadLE(UInt32.self, at: offset))
            offset += 4
            if length == 0 {
                metadataStore.append(nil)
            } else {
                guard offset + length <= fileData.count else {
                    throw PersistenceError.corruptedData("SparseIndex metadata bytes truncated")
                }
                metadataStore.append(fileData.subdata(in: offset..<(offset + length)))
                offset += length
            }
        }

        let snapshot = SparseIndexSnapshot(
            configuration: configuration,
            tokenCounts: tokenCounts,
            docLengths: docLengths,
            metadataStore: metadataStore,
            nodeToUUID: nodeToUUID,
            postings: postings
        )
        return SparseIndex(restoring: snapshot, tokenizer: tokenizer)
    }
}

// MARK: - Header Writer

private func sparseWriteHeader(
    _ data: inout Data,
    count: UInt32,
    k1: Float,
    b: Float,
    totalLiveTokens: UInt64,
    postingsOffset: UInt32,
    metadataOffset: UInt32
) {
    sparseWriteUInt32(&data, sparseMagic, at: 0)
    sparseWriteUInt32(&data, sparseFormatVersion, at: 4)
    sparseWriteUInt32(&data, 2, at: 8)                         // indexType = sparse
    sparseWriteUInt32(&data, count, at: 12)
    sparseWriteUInt32(&data, k1.bitPattern, at: 16)
    sparseWriteUInt32(&data, b.bitPattern, at: 20)
    sparseWriteUInt64(&data, totalLiveTokens, at: 24)
    sparseWriteUInt32(&data, postingsOffset, at: 32)
    sparseWriteUInt32(&data, metadataOffset, at: 36)
    // Bytes 40..64 remain zero (reserved).
}

// MARK: - Little-Endian Write Helpers

private func sparseAppendUInt32(_ data: inout Data, _ value: UInt32) {
    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
}

private func sparseAppendUUID(_ data: inout Data, _ uuid: UUID) {
    withUnsafeBytes(of: uuid.uuid) { data.append(contentsOf: $0) }
}

private func sparseWriteUInt32(_ data: inout Data, _ value: UInt32, at position: Int) {
    withUnsafeBytes(of: value.littleEndian) { bytes in
        for (i, byte) in bytes.enumerated() {
            data[position + i] = byte
        }
    }
}

private func sparseWriteUInt64(_ data: inout Data, _ value: UInt64, at position: Int) {
    withUnsafeBytes(of: value.littleEndian) { bytes in
        for (i, byte) in bytes.enumerated() {
            data[position + i] = byte
        }
    }
}
