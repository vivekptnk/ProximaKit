// CustomRAGWrapperRecipeTests.swift
// ProximaKitTests
//
// A compiled consumer recipe for building a custom RAG wrapper directly on
// raw HNSWIndex while keeping chunk metadata crash-consistent with vectors.

import Foundation
import XCTest
@testable import ProximaKit

/// THE copyable template: a tiny RAG wrapper that owns raw ``HNSWIndex`` and
/// stores each chunk record in the index's per-vector metadata slot.
///
/// Strategy A is the important bit: ``ChunkRecord`` is JSON-encoded into the
/// `metadata` argument of ``HNSWIndex/add(_:id:metadata:)``. That same metadata
/// is persisted in `.pxkt` snapshots and included in WAL `add` records, so a
/// recovered index can derive its chunk catalog from the vectors themselves.
/// There is no sidecar to keep in sync.
///
/// For journaled use, create a base once with ``checkpoint()``, then keep
/// adding chunks. Reopen with ``openJournaled(baseURL:walURL:durability:)``;
/// it loads the base, replays the WAL to its longest valid prefix, and attaches
/// the journal for future appends.
///
/// Do not blindly retry a failed journaled mutation or failed automatic
/// checkpoint wrapper built on this pattern. At the index level the vector may
/// already be present in memory or durable in the WAL; retrying with a fresh
/// UUID can duplicate the chunk. Reopen/reconcile from recovered metadata, or
/// retry only with an idempotency policy you own.
actor RecipeRAGIndex {
    struct ChunkRecord: Codable, Equatable, Hashable, Sendable {
        let sourcePath: String
        let byteOffset: Int64
        let text: String
    }

    private let index: HNSWIndex
    private let baseURL: URL?
    private let walURL: URL?
    private let durability: WALDurability
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(
        dimension: Int,
        metric: any DistanceMetric = EuclideanDistance(),
        config: HNSWConfiguration = HNSWConfiguration(),
        baseURL: URL? = nil,
        walURL: URL? = nil,
        durability: WALDurability = .everyRecord
    ) {
        self.index = HNSWIndex(dimension: dimension, metric: metric, config: config)
        self.baseURL = baseURL
        self.walURL = walURL
        self.durability = durability
    }

    private init(index: HNSWIndex, baseURL: URL?, walURL: URL?, durability: WALDurability) {
        self.index = index
        self.baseURL = baseURL
        self.walURL = walURL
        self.durability = durability
    }

    static func load(from url: URL) throws -> RecipeRAGIndex {
        RecipeRAGIndex(
            index: try HNSWIndex.load(from: url),
            baseURL: nil,
            walURL: nil,
            durability: .everyRecord
        )
    }

    static func openJournaled(
        baseURL: URL,
        walURL: URL,
        durability: WALDurability = .everyRecord
    ) async throws -> RecipeRAGIndex {
        let index = try await HNSWIndex.open(
            baseURL: baseURL,
            walURL: walURL,
            durability: durability
        )
        return RecipeRAGIndex(
            index: index,
            baseURL: baseURL,
            walURL: walURL,
            durability: durability
        )
    }

    @discardableResult
    func addChunk(
        text: String,
        sourcePath: String,
        offset: Int64,
        embedding: Vector,
        id: UUID = UUID()
    ) async throws -> UUID {
        let record = ChunkRecord(sourcePath: sourcePath, byteOffset: offset, text: text)
        try await index.add(embedding, id: id, metadata: encoder.encode(record))
        return id
    }

    func search(query: Vector, k: Int) async throws -> [(ChunkRecord, Float)] {
        let results = await index.search(query: query, k: k)
        return try results.map { result in
            guard let metadata = result.metadata else {
                throw RecipeRAGIndexError.missingChunkMetadata(result.id)
            }
            return (try decoder.decode(ChunkRecord.self, from: metadata), result.distance)
        }
    }

    func save(to url: URL) async throws {
        try await index.save(to: url)
    }

    func checkpoint() async throws {
        guard let baseURL, let walURL else {
            throw RecipeRAGIndexError.missingJournalPaths
        }
        try await index.checkpoint(baseURL: baseURL, walURL: walURL, durability: durability)
    }
}

private enum RecipeRAGIndexError: Error, Equatable {
    case missingChunkMetadata(UUID)
    case missingJournalPaths
}

/*
 STRATEGY B sketch: keep an owned sidecar keyed by vector UUID.

 actor SidecarRAGIndex {
     private let index: HNSWIndex
     private var chunksByID: [UUID: RecipeRAGIndex.ChunkRecord]

     func addChunk(...) async throws -> UUID {
         let id = UUID()
         try await index.add(embedding, id: id)
         chunksByID[id] = ChunkRecord(sourcePath: path, byteOffset: offset, text: text)
         try saveSidecar(chunksByID)
         return id
     }
 }

 This is tinybrain's current shape when it owns richer sidecar state, but it
 also owns the crash-consistency obligation: the HNSW WAL can replay an id that
 the sidecar write never recorded, or a sidecar can contain an id that never
 reached the index. ProximaKit's store layer avoids that split brain by
 deriving sidecars from HNSWIndex.liveEntries() after WAL replay. Strategy A
 gets that property directly because the chunk record is the vector metadata.
 */

final class CustomRAGWrapperRecipeTests: XCTestCase {
    private let dimension = 8

    func testSnapshotRoundTripReturnsIdenticalChunkRecords() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }
        let snapshot = dir.appendingPathComponent("rag.pxkt")
        let chunks = makeChunks(count: 5)

        let rag = RecipeRAGIndex(dimension: dimension, config: seededConfig())
        try await add(chunks, to: rag)
        try await rag.save(to: snapshot)

        let loaded = try RecipeRAGIndex.load(from: snapshot)
        let results = try await loaded.search(query: chunks[0].embedding, k: chunks.count)

        XCTAssertEqual(results.first?.0, chunks[0].record)
        XCTAssertEqual(Set(results.map(\.0)), Set(chunks.map(\.record)))
    }

    func testJournaledRecoveryReplaysChunkMetadataFromWAL() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }
        let base = dir.appendingPathComponent("rag.pxkt")
        let wal = dir.appendingPathComponent("rag.pxwal")
        let chunks = makeChunks(count: 6)

        do {
            let rag = RecipeRAGIndex(
                dimension: dimension,
                config: seededConfig(),
                baseURL: base,
                walURL: wal,
                durability: .everyRecord
            )
            try await rag.checkpoint()
            try await add(chunks, to: rag)
        }

        let recovered = try await RecipeRAGIndex.openJournaled(
            baseURL: base,
            walURL: wal,
            durability: .everyRecord
        )
        let results = try await recovered.search(query: chunks[0].embedding, k: chunks.count)

        XCTAssertEqual(results.first?.0, chunks[0].record)
        XCTAssertEqual(Set(results.map(\.0)), Set(chunks.map(\.record)))
    }

    func testCheckpointThenReopenReturnsChunkRecordsFromBase() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }
        let base = dir.appendingPathComponent("rag.pxkt")
        let wal = dir.appendingPathComponent("rag.pxwal")
        let chunks = makeChunks(count: 6)

        do {
            let rag = RecipeRAGIndex(
                dimension: dimension,
                config: seededConfig(),
                baseURL: base,
                walURL: wal,
                durability: .everyRecord
            )
            try await rag.checkpoint()
            try await add(chunks, to: rag)
            try await rag.checkpoint()
        }

        let reopened = try await RecipeRAGIndex.openJournaled(
            baseURL: base,
            walURL: wal,
            durability: .everyRecord
        )
        let results = try await reopened.search(query: chunks[0].embedding, k: chunks.count)

        XCTAssertEqual(results.first?.0, chunks[0].record)
        XCTAssertEqual(Set(results.map(\.0)), Set(chunks.map(\.record)))
    }

    func testTornWALTailRecoversLongestValidChunkMetadataPrefix() async throws {
        let dir = tempDir()
        defer { cleanup(dir) }
        let base = dir.appendingPathComponent("rag.pxkt")
        let wal = dir.appendingPathComponent("rag.pxwal")
        let chunks = makeChunks(count: 6)

        do {
            let rag = RecipeRAGIndex(
                dimension: dimension,
                config: seededConfig(),
                baseURL: base,
                walURL: wal,
                durability: .everyRecord
            )
            try await rag.checkpoint()
            try await add(chunks, to: rag)
        }

        let fullWAL = try Data(contentsOf: wal)
        let ends = walRecordEndOffsets(in: fullWAL)
        XCTAssertEqual(ends.count, chunks.count + 1)

        let validPrefixCount = 3
        let cut = ends[validPrefixCount] + 12
        XCTAssertLessThan(cut, ends[validPrefixCount + 1])
        try fullWAL.prefix(cut).write(to: wal)

        let recovered: RecipeRAGIndex
        do {
            recovered = try await RecipeRAGIndex.openJournaled(
                baseURL: base,
                walURL: wal,
                durability: .everyRecord
            )
        } catch {
            XCTFail("mid-record WAL truncation should recover a prefix, got \(error)")
            return
        }

        let expected = Array(chunks.prefix(validPrefixCount))
        let results = try await recovered.search(query: chunks[0].embedding, k: chunks.count)
        XCTAssertEqual(results.first?.0, expected[0].record)
        XCTAssertEqual(Set(results.map(\.0)), Set(expected.map(\.record)))

        let stub = dir.appendingPathComponent("stub.pxwal")
        try fullWAL.prefix(WALFormat.headerSize - 1).write(to: stub)
        do {
            _ = try await RecipeRAGIndex.openJournaled(
                baseURL: base,
                walURL: stub,
                durability: .everyRecord
            )
            XCTFail("sub-header WAL truncation should throw a typed persistence error")
        } catch PersistenceError.fileTooSmall {
        } catch {
            XCTFail("expected typed PersistenceError.fileTooSmall, got \(error)")
        }
    }

    private struct FixtureChunk {
        let id: UUID
        let record: RecipeRAGIndex.ChunkRecord
        let embedding: Vector
    }

    private func add(_ chunks: [FixtureChunk], to rag: RecipeRAGIndex) async throws {
        for chunk in chunks {
            try await rag.addChunk(
                text: chunk.record.text,
                sourcePath: chunk.record.sourcePath,
                offset: chunk.record.byteOffset,
                embedding: chunk.embedding,
                id: chunk.id
            )
        }
    }

    private func makeChunks(count: Int) -> [FixtureChunk] {
        (0..<count).map { index in
            FixtureChunk(
                id: uuid(index),
                record: RecipeRAGIndex.ChunkRecord(
                    sourcePath: "Fixtures/Docs/doc-\(index / 2).md",
                    byteOffset: Int64(index * 137),
                    text: "seeded chunk \(index)"
                ),
                embedding: vector(index)
            )
        }
    }

    private func vector(_ index: Int) -> Vector {
        var generator = SplitMix64(seed: 0xC0FF_EE00 &+ UInt64(index))
        return Vector((0..<dimension).map { _ in
            Float(UInt32(truncatingIfNeeded: generator.next()) % 10_000) / 10_000.0
        })
    }

    private func seededConfig() -> HNSWConfiguration {
        HNSWConfiguration(
            m: 6,
            efConstruction: 40,
            efSearch: 20,
            autoCompactionThreshold: nil,
            levelSeed: 0x5EED
        )
    }

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("rag-recipe-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func cleanup(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func uuid(_ index: Int) -> UUID {
        var bytes = (
            UInt8(0), UInt8(0), UInt8(0), UInt8(0),
            UInt8(0), UInt8(0), UInt8(0), UInt8(0),
            UInt8(0), UInt8(0), UInt8(0), UInt8(0),
            UInt8(0), UInt8(0), UInt8(0), UInt8(0)
        )
        withUnsafeMutableBytes(of: &bytes) {
            $0.storeBytes(of: UInt64(index).littleEndian, as: UInt64.self)
        }
        return UUID(uuid: bytes)
    }

    private func walRecordEndOffsets(in data: Data) -> [Int] {
        var ends = [WALFormat.headerSize]
        var offset = WALFormat.headerSize
        while offset + 8 <= data.count {
            let payloadLength = Int(littleEndianUInt32(in: data, at: offset))
            let end = offset + 8 + payloadLength
            guard payloadLength > 0, end <= data.count else { break }
            ends.append(end)
            offset = end
        }
        return ends
    }

    private func littleEndianUInt32(in data: Data, at offset: Int) -> UInt32 {
        var value: UInt32 = 0
        for byteOffset in 0..<4 {
            value |= UInt32(data[offset + byteOffset]) << UInt32(byteOffset * 8)
        }
        return value
    }
}
