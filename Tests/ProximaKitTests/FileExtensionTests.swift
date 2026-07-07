// FileExtensionTests.swift
// ProximaKitTests
//
// Pins the public persistence file-extension constants and verifies the store
// paths that dogfood them.

import XCTest
@testable import ProximaKit

private struct FileExtensionTestEmbedder: TextEmbedder {
    let dimension: Int

    func embed(_ text: String) async throws -> Vector {
        var hasher = Hasher()
        hasher.combine(text)
        let hash = abs(hasher.finalize())
        let base = Float(hash % 1000) / 1000.0
        return Vector((0..<dimension).map { base + Float($0) * 0.001 })
    }
}

final class FileExtensionTests: XCTestCase {

    private let dimension = 8
    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-FileExtensions-\(UUID().uuidString)")
    }

    override func tearDown() {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        super.tearDown()
    }

    func testPublicFileExtensionConstantsArePinned() {
        // These strings are public API contracts; changing one is breaking.
        XCTAssertEqual(ProximaKit.FileExtension.index, "pxkt")
        XCTAssertEqual(ProximaKit.FileExtension.writeAheadLog, "pxwal")
        XCTAssertEqual(ProximaKit.FileExtension.sparseIndex, "pxbm")
    }

    func testStoreWrittenFilesUsePublicExtensions() async throws {
        let embedder = FileExtensionTestEmbedder(dimension: dimension)
        let config = HNSWConfiguration(m: 4, efConstruction: 20, efSearch: 20)

        let vectorStore = try VectorStore(
            name: "vector",
            embedder: embedder,
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: config
        )
        try await vectorStore.addChunks(
            ["dense document"],
            metadata: [ChunkMetadata(documentId: "dense", chunkIndex: 0, text: "dense document")]
        )
        try await vectorStore.save()

        let journaledStore = try await VectorStore.open(
            name: "journaled",
            embedder: embedder,
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            config: config
        )
        try await journaledStore.addChunks(
            ["journaled document"],
            metadata: [
                ChunkMetadata(
                    documentId: "journaled",
                    chunkIndex: 0,
                    text: "journaled document"
                ),
            ]
        )
        try await journaledStore.save()

        let hybridStore = try HybridVectorStore(
            name: "hybrid",
            embedder: embedder,
            storageDirectory: tempDir,
            metric: EuclideanDistance(),
            hnswConfig: config
        )
        try await hybridStore.addChunks(
            ["sparse document"],
            metadata: [ChunkMetadata(documentId: "sparse", chunkIndex: 0, text: "sparse document")]
        )
        try await hybridStore.save()

        assertFileExists(named: "index", fileExtension: ProximaKit.FileExtension.index, in: "vector")
        assertFileExists(named: "index", fileExtension: ProximaKit.FileExtension.index, in: "journaled")
        assertFileExists(
            named: "index",
            fileExtension: ProximaKit.FileExtension.writeAheadLog,
            in: "journaled"
        )
        assertFileExists(named: "index", fileExtension: ProximaKit.FileExtension.index, in: "hybrid")
        assertFileExists(named: "index", fileExtension: ProximaKit.FileExtension.sparseIndex, in: "hybrid")
    }

    private func assertFileExists(
        named basename: String,
        fileExtension: String,
        in storeName: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let url = tempDir.appendingPathComponent(storeName)
            .appendingPathComponent(basename)
            .appendingPathExtension(fileExtension)
        XCTAssertEqual(url.pathExtension, fileExtension, file: file, line: line)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), file: file, line: line)
    }
}
