import XCTest
@testable import ProximaKit

final class HNSWGraphSnapshotTests: XCTestCase {

    private func fixedID(_ value: UInt8) -> UUID {
        UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, value))
    }

    private func randomVector(dim: Int, rng: inout SeededRandom) -> Vector {
        Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
    }

    private func makeIndex(
        dimension: Int = 4,
        seed: UInt64 = 0x6A11_5EED
    ) -> HNSWIndex {
        let config = HNSWConfiguration(
            m: 4,
            efConstruction: 48,
            efSearch: 32,
            autoCompactionThreshold: nil,
            levelSeed: seed
        )
        return HNSWIndex(dimension: dimension, metric: EuclideanDistance(), config: config)
    }

    func testLiveGraphSnapshotDoesNotCompactOrMutateTombstonedIndex() async throws {
        let index = makeIndex(dimension: 2, seed: 0x6A11_5EED_0001)
        let ids = (0..<8).map { fixedID(UInt8($0 + 1)) }
        for (i, id) in ids.enumerated() {
            try await index.add(Vector([Float(i), 0]), id: id, metadata: Data("node-\(i)".utf8))
        }
        for i in [1, 3, 6] {
            await index.remove(id: ids[i])
        }

        let beforeCount = await index.count
        let beforeLiveCount = await index.liveCount
        let beforeFingerprint = await index.structuralFingerprint

        let snapshot = await index.liveGraphSnapshot()

        let afterCount = await index.count
        let afterLiveCount = await index.liveCount
        let afterFingerprint = await index.structuralFingerprint

        XCTAssertEqual(snapshot.liveCount, 5)
        XCTAssertEqual(beforeCount, afterCount)
        XCTAssertEqual(beforeLiveCount, afterLiveCount)
        XCTAssertEqual(beforeFingerprint, afterFingerprint)

        let contractLine = "before count=\(beforeCount) live=\(beforeLiveCount); " +
            "snapshot live=\(snapshot.liveCount); after count=\(afterCount) live=\(afterLiveCount)"
        XCTAssertEqual(
            contractLine,
            "before count=8 live=5; snapshot live=5; after count=8 live=5"
        )
    }

    func testLiveGraphSnapshotMatchesLiveEntriesLevelsAndLiveNeighborIDs() async throws {
        let index = makeIndex(seed: 0x6A11_5EED_0002)
        let ids = (0..<24).map { fixedID(UInt8($0 + 1)) }
        var rng = SeededRandom(seed: 0x6A11_5EED_1002)
        for id in ids {
            let metadata = Data("meta-\(id.uuidString.suffix(2))".utf8)
            try await index.add(randomVector(dim: 4, rng: &rng), id: id, metadata: metadata)
        }
        for i in [2, 5, 11, 17] {
            await index.remove(id: ids[i])
        }

        let snapshot = await index.liveGraphSnapshot()
        let liveEntries = await index.liveEntries()
        let fingerprint = await index.structuralFingerprint

        XCTAssertEqual(snapshot.liveCount, liveEntries.count)
        XCTAssertEqual(snapshot.nodes.map(\.id), liveEntries.map(\.id))
        XCTAssertEqual(snapshot.nodes.map(\.metadata), liveEntries.map(\.metadata))

        let expectedLevelsByID = Dictionary(uniqueKeysWithValues: fingerprint.nodeToUUID.enumerated().compactMap { node, id -> (UUID, Int)? in
            fingerprint.uuidToNode[id] == node ? (id, fingerprint.nodeLevels[node]) : nil
        })
        let liveIDs = Set(liveEntries.map(\.id))

        for node in snapshot.nodes {
            XCTAssertEqual(node.level, expectedLevelsByID[node.id])
            XCTAssertTrue(
                node.layer0NeighborIDs.allSatisfy { liveIDs.contains($0) },
                "snapshot layer-0 neighbors must be live UUIDs"
            )
        }

        let expectedNodesPerLayer = (0...max(snapshot.maxLevel, 0)).map { layer in
            snapshot.nodes.filter { $0.level >= layer }.count
        }
        XCTAssertEqual(snapshot.nodesPerLayer, expectedNodesPerLayer)
        XCTAssertEqual(snapshot.maxLevel, snapshot.nodes.map(\.level).max() ?? -1)
    }

    func testLiveGraphSnapshotFiltersTombstonedLayer0Neighbors() async throws {
        let index = makeIndex(dimension: 2, seed: 0x6A11_5EED_0003)
        let ids = (0..<6).map { fixedID(UInt8($0 + 1)) }
        for (i, id) in ids.enumerated() {
            try await index.add(Vector([Float(i), 0]), id: id)
        }

        await index.remove(id: ids[2])
        let inserted = await index.insertLayer0EdgeForTesting(from: ids[1], toStoredID: ids[2])
        XCTAssertTrue(inserted)

        let dangling = await index.hasDanglingEdges
        XCTAssertTrue(dangling, "fixture must contain a raw edge to a tombstoned slot")

        let snapshot = await index.liveGraphSnapshot()
        let source = try XCTUnwrap(snapshot.nodes.first { $0.id == ids[1] })
        XCTAssertFalse(source.layer0NeighborIDs.contains(ids[2]))
        XCTAssertFalse(snapshot.nodes.contains { $0.id == ids[2] })
    }

    func testLiveGraphSnapshotEmptyIndex() async {
        let index = makeIndex(dimension: 3, seed: 0x6A11_5EED_0004)

        let snapshot = await index.liveGraphSnapshot()

        XCTAssertEqual(snapshot.liveCount, 0)
        XCTAssertEqual(snapshot.maxLevel, -1)
        XCTAssertTrue(snapshot.nodes.isEmpty)
        XCTAssertEqual(snapshot.nodesPerLayer, [])
    }
}
