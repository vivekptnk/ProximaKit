// PQDeterminismTests.swift
// ProximaKitTests
//
// Determinism tests for seeded PQ k-means training (PQConfiguration.seed).
//
// Before the seed existed, ProductQuantizer.train drew its centroid
// initialization from the system RNG, so trained codebooks — and every
// PQ recall number derived from them — varied run to run. These tests
// pin the seeded contract: same seed + same vectors → byte-identical
// codebooks and codes; the seed is a training-time knob that is neither
// persisted by the PQTT codec (ADR-011) nor encoded by Codable.

import Foundation
import XCTest
@testable import ProximaKit

final class PQDeterminismTests: XCTestCase {

    // ── Helpers ──────────────────────────────────────────────────────

    /// Deterministic training data, so assertions compare training runs
    /// rather than datasets (see SeededRandom.swift).
    private func seededVectors(count: Int, dimension: Int, dataSeed: UInt64) -> [Float] {
        var rng = SeededRandom(seed: dataSeed)
        return (0..<count * dimension).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    /// Flattens all codebooks to raw bytes for byte-identity assertions.
    /// (Stricter than `[Float]` equality: distinguishes -0.0 from +0.0.)
    private func codebookBytes(_ pq: ProductQuantizer) -> Data {
        var data = Data()
        for cb in pq.codebooks {
            cb.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        return data
    }

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("pqtt")
    }

    // ── Configuration Surface ────────────────────────────────────────

    func testSeedDefaultsToNil() {
        // The pre-seed call shape must keep its pre-seed behavior.
        let config = PQConfiguration(subspaceCount: 8)
        XCTAssertNil(config.seed)
        XCTAssertEqual(config.centroidsPerSubspace, 256)
        XCTAssertEqual(config.trainingIterations, 25)
    }

    func testSeedParticipatesInEquatable() {
        XCTAssertEqual(
            PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 7),
            PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 7)
        )
        XCTAssertNotEqual(
            PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 7),
            PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 8)
        )
        XCTAssertNotEqual(
            PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 7),
            PQConfiguration(subspaceCount: 8, trainingIterations: 10)
        )
    }

    // ── Same Seed → Byte-Identical Training ──────────────────────────

    func testSameSeedSameVectorsYieldsByteIdenticalCodebooksAndCodes() throws {
        let dim = 32
        let M = 8
        let n = 400
        let vectors = seededVectors(count: n, dimension: dim, dataSeed: 0xDA7A_0001)
        let config = PQConfiguration(subspaceCount: M, trainingIterations: 10, seed: 42)

        let pqA = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim, config: config
        )
        let pqB = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim, config: config
        )

        XCTAssertEqual(
            codebookBytes(pqA), codebookBytes(pqB),
            "Same seed + same vectors must yield byte-identical codebooks"
        )

        // Codes follow from codebooks, but assert the user-visible artifact
        // independently across every training vector.
        for i in 0..<n {
            let v = Array(vectors[i * dim..<(i + 1) * dim])
            XCTAssertEqual(
                pqA.encode(v), pqB.encode(v),
                "Same seed must yield identical codes (vector \(i))"
            )
        }
    }

    func testSameSeedDeterminismThroughVectorOverload() throws {
        let dim = 16
        let n = 300
        var rng = SeededRandom(seed: 0xDA7A_0002)
        let vectors = (0..<n).map { _ in
            Vector((0..<dim).map { _ in Float.random(in: -1...1, using: &rng) })
        }
        let config = PQConfiguration(subspaceCount: 4, trainingIterations: 8, seed: 99)

        let pqA = try ProductQuantizer.train(vectors: vectors, config: config)
        let pqB = try ProductQuantizer.train(vectors: vectors, config: config)

        XCTAssertEqual(
            codebookBytes(pqA), codebookBytes(pqB),
            "The [Vector] overload must be deterministic under a seed too"
        )
    }

    // ── Different Seeds → Different Codebooks (sanity) ───────────────

    func testDifferentSeedsYieldDifferentCodebooks() throws {
        let dim = 32
        let M = 8
        let n = 400
        let vectors = seededVectors(count: n, dimension: dim, dataSeed: 0xDA7A_0003)

        let pq1 = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5, seed: 1)
        )
        let pq2 = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 5, seed: 2)
        )

        XCTAssertNotEqual(
            codebookBytes(pq1), codebookBytes(pq2),
            "Different seeds should initialize different centroids and converge differently"
        )
    }

    // ── Nil Seed: Unchanged Behavior ─────────────────────────────────

    func testNilSeedTrainingStillProducesValidQuantizer() throws {
        let dim = 32
        let M = 8
        let n = 400
        let vectors = seededVectors(count: n, dimension: dim, dataSeed: 0xDA7A_0004)

        // No seed: the system-RNG path (the pre-seed code path) still trains
        // a structurally valid quantizer with usable codes.
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: M, trainingIterations: 10)
        )

        XCTAssertNil(pq.config.seed)
        XCTAssertEqual(pq.codebooks.count, M)
        for cb in pq.codebooks {
            XCTAssertEqual(cb.count, 256 * (dim / M))
        }
        let codes = pq.encode(Array(vectors[0..<dim]))
        XCTAssertEqual(codes.count, M)
        XCTAssertEqual(pq.decode(codes).count, dim)
    }

    // ── Codec: Seed Is a Training-Time Knob, Never Persisted ─────────

    func testSeedIsNotPersistedByPQTTCodec() throws {
        let dim = 16
        let n = 300
        let vectors = seededVectors(count: n, dimension: dim, dataSeed: 0xDA7A_0005)
        let pq = try ProductQuantizer.train(
            vectors: vectors, vectorCount: n, dimension: dim,
            config: PQConfiguration(subspaceCount: 4, trainingIterations: 8, seed: 1234)
        )

        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url) }
        try pq.save(to: url)
        let loaded = try ProductQuantizer.load(from: url)

        // The PQTT header carries subspaceCount/trainingIterations as UInt32
        // fields and has no slot for a seed (ADR-011); a loaded configuration
        // reports nil, and the trained artifact survives byte-identically.
        XCTAssertNil(loaded.config.seed, "seed must not survive a PQTT round trip")
        XCTAssertEqual(loaded.config.subspaceCount, pq.config.subspaceCount)
        XCTAssertEqual(loaded.config.trainingIterations, pq.config.trainingIterations)
        XCTAssertEqual(codebookBytes(loaded), codebookBytes(pq))

        // File size proves no extra header field appeared: 24-byte header +
        // M * K * ds Float32 codebooks, exactly as before the seed existed.
        let fileSize = try Data(contentsOf: url).count
        XCTAssertEqual(fileSize, 24 + 4 * 256 * (dim / 4) * 4)
    }

    func testSeedIsExcludedFromCodableEncoding() throws {
        let config = PQConfiguration(subspaceCount: 8, trainingIterations: 10, seed: 555)
        let encoded = try JSONEncoder().encode(config)

        // The wholesale-Codable layout must stay identical to pre-seed
        // releases: exactly the three legacy keys, no "seed".
        let object = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        )
        XCTAssertEqual(
            Set(object.keys),
            ["subspaceCount", "centroidsPerSubspace", "trainingIterations"]
        )

        let decoded = try JSONDecoder().decode(PQConfiguration.self, from: encoded)
        XCTAssertNil(decoded.seed)
        XCTAssertEqual(decoded.subspaceCount, 8)
        XCTAssertEqual(decoded.centroidsPerSubspace, 256)
        XCTAssertEqual(decoded.trainingIterations, 10)
    }

    func testLegacyJSONWithoutSeedKeyStillDecodes() throws {
        // Payload shape produced by releases that predate the seed.
        let legacy = Data(
            #"{"subspaceCount":48,"centroidsPerSubspace":256,"trainingIterations":25}"#.utf8
        )
        let decoded = try JSONDecoder().decode(PQConfiguration.self, from: legacy)
        XCTAssertEqual(decoded.subspaceCount, 48)
        XCTAssertEqual(decoded.centroidsPerSubspace, 256)
        XCTAssertEqual(decoded.trainingIterations, 25)
        XCTAssertNil(decoded.seed)
    }
}
