// SparseIndexTests.swift
// ProximaKit
//
// Tests for SparseIndex (BM25):
// - Basic add / search / remove / count
// - Tombstoning + compaction
// - BM25 top-k parity against an oracle on a 1K synthetic corpus
// - Persistence round-trip (.pxbm)
// - Filter hooks

import XCTest
@testable import ProximaKit

// MARK: - Oracle BM25 (reference implementation for parity checks)

/// Straightforward BM25 reference. Intentionally written for clarity over speed —
/// this is the oracle the production index is compared against.
private struct OracleBM25 {
    struct Doc {
        let id: UUID
        let tokens: [String]
    }

    let k1: Double
    let b: Double
    let docs: [Doc]

    func search(query: [String], k: Int) -> [(id: UUID, score: Double)] {
        guard !docs.isEmpty else { return [] }

        let N = Double(docs.count)
        let avgdl = Double(docs.map(\.tokens.count).reduce(0, +)) / N

        // Document frequency per term.
        var df: [String: Int] = [:]
        for doc in docs {
            let unique = Set(doc.tokens)
            for term in unique {
                df[term, default: 0] += 1
            }
        }

        // De-dup query tokens (BM25 ignores query tf).
        var seen = Set<String>()
        let uniqueQuery = query.filter { seen.insert($0).inserted }

        var scored: [(id: UUID, score: Double)] = []
        for doc in docs {
            var score = 0.0
            let dl = Double(doc.tokens.count)
            var counts: [String: Int] = [:]
            for t in doc.tokens { counts[t, default: 0] += 1 }
            for qt in uniqueQuery {
                guard let tf = counts[qt], tf > 0, let d = df[qt], d > 0 else { continue }
                let idf = log(1.0 + (N - Double(d) + 0.5) / (Double(d) + 0.5))
                let norm = (avgdl > 0) ? (1 - b + b * (dl / avgdl)) : 1
                score += idf * (Double(tf) * (k1 + 1)) / (Double(tf) + k1 * norm)
            }
            if score > 0 {
                scored.append((id: doc.id, score: score))
            }
        }
        scored.sort { $0.score > $1.score }
        if scored.count > k {
            scored = Array(scored.prefix(k))
        }
        return scored
    }
}

// MARK: - Seeded RNG (deterministic corpus generation)

/// Simple xorshift — we need determinism across test runs and xorshift is
/// enough for generating unbiased-enough synthetic corpora.
private struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

// MARK: - Basic SparseIndex Tests

final class SparseIndexTests: XCTestCase {

    func testEmptyIndexReturnsNoResults() async throws {
        let index = SparseIndex()
        let results = await index.search(query: "anything", k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    func testAddAndSearch() async throws {
        let index = SparseIndex()
        let id1 = UUID()
        let id2 = UUID()
        let id3 = UUID()
        try await index.add(text: "the quick brown fox", id: id1)
        try await index.add(text: "the lazy dog slept", id: id2)
        try await index.add(text: "a fox jumped over the dog", id: id3)

        let count = await index.count
        XCTAssertEqual(count, 3)

        let results = await index.search(query: "fox", k: 5)
        let ids = Set(results.map(\.id))
        XCTAssertTrue(ids.contains(id1))
        XCTAssertTrue(ids.contains(id3))
        XCTAssertFalse(ids.contains(id2))
    }

    func testSearchOrdering() async throws {
        let index = SparseIndex()
        let rare = UUID()
        let common = UUID()
        let unrelated = UUID()
        try await index.add(text: "zebra", id: rare)
        try await index.add(text: "zebra zebra zebra", id: common)
        try await index.add(text: "hello world", id: unrelated)

        let results = await index.search(query: "zebra", k: 3)

        // Both zebra docs score; unrelated must be filtered out.
        XCTAssertEqual(results.count, 2)
        let ids = results.map(\.id)
        // Higher tf wins at the same doc length category.
        XCTAssertEqual(ids.first, common)
        XCTAssertTrue(ids.contains(rare))
    }

    func testReplaceOnDuplicateId() async throws {
        let index = SparseIndex()
        let id = UUID()
        try await index.add(text: "first version", id: id)
        try await index.add(text: "second version with search term", id: id)

        let count = await index.count
        XCTAssertEqual(count, 1, "Duplicate id should replace, not duplicate")

        let results = await index.search(query: "search", k: 5)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, id)
    }

    func testEmptyQueryReturnsNothing() async throws {
        let index = SparseIndex()
        try await index.add(text: "anything", id: UUID())
        let results = await index.search(query: "", k: 5)
        XCTAssertTrue(results.isEmpty)
    }

    // MARK: - Tombstoning + Compaction

    func testRemoveTombstones() async throws {
        let index = SparseIndex()
        let id1 = UUID()
        let id2 = UUID()
        try await index.add(text: "alpha beta", id: id1)
        try await index.add(text: "alpha gamma", id: id2)

        let removed = await index.remove(id: id1)
        XCTAssertTrue(removed)

        let count = await index.count
        XCTAssertEqual(count, 1)

        let results = await index.search(query: "alpha", k: 5)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, id2)
    }

    func testRemoveNonExistent() async throws {
        let index = SparseIndex()
        let removed = await index.remove(id: UUID())
        XCTAssertFalse(removed)
    }

    func testCompactReclaimsSlots() async throws {
        // Disable auto-compaction so we can drive it manually.
        let index = SparseIndex(
            configuration: BM25Configuration(autoCompactionThreshold: nil)
        )
        let ids = (0..<10).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            try await index.add(text: "word\(i) shared", id: id)
        }

        for id in ids.prefix(7) {
            _ = await index.remove(id: id)
        }

        let preCount = await index.count
        let preSlots = await index.slotCount
        XCTAssertEqual(preCount, 3)
        XCTAssertEqual(preSlots, 10)

        await index.compact()

        let postCount = await index.count
        let postSlots = await index.slotCount
        XCTAssertEqual(postCount, 3)
        XCTAssertEqual(postSlots, 3)

        // Still searchable post-compaction.
        let results = await index.search(query: "shared", k: 10)
        XCTAssertEqual(results.count, 3)
    }

    func testAutoCompactionFiresBelowThreshold() async throws {
        let index = SparseIndex(
            configuration: BM25Configuration(autoCompactionThreshold: 0.5)
        )
        let ids = (0..<10).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            try await index.add(text: "token\(i)", id: id)
        }

        // Remove 6/10 → live ratio 0.4 < 0.5 → should auto-compact.
        for id in ids.prefix(6) {
            _ = await index.remove(id: id)
        }

        let slots = await index.slotCount
        let count = await index.count
        XCTAssertEqual(count, 4)
        XCTAssertEqual(slots, 4, "Auto-compaction should have collapsed slots to live count")
    }

    // MARK: - Filter

    func testFilterRejectsIds() async throws {
        let index = SparseIndex()
        let a = UUID(), b = UUID()
        try await index.add(text: "match here", id: a)
        try await index.add(text: "match there", id: b)

        let onlyA = await index.search(query: "match", k: 5, filter: { $0 == a })
        XCTAssertEqual(onlyA.map(\.id), [a])
    }

    // MARK: - Metadata

    func testMetadataSurvivesSearch() async throws {
        struct Payload: Codable, Equatable { let label: String }
        let index = SparseIndex()
        let id = UUID()
        let payload = Payload(label: "note-1")
        let data = try JSONEncoder().encode(payload)
        try await index.add(text: "metadata check", id: id, metadata: data)

        let results = await index.search(query: "metadata", k: 1)
        XCTAssertEqual(results.count, 1)
        let decoded = results[0].decodeMetadata(as: Payload.self)
        XCTAssertEqual(decoded, payload)
    }

    // MARK: - BM25 Parity (1K synthetic corpus)

    func testBM25ParityAgainstOracle() async throws {
        // Generate a deterministic 1K-doc corpus from a small vocabulary.
        // Small vocab => lots of term overlap => meaningful IDF work for the
        // oracle to disagree with if we had the formula wrong.
        var rng = SeededRNG(state: 0xA7B5_CC1F_0011_2233)
        let vocab = (0..<50).map { "w\($0)" }

        var docs: [OracleBM25.Doc] = []
        docs.reserveCapacity(1000)

        let index = SparseIndex()

        for _ in 0..<1000 {
            let docLen = Int.random(in: 5...25, using: &rng)
            let tokens = (0..<docLen).map { _ -> String in
                let idx = Int.random(in: 0..<vocab.count, using: &rng)
                return vocab[idx]
            }
            let id = UUID()
            docs.append(OracleBM25.Doc(id: id, tokens: tokens))
            try await index.add(text: tokens.joined(separator: " "), id: id)
        }

        let oracle = OracleBM25(k1: 1.2, b: 0.75, docs: docs)

        // Run several queries of varying term count.
        let queries: [[String]] = [
            ["w3"],
            ["w0", "w10"],
            ["w12", "w14", "w25"],
            ["w30", "w31", "w32", "w33"],
        ]

        // k=10 is the product surface we care about, but BM25 ties are common
        // on a small vocabulary and neither impl guarantees a tie-break order.
        // Oversample both sides by a healthy slack so any bucket straddling
        // the top-k boundary is fully realized, and compare complete buckets.
        let k = 10
        let slack = 50

        for query in queries {
            let oracleFull = oracle.search(query: query, k: k + slack)
            let indexFull = await index.search(
                query: query.joined(separator: " "),
                k: k + slack
            )

            let oracleBuckets = bucketByScore(oracleFull.map { ($0.id, $0.score) })
            let indexBuckets = bucketByScore(indexFull.map { ($0.id, -Double($0.distance)) })

            // For fully-realized buckets (both sides drew their full tied set
            // out of the corpus), score AND id set must match exactly. A
            // bucket is fully realized when oracle's running count stays at
            // or below k + slack AND index returns the same bucket size.
            var running = 0
            for (i, oracleBucket) in oracleBuckets.enumerated() {
                // Only assert on buckets that are relevant to the top-k
                // product surface — once we've covered more than k docs, we
                // have enough evidence that BM25 scoring agrees with the
                // oracle across the top-k window.
                if running >= k { break }

                XCTAssertLessThan(
                    i, indexBuckets.count,
                    "Index missing bucket \(i) for query \(query)"
                )
                let indexBucket = indexBuckets[i]

                XCTAssertEqual(
                    oracleBucket.score,
                    indexBucket.score,
                    accuracy: 1e-5,
                    "Bucket \(i) score must match oracle for query \(query)"
                )
                XCTAssertEqual(
                    oracleBucket.ids,
                    indexBucket.ids,
                    "Bucket \(i) id set must match oracle for query \(query)"
                )

                running += oracleBucket.ids.count
            }
        }
    }

    /// Groups consecutive (id, score) pairs into buckets of equal score, in
    /// descending score order. Within each bucket, ids are stored as a Set so
    /// tie-breaking order does not affect the comparison.
    private func bucketByScore(
        _ entries: [(id: UUID, score: Double)]
    ) -> [(score: Double, ids: Set<UUID>)] {
        var buckets: [(score: Double, ids: Set<UUID>)] = []
        for entry in entries {
            if var last = buckets.last,
               abs(last.score - entry.score) < 1e-6 {
                last.ids.insert(entry.id)
                buckets[buckets.count - 1] = last
            } else {
                buckets.append((score: entry.score, ids: [entry.id]))
            }
        }
        return buckets
    }

    // MARK: - Persistence

    func testPersistenceRoundtrip() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-SparseIndex-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = tempDir.appendingPathComponent("test.pxbm")

        let original = SparseIndex()
        let ids = (0..<20).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            let text = "alpha beta gamma delta \(i) common"
            try await original.add(
                text: text,
                id: id,
                metadata: Data("meta-\(i)".utf8)
            )
        }

        try await original.save(to: url)

        let reloaded = try SparseIndex.load(from: url)
        let reloadedCount = await reloaded.count
        XCTAssertEqual(reloadedCount, 20)

        let query = "gamma"
        let originalResults = await original.search(query: query, k: 10)
        let reloadedResults = await reloaded.search(query: query, k: 10)

        XCTAssertEqual(originalResults.count, reloadedResults.count)
        for (a, b) in zip(originalResults, reloadedResults) {
            XCTAssertEqual(a.id, b.id)
            XCTAssertEqual(a.distance, b.distance, accuracy: 1e-5)
            XCTAssertEqual(a.metadata, b.metadata)
        }
    }

    func testPersistenceCompactsTombstonesOnSave() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProximaKitTests-SparseIndex-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let url = tempDir.appendingPathComponent("test.pxbm")

        let index = SparseIndex(
            configuration: BM25Configuration(autoCompactionThreshold: nil)
        )
        let ids = (0..<5).map { _ in UUID() }
        for (i, id) in ids.enumerated() {
            try await index.add(text: "term\(i) keep", id: id)
        }
        _ = await index.remove(id: ids[0])
        _ = await index.remove(id: ids[1])

        let preSaveSlots = await index.slotCount
        XCTAssertEqual(preSaveSlots, 5)

        try await index.save(to: url)

        let reloaded = try SparseIndex.load(from: url)
        let reloadedSlots = await reloaded.slotCount
        let reloadedCount = await reloaded.count
        XCTAssertEqual(reloadedSlots, 3, "Tombstoned slots must not persist")
        XCTAssertEqual(reloadedCount, 3)
    }
}

// MARK: - Tokenizer Tests

final class DefaultBM25TokenizerTests: XCTestCase {

    func testLowercases() {
        let t = DefaultBM25Tokenizer()
        XCTAssertEqual(t.tokenize("Hello World"), ["hello", "world"])
    }

    func testSplitsOnPunctuation() {
        let t = DefaultBM25Tokenizer()
        XCTAssertEqual(t.tokenize("foo, bar! baz?"), ["foo", "bar", "baz"])
    }

    func testEmptyString() {
        let t = DefaultBM25Tokenizer()
        XCTAssertEqual(t.tokenize(""), [])
    }

    func testUnicodeWords() {
        let t = DefaultBM25Tokenizer()
        let tokens = t.tokenize("café résumé")
        XCTAssertEqual(tokens, ["café", "résumé"])
    }

    func testPreservesDuplicatesForTF() {
        let t = DefaultBM25Tokenizer()
        XCTAssertEqual(t.tokenize("cat cat cat"), ["cat", "cat", "cat"])
    }
}
