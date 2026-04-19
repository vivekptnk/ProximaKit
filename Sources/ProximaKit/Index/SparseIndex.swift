// SparseIndex.swift
// ProximaKit
//
// BM25 lexical index. Sibling of HNSWIndex / BruteForceIndex — same
// actor-isolated model, same tombstoning + compaction strategy, same binary
// persistence contract (`.pxbm`).
//
// BM25 reference: Robertson & Zaragoza, "The Probabilistic Relevance Framework:
// BM25 and Beyond" (2009). Uses the +0.5 IDF variant (Lucene-style) so scores
// remain non-negative even for terms that appear in every document.

import Foundation

/// Configuration for ``SparseIndex``.
///
/// Defaults match the Lucene / Elasticsearch BM25 presets (`k1 = 1.2`, `b = 0.75`),
/// which are the consensus "good enough across most English corpora" values.
public struct BM25Configuration: Sendable, Equatable {
    /// Term-frequency saturation. Higher = tf grows more before plateauing.
    ///
    /// Typical range: `1.2 ... 2.0`. `1.2` is the Lucene default.
    public let k1: Double

    /// Length normalization. `0` disables it, `1` is full normalization.
    ///
    /// Typical: `0.75` (Lucene default). Higher penalizes long documents more.
    public let b: Double

    /// Auto-compaction threshold — when the live/total ratio drops below this
    /// after a removal, the index rebuilds itself to reclaim tombstone slots.
    ///
    /// `nil` disables auto-compaction (manual `compact()` only). Default `0.7`,
    /// mirroring ``HNSWConfiguration/autoCompactionThreshold``.
    public let autoCompactionThreshold: Double?

    public init(
        k1: Double = 1.2,
        b: Double = 0.75,
        autoCompactionThreshold: Double? = 0.7
    ) {
        precondition(k1 >= 0, "k1 must be non-negative")
        precondition(b >= 0 && b <= 1, "b must be in [0, 1]")
        if let threshold = autoCompactionThreshold {
            precondition(threshold > 0 && threshold < 1, "autoCompactionThreshold must be in (0, 1)")
        }
        self.k1 = k1
        self.b = b
        self.autoCompactionThreshold = autoCompactionThreshold
    }
}

/// BM25 sparse (lexical) index with actor-isolated mutation.
///
/// `SparseIndex` is the lexical counterpart to ``HNSWIndex`` — together they form
/// the dense + sparse legs of ``HybridIndex``. It maintains an inverted index
/// from terms to postings (internal node id + term frequency) plus per-document
/// length statistics, and scores queries with the BM25 Okapi formula.
///
/// Add / remove are O(|document tokens|). Search is O(|query tokens| ×
/// average postings length), so performance tracks corpus vocabulary density
/// rather than raw document count.
///
/// ```swift
/// let index = SparseIndex()
/// let id = UUID()
/// try await index.add(text: "swift vector search", id: id)
/// let hits = await index.search(query: "vector", k: 3)
/// ```
public actor SparseIndex: SparseVectorIndex {

    // ── Configuration (immutable) ─────────────────────────────────────

    public nonisolated let tokenizer: any BM25Tokenizer
    public nonisolated let configuration: BM25Configuration

    // ── Node storage ──────────────────────────────────────────────────
    //
    // Docs are assigned a monotonically increasing internal node id on insert.
    // Removal is tombstoning — the slot survives in the parallel arrays and
    // postings lists, but the UUID is dropped from `uuidToNode`. Search skips
    // any posting whose node is no longer in `uuidToNode`.

    /// Internal node → token counts for this doc (used by compact()).
    /// `nil` for tombstoned slots.
    private var tokenCounts: [[String: Int]?] = []

    /// Per-doc token length (0 for tombstoned slots).
    private var docLengths: [Int] = []

    /// Metadata (JSON) for each slot.
    private var metadataStore: [Data?] = []

    /// Internal node index → external UUID.
    private var nodeToUUID: [UUID] = []

    /// External UUID → internal node index. Also the tombstone oracle:
    /// if an id is absent, its node is tombstoned.
    private var uuidToNode: [UUID: Int] = [:]

    /// Inverted index: token → [(nodeIndex, tf)].
    /// Postings may reference tombstoned nodes until the next `compact()`;
    /// search filters them out via `uuidToNode` membership.
    private var postings: [String: [(node: Int, tf: Int)]] = [:]

    /// Sum of live doc lengths. Used for avgdl during scoring.
    private var totalLiveTokens: Int = 0

    // ── SparseVectorIndex conformance ────────────────────────────────

    /// The number of live (non-tombstoned) documents.
    public var count: Int { uuidToNode.count }

    /// Total slot count including tombstones. Use ``count`` for the searchable
    /// document count; this is the allocation for persistence / compaction bookkeeping.
    public var slotCount: Int { nodeToUUID.count }

    /// The average document length over live documents, or `0` if the index is empty.
    /// Exposed for diagnostics and tests.
    public var averageDocumentLength: Double {
        let live = uuidToNode.count
        return live > 0 ? Double(totalLiveTokens) / Double(live) : 0
    }

    // ── Initialization ────────────────────────────────────────────────

    /// Creates an empty BM25 index.
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer used both at insert time and at query time.
    ///     The same instance (or an equivalent one) must be used for persistence
    ///     round-trips — this library does not serialize tokenizer identity.
    ///   - configuration: BM25 tuning parameters.
    public init(
        tokenizer: any BM25Tokenizer = DefaultBM25Tokenizer(),
        configuration: BM25Configuration = BM25Configuration()
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    /// Restores a SparseIndex from a persistence snapshot.
    public init(
        restoring snapshot: SparseIndexSnapshot,
        tokenizer: any BM25Tokenizer = DefaultBM25Tokenizer()
    ) {
        self.tokenizer = tokenizer
        self.configuration = snapshot.configuration
        self.tokenCounts = snapshot.tokenCounts
        self.docLengths = snapshot.docLengths
        self.metadataStore = snapshot.metadataStore
        self.nodeToUUID = snapshot.nodeToUUID
        self.postings = snapshot.postings
        self.totalLiveTokens = snapshot.docLengths.reduce(0, +)
        self.uuidToNode = [:]
        for (i, uuid) in snapshot.nodeToUUID.enumerated() {
            uuidToNode[uuid] = i
        }
    }

    // ── Add ───────────────────────────────────────────────────────────

    public func add(text: String, id: UUID, metadata: Data? = nil) throws {
        // Replace-on-duplicate, matching HNSWIndex / BruteForceIndex.
        if uuidToNode[id] != nil {
            _ = remove(id: id)
        }

        let tokens = tokenizer.tokenize(text)

        // Count term frequencies for this doc.
        var counts: [String: Int] = [:]
        counts.reserveCapacity(tokens.count)
        for token in tokens {
            counts[token, default: 0] += 1
        }

        let node = nodeToUUID.count
        nodeToUUID.append(id)
        uuidToNode[id] = node
        tokenCounts.append(counts)
        docLengths.append(tokens.count)
        metadataStore.append(metadata)
        totalLiveTokens += tokens.count

        for (token, tf) in counts {
            postings[token, default: []].append((node: node, tf: tf))
        }
    }

    /// Convenience add without metadata.
    public func add(text: String, id: UUID) throws {
        try add(text: text, id: id, metadata: nil)
    }

    // ── Remove (tombstoning) ──────────────────────────────────────────

    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let node = uuidToNode.removeValue(forKey: id) else { return false }

        totalLiveTokens -= docLengths[node]
        docLengths[node] = 0
        tokenCounts[node] = nil
        // metadataStore[node] kept until compact(); not harmful because
        // the node is no longer reachable via uuidToNode.

        if let threshold = configuration.autoCompactionThreshold,
           nodeToUUID.count > 0,
           Double(uuidToNode.count) / Double(nodeToUUID.count) < threshold {
            compactInternal()
        }

        return true
    }

    // ── Search (BM25 Okapi) ───────────────────────────────────────────

    public func search(
        query: String,
        k: Int,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        guard k > 0 else { return [] }
        let liveCount = uuidToNode.count
        guard liveCount > 0 else { return [] }

        let queryTokens = tokenizer.tokenize(query)
        guard !queryTokens.isEmpty else { return [] }

        // De-dup query terms but keep one count per term — BM25 ignores query tf.
        var queryTermSet = Set<String>()
        var uniqueQueryTokens: [String] = []
        for token in queryTokens where queryTermSet.insert(token).inserted {
            uniqueQueryTokens.append(token)
        }

        let k1 = configuration.k1
        let b = configuration.b
        let N = Double(liveCount)
        let avgdl = averageDocumentLength

        // Accumulate BM25 scores for each doc that contains at least one query term.
        var scores: [Int: Double] = [:]

        for token in uniqueQueryTokens {
            guard let termPostings = postings[token] else { continue }

            // Document frequency counts only live postings.
            var df = 0
            for posting in termPostings where tokenCounts[posting.node] != nil {
                df += 1
            }
            guard df > 0 else { continue }

            // Lucene-style +0.5 IDF: log(1 + (N - df + 0.5) / (df + 0.5)).
            // Always non-negative; nudges by +1 so a term in every doc still
            // contributes a tiny positive score rather than zero.
            let idf = log(1.0 + (N - Double(df) + 0.5) / (Double(df) + 0.5))

            for posting in termPostings {
                guard tokenCounts[posting.node] != nil else { continue }
                let tf = Double(posting.tf)
                let dl = Double(docLengths[posting.node])
                let norm: Double
                if avgdl > 0 {
                    norm = 1.0 - b + b * (dl / avgdl)
                } else {
                    norm = 1.0
                }
                let termScore = idf * ((tf * (k1 + 1.0)) / (tf + k1 * norm))
                scores[posting.node, default: 0] += termScore
            }
        }

        guard !scores.isEmpty else { return [] }

        // Build results. Convert "higher is better" BM25 score → "lower distance"
        // by negating, matching the library-wide SearchResult.distance convention.
        var results: [SearchResult] = []
        results.reserveCapacity(scores.count)
        for (node, score) in scores {
            let uuid = nodeToUUID[node]
            if let filter, !filter(uuid) { continue }
            results.append(SearchResult(
                id: uuid,
                distance: Float(-score),
                metadata: metadataStore[node]
            ))
        }

        results.sort()
        if results.count > k {
            results = Array(results.prefix(k))
        }
        return results
    }

    // ── Compaction ────────────────────────────────────────────────────

    /// Rebuilds the index without tombstoned slots.
    ///
    /// O(total live tokens). Call after large removal bursts, or let the
    /// configured ``BM25Configuration/autoCompactionThreshold`` trigger it.
    public func compact() {
        compactInternal()
    }

    private func compactInternal() {
        // Snapshot live docs keyed by their NEW node index (insertion order preserved).
        var newTokenCounts: [[String: Int]?] = []
        var newDocLengths: [Int] = []
        var newMetadata: [Data?] = []
        var newNodeToUUID: [UUID] = []
        var newPostings: [String: [(node: Int, tf: Int)]] = [:]

        newTokenCounts.reserveCapacity(uuidToNode.count)
        newDocLengths.reserveCapacity(uuidToNode.count)
        newMetadata.reserveCapacity(uuidToNode.count)
        newNodeToUUID.reserveCapacity(uuidToNode.count)

        for (oldNode, uuid) in nodeToUUID.enumerated() {
            guard uuidToNode[uuid] != nil,
                  let counts = tokenCounts[oldNode] else { continue }

            let newNode = newNodeToUUID.count
            newNodeToUUID.append(uuid)
            newTokenCounts.append(counts)
            newDocLengths.append(docLengths[oldNode])
            newMetadata.append(metadataStore[oldNode])

            for (token, tf) in counts {
                newPostings[token, default: []].append((node: newNode, tf: tf))
            }
        }

        // Commit.
        tokenCounts = newTokenCounts
        docLengths = newDocLengths
        metadataStore = newMetadata
        nodeToUUID = newNodeToUUID
        postings = newPostings
        uuidToNode = [:]
        for (i, uuid) in nodeToUUID.enumerated() {
            uuidToNode[uuid] = i
        }
        totalLiveTokens = docLengths.reduce(0, +)
    }

    // ── Persistence ───────────────────────────────────────────────────

    /// Returns a snapshot for binary persistence. Compacts first if needed.
    public func persistenceSnapshot() -> SparseIndexSnapshot {
        if uuidToNode.count < nodeToUUID.count {
            compactInternal()
        }
        return SparseIndexSnapshot(
            configuration: configuration,
            tokenCounts: tokenCounts,
            docLengths: docLengths,
            metadataStore: metadataStore,
            nodeToUUID: nodeToUUID,
            postings: postings
        )
    }

    /// Saves the index to a `.pxbm` binary file.
    public func save(to url: URL) throws {
        let snapshot = persistenceSnapshot()
        try PersistenceEngine.save(snapshot, to: url)
    }

    /// Loads a SparseIndex from a `.pxbm` binary file, using the supplied tokenizer
    /// for subsequent queries.
    public static func load(
        from url: URL,
        tokenizer: any BM25Tokenizer = DefaultBM25Tokenizer()
    ) throws -> SparseIndex {
        try PersistenceEngine.loadSparse(from: url, tokenizer: tokenizer)
    }
}
