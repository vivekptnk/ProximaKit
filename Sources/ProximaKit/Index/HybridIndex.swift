// HybridIndex.swift
// ProximaKit
//
// Fuses a dense ``VectorIndex`` (e.g., ``HNSWIndex``) with a sparse
// ``SparseVectorIndex`` (``SparseIndex`` / BM25). Both legs are queried
// concurrently with structured concurrency (`async let`); the two result
// lists are combined with a configurable ``HybridFusionStrategy``.
//
// Why two legs? Dense retrieval catches paraphrase / semantic matches;
// sparse retrieval catches rare terms, product codes, and keyword overlaps
// where the embedding model dilutes the signal. Fusion is robust to a
// weak leg — see Cormack et al., "Reciprocal rank fusion outperforms
// Condorcet and individual rank learning methods" (SIGIR 2009).

import Foundation

// MARK: - Fusion Strategy

/// How to combine ranked result lists from the dense and sparse legs.
public enum HybridFusionStrategy: Sendable, Equatable {
    /// Reciprocal Rank Fusion. Robust default — score = Σ 1 / (k + rank).
    ///
    /// Does not depend on raw score scales, which makes it the correct first
    /// pass when one leg's score distribution is unknown or drifts across
    /// corpora. Default `k = 60` follows the Cormack et al. paper.
    case rrf(k: Double = 60.0)

    /// Weighted sum of min-max normalized per-leg scores.
    ///
    /// `alpha` is the weight of the dense leg; the sparse leg is weighted by
    /// `1 - alpha`. `alpha = 1.0` degenerates to dense-only, `alpha = 0.0` to
    /// sparse-only. Use this when you've validated both legs' score
    /// distributions on your own data and want finer control.
    case weightedSum(alpha: Double)

    public static var rrf: HybridFusionStrategy { .rrf() }
}

// MARK: - HybridIndex

/// A hybrid dense + sparse index. Writes fan out to both legs; reads fuse the
/// two ranked lists with a configurable ``HybridFusionStrategy``.
///
/// ```swift
/// let dense = HNSWIndex(dimension: 384)
/// let sparse = SparseIndex()
/// let hybrid = HybridIndex(dense: dense, sparse: sparse)
///
/// try await hybrid.add(text: "hybrid search", vector: embedding, id: UUID())
/// let hits = try await hybrid.search(
///     queryText: "hybrid",
///     queryVector: queryEmbedding,
///     k: 10
/// )
/// ```
///
/// The unified ``add(text:vector:id:metadata:)`` entry point is the enforced
/// way to keep document IDs consistent across the two legs. Direct access to
/// `dense` and `sparse` is possible for advanced use (e.g., bulk import) but
/// comes with the responsibility of keeping IDs synchronized.
public actor HybridIndex {

    // ── Legs (exposed for advanced read access) ──────────────────────

    /// The dense vector index leg (e.g., ``HNSWIndex``).
    public nonisolated let dense: any VectorIndex
    /// The sparse (BM25) leg.
    public nonisolated let sparse: any SparseVectorIndex

    /// Current fusion strategy. Mutable — swap it per-query or per-index-lifetime.
    public var fusion: HybridFusionStrategy

    // ── Initialization ────────────────────────────────────────────────

    public init(
        dense: any VectorIndex,
        sparse: any SparseVectorIndex,
        fusion: HybridFusionStrategy = .rrf()
    ) {
        self.dense = dense
        self.sparse = sparse
        self.fusion = fusion
    }

    // ── Counts ────────────────────────────────────────────────────────

    /// The number of documents in the dense leg.
    public var denseCount: Int {
        get async { await dense.count }
    }

    /// The number of live documents in the sparse leg.
    public var sparseCount: Int {
        get async { await sparse.count }
    }

    // ── Unified add ──────────────────────────────────────────────────

    /// Embeds a document into both legs under the same UUID.
    ///
    /// - Parameters:
    ///   - text: Raw text for the sparse leg to tokenize.
    ///   - vector: Pre-computed dense embedding for the dense leg.
    ///   - id: UUID shared across both legs. Must match in every subsequent
    ///     operation (remove, filter) for the leg IDs to stay aligned.
    ///   - metadata: Optional JSON metadata — stored on *both* legs so search
    ///     results from either leg carry the same payload.
    public func add(
        text: String,
        vector: Vector,
        id: UUID,
        metadata: Data? = nil
    ) async throws {
        try await dense.add(vector, id: id, metadata: metadata)
        try await sparse.add(text: text, id: id, metadata: metadata)
    }

    // ── Remove ────────────────────────────────────────────────────────

    /// Removes a document from both legs. Returns `true` if either leg had the ID.
    @discardableResult
    public func remove(id: UUID) async -> Bool {
        // Fire both concurrently — the actors serialize internally, but we
        // avoid one leg waiting on the other's queue.
        async let removedDense = dense.remove(id: id)
        async let removedSparse = sparse.remove(id: id)
        let (d, s) = await (removedDense, removedSparse)
        return d || s
    }

    // ── Search ────────────────────────────────────────────────────────

    /// Searches both legs concurrently and fuses their rankings.
    ///
    /// - Parameters:
    ///   - queryText: Raw query text for the sparse leg.
    ///   - queryVector: Pre-computed dense query vector.
    ///   - k: Final top-k to return after fusion.
    ///   - candidatePoolK: How many candidates to draw from **each** leg before
    ///     fusing. Larger pools improve recall at the cost of per-query work.
    ///     Defaults to `max(k * 5, 50)`, chosen so k=10 draws 50 per leg —
    ///     enough to fill the fusion buffer even if the legs barely overlap.
    ///   - filter: Applied on both legs before fusion.
    public func search(
        queryText: String,
        queryVector: Vector,
        k: Int,
        candidatePoolK: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) async -> [SearchResult] {
        guard k > 0 else { return [] }

        let pool = candidatePoolK ?? max(k * 5, 50)

        async let denseResults = dense.search(
            query: queryVector,
            k: pool,
            efSearch: nil,
            filter: filter
        )
        async let sparseResults = sparse.search(
            query: queryText,
            k: pool,
            filter: filter
        )

        let (denseList, sparseList) = await (denseResults, sparseResults)

        return HybridIndex.fuse(
            dense: denseList,
            sparse: sparseList,
            strategy: fusion,
            k: k
        )
    }

    // ── Fusion ────────────────────────────────────────────────────────

    /// Fuses two ranked lists with the given strategy. Internal but exposed
    /// `static` so tests can exercise the math without standing up both legs.
    static func fuse(
        dense: [SearchResult],
        sparse: [SearchResult],
        strategy: HybridFusionStrategy,
        k: Int
    ) -> [SearchResult] {
        guard k > 0 else { return [] }
        if dense.isEmpty && sparse.isEmpty { return [] }

        switch strategy {
        case .rrf(let rrfK):
            return fuseRRF(dense: dense, sparse: sparse, k: k, rrfK: rrfK)
        case .weightedSum(let alpha):
            return fuseWeightedSum(dense: dense, sparse: sparse, k: k, alpha: alpha)
        }
    }

    private static func fuseRRF(
        dense: [SearchResult],
        sparse: [SearchResult],
        k: Int,
        rrfK: Double
    ) -> [SearchResult] {
        // Per-list rank is 1-based in RRF.
        var scores: [UUID: Double] = [:]
        var metadata: [UUID: Data] = [:]

        for (rank, result) in dense.enumerated() {
            scores[result.id, default: 0] += 1.0 / (rrfK + Double(rank + 1))
            if let meta = result.metadata {
                metadata[result.id] = meta
            }
        }
        for (rank, result) in sparse.enumerated() {
            scores[result.id, default: 0] += 1.0 / (rrfK + Double(rank + 1))
            // Prefer whichever leg wrote metadata first (dense runs first above);
            // if only the sparse leg has it, take it.
            if metadata[result.id] == nil, let meta = result.metadata {
                metadata[result.id] = meta
            }
        }

        return finalizeFusion(scores: scores, metadata: metadata, k: k)
    }

    private static func fuseWeightedSum(
        dense: [SearchResult],
        sparse: [SearchResult],
        k: Int,
        alpha: Double
    ) -> [SearchResult] {
        let denseNorm = minMaxNormalize(dense)
        let sparseNorm = minMaxNormalize(sparse)

        var scores: [UUID: Double] = [:]
        var metadata: [UUID: Data] = [:]

        for (id, score) in denseNorm {
            scores[id, default: 0] += alpha * score
        }
        for result in dense {
            if let meta = result.metadata {
                metadata[result.id] = meta
            }
        }

        for (id, score) in sparseNorm {
            scores[id, default: 0] += (1.0 - alpha) * score
        }
        for result in sparse where metadata[result.id] == nil {
            if let meta = result.metadata {
                metadata[result.id] = meta
            }
        }

        return finalizeFusion(scores: scores, metadata: metadata, k: k)
    }

    /// Min-max normalizes a list of results into [0, 1] where 1 is best.
    ///
    /// Input distances are "lower is better"; we negate before normalizing so
    /// the output is "higher is better" to make the weighted sum math read
    /// naturally (`alpha*dense + (1-alpha)*sparse`).
    private static func minMaxNormalize(_ results: [SearchResult]) -> [UUID: Double] {
        guard !results.isEmpty else { return [:] }

        let negated = results.map { (id: $0.id, score: -Double($0.distance)) }
        guard let min = negated.map(\.score).min(),
              let max = negated.map(\.score).max() else {
            return [:]
        }

        let range = max - min
        var out: [UUID: Double] = [:]
        out.reserveCapacity(negated.count)

        if range == 0 {
            // Every result tied — give each the midpoint value. Avoids NaN.
            for entry in negated {
                out[entry.id] = 0.5
            }
        } else {
            for entry in negated {
                out[entry.id] = (entry.score - min) / range
            }
        }
        return out
    }

    private static func finalizeFusion(
        scores: [UUID: Double],
        metadata: [UUID: Data],
        k: Int
    ) -> [SearchResult] {
        // Convert "higher is better" fused score → "lower distance" for SearchResult.
        var results: [SearchResult] = []
        results.reserveCapacity(scores.count)
        for (id, score) in scores {
            results.append(SearchResult(
                id: id,
                distance: Float(-score),
                metadata: metadata[id]
            ))
        }

        results.sort()
        if results.count > k {
            results = Array(results.prefix(k))
        }
        return results
    }
}
