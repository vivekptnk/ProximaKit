// QuantizedHNSWIndex.swift
// ProximaKit
//
// Memory-efficient HNSW index using product quantization.
//
// Stores the HNSW graph structure with PQ codes instead of full Float32 vectors.
// Search uses asymmetric distance computation (ADC): full-precision query
// against PQ-compressed database vectors.
//
// Typical memory reduction: 384d Float32 (1536 bytes) → 48-byte PQ code = 32×.
// Graph overhead (~200 bytes/node for M=16) is shared with the standard index.
//
// Build workflow:
//   1. Insert all vectors into a full-precision HNSWIndex (accurate graph)
//   2. Train a ProductQuantizer on the same vectors
//   3. Call QuantizedHNSWIndex.build(...) to extract graph + PQ codes
//   4. Discard the original vectors — search uses ADC only
//
// Opt-in reranking (ADR-012): build(retainOriginals: true) keeps the full
// vectors alongside the codes, and search(rerankDepth:) re-scores the top
// ADC candidates with exact Euclidean distance before truncating to k.
// This trades the memory story for recall — see ADR-012 for the math.

import Foundation

/// Errors thrown by `QuantizedHNSWIndex` search operations.
public enum QuantizedIndexError: Error, LocalizedError, Sendable, Equatable {
    /// A positive `rerankDepth` was requested, but the index was built
    /// without `retainOriginals: true` (or loaded from a file saved without
    /// originals), so there are no full-precision vectors to re-score
    /// against. Rebuild with `retainOriginals: true` to enable reranking.
    case originalsNotRetained

    public var errorDescription: String? {
        switch self {
        case .originalsNotRetained:
            return "Reranking requires retained originals — build the index "
                + "with retainOriginals: true (ADR-012)"
        }
    }
}

/// A memory-efficient HNSW index that stores PQ codes instead of full vectors.
///
/// Build a quantized index from training data:
/// ```swift
/// let qIndex = try QuantizedHNSWIndex.build(
///     vectors: vectors,
///     ids: ids,
///     dimension: 384,
///     hnswConfig: HNSWConfiguration(),
///     pqConfig: PQConfiguration(subspaceCount: 48)
/// )
///
/// let results = qIndex.search(query: queryVec, k: 10)
/// // Memory per vector: 48 bytes instead of 1536 bytes
/// ```
public actor QuantizedHNSWIndex {

    // ── Configuration ────────────────────────────────────────────────

    /// The full vector dimension this index was built for.
    public nonisolated let dimension: Int

    /// HNSW graph configuration.
    let hnswConfig: HNSWConfiguration

    /// The trained product quantizer.
    public let quantizer: ProductQuantizer

    // ── Graph Structure ──────────────────────────────────────────────
    //
    // Same multi-layer adjacency list layout as HNSWIndex.
    // layers[l][n] = neighbor indices for node n on layer l.

    var layers: [[[Int]]]

    /// Reverse adjacency: `inEdges[l][n]` is the set of nodes whose layer-`l`
    /// neighbor list contains `n` — the exact transpose of `layers`. Maintained
    /// so `remove(id:)` can delete every edge pointing INTO a node in
    /// O(in-degree) instead of sweeping every edge list on the layer (O(E_l)).
    ///
    /// Derived state: NOT persisted (the on-disk format is unchanged, ADR-010)
    /// and NOT built in the memberwise initializer. It is materialized lazily
    /// by `ensureInEdges()` at first use — mirroring `HNSWIndex`, whose
    /// transposing `init(restoring:)` only ever receives loader-validated
    /// layers. The quantized memberwise initializer, by contrast, is the
    /// documented entry point for arbitrary components (the persistence
    /// corruption suite constructs indexes whose adjacency is deliberately out
    /// of bounds to prove `load(from:)` rejects the saved bytes with a typed
    /// `PersistenceError`). Transposing eagerly in the initializer would index
    /// `layers` BEFORE that load-time gate and turn the contracted typed error
    /// into a process trap. Lazily, the map is only ever built after the graph
    /// is known well-formed: post-`build` (graph from a healthy `HNSWIndex`)
    /// or post-`load` (every neighbor bounds-checked before construction).
    ///
    /// `nil` means "not yet materialized" — an Optional rather than an empty
    /// array + flag because it is the honest representation AND because
    /// reflection-based storage vetting (the scalar twin's
    /// no-retained-vectors test) would false-positive on an empty array:
    /// Swift's runtime lets an empty `[[Set<Int>]]` conditionally cast to any
    /// array type, `[Vector]` included. Once materialized, `remove(id:)`
    /// maintains the map incrementally.
    private var inEdges: [[Set<Int>]]?

    var nodeLevels: [Int]
    var entryPointNode: Int?
    var maxLevel: Int

    // ── PQ Code Storage ──────────────────────────────────────────────
    //
    // Instead of [Vector], we store M-byte PQ codes per node.
    // Flat layout: codes[i] has M UInt8 values for node i.

    var codes: [[UInt8]]

    // ── Original Vector Retention (ADR-012) ──────────────────────────
    //
    // Present iff the index was built with retainOriginals: true (or loaded
    // from a PQHW v2 file that carried originals). Slot-aligned with `codes`:
    // originals[i] is the full-precision vector PQ-encoded into codes[i].
    // Tombstoned slots keep their (never-read) original until save-time
    // compaction drops them alongside every other per-slot section.

    var originals: [Vector]?

    // ── ID & Metadata ────────────────────────────────────────────────

    var nodeToUUID: [UUID]
    var uuidToNode: [UUID: Int]
    var metadata: [Data?]

    /// The number of slots in the index, **including tombstoned (removed) nodes**.
    /// Use `liveCount` to get the number of searchable vectors.
    /// Equal to `liveCount` right after `build(...)`; diverges after `remove(id:)`.
    public var count: Int { codes.count }

    /// The number of live (non-tombstoned) vectors available for search.
    public var liveCount: Int { uuidToNode.count }

    // ── Memory Statistics ────────────────────────────────────────────

    /// Approximate memory used by PQ codes in bytes.
    public var codeStorageBytes: Int {
        codes.count * quantizer.config.subspaceCount
    }

    /// What a full-precision vector store would cost (for comparison).
    public var equivalentFullPrecisionBytes: Int {
        codes.count * dimension * MemoryLayout<Float>.size
    }

    /// Whether full-precision originals are retained for reranking (ADR-012).
    public var retainsOriginals: Bool { originals != nil }

    /// Approximate memory used by retained original vectors, in bytes.
    /// Zero unless the index was built with `retainOriginals: true`.
    public var originalStorageBytes: Int {
        (originals?.count ?? 0) * dimension * MemoryLayout<Float>.size
    }

    /// Memory savings ratio (full precision / actual vector storage).
    ///
    /// Retained originals count against the savings: an index built with
    /// `retainOriginals: true` stores originals **plus** codes, so this
    /// drops below 1.0 — retention trades the compression story for
    /// reranked recall (ADR-012).
    public var memorySavingsRatio: Float {
        let stored = codeStorageBytes + originalStorageBytes
        guard stored > 0 else { return 0 }
        return Float(equivalentFullPrecisionBytes) / Float(stored)
    }

    // ── Initialization ───────────────────────────────────────────────

    /// Restores a quantized index from its components (used by persistence and build).
    public init(
        dimension: Int,
        hnswConfig: HNSWConfiguration,
        quantizer: ProductQuantizer,
        layers: [[[Int]]],
        nodeLevels: [Int],
        entryPointNode: Int?,
        maxLevel: Int,
        codes: [[UInt8]],
        nodeToUUID: [UUID],
        uuidToNode: [UUID: Int],
        metadata: [Data?],
        originals: [Vector]? = nil
    ) {
        if let originals {
            precondition(originals.count == codes.count,
                "originals must be slot-aligned with codes (\(originals.count) vs \(codes.count))")
            precondition(originals.allSatisfy { $0.dimension == dimension },
                "every retained original must have dimension \(dimension)")
        }
        self.dimension = dimension
        self.hnswConfig = hnswConfig
        self.quantizer = quantizer
        self.layers = layers
        // `inEdges` is deliberately NOT built here: this initializer accepts
        // arbitrary components (the corruption suite feeds it out-of-bounds
        // adjacency to prove load(from:) throws typed), so any indexing of
        // `layers` would trap before the load-time validation gate. The map
        // is materialized lazily by `ensureInEdges()` — see its property doc.
        self.nodeLevels = nodeLevels
        self.entryPointNode = entryPointNode
        self.maxLevel = maxLevel
        self.codes = codes
        self.nodeToUUID = nodeToUUID
        self.uuidToNode = uuidToNode
        self.metadata = metadata
        self.originals = originals
    }

    /// Builds the reverse-adjacency transpose of a layer set: for every edge
    /// `u → v` on layer `l`, records `u` in the in-set of `v`. Strict by
    /// design — an out-of-bounds neighbor traps, so callers must only pass
    /// well-formed layers (see `ensureInEdges()`); tolerating bad indices here
    /// would mask corruption instead of surfacing it at the load gate.
    private static func transpose(of layers: [[[Int]]]) -> [[Set<Int>]] {
        var result: [[Set<Int>]] = layers.map { Array(repeating: Set<Int>(), count: $0.count) }
        for (l, layer) in layers.enumerated() {
            for (from, neighbors) in layer.enumerated() {
                for to in neighbors {
                    result[l][to].insert(from)
                }
            }
        }
        return result
    }

    /// Materializes `inEdges` from `layers` on first use. Every caller sits
    /// behind the graph-validity gates — `build(...)` produces a well-formed
    /// graph and `load(from:)` bounds-checks every neighbor before
    /// constructing the index — so the strict transpose is safe here even
    /// though it is not safe in the memberwise initializer.
    private func ensureInEdges() {
        guard inEdges == nil else { return }
        inEdges = Self.transpose(of: layers)
    }

    // ── Build ────────────────────────────────────────────────────────

    /// Builds a quantized HNSW index from a set of vectors.
    ///
    /// This is a two-phase process:
    /// 1. Inserts all vectors into a full-precision `HNSWIndex` for accurate graph construction
    /// 2. Trains a `ProductQuantizer` on the vectors and encodes them to PQ codes
    ///
    /// The full vectors are discarded after building — only the graph and PQ
    /// codes are kept — unless `retainOriginals` is `true`, in which case the
    /// originals are kept slot-aligned with the codes so that
    /// `search(query:k:efSearch:rerankDepth:filter:)` can re-score the top
    /// ADC candidates exactly. Retention costs the full `4 * dimension`
    /// bytes/vector again, forfeiting the compression story (ADR-012).
    ///
    /// Duplicate ids follow `HNSWIndex`'s replace-on-duplicate semantics: the
    /// *last* vector for a given id wins, and the built index contains one node
    /// per distinct id (`count == liveCount`, both equal to the distinct-id count).
    ///
    /// - Parameters:
    ///   - vectors: The vectors to index.
    ///   - ids: UUID for each vector. Must have the same count as `vectors`.
    ///   - metadata: Optional metadata for each vector.
    ///   - dimension: Vector dimension.
    ///   - hnswConfig: Configuration for the HNSW graph.
    ///   - pqConfig: Configuration for product quantization.
    ///   - retainOriginals: Keep the full-precision vectors for exact
    ///     reranking. Defaults to `false` (codes only). See ADR-012.
    /// - Returns: A `QuantizedHNSWIndex` ready for search.
    /// - Throws: If PQ training fails or dimensions are invalid.
    public static func build(
        vectors: [Vector],
        ids: [UUID],
        metadata: [Data?]? = nil,
        dimension: Int,
        hnswConfig: HNSWConfiguration = HNSWConfiguration(),
        pqConfig: PQConfiguration,
        retainOriginals: Bool = false
    ) async throws -> QuantizedHNSWIndex {
        precondition(vectors.count == ids.count, "vectors and ids must have the same count")
        if let md = metadata {
            precondition(md.count == vectors.count, "metadata must have the same count as vectors")
        }

        // Phase 1: Build full-precision HNSW for accurate graph construction.
        let fullIndex = HNSWIndex(
            dimension: dimension,
            metric: EuclideanDistance(),
            config: hnswConfig
        )

        let metadataArray = metadata ?? Array(repeating: nil, count: vectors.count)
        for i in 0..<vectors.count {
            try await fullIndex.add(vectors[i], id: ids[i], metadata: metadataArray[i])
        }

        // Phase 2: Train PQ on the vectors.
        let pq = try ProductQuantizer.train(
            vectors: vectors,
            config: pqConfig
        )

        // Phase 3: Extract graph structure from the full index.
        // persistenceSnapshot() COMPACTS when slots were tombstoned (e.g. a
        // duplicate id replaced an earlier vector), renumbering every node.
        let snapshot = try await fullIndex.persistenceSnapshot()

        // Phase 4: Encode PQ codes from the snapshot's vectors — NOT from the
        // raw input — so codes/metadata stay positionally aligned with the
        // snapshot's node order (`snapshot.layers` / `snapshot.nodeToUUID`).
        // Encoding from the input would silently shift every code and metadata
        // payload after a compacted slot.
        let pqCodes = snapshot.vectors.map { pq.encode($0) }

        // Build the UUID lookup.
        var uuidToNode: [UUID: Int] = [:]
        for (i, uuid) in snapshot.nodeToUUID.enumerated() {
            uuidToNode[uuid] = i
        }

        return QuantizedHNSWIndex(
            dimension: dimension,
            hnswConfig: hnswConfig,
            quantizer: pq,
            layers: snapshot.layers,
            nodeLevels: snapshot.nodeLevels,
            entryPointNode: snapshot.entryPointNode,
            maxLevel: snapshot.maxLevel,
            codes: pqCodes,
            nodeToUUID: snapshot.nodeToUUID,
            uuidToNode: uuidToNode,
            metadata: snapshot.metadata,
            // Retained originals come from the SAME snapshot the codes were
            // encoded from, so they share its compacted node order and stay
            // slot-aligned even after duplicate-id replacement.
            originals: retainOriginals ? snapshot.vectors : nil
        )
    }

    // ── Search ───────────────────────────────────────────────────────

    /// Searches for the k nearest vectors to the query using asymmetric distance computation.
    ///
    /// The search navigates the HNSW graph using ADC distances:
    /// - Precomputes a distance table (M × 256) from the query to all centroids
    /// - Each node's distance is computed as M table lookups (O(M) per node)
    ///
    /// - Parameters:
    ///   - query: The full-precision query vector.
    ///   - k: Number of results to return.
    ///   - efSearch: Beam width override. Defaults to the HNSW config value.
    ///   - filter: Optional predicate to exclude vectors by ID. Applied
    ///     graph-aware DURING the layer-0 ADC beam (ADR-008 second addendum).
    /// - Returns: Up to `k` results, sorted by ascending distance.
    ///
    /// - Important: If `query.dimension` does not match the index dimension,
    ///   this returns `[]` rather than throwing. If you get empty results from
    ///   a non-empty index, check that the query embedder produces vectors of
    ///   the index's dimension.
    ///
    /// - Note: When the index retains originals (`retainOriginals: true` at
    ///   build), this entry point reranks **by default** at depth `4 * k` —
    ///   retention is an explicit opt-in to recall recovery, so the default
    ///   search uses it. Pass `rerankDepth: 0` to the throwing overload for
    ///   pure-ADC results, or any other depth to tune the recall/latency
    ///   trade (ADR-012). Indexes without originals are unaffected.
    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        // Auto-rerank depth: 4*k when originals are retained, off otherwise.
        // Cannot throw — the rerank path only engages when originals exist.
        let autoDepth = originals != nil ? 4 * max(k, 0) : 0
        return searchImpl(
            query: query, k: k, efSearch: efSearch,
            rerankDepth: autoDepth, filter: filter
        )
    }

    /// Searches with explicit control over full-precision reranking (ADR-012).
    ///
    /// ADC navigates the graph as usual, but the layer-0 beam is widened to
    /// at least `rerankDepth` candidates; the top `rerankDepth` live,
    /// filter-passing candidates (by ADC distance) are then re-scored with
    /// **exact Euclidean distance** against the retained originals before
    /// sorting and truncating to `k`. Reranked results carry exact L2
    /// distances (the same scale as `HNSWIndex`); with reranking disabled,
    /// distances remain squared-L2 ADC approximations as before.
    ///
    /// A positive `rerankDepth` smaller than `k` caps the result count at
    /// `rerankDepth`: only re-scored candidates are returned, never padded
    /// with ADC-scale distances (mixing the two scales would corrupt
    /// ordering). Use `rerankDepth >= k` (typically `4 * k`) in practice.
    ///
    /// - Parameters:
    ///   - query: The full-precision query vector.
    ///   - k: Number of results to return.
    ///   - efSearch: Beam width override. Defaults to the HNSW config value;
    ///     reranking raises the effective beam to at least `rerankDepth`.
    ///   - rerankDepth: How many top ADC candidates to re-score exactly.
    ///     `nil` or any value `<= 0` disables reranking — results are then
    ///     byte-identical to the pure-ADC path. A common starting point is
    ///     `4 * k`.
    ///   - filter: Optional predicate to exclude vectors by ID. Applied
    ///     graph-aware DURING the layer-0 ADC beam (ADR-008 second addendum):
    ///     the beam routes through non-matching nodes and adaptively widens so
    ///     selective filters still surface enough matching candidates. Only
    ///     live, filter-passing candidates count toward `rerankDepth`, which
    ///     the exact re-score then honours — composing with `rerankDepth`
    ///     exactly as the retired post-filter path did.
    /// - Returns: Up to `k` results, sorted by ascending distance.
    /// - Throws: `QuantizedIndexError.originalsNotRetained` if a positive
    ///   `rerankDepth` is requested but the index was built without
    ///   `retainOriginals: true`. Reranking without originals is impossible —
    ///   there is nothing full-precision to re-score against — and silently
    ///   falling back to ADC would hide a ~30%-recall misconfiguration, so
    ///   this fails fast instead (ADR-012).
    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        rerankDepth: Int?,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) throws -> [SearchResult] {
        let depth = rerankDepth ?? 0
        if depth > 0, originals == nil {
            throw QuantizedIndexError.originalsNotRetained
        }
        return searchImpl(
            query: query, k: k, efSearch: efSearch,
            rerankDepth: depth, filter: filter
        )
    }

    /// Shared search core. `rerankDepth <= 0` is the pure-ADC path; callers
    /// guarantee `originals != nil` whenever `rerankDepth > 0`.
    private func searchImpl(
        query: Vector,
        k: Int,
        efSearch: Int?,
        rerankDepth: Int,
        filter: (@Sendable (UUID) -> Bool)?
    ) -> [SearchResult] {
        guard query.dimension == dimension else { return [] }
        guard let ep = entryPointNode else { return [] }
        guard k > 0 else { return [] }

        let reranking = rerankDepth > 0 && originals != nil

        // Overscan: the layer-0 beam must surface at least rerankDepth
        // candidates for the exact re-scoring pass to choose from.
        let ef = max(efSearch ?? hnswConfig.efSearch, k, reranking ? rerankDepth : 0)

        // Precompute the ADC distance table once.
        let distTable = quantizer.buildDistanceTable(query: query)

        var currentEntry = ep

        // Phase 1: Greedy descent on upper layers (ef=1).
        for level in stride(from: maxLevel, through: 1, by: -1) {
            let nearest = searchLayerADC(
                distTable: distTable, entryPoint: currentEntry, ef: 1, layer: level
            )
            if let closest = nearest.first {
                currentEntry = closest.node
            }
        }

        // Phase 2: Full beam search on layer 0.
        //
        // Filtered queries take the graph-aware path (ADR-008 second
        // addendum): the predicate is applied DURING the beam, so rejected
        // nodes route but never occupy result candidacy, and the beam width
        // adapts to the observed acceptance rate so selective filters still
        // surface enough live, matching candidates. Unfiltered queries keep
        // the original filter-blind ADC beam below — that path is unchanged
        // (zero overhead). When reranking, the adaptive target is
        // `rerankDepth`, not `k`: the filtered beam surfaces up to
        // `rerankDepth` accepted candidates so the exact re-score below sees
        // the same depth the retired post-filter pipeline fed it — composing
        // with `rerankDepth` exactly as post-filter did, on filtered
        // candidates only (ADR-012 × ADR-008).
        let candidates: [(node: Int, distance: Float)]
        if let filter = filter {
            candidates = searchLayer0FilteredADC(
                distTable: distTable, entryPoint: currentEntry, ef: ef,
                target: reranking ? rerankDepth : k, filter: filter
            )
        } else {
            candidates = searchLayerADC(
                distTable: distTable, entryPoint: currentEntry, ef: ef, layer: 0
            )
        }

        // Build results. The materialization loops below re-apply the
        // identity-liveness gate and the predicate; for filtered candidates
        // those are already satisfied (idempotent), so a single, unchanged
        // materialization path serves both beams.
        var results: [SearchResult] = []
        if reranking, let originals {
            // Phase 3 (rerank): candidates arrive sorted by ascending ADC
            // distance, so the first rerankDepth live, filter-passing entries
            // are the ADC top-N. Re-score those exactly on the originals;
            // the final sort below then ranks by exact distance.
            let metric = EuclideanDistance()
            var taken = 0
            for (node, _) in candidates {
                guard taken < rerankDepth else { break }
                let uuid = nodeToUUID[node]
                // Identity check: a slot is live only if its UUID maps back
                // to it (a re-added UUID maps to a newer node, tombstoning
                // this slot).
                guard uuidToNode[uuid] == node else { continue }
                if let filter = filter, !filter(uuid) { continue }
                taken += 1
                results.append(SearchResult(
                    id: uuid,
                    distance: metric.distance(query, originals[node]),
                    metadata: metadata[node]
                ))
            }
        } else {
            for (node, distance) in candidates {
                let uuid = nodeToUUID[node]
                // Identity check: a slot is live only if its UUID maps back to it
                // (a re-added UUID maps to a newer node, tombstoning this slot).
                guard uuidToNode[uuid] == node else { continue }
                if let filter = filter, !filter(uuid) { continue }
                results.append(SearchResult(
                    id: uuid,
                    distance: distance,
                    metadata: metadata[node]
                ))
            }
        }

        results.sort()
        if results.count > k {
            results = Array(results.prefix(k))
        }

        return results
    }

    // ── Remove ───────────────────────────────────────────────────────

    /// Removes a vector by its ID (tombstone — graph edges are disconnected).
    ///
    /// The graph was built by a full-precision `HNSWIndex`, whose insertion-time
    /// pruning is one-sided: a live node can hold an edge INTO this node without
    /// a reciprocal edge, so the removed node's own neighbor list does not name
    /// every node pointing at it. The maintained reverse-adjacency map
    /// (`inEdges`) lists exactly those nodes, so the incoming-edge cleanup is
    /// O(in-degree) per layer instead of the former O(E_l) whole-layer sweep.
    /// Unlike `HNSWIndex.remove(id:)`, no neighbor reconnection is performed (the
    /// full vectors the diversity heuristic needs were discarded at build time);
    /// heavy removal workloads should rebuild the index for best recall.
    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }
        let level = nodeLevels[node]
        ensureInEdges()

        for l in 0...level where l < layers.count {
            // Step 1: reverse-adjacency cleanup. `inEdges[l][node]` is exactly
            // the set of nodes whose layer-`l` list points at `node`, so only
            // those lists are touched — O(in-degree), not a whole-layer sweep.
            // (`inEdges!` is safe: `ensureInEdges()` above materialized it.)
            for from in inEdges![l][node] {
                layers[l][from].removeAll { $0 == node }
            }
            inEdges![l][node] = []
            // Step 2: drop the node's own out-edges, keeping each target's
            // in-set in sync so `inEdges` stays the exact transpose of `layers`.
            for to in layers[l][node] {
                inEdges![l][to].remove(node)
            }
            layers[l][node] = []
        }

        uuidToNode.removeValue(forKey: id)

        if entryPointNode == node {
            entryPointNode = nil
            maxLevel = -1
            for (n, nUUID) in nodeToUUID.enumerated() {
                guard uuidToNode[nUUID] == n else { continue }
                if nodeLevels[n] > maxLevel {
                    maxLevel = nodeLevels[n]
                    entryPointNode = n
                }
            }
        }

        return true
    }

    /// The pre-optimization O(E_l)-per-layer removal (a full reverse-edge
    /// sweep), kept **internal** solely so the differential test harness can
    /// prove the new `inEdges`-based `remove(id:)` produces byte-identical graph
    /// state. It deliberately does NOT maintain `inEdges` (a control index that
    /// uses only this method never has its map inspected). Not public API; do
    /// not call on a production removal path.
    @discardableResult
    internal func removeUsingFullSweep(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }
        let level = nodeLevels[node]

        for l in 0...level where l < layers.count {
            // Full sweep: edges into this node can exist outside its own list.
            for n in layers[l].indices where n != node {
                layers[l][n].removeAll { $0 == node }
            }
            layers[l][node] = []
        }

        uuidToNode.removeValue(forKey: id)

        if entryPointNode == node {
            entryPointNode = nil
            maxLevel = -1
            for (n, nUUID) in nodeToUUID.enumerated() {
                guard uuidToNode[nUUID] == n else { continue }
                if nodeLevels[n] > maxLevel {
                    maxLevel = nodeLevels[n]
                    entryPointNode = n
                }
            }
        }

        return true
    }

    // ── ADC-based Layer Search ───────────────────────────────────────
    //
    // Same beam search algorithm as HNSWIndex.searchLayer, but uses
    // precomputed ADC distance table instead of full-precision metric.

    private func searchLayerADC(
        distTable: ProductQuantizer.DistanceTable,
        entryPoint: Int,
        ef: Int,
        layer: Int
    ) -> [(node: Int, distance: Float)] {
        let epDist = quantizer.asymmetricDistance(table: distTable, codes: codes[entryPoint])

        var candidates = Heap<(node: Int, distance: Float)>(comparator: { $0.distance < $1.distance })
        candidates.push((entryPoint, epDist))

        var results = Heap<(node: Int, distance: Float)>(comparator: { $0.distance > $1.distance })
        results.push((entryPoint, epDist))

        var visited = Set<Int>([entryPoint])

        while !candidates.isEmpty {
            let nearest = candidates.pop()!

            if let furthest = results.peek(), nearest.distance > furthest.distance {
                break
            }

            for neighbor in layers[layer][nearest.node] {
                guard !visited.contains(neighbor) else { continue }
                visited.insert(neighbor)

                let dist = quantizer.asymmetricDistance(table: distTable, codes: codes[neighbor])

                let shouldAdd: Bool
                if results.count < ef {
                    shouldAdd = true
                } else if let furthest = results.peek(), dist < furthest.distance {
                    shouldAdd = true
                } else {
                    shouldAdd = false
                }

                if shouldAdd {
                    candidates.push((neighbor, dist))
                    results.push((neighbor, dist))
                    if results.count > ef {
                        results.pop()
                    }
                }
            }
        }

        var output: [(node: Int, distance: Float)] = []
        output.reserveCapacity(results.count)
        while let item = results.pop() {
            output.append(item)
        }
        output.reverse()
        return output
    }

    // ── Liveness ──────────────────────────────────────────────────────

    /// A slot is live iff its UUID still maps back to it (a re-added UUID
    /// maps to a newer node, tombstoning the old slot). Mirrors
    /// `HNSWIndex.isLiveNode` so the filtered beam's liveness gate matches
    /// the full-precision port and the post-filter materialization contract.
    private func isLiveNode(_ node: Int) -> Bool {
        uuidToNode[nodeToUUID[node]] == node
    }

    // ── Test Hooks (internal) ─────────────────────────────────────────
    //
    // The reverse-adjacency map is derived state, so tests assert two things:
    // (1) `inEdges` stays the exact transpose of `layers` through every remove
    // and across a save/load rebuild, and (2) the O(in-degree) `remove(id:)`
    // leaves the graph byte-identical to the retired O(E_l) full sweep
    // (`removeUsingFullSweep`), which is kept solely to drive that proof.

    /// Structural invariant: `inEdges` is exactly the transpose of `layers` —
    /// every edge `u → v` on layer `l` is mirrored by `v`'s in-set containing
    /// `u`, and nothing else. O(total edges). Materializes the map first, so
    /// on a fresh index this checks `transpose(of:)` against the inline
    /// expectation below; after removes it checks that the incremental
    /// maintenance in `remove(id:)` kept the map in exact sync.
    internal var reverseAdjacencyIsConsistent: Bool {
        ensureInEdges()
        guard let inEdges, inEdges.count == layers.count else { return false }
        for l in layers.indices {
            guard inEdges[l].count == layers[l].count else { return false }
            var expected = Array(repeating: Set<Int>(), count: layers[l].count)
            for (from, neighbors) in layers[l].enumerated() {
                for to in neighbors {
                    expected[to].insert(from)
                }
            }
            if expected != inEdges[l] { return false }
        }
        return true
    }

    /// Whether any adjacency list still references a tombstoned slot — i.e.
    /// `remove(id:)` left a dangling incoming edge. Proves the O(in-degree)
    /// cleanup clears every edge the full sweep would have.
    internal var hasDanglingEdges: Bool {
        for layer in layers {
            for neighbors in layer {
                for neighbor in neighbors where !isLiveNode(neighbor) {
                    return true
                }
            }
        }
        return false
    }

    /// The graph state `remove(id:)` mutates, captured for exact differential
    /// equality between the `inEdges` path and the retained full-sweep path.
    internal struct GraphFingerprint: Equatable {
        let layers: [[[Int]]]
        let nodeLevels: [Int]
        let entryPointNode: Int?
        let maxLevel: Int
        let nodeToUUID: [UUID]
        let uuidToNode: [UUID: Int]
    }

    internal var graphFingerprint: GraphFingerprint {
        GraphFingerprint(
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPointNode,
            maxLevel: maxLevel,
            nodeToUUID: nodeToUUID,
            uuidToNode: uuidToNode
        )
    }

    // ── Core: Graph-Aware Filtered Beam on Layer 0 (ADR-008 2nd addendum) ─
    //
    // Ports `HNSWIndex.searchLayer0Filtered` onto the ADC scoring path.
    // Differs from `searchLayerADC` in two ways:
    //   1. Result candidacy: only live, predicate-passing nodes enter the
    //      result heap. Rejected nodes (tombstoned or filtered out) still
    //      join the candidate frontier, so the beam routes THROUGH them
    //      toward matching regions instead of spending result slots on them.
    //   2. Adaptive widening: the effective ef is recomputed from the
    //      acceptance rate observed so far — roughly target / rate, add-one
    //      smoothed, clamped to [ef, efCap] — so selective predicates keep
    //      the beam exploring until `target` matching results are plausible.
    //      `target` is `k` for the pure-ADC path and `rerankDepth` when
    //      reranking, so the exact re-score is fed the same candidate depth
    //      post-filter fed it. When (nearly) every node passes, the
    //      recomputed value stays at `ef` and the beam behaves like the
    //      unfiltered one.
    //
    // Liveness is checked BEFORE the predicate (the filter is never invoked
    // for tombstoned slots), matching the post-filter path's contract.
    //
    // Termination: the beam stops early only once the result heap holds
    // `effectiveEf` ACCEPTED nodes and the nearest frontier candidate is
    // farther than the worst of them. With fewer matching nodes reachable
    // than `effectiveEf`, the beam explores the whole connected component —
    // selective filters trade latency for filled, high-recall results
    // (worst case O(liveCount) ADC evaluations), the same trade the
    // full-precision port makes.

    private func searchLayer0FilteredADC(
        distTable: ProductQuantizer.DistanceTable,
        entryPoint: Int,
        ef: Int,
        target: Int,
        filter: (UUID) -> Bool
    ) -> [(node: Int, distance: Float)] {
        let efCap = max(ef, min(liveCount, 16 * max(ef, target)))
        var effectiveEf = ef

        // Acceptance bookkeeping counts predicate *evaluations*, not visits:
        // far nodes that never pass frontier admission are never evaluated.
        var evaluated = 0
        var accepted = 0

        func acceptsIntoResults(_ node: Int) -> Bool {
            evaluated += 1
            guard isLiveNode(node) else { return false }
            guard filter(nodeToUUID[node]) else { return false }
            accepted += 1
            return true
        }

        // effectiveEf ≈ target / acceptance-rate, add-one smoothed so an
        // early run of rejections widens gradually rather than jumping to the
        // cap. Recomputed (not monotonic): the estimate self-corrects.
        func rewiden() {
            let rate = Double(accepted + 1) / Double(evaluated + 1)
            let needed = Int((Double(target) / rate).rounded(.up))
            effectiveEf = min(efCap, max(ef, needed))
        }

        let epDist = quantizer.asymmetricDistance(table: distTable, codes: codes[entryPoint])

        var candidates = Heap<(node: Int, distance: Float)>(comparator: { $0.distance < $1.distance })
        candidates.push((entryPoint, epDist))

        var results = Heap<(node: Int, distance: Float)>(comparator: { $0.distance > $1.distance })
        if acceptsIntoResults(entryPoint) {
            results.push((entryPoint, epDist))
        }
        rewiden()

        var visited = Set<Int>([entryPoint])

        while !candidates.isEmpty {
            let nearest = candidates.pop()!

            // Early exit only once the (widened) beam is full of accepted
            // nodes — an under-filled beam keeps routing through rejects.
            if results.count >= effectiveEf,
               let furthest = results.peek(), nearest.distance > furthest.distance {
                break
            }

            for neighbor in layers[0][nearest.node] {
                guard !visited.contains(neighbor) else { continue }
                visited.insert(neighbor)

                let dist = quantizer.asymmetricDistance(table: distTable, codes: codes[neighbor])

                let shouldExplore: Bool
                if results.count < effectiveEf {
                    shouldExplore = true
                } else if let furthest = results.peek(), dist < furthest.distance {
                    shouldExplore = true
                } else {
                    shouldExplore = false
                }
                guard shouldExplore else { continue }

                candidates.push((neighbor, dist))
                if acceptsIntoResults(neighbor) {
                    results.push((neighbor, dist))
                }
                rewiden()
                while results.count > effectiveEf {
                    results.pop()
                }
            }
        }

        var output: [(node: Int, distance: Float)] = []
        output.reserveCapacity(results.count)
        while let item = results.pop() {
            output.append(item)
        }
        output.reverse()
        return output
    }
}
