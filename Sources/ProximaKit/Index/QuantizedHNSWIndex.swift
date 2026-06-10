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
        self.nodeLevels = nodeLevels
        self.entryPointNode = entryPointNode
        self.maxLevel = maxLevel
        self.codes = codes
        self.nodeToUUID = nodeToUUID
        self.uuidToNode = uuidToNode
        self.metadata = metadata
        self.originals = originals
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
    ///   - filter: Optional predicate to exclude vectors by ID.
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
    ///     before candidates count toward `rerankDepth` (post-filter,
    ///     ADR-008).
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
        let candidates = searchLayerADC(
            distTable: distTable, entryPoint: currentEntry, ef: ef, layer: 0
        )

        // Build results.
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
    /// pruning is one-sided: a live node can hold an edge to this node without a
    /// reciprocal edge. A full reverse-edge sweep — O(E_l) per layer — is therefore
    /// required to leave no dangling references. Unlike `HNSWIndex.remove(id:)`,
    /// no neighbor reconnection is performed (the full vectors needed for the
    /// diversity heuristic were discarded at build time); heavy removal workloads
    /// should rebuild the index for best recall.
    @discardableResult
    public func remove(id: UUID) -> Bool {
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
}
