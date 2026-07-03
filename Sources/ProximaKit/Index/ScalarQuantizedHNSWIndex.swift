// ScalarQuantizedHNSWIndex.swift
// ProximaKit
//
// Memory-efficient HNSW index using INT8 scalar quantization.
//
// Stores the HNSW graph structure with Int8 codes + per-vector scales instead
// of full Float32 vectors. Search dequantizes each candidate on the fly and
// applies the configured distance metric — so unlike QuantizedHNSWIndex's
// L2-only ADC, this index supports cosine, euclidean, dot product, Manhattan,
// Chebyshev, and Bray-Curtis. Hamming is the exception: its exact bit-equality
// semantics are destroyed by lossy reconstruction — use the full-precision
// HNSWIndex for Hamming workloads. See ADR-007.
//
// Typical memory reduction: 384d Float32 (1536 bytes) → 388 bytes ≈ 3.96×,
// at a ~1-2% Recall@10 cost. Graph overhead (~200 bytes/node for M=16) is
// shared with the standard index.
//
// Build workflow:
//   1. Insert all vectors into a full-precision HNSWIndex (accurate graph)
//   2. Encode each vector to Int8 codes + a Float32 scale (no training phase)
//   3. Discard the original vectors — search uses dequantized candidates only

import Foundation

/// A memory-efficient HNSW index that stores INT8 codes instead of full vectors.
///
/// Build a scalar-quantized index from your vectors (no training required):
/// ```swift
/// let sqIndex = try await ScalarQuantizedHNSWIndex.build(
///     vectors: vectors,
///     ids: ids,
///     dimension: 384,
///     hnswConfig: HNSWConfiguration(),
///     metric: .cosine
/// )
///
/// let results = await sqIndex.search(query: queryVec, k: 10)
/// // Memory per vector: 388 bytes instead of 1536 bytes
/// ```
public actor ScalarQuantizedHNSWIndex {

    // ── Configuration ────────────────────────────────────────────────

    /// The full vector dimension this index was built for.
    public nonisolated let dimension: Int

    /// HNSW graph configuration.
    nonisolated let hnswConfig: HNSWConfiguration

    /// The configuration this index was created with. Readable without `await`.
    public nonisolated var configuration: HNSWConfiguration { hnswConfig }

    /// The serializable distance metric this index was built (and searches) with.
    public nonisolated let metricType: DistanceMetricType

    /// The metric instance used at search time (always reconstructible from
    /// `metricType`, so persistence can never hit `unserializableMetric`).
    private let metric: any DistanceMetric

    /// The stateless scalar quantizer (dimension only — nothing trained).
    public let quantizer: ScalarQuantizer

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
    /// layers. The scalar-quantized memberwise initializer, by contrast, is
    /// the documented entry point for arbitrary components (the persistence
    /// corruption suite constructs indexes whose adjacency is deliberately out
    /// of bounds to prove `load(from:)` rejects the saved bytes with a typed
    /// `PersistenceError`). Transposing eagerly in the initializer would index
    /// `layers` BEFORE that load-time gate and turn the contracted typed error
    /// into a process trap. Lazily, the map is only ever built after the graph
    /// is known well-formed: post-`build` (graph from a healthy `HNSWIndex`)
    /// or post-`load` (every neighbor bounds-checked before construction).
    ///
    /// `nil` means "not yet materialized" — an Optional rather than an empty
    /// array + flag because it is the honest representation AND because the
    /// reflection-based storage vetting in
    /// `testNoFullPrecisionVectorsRetainedAfterBuild` would false-positive on
    /// an empty array: Swift's runtime lets an empty `[[Set<Int>]]`
    /// conditionally cast to any array type, `[Vector]` included. Once
    /// materialized, `remove(id:)` maintains the map incrementally.
    private var inEdges: [[Set<Int>]]?

    var nodeLevels: [Int]
    var entryPointNode: Int?
    var maxLevel: Int

    // ── INT8 Code Storage ────────────────────────────────────────────
    //
    // Instead of [Vector], we store dimension-byte Int8 codes plus one
    // Float32 scale per node. codes[i] / scales[i] reconstruct node i.

    var codes: [[Int8]]
    var scales: [Float]

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

    /// Approximate memory used by quantized vector storage in bytes:
    /// `dimension` Int8 codes + one Float32 scale per slot.
    public var codeStorageBytes: Int {
        codes.count * (dimension + MemoryLayout<Float>.size)
    }

    /// What a full-precision vector store would cost (for comparison).
    public var equivalentFullPrecisionBytes: Int {
        codes.count * dimension * MemoryLayout<Float>.size
    }

    /// Memory savings ratio (full precision / quantized storage). ≈ 3.96 at 384d.
    public var memorySavingsRatio: Float {
        guard codeStorageBytes > 0 else { return 0 }
        return Float(equivalentFullPrecisionBytes) / Float(codeStorageBytes)
    }

    // ── Initialization ───────────────────────────────────────────────

    /// Restores a scalar-quantized index from its components (used by persistence and build).
    public init(
        dimension: Int,
        hnswConfig: HNSWConfiguration,
        metricType: DistanceMetricType,
        layers: [[[Int]]],
        nodeLevels: [Int],
        entryPointNode: Int?,
        maxLevel: Int,
        codes: [[Int8]],
        scales: [Float],
        nodeToUUID: [UUID],
        uuidToNode: [UUID: Int],
        metadata: [Data?]
    ) {
        self.dimension = dimension
        self.hnswConfig = hnswConfig
        self.metricType = metricType
        self.metric = metricType.makeMetric()
        self.quantizer = ScalarQuantizer(dimension: dimension)
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
        self.scales = scales
        self.nodeToUUID = nodeToUUID
        self.uuidToNode = uuidToNode
        self.metadata = metadata
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

    /// Builds a scalar-quantized HNSW index from a set of vectors.
    ///
    /// This is a two-phase process:
    /// 1. Inserts all vectors into a full-precision `HNSWIndex` (with the same
    ///    metric) for accurate graph construction
    /// 2. Encodes each vector to Int8 codes + a Float32 scale
    ///
    /// There is **no training phase** (contrast with `QuantizedHNSWIndex`,
    /// which trains PQ codebooks). The full vectors are discarded after
    /// building — only the graph, codes, and scales are kept.
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
    ///   - metric: The distance metric to build and search with. Any
    ///     serializable metric except Hamming is supported (unlike PQ's
    ///     L2-only ADC). Hamming compares exact bit equality, which lossy
    ///     reconstruction destroys — use `HNSWIndex` for Hamming workloads.
    /// - Returns: A `ScalarQuantizedHNSWIndex` ready for search.
    /// - Throws: `IndexError.dimensionMismatch` if a vector has the wrong dimension.
    public static func build(
        vectors: [Vector],
        ids: [UUID],
        metadata: [Data?]? = nil,
        dimension: Int,
        hnswConfig: HNSWConfiguration = HNSWConfiguration(),
        metric: DistanceMetricType = .cosine
    ) async throws -> ScalarQuantizedHNSWIndex {
        precondition(vectors.count == ids.count, "vectors and ids must have the same count")
        if let md = metadata {
            precondition(md.count == vectors.count, "metadata must have the same count as vectors")
        }

        // Phase 1: Build full-precision HNSW for accurate graph construction,
        // using the SAME metric the quantized index will search with.
        let fullIndex = HNSWIndex(
            dimension: dimension,
            metric: metric.makeMetric(),
            config: hnswConfig
        )

        let metadataArray = metadata ?? Array(repeating: nil, count: vectors.count)
        for i in 0..<vectors.count {
            try await fullIndex.add(vectors[i], id: ids[i], metadata: metadataArray[i])
        }

        // Phase 2: Extract graph structure from the full index.
        // persistenceSnapshot() COMPACTS when slots were tombstoned (e.g. a
        // duplicate id replaced an earlier vector), renumbering every node.
        let snapshot = try await fullIndex.persistenceSnapshot()

        // Phase 3: Encode codes/scales from the snapshot's vectors — NOT from
        // the raw input — so they stay positionally aligned with the
        // snapshot's node order (`snapshot.layers` / `snapshot.nodeToUUID`).
        // Encoding from the input would silently shift every code and metadata
        // payload after a compacted slot.
        let quantizer = ScalarQuantizer(dimension: dimension)
        let (sqCodes, sqScales) = quantizer.encodeBatch(snapshot.vectors)

        // Build the UUID lookup.
        var uuidToNode: [UUID: Int] = [:]
        for (i, uuid) in snapshot.nodeToUUID.enumerated() {
            uuidToNode[uuid] = i
        }

        return ScalarQuantizedHNSWIndex(
            dimension: dimension,
            hnswConfig: hnswConfig,
            metricType: metric,
            layers: snapshot.layers,
            nodeLevels: snapshot.nodeLevels,
            entryPointNode: snapshot.entryPointNode,
            maxLevel: snapshot.maxLevel,
            codes: sqCodes,
            scales: sqScales,
            nodeToUUID: snapshot.nodeToUUID,
            uuidToNode: uuidToNode,
            metadata: snapshot.metadata
        )
    }

    // ── Search ───────────────────────────────────────────────────────

    /// Searches for the k nearest vectors to the query.
    ///
    /// The search navigates the HNSW graph using query-time reconstruction:
    /// each candidate's Int8 codes are dequantized back to Float32 on the fly
    /// and compared to the full-precision query with the configured metric.
    /// Reconstruction error is bounded by `scale / 2` per component, so recall
    /// typically trails full precision by only ~1-2% (see ADR-007).
    ///
    /// - Parameters:
    ///   - query: The full-precision query vector.
    ///   - k: Number of results to return.
    ///   - efSearch: Beam width override. Defaults to the HNSW config value.
    ///   - filter: Optional predicate to exclude vectors by ID. Applied
    ///     graph-aware DURING the layer-0 reconstruction beam (ADR-008 second
    ///     addendum): the beam routes through non-matching nodes and
    ///     adaptively widens so selective filters still fill k.
    /// - Returns: Up to `k` results, sorted by ascending distance.
    ///
    /// - Important: If `query.dimension` does not match the index dimension,
    ///   this returns `[]` rather than throwing. If you get empty results from
    ///   a non-empty index, check that the query embedder produces vectors of
    ///   the index's dimension.
    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        guard query.dimension == dimension else { return [] }
        guard let ep = entryPointNode else { return [] }
        guard k > 0 else { return [] }

        let ef = max(efSearch ?? hnswConfig.efSearch, k)

        var currentEntry = ep

        // Phase 1: Greedy descent on upper layers (ef=1).
        for level in stride(from: maxLevel, through: 1, by: -1) {
            let nearest = searchLayerSQ(
                query: query, entryPoint: currentEntry, ef: 1, layer: level
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
        // fill k. Unfiltered queries keep the original filter-blind
        // reconstruction beam below — that path is unchanged (zero overhead).
        let candidates: [(node: Int, distance: Float)]
        if let filter = filter {
            candidates = searchLayer0FilteredSQ(
                query: query, entryPoint: currentEntry, ef: ef, k: k, filter: filter
            )
        } else {
            candidates = searchLayerSQ(
                query: query, entryPoint: currentEntry, ef: ef, layer: 0
            )
        }

        // Build results. The materialization loop re-applies the
        // identity-liveness gate and the predicate; for filtered candidates
        // both are already satisfied (idempotent), so one unchanged
        // materialization path serves both beams.
        var results: [SearchResult] = []
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

    // ── Reconstruction-based Layer Search ────────────────────────────
    //
    // Same beam search algorithm as HNSWIndex.searchLayer, but each node's
    // distance is computed against its dequantized vector instead of a
    // stored full-precision vector.

    /// Dequantizes the stored codes for a node back to a Float32 vector.
    private func reconstruct(_ node: Int) -> Vector {
        quantizer.decodeToVector(codes[node], scale: scales[node])
    }

    private func searchLayerSQ(
        query: Vector,
        entryPoint: Int,
        ef: Int,
        layer: Int
    ) -> [(node: Int, distance: Float)] {
        let epDist = metric.distance(query, reconstruct(entryPoint))

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

                let dist = metric.distance(query, reconstruct(neighbor))

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
    // Ports `HNSWIndex.searchLayer0Filtered` onto the dequantize-scoring
    // path. Differs from `searchLayerSQ` in two ways:
    //   1. Result candidacy: only live, predicate-passing nodes enter the
    //      result heap. Rejected nodes (tombstoned or filtered out) still
    //      join the candidate frontier, so the beam routes THROUGH them
    //      toward matching regions instead of spending result slots on them.
    //   2. Adaptive widening: the effective ef is recomputed from the
    //      acceptance rate observed so far — roughly k / rate, add-one
    //      smoothed, clamped to [ef, efCap] — so selective predicates keep
    //      the beam exploring until k matching results are plausible. When
    //      (nearly) every node passes, the recomputed value stays at `ef` and
    //      the beam behaves like the unfiltered one.
    //
    // Liveness is checked BEFORE the predicate (the filter is never invoked
    // for tombstoned slots), matching the post-filter path's contract.
    //
    // Termination: the beam stops early only once the result heap holds
    // `effectiveEf` ACCEPTED nodes and the nearest frontier candidate is
    // farther than the worst of them. With fewer matching nodes reachable
    // than `effectiveEf`, the beam explores the whole connected component —
    // selective filters trade latency for filled, high-recall results (worst
    // case O(liveCount) reconstruction+distance evaluations), the same trade
    // the full-precision port makes.

    private func searchLayer0FilteredSQ(
        query: Vector,
        entryPoint: Int,
        ef: Int,
        k: Int,
        filter: (UUID) -> Bool
    ) -> [(node: Int, distance: Float)] {
        let efCap = max(ef, min(liveCount, 16 * max(ef, k)))
        var effectiveEf = ef

        // Acceptance bookkeeping counts predicate *evaluations*, not visits.
        var evaluated = 0
        var accepted = 0

        func acceptsIntoResults(_ node: Int) -> Bool {
            evaluated += 1
            guard isLiveNode(node) else { return false }
            guard filter(nodeToUUID[node]) else { return false }
            accepted += 1
            return true
        }

        // effectiveEf ≈ k / acceptance-rate, add-one smoothed so an early run
        // of rejections widens gradually rather than jumping to the cap.
        func rewiden() {
            let rate = Double(accepted + 1) / Double(evaluated + 1)
            let needed = Int((Double(k) / rate).rounded(.up))
            effectiveEf = min(efCap, max(ef, needed))
        }

        let epDist = metric.distance(query, reconstruct(entryPoint))

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

                let dist = metric.distance(query, reconstruct(neighbor))

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
