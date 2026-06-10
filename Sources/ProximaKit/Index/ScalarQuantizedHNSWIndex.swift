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
        self.nodeLevels = nodeLevels
        self.entryPointNode = entryPointNode
        self.maxLevel = maxLevel
        self.codes = codes
        self.scales = scales
        self.nodeToUUID = nodeToUUID
        self.uuidToNode = uuidToNode
        self.metadata = metadata
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
    ///     serializable metric is supported (unlike PQ's L2-only ADC).
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
    ///   - filter: Optional predicate to exclude vectors by ID (post-filter,
    ///     ADR-008 — applied while materializing layer-0 candidates).
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
        let candidates = searchLayerSQ(
            query: query, entryPoint: currentEntry, ef: ef, layer: 0
        )

        // Build results.
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
}
