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

import Foundation

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

    // ── ID & Metadata ────────────────────────────────────────────────

    var nodeToUUID: [UUID]
    var uuidToNode: [UUID: Int]
    var metadata: [Data?]

    /// The number of vectors in the index.
    public var count: Int { codes.count }

    /// The number of live (non-tombstoned) vectors.
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

    /// Memory savings ratio (full precision / PQ storage).
    public var memorySavingsRatio: Float {
        guard codeStorageBytes > 0 else { return 0 }
        return Float(equivalentFullPrecisionBytes) / Float(codeStorageBytes)
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
        metadata: [Data?]
    ) {
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
    }

    // ── Build ────────────────────────────────────────────────────────

    /// Builds a quantized HNSW index from a set of vectors.
    ///
    /// This is a two-phase process:
    /// 1. Inserts all vectors into a full-precision `HNSWIndex` for accurate graph construction
    /// 2. Trains a `ProductQuantizer` on the vectors and encodes them to PQ codes
    ///
    /// The full vectors are discarded after building — only the graph and PQ codes are kept.
    ///
    /// - Parameters:
    ///   - vectors: The vectors to index.
    ///   - ids: UUID for each vector. Must have the same count as `vectors`.
    ///   - metadata: Optional metadata for each vector.
    ///   - dimension: Vector dimension.
    ///   - hnswConfig: Configuration for the HNSW graph.
    ///   - pqConfig: Configuration for product quantization.
    /// - Returns: A `QuantizedHNSWIndex` ready for search.
    /// - Throws: If PQ training fails or dimensions are invalid.
    public static func build(
        vectors: [Vector],
        ids: [UUID],
        metadata: [Data?]? = nil,
        dimension: Int,
        hnswConfig: HNSWConfiguration = HNSWConfiguration(),
        pqConfig: PQConfiguration
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

        // Phase 3: Encode all vectors to PQ codes.
        let pqCodes = vectors.map { pq.encode($0) }

        // Phase 4: Extract graph structure from the full index.
        let snapshot = try await fullIndex.persistenceSnapshot()

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
            metadata: metadataArray
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
        for (node, distance) in candidates {
            let uuid = nodeToUUID[node]
            guard uuidToNode[uuid] != nil else { continue }
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
    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }
        let level = nodeLevels[node]

        for l in 0...level where l < layers.count {
            for neighbor in layers[l][node] {
                layers[l][neighbor].removeAll { $0 == node }
            }
            layers[l][node] = []
        }

        uuidToNode.removeValue(forKey: id)

        if entryPointNode == node {
            entryPointNode = nil
            maxLevel = -1
            for (n, nUUID) in nodeToUUID.enumerated() {
                guard uuidToNode[nUUID] != nil else { continue }
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
