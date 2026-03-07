// HNSWIndex.swift
// ProximaKit
//
// Hierarchical Navigable Small World index for approximate nearest-neighbor search.
// PK-005 implements single-layer NSW (layer 0 only).
// PK-006 will add multi-layer support.
//
// Based on: "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)

import Accelerate
import Foundation

/// Configuration for the HNSW index.
///
/// - `m`: Max connections per node per layer. Higher = better recall, more memory.
///   Default 16 is a good balance for most use cases.
/// - `efConstruction`: Beam width during insertion. Higher = better graph quality,
///   slower build. Default 200 gives high-quality graphs.
/// - `efSearch`: Default beam width during queries. Can be overridden per query.
///   Higher = better recall, slower queries. Default 50 gives >95% recall.
public struct HNSWConfiguration: Sendable {
    public let m: Int
    public let efConstruction: Int
    public let efSearch: Int

    public init(m: Int = 16, efConstruction: Int = 200, efSearch: Int = 50) {
        precondition(m > 0, "M must be positive")
        precondition(efConstruction > 0, "efConstruction must be positive")
        precondition(efSearch > 0, "efSearch must be positive")
        self.m = m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
    }
}

/// Approximate nearest-neighbor index using Navigable Small World graphs.
///
/// Vectors are stored as nodes in a graph. Each node connects to up to `M`
/// neighbors. Search starts at an entry point and greedily traverses the graph,
/// always moving toward the query vector. This achieves O(log n) query time
/// with >95% recall at default settings.
///
/// ```swift
/// let config = HNSWConfiguration(m: 16, efConstruction: 200, efSearch: 50)
/// let index = HNSWIndex(dimension: 384, metric: CosineDistance(), config: config)
/// try await index.add(vector, id: UUID())
/// let results = try await index.search(query: queryVec, k: 10)
/// ```
public actor HNSWIndex: VectorIndex {

    // ── Configuration ─────────────────────────────────────────────────

    public let dimension: Int
    private let metric: any DistanceMetric
    private let config: HNSWConfiguration

    // ── Node Storage ──────────────────────────────────────────────────
    //
    // Nodes are identified by an internal Int index (0, 1, 2, ...).
    // This allows O(1) vector lookup and cache-friendly adjacency lists.
    // The external UUID is mapped separately.
    //
    // adjacency[i] = neighbor indices for node i (max M entries)
    // vectors[i]   = the vector data for node i
    // nodeToUUID[i] = the external UUID for node i
    // uuidToNode    = reverse lookup

    /// Adjacency lists: neighbors[i] = [Int] of neighbor node indices.
    private var adjacency: [[Int]] = []

    /// Vector data for each node, indexed by node ID.
    private var vectors: [Vector] = []

    /// Metadata for each node, indexed by node ID.
    private var metadata: [Data?] = []

    /// Maps internal node index → external UUID.
    private var nodeToUUID: [UUID] = []

    /// Maps external UUID → internal node index.
    private var uuidToNode: [UUID: Int] = [:]

    /// The entry point for graph search (first node inserted).
    private var entryPoint: Int?

    /// The number of vectors in the index.
    public var count: Int { vectors.count }

    // ── Initialization ────────────────────────────────────────────────

    public init(
        dimension: Int,
        metric: any DistanceMetric = CosineDistance(),
        config: HNSWConfiguration = HNSWConfiguration()
    ) {
        precondition(dimension > 0, "Dimension must be positive")
        self.dimension = dimension
        self.metric = metric
        self.config = config
    }

    // ── VectorIndex: Add ──────────────────────────────────────────────

    public func add(_ vector: Vector, id: UUID, metadata: Data? = nil) throws {
        guard vector.dimension == dimension else {
            throw IndexError.dimensionMismatch(expected: dimension, got: vector.dimension)
        }

        // If this UUID already exists, remove it first.
        if uuidToNode[id] != nil {
            _ = remove(id: id)
        }

        let newNode = vectors.count

        // Store the node data.
        vectors.append(vector)
        adjacency.append([])
        self.metadata.append(metadata)
        nodeToUUID.append(id)
        uuidToNode[id] = newNode

        // First node becomes the entry point.
        guard let ep = entryPoint else {
            entryPoint = newNode
            return
        }

        // Find nearest neighbors using greedy search from the entry point.
        let neighbors = searchLayer(
            query: vector,
            entryPoint: ep,
            ef: config.efConstruction
        )

        // Connect to the top M nearest neighbors.
        let topM = Array(neighbors.prefix(config.m))
        for (neighborNode, _) in topM {
            connect(newNode, neighborNode)
        }
    }

    // ── VectorIndex: Search ───────────────────────────────────────────

    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        guard query.dimension == dimension else { return [] }
        guard let ep = entryPoint else { return [] }
        guard k > 0 else { return [] }

        let ef = max(efSearch ?? config.efSearch, k)

        // Search the graph starting from the entry point.
        let candidates = searchLayer(query: query, entryPoint: ep, ef: ef)

        // Build results, applying filter.
        var results: [SearchResult] = []
        for (node, distance) in candidates {
            let uuid = nodeToUUID[node]
            if let filter = filter, !filter(uuid) { continue }
            results.append(SearchResult(
                id: uuid,
                distance: distance,
                metadata: metadata[node]
            ))
        }

        // Sort and trim to k.
        results.sort()
        if results.count > k {
            results = Array(results.prefix(k))
        }

        return results
    }

    // ── VectorIndex: Remove ───────────────────────────────────────────

    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }

        // Disconnect this node from all neighbors.
        for neighbor in adjacency[node] {
            adjacency[neighbor].removeAll { $0 == node }
        }
        adjacency[node] = []

        // Clear the UUID mapping.
        uuidToNode.removeValue(forKey: id)

        // Note: We don't compact arrays (would invalidate all indices).
        // The node slot becomes a "tombstone" — searchLayer skips it
        // because it has no neighbors and won't be found by traversal.
        // For a production implementation, you'd periodically rebuild.

        // If we removed the entry point, pick a new one.
        if entryPoint == node {
            entryPoint = uuidToNode.values.first
        }

        return true
    }

    // ── Core Algorithm: Greedy Search on Single Layer ─────────────────
    //
    // This is Algorithm 2 from the HNSW paper, adapted for a single layer.
    //
    // Starting from the entry point, we greedily explore the graph:
    // 1. Maintain a min-heap of candidates (nodes to explore next)
    // 2. Maintain a max-heap of results (best nodes found so far, capped at ef)
    // 3. Always explore the closest candidate first
    // 4. Stop when the closest candidate is farther than the worst result
    //
    // Returns up to `ef` (node, distance) pairs sorted by distance ascending.

    private func searchLayer(
        query: Vector,
        entryPoint: Int,
        ef: Int
    ) -> [(node: Int, distance: Float)] {
        let epDist = metric.distance(query, vectors[entryPoint])

        // Candidates: min-heap (explore closest first)
        var candidates = Heap<(node: Int, distance: Float)>(comparator: { $0.distance < $1.distance })
        candidates.push((entryPoint, epDist))

        // Results: max-heap (evict the farthest when full)
        var results = Heap<(node: Int, distance: Float)>(comparator: { $0.distance > $1.distance })
        results.push((entryPoint, epDist))

        var visited = Set<Int>([entryPoint])

        while !candidates.isEmpty {
            // Pop the closest unexplored candidate.
            let nearest = candidates.pop()!

            // If the closest candidate is farther than our worst result,
            // we can't improve — stop exploring.
            if let furthest = results.peek(), nearest.distance > furthest.distance {
                break
            }

            // Explore this candidate's neighbors.
            for neighbor in adjacency[nearest.node] {
                guard !visited.contains(neighbor) else { continue }
                visited.insert(neighbor)

                let dist = metric.distance(query, vectors[neighbor])

                // Add to results if it improves them (or results aren't full yet).
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
                        results.pop()  // evict the farthest
                    }
                }
            }
        }

        // Drain the max-heap into an array sorted by distance ascending.
        var output: [(node: Int, distance: Float)] = []
        output.reserveCapacity(results.count)
        while let item = results.pop() {
            output.append(item)
        }
        output.reverse()  // max-heap pops largest first, so reverse for ascending
        return output
    }

    // ── Graph Mutation ────────────────────────────────────────────────

    /// Adds a bidirectional edge between two nodes.
    /// If either node exceeds M connections, prunes to keep the M nearest.
    private func connect(_ a: Int, _ b: Int) {
        // Add edge a → b (if not already connected)
        if !adjacency[a].contains(b) {
            adjacency[a].append(b)
        }
        // Add edge b → a
        if !adjacency[b].contains(a) {
            adjacency[b].append(a)
        }

        // Prune if over capacity.
        pruneIfNeeded(a)
        pruneIfNeeded(b)
    }

    /// If a node has more than M connections, keep only the M nearest.
    /// This is "simple selection" — PK-006 will add heuristic selection.
    private func pruneIfNeeded(_ node: Int) {
        guard adjacency[node].count > config.m else { return }

        let nodeVector = vectors[node]
        // Sort neighbors by distance to this node, keep the M closest.
        let sorted = adjacency[node].sorted { a, b in
            metric.distance(nodeVector, vectors[a]) < metric.distance(nodeVector, vectors[b])
        }
        adjacency[node] = Array(sorted.prefix(config.m))
    }
}
