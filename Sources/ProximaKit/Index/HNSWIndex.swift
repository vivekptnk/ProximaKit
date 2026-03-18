// HNSWIndex.swift
// ProximaKit
//
// Hierarchical Navigable Small World index for approximate nearest-neighbor search.
// Multi-layer graph with greedy descent on upper layers and beam search on layer 0.
//
// Based on: "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)

import Accelerate
import Foundation

/// Configuration for the HNSW index.
///
/// - `m`: Max connections per node on upper layers. Layer 0 allows `2 * m`.
/// - `efConstruction`: Beam width during insertion. Higher = better graph quality.
/// - `efSearch`: Default beam width during queries. Tunable per query.
public struct HNSWConfiguration: Sendable {
    /// Max connections per node on layers 1+.
    public let m: Int

    /// Max connections per node on layer 0 (2 * m per the paper).
    public let mMax0: Int

    /// Beam width during index construction.
    public let efConstruction: Int

    /// Default beam width during search queries.
    public let efSearch: Int

    /// Auto-compaction threshold as a fraction of live/total nodes.
    ///
    /// When `liveCount / count` drops below this value after a removal,
    /// the index automatically compacts to reclaim tombstone slots.
    /// Set to `nil` to disable auto-compaction (manual `compact()` only).
    ///
    /// Default: `0.7` (compact when fewer than 70% of slots are live).
    public let autoCompactionThreshold: Double?

    public init(
        m: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 50,
        autoCompactionThreshold: Double? = 0.7
    ) {
        precondition(m > 0, "M must be positive")
        precondition(efConstruction > 0, "efConstruction must be positive")
        precondition(efSearch > 0, "efSearch must be positive")
        if let threshold = autoCompactionThreshold {
            precondition(threshold > 0 && threshold < 1, "autoCompactionThreshold must be in (0, 1)")
        }
        self.m = m
        self.mMax0 = 2 * m
        self.efConstruction = efConstruction
        self.efSearch = efSearch
        self.autoCompactionThreshold = autoCompactionThreshold
    }
}

/// Approximate nearest-neighbor index using Hierarchical Navigable Small World graphs.
///
/// The index organizes vectors into a multi-layer graph:
/// - **Upper layers** (sparse): few nodes with long-range edges for fast navigation
/// - **Layer 0** (dense): all nodes, where the final beam search happens
///
/// Search descends from the top layer greedily (ef=1) to find a good entry point
/// for layer 0, then does a full beam search with `efSearch` candidates.
///
/// ```swift
/// let index = HNSWIndex(dimension: 384, metric: CosineDistance())
/// try await index.add(vector, id: UUID())
/// let results = try await index.search(query: queryVec, k: 10)
/// ```
public actor HNSWIndex: VectorIndex {

    // ── Configuration ─────────────────────────────────────────────────

    // `dimension` and `config` are set at init and never mutated.
    // `nonisolated` makes them readable without `await` — safe because
    // immutable stored `let` properties can't cause data races.
    public nonisolated let dimension: Int
    private let metric: any DistanceMetric
    private let config: HNSWConfiguration

    /// The configuration this index was created with. Readable without `await`.
    public nonisolated var configuration: HNSWConfiguration { config }

    /// Level multiplier: mL = 1/ln(M). Used for exponential layer assignment.
    private let levelMultiplier: Double

    // ── Node Storage ──────────────────────────────────────────────────
    //
    // Nodes are identified by internal Int indices (0, 1, 2, ...).
    //
    // layers[l][n] = neighbor indices for node n on layer l.
    // Nodes that don't exist on layer l have an empty array at layers[l][n].
    // This wastes trivial memory (empty Swift arrays are inline) but gives
    // O(1) neighbor lookup on any layer — critical for search performance.

    /// Per-layer adjacency lists. layers[l][n] = [neighbor indices].
    private var layers: [[[Int]]] = []

    /// The maximum layer each node was assigned to.
    private var nodeLevels: [Int] = []

    /// Vector data for each node.
    private var vectors: [Vector] = []

    /// Metadata for each node.
    private var metadata: [Data?] = []

    /// Internal node index → external UUID.
    private var nodeToUUID: [UUID] = []

    /// External UUID → internal node index.
    private var uuidToNode: [UUID: Int] = [:]

    /// The entry point node (always the node at the highest layer).
    private var entryPointNode: Int?

    /// The highest layer currently in the index.
    private var maxLevel: Int = -1

    /// The number of slots in the index, including tombstoned (removed) nodes.
    /// Use `liveCount` to get the number of searchable vectors.
    public var count: Int { vectors.count }

    /// The number of live (non-tombstoned) vectors available for search.
    /// After removals, `liveCount <= count`. After `compact()`, they are equal.
    public var liveCount: Int { uuidToNode.count }

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
        self.levelMultiplier = 1.0 / log(Double(config.m))
    }

    // ── VectorIndex: Add (Algorithm 1 from paper) ─────────────────────

    public func add(_ vector: Vector, id: UUID, metadata: Data? = nil) throws {
        guard vector.dimension == dimension else {
            throw IndexError.dimensionMismatch(expected: dimension, got: vector.dimension)
        }

        if uuidToNode[id] != nil {
            _ = remove(id: id)
        }

        let newNode = vectors.count
        let newLevel = assignLevel()

        // Store node data.
        vectors.append(vector)
        self.metadata.append(metadata)
        nodeToUUID.append(id)
        uuidToNode[id] = newNode
        nodeLevels.append(newLevel)

        // Ensure layers exist for this node's level.
        ensureLayer(newLevel)

        // Add empty neighbor slots for this new node on all existing layers.
        for l in 0..<layers.count {
            if layers[l].count <= newNode {
                layers[l].append([])
            }
        }

        // First node: becomes the entry point, no connections needed.
        guard let ep = entryPointNode else {
            entryPointNode = newNode
            maxLevel = newLevel
            return
        }

        var currentEntry = ep

        // Phase 1: Greedy descent on layers above the new node's level.
        // Single-result search (ef=1) to find the closest node on each layer.
        if maxLevel > newLevel {
            for level in stride(from: maxLevel, through: newLevel + 1, by: -1) {
                let nearest = searchLayer(query: vector, entryPoint: currentEntry, ef: 1, layer: level)
                if let closest = nearest.first {
                    currentEntry = closest.node
                }
            }
        }

        // Phase 2: Insert on layers min(newLevel, maxLevel) down to 0.
        // Beam search + connect with heuristic neighbor selection.
        let insertionTop = min(newLevel, maxLevel)
        for level in stride(from: insertionTop, through: 0, by: -1) {
            let candidates = searchLayer(
                query: vector,
                entryPoint: currentEntry,
                ef: config.efConstruction,
                layer: level
            )

            let maxConnections = (level == 0) ? config.mMax0 : config.m
            let selected = selectNeighborsHeuristic(
                target: vector,
                candidates: candidates,
                m: maxConnections
            )

            // Connect new node to selected neighbors on this layer.
            for (neighbor, _) in selected {
                layers[level][newNode].append(neighbor)
                layers[level][neighbor].append(newNode)

                // Prune neighbor if over capacity.
                let neighborMax = (level == 0) ? config.mMax0 : config.m
                if layers[level][neighbor].count > neighborMax {
                    pruneConnections(node: neighbor, layer: level, maxConnections: neighborMax)
                }
            }

            // Use closest candidate as entry for the next layer down.
            if let closest = candidates.first {
                currentEntry = closest.node
            }
        }

        // If the new node has a higher level than the current max,
        // it becomes the new entry point.
        if newLevel > maxLevel {
            entryPointNode = newNode
            maxLevel = newLevel
        }
    }

    // ── VectorIndex: Search (Algorithm 5 from paper) ──────────────────

    public func search(
        query: Vector,
        k: Int,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) -> [SearchResult] {
        guard query.dimension == dimension else { return [] }
        guard let ep = entryPointNode else { return [] }
        guard k > 0 else { return [] }

        let ef = max(efSearch ?? config.efSearch, k)
        var currentEntry = ep

        // Phase 1: Greedy descent on upper layers (ef=1).
        // Each layer narrows down to a better starting point for the next.
        for level in stride(from: maxLevel, through: 1, by: -1) {
            let nearest = searchLayer(query: query, entryPoint: currentEntry, ef: 1, layer: level)
            if let closest = nearest.first {
                currentEntry = closest.node
            }
        }

        // Phase 2: Full beam search on layer 0.
        let candidates = searchLayer(query: query, entryPoint: currentEntry, ef: ef, layer: 0)

        // Build results, skipping tombstoned nodes and applying filter.
        var results: [SearchResult] = []
        for (node, distance) in candidates {
            let uuid = nodeToUUID[node]
            // Skip tombstoned nodes (removed but slot not compacted).
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

    // ── VectorIndex: Remove ───────────────────────────────────────────

    @discardableResult
    public func remove(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }
        let level = nodeLevels[node]

        // Disconnect this node on all its layers.
        for l in 0...level where l < layers.count {
            for neighbor in layers[l][node] {
                layers[l][neighbor].removeAll { $0 == node }
            }
            layers[l][node] = []
        }

        uuidToNode.removeValue(forKey: id)

        // If we removed the entry point, find the node with the highest level.
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

        // Auto-compact if the live ratio dropped below the threshold.
        if let threshold = config.autoCompactionThreshold,
           count > 0,
           Double(liveCount) / Double(count) < threshold {
            try? compact()
        }

        return true
    }

    // ── Persistence ────────────────────────────────────────────────────

    /// Restores an HNSW index from a persistence snapshot.
    /// Directly sets all internal state — no re-insertion needed.
    public init(restoring snapshot: HNSWSnapshot) {
        self.dimension = snapshot.dimension
        self.metric = snapshot.metricType.makeMetric()
        self.config = snapshot.config
        self.levelMultiplier = 1.0 / log(Double(snapshot.config.m))
        self.layers = snapshot.layers
        self.nodeLevels = snapshot.nodeLevels
        self.vectors = snapshot.vectors
        self.metadata = snapshot.metadata
        self.nodeToUUID = snapshot.nodeToUUID
        self.entryPointNode = snapshot.entryPointNode
        self.maxLevel = snapshot.maxLevel
        // Rebuild the reverse lookup.
        self.uuidToNode = [:]
        for (i, uuid) in snapshot.nodeToUUID.enumerated() {
            self.uuidToNode[uuid] = i
        }
    }

    /// Returns a snapshot of this index for binary persistence.
    /// Compacts first if there are tombstoned nodes.
    public func persistenceSnapshot() throws -> HNSWSnapshot {
        guard let metricType = DistanceMetricType(metric: metric) else {
            throw PersistenceError.unserializableMetric
        }
        // Compact to remove tombstones before saving.
        if liveCount < count {
            try compact()
        }
        return HNSWSnapshot(
            dimension: dimension,
            config: config,
            metricType: metricType,
            vectors: vectors,
            metadata: metadata,
            nodeToUUID: nodeToUUID,
            layers: layers,
            nodeLevels: nodeLevels,
            entryPointNode: entryPointNode,
            maxLevel: maxLevel
        )
    }

    /// Saves this index to a binary file.
    public func save(to url: URL) throws {
        let snapshot = try persistenceSnapshot()
        try PersistenceEngine.save(snapshot, to: url)
    }

    /// Loads an HNSW index from a binary file.
    public static func load(from url: URL) throws -> HNSWIndex {
        try PersistenceEngine.loadHNSW(from: url)
    }

    // ── Compaction ────────────────────────────────────────────────────

    /// Rebuilds the index, removing all tombstoned (deleted) nodes.
    ///
    /// After removals, `count` includes tombstone slots but `liveCount` does not.
    /// Compaction reclaims that memory and resets `count == liveCount`.
    ///
    /// The graph is fully rebuilt, so this is O(n log n) — call it when the
    /// ratio `liveCount / count` drops below your acceptable threshold (e.g., 0.7).
    ///
    /// ```swift
    /// await index.compact()
    /// // Now count == liveCount
    /// ```
    public func compact() throws {
        // Snapshot all live nodes before clearing state.
        var live: [(id: UUID, vector: Vector, metadata: Data?)] = []
        for (node, uuid) in nodeToUUID.enumerated() {
            guard uuidToNode[uuid] != nil else { continue }
            live.append((id: uuid, vector: vectors[node], metadata: metadata[node]))
        }

        // Reset all storage.
        layers = []
        nodeLevels = []
        vectors = []
        metadata = []
        nodeToUUID = []
        uuidToNode = [:]
        entryPointNode = nil
        maxLevel = -1

        // Re-insert all live nodes — this rebuilds a clean graph.
        for entry in live {
            try add(entry.vector, id: entry.id, metadata: entry.metadata)
        }
    }

    // ── Layer Assignment ──────────────────────────────────────────────
    //
    // level = floor(-ln(random) * mL)  where mL = 1/ln(M)
    //
    // This produces a geometric distribution:
    //   ~93.75% of nodes at level 0  (for M=16)
    //   ~5.86% at level 1
    //   ~0.37% at level 2
    //   etc.
    // Like skip lists: most nodes are in the "ground floor" (layer 0),
    // and each upper layer has ~1/M as many nodes.

    private func assignLevel() -> Int {
        let random = Double.random(in: Double.leastNonzeroMagnitude...1.0)
        return Int(floor(-log(random) * levelMultiplier))
    }

    /// Ensures the layers array has capacity through the given level.
    private func ensureLayer(_ level: Int) {
        while layers.count <= level {
            layers.append(Array(repeating: [], count: vectors.count))
        }
    }

    // ── Core: Beam Search on a Single Layer (Algorithm 2) ─────────────
    //
    // Unchanged from PK-005 except it now takes a `layer` parameter
    // to index into the correct adjacency list.

    private func searchLayer(
        query: Vector,
        entryPoint: Int,
        ef: Int,
        layer: Int
    ) -> [(node: Int, distance: Float)] {
        let epDist = metric.distance(query, vectors[entryPoint])

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

            // The ONLY change from PK-005: layers[layer][...] instead of adjacency[...]
            for neighbor in layers[layer][nearest.node] {
                guard !visited.contains(neighbor) else { continue }
                visited.insert(neighbor)

                let dist = metric.distance(query, vectors[neighbor])

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

    // ── Heuristic Neighbor Selection (Algorithm 4) ────────────────────
    //
    // Instead of keeping the M nearest neighbors (simple selection),
    // we keep the M most *diverse* neighbors. For each candidate:
    //   - If it's closer to the target than to any already-selected neighbor → keep it
    //   - Otherwise → skip it (it's "covered" by an existing neighbor)
    //
    // This creates longer-range edges across clusters, improving recall
    // from ~88% to ~96% on clustered data (see ADR-004).

    private func selectNeighborsHeuristic(
        target: Vector,
        candidates: [(node: Int, distance: Float)],
        m: Int
    ) -> [(node: Int, distance: Float)] {
        guard !candidates.isEmpty else { return [] }

        let sorted = candidates.sorted { $0.distance < $1.distance }
        var selected: [(node: Int, distance: Float)] = []
        selected.reserveCapacity(m)

        for candidate in sorted {
            guard selected.count < m else { break }

            // Check: is this candidate closer to the target than
            // to any already-selected neighbor?
            let candidateVector = vectors[candidate.node]
            var isGood = true

            for (selectedNode, _) in selected {
                let distToSelected = metric.distance(candidateVector, vectors[selectedNode])
                if distToSelected < candidate.distance {
                    isGood = false
                    break
                }
            }

            if isGood {
                selected.append(candidate)
            }
        }

        // If heuristic was too strict and we have fewer than M,
        // fill with the closest remaining candidates.
        if selected.count < m {
            let selectedNodes = Set(selected.map(\.node))
            for candidate in sorted where selected.count < m {
                if !selectedNodes.contains(candidate.node) {
                    selected.append(candidate)
                }
            }
        }

        return selected
    }

    // ── Connection Pruning ────────────────────────────────────────────

    /// Prunes a node's connections on a given layer using heuristic selection.
    private func pruneConnections(node: Int, layer: Int, maxConnections: Int) {
        guard layers[layer][node].count > maxConnections else { return }

        let nodeVector = vectors[node]
        let neighbors = layers[layer][node].map { neighbor in
            (node: neighbor, distance: metric.distance(nodeVector, vectors[neighbor]))
        }

        let selected = selectNeighborsHeuristic(
            target: nodeVector,
            candidates: neighbors,
            m: maxConnections
        )

        layers[layer][node] = selected.map(\.node)
    }
}
