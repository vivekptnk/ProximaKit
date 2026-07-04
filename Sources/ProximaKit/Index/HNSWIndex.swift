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

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

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

    /// Seeds the random layer-assignment draw so graph construction is
    /// reproducible: the same insertion sequence yields the same topology.
    ///
    /// `nil` (the default) uses the system RNG. The seed is a build-time
    /// knob only — it is NOT persisted, because it affects how a graph is
    /// constructed, never how an existing graph is searched.
    public let levelSeed: UInt64?

    public init(
        m: Int = 16,
        efConstruction: Int = 200,
        efSearch: Int = 50,
        autoCompactionThreshold: Double? = 0.7,
        levelSeed: UInt64? = nil
    ) {
        // m == 1 would make levelMultiplier = 1/ln(1) = +inf, which traps on
        // the first add() when assignLevel() converts the level to Int.
        // m == 1 is also degenerate for the graph (mMax0 = 2), so reject it here.
        precondition(m >= 2, "M must be at least 2 (m = 1 yields an infinite level multiplier)")
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
        self.levelSeed = levelSeed
    }
}

/// SplitMix64 — deterministic generator for seeded layer assignment.
/// Not cryptographic; reproducible-construction use only.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Backward-compatible spelling for ``IndexResidency`` on full-precision HNSW APIs.
public typealias HNSWOpenMode = IndexResidency

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
/// let results = await index.search(query: queryVec, k: 10)  // search doesn't throw
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

    /// Seeded generator for layer assignment when `config.levelSeed` is set;
    /// `nil` means draw from the system RNG (the default, non-reproducible).
    private var levelRNG: SplitMix64?

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

    /// Reverse adjacency: inEdges[l][n] = the set of nodes whose layer-l
    /// neighbor list contains n. This realizes the incoming-edge map
    /// (node → {(layer, fromNode)}) in layer-major parallel-array form,
    /// mirroring `layers`.
    ///
    /// Maintained on every edge mutation (insert-time connect, pruning,
    /// remove-time reconnection) so `remove()` can delete all edges into a
    /// node in O(in-degree) instead of sweeping every edge list on the
    /// layer (O(E_l)). Derived state: NOT persisted — the snapshot format
    /// is unchanged (ADR-010) and `init(restoring:)` rebuilds it from
    /// the snapshot's layers.
    private var inEdges: [[Set<Int>]] = []

    /// The maximum layer each node was assigned to.
    private var nodeLevels: [Int] = []

    /// Vector data for each node, served through the resident-or-paged provider
    /// (ADR-013 Stage 2). Resident by default (byte-identical to the historical
    /// `[Vector]`); a `.paged` open maps the base's vector section and keeps
    /// post-snapshot adds in a resident tail. `count` slots come from here.
    private var vectorProvider: VectorProvider

    /// Whether this index was opened in `.paged` mode. Governs the
    /// checkpoint-remap story: a paged index re-maps the freshly written base
    /// after a checkpoint so it keeps serving vectors from the file, not
    /// resident memory. Resident indexes leave it `false` and behave exactly as
    /// before. (`compact` materializes vectors resident transiently; the next
    /// checkpoint re-establishes the mapping.)
    private var isPagedMode = false

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

    // ── Write-ahead journaling (ADR-013, opt-in) ──────────────────────
    //
    // Off by default: a nil journal means this index behaves exactly as it
    // always has (the non-journaled path is byte-identical). When a journal is
    // attached via ``checkpoint(baseURL:walURL:durability:)`` or
    // ``open(baseURL:walURL:durability:)``, every `add`/`remove` appends its
    // primitive record (add carries the *assigned level* so replay is
    // deterministic) synchronously inside the actor — WAL order is actor
    // serialization order.

    /// The attached WAL sink, or nil when journaling is disabled.
    private var journal: WALJournal?

    /// True while compaction or replay drives internal `add`/`remove` calls, so
    /// those internal mutations are not re-journaled (compaction is a
    /// checkpoint event; replay is reconstructing from the journal itself).
    private var suppressJournaling = false

    /// Generation of the last base snapshot this index was checkpointed to.
    /// Bound into the WAL header so a stale WAL is rejected on open.
    private var snapshotGeneration: UInt64 = 0

    /// Byte size of the last base snapshot written/loaded, for the checkpoint
    /// policy's "WAL bytes > fraction of base" rule.
    private var baseByteCount: Int = 0

    /// The number of slots in the index, **including tombstoned (removed) nodes**.
    /// Use `liveCount` to get the number of searchable vectors.
    ///
    /// - Note: Sibling index types differ here: ``SparseIndex`` reports its
    ///   *live* document count from `count` (its slot total is `slotCount`).
    ///   When comparing the two legs of a ``HybridIndex`` after removals,
    ///   compare this index's `liveCount` against the sparse leg's `count`.
    public var count: Int { vectorProvider.count }

    /// The number of live (non-tombstoned) vectors available for search.
    /// After removals, `liveCount <= count`. After `compact()`, they are equal.
    public var liveCount: Int { uuidToNode.count }

    /// Whether the index contains no live vectors.
    /// `true` for a fresh index and after every live vector has been removed,
    /// even if tombstoned slots remain (`count > 0`).
    public var isEmpty: Bool { uuidToNode.isEmpty }

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
        self.levelRNG = config.levelSeed.map(SplitMix64.init(seed:))
        self.vectorProvider = .resident(dimension: dimension)
    }

    // ── VectorIndex: Add (Algorithm 1 from paper) ─────────────────────

    public func add(_ vector: Vector, id: UUID, metadata: Data? = nil) throws {
        guard vector.dimension == dimension else {
            throw IndexError.dimensionMismatch(expected: dimension, got: vector.dimension)
        }

        if uuidToNode[id] != nil {
            _ = remove(id: id)
        }

        let newLevel = assignLevel()
        insertNode(vector, id: id, metadata: metadata, level: newLevel)

        // Journal AFTER the in-memory mutation succeeds, carrying the assigned
        // level so replay is byte-exact (no RNG re-draw). Skipped during
        // compaction/replay. Errors deferred from a prior non-throwing
        // `remove` surface here.
        if !suppressJournaling, let journal = journal {
            try journal.appendAdd(
                id: id, level: newLevel, vector: Array(vector.components), metadata: metadata
            )
        }
    }

    /// Inserts a node with an explicit, caller-supplied level — the primitive
    /// shared by `add` (which draws the level) and WAL replay (which feeds the
    /// journaled level). Assumes `id` is not currently live (the caller, or a
    /// preceding `remove`, guarantees this), performs no journaling, and does
    /// not draw from the RNG. Everything the original `add` body did from the
    /// node allocation onward lives here.
    private func insertNode(_ vector: Vector, id: UUID, metadata: Data?, level newLevel: Int) {
        let newNode = vectorProvider.count

        // Store node data.
        vectorProvider.append(vector)
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
                inEdges[l].append([])
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
                addEdge(from: newNode, to: neighbor, layer: level)
                addEdge(from: neighbor, to: newNode, layer: level)

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

    /// Searches for the `k` nearest neighbors of `query`.
    ///
    /// - Important: If `query.dimension` does not match the index dimension,
    ///   this returns `[]` rather than throwing — unlike ``add(_:id:metadata:)``,
    ///   which throws `IndexError.dimensionMismatch` for the same condition.
    ///   If you get empty results from a non-empty index, check that the query
    ///   embedder produces vectors of the index's dimension.
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
        //
        // Filtered queries take the graph-aware path (ADR-008 addendum):
        // the predicate is applied DURING the beam, so rejected nodes are
        // traversed for routing but never occupy result candidacy, and the
        // beam width adapts to the observed acceptance rate so selective
        // filters still fill k. Unfiltered queries keep the original
        // filter-blind beam below — that path is unchanged (zero overhead).
        if let filter = filter {
            let accepted = searchLayer0Filtered(
                query: query, entryPoint: currentEntry, ef: ef, k: k, filter: filter
            )
            // Every accepted candidate already passed the liveness and
            // predicate checks inside the beam — materialize directly.
            var results = accepted.map { (node, distance) in
                SearchResult(id: nodeToUUID[node], distance: distance, metadata: metadata[node])
            }
            results.sort()
            if results.count > k {
                results = Array(results.prefix(k))
            }
            return results
        }

        let candidates = searchLayer(query: query, entryPoint: currentEntry, ef: ef, layer: 0)

        // Build results, skipping tombstoned nodes.
        var results: [SearchResult] = []
        for (node, distance) in candidates {
            let uuid = nodeToUUID[node]
            // Skip tombstoned nodes (removed but slot not compacted).
            // Identity check: after a re-add of the same UUID, the UUID maps to
            // a NEW node — the old slot is live only if it maps back to itself.
            guard uuidToNode[uuid] == node else { continue }
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

    /// Removes a vector by ID, tombstoning its slot and repairing the graph.
    ///
    /// Repair policy, applied per layer the node lived on:
    /// 1. **Reverse-edge cleanup** — every edge pointing *into* the removed
    ///    node is deleted. Insertion-time pruning is one-sided
    ///    (`pruneConnections` trims only the over-capacity node's own list),
    ///    so a live node X can hold an edge to this node without a reciprocal
    ///    edge; iterating only the removed node's own adjacency would leave
    ///    X's edge dangling and waste beam slots on a dead end during search.
    ///    The maintained reverse-adjacency map (`inEdges`) lists exactly the
    ///    nodes holding such edges, so the cleanup is O(in-degree) per layer
    ///    rather than a full O(E_l) sweep of every edge list on the layer.
    /// 2. **Neighbor reconnection** — the removed node's former neighbors are
    ///    offered each other as candidate edges and re-selected with the same
    ///    diversity heuristic used at insertion, so paths that ran through the
    ///    removed node are bridged rather than severed. O(d² · D) per layer,
    ///    where d = the removed node's degree and D = the vector dimension.
    ///
    /// The slot itself is kept as a tombstone (`count` unchanged, `liveCount`
    /// decremented) until `compact()` or auto-compaction reclaims it.
    ///
    /// **Durability asymmetry** — `add` throws, so its `appendAdd` surfaces a
    /// WAL write error on the same call; `remove` is non-throwing, so its
    /// `appendRemove` can only *defer* a write failure into the journal's
    /// `pendingError`, left for the next throwing journaled op (another `add`,
    /// a checkpoint, or an explicit sync) to raise. If the process crashes
    /// before any such op runs, a `remove` that already returned `true` was
    /// never durably journaled: WAL replay on recovery omits it, so the
    /// "removed" vector reappears.
    @discardableResult
    public func remove(id: UUID) -> Bool {
        let removed = primitiveRemove(id: id)
        guard removed else { return false }

        // Journal the removal (non-throwing path: a write error is captured and
        // re-raised by the next throwing journaled op). Skipped during
        // compaction/replay.
        if !suppressJournaling {
            journal?.appendRemove(id: id)
        }

        // Auto-compaction reclaims tombstones but changes `count`, which an
        // append-only WAL cannot reproduce on replay. So a *journaled* index
        // defers compaction to the next checkpoint (which rewrites the base and
        // truncates the WAL). Non-journaled indexes keep today's exact behavior.
        if journal == nil,
           let threshold = config.autoCompactionThreshold,
           count > 0,
           Double(liveCount) / Double(count) < threshold {
            try? compact()
        }

        return true
    }

    /// The graph-repair core of `remove` (reverse-edge cleanup + neighbor
    /// reconnection + entry-point recompute), without journaling or
    /// auto-compaction. Shared by public `remove` and WAL replay.
    @discardableResult
    private func primitiveRemove(id: UUID) -> Bool {
        guard let node = uuidToNode[id] else { return false }
        let level = nodeLevels[node]

        // Tombstone first so the reconnection step never selects this node.
        uuidToNode.removeValue(forKey: id)

        for l in 0...level where l < layers.count {
            // Capture the node's own neighbor list before clearing — these are
            // the nodes whose connectivity may have run through the removed node.
            let formerNeighbors = layers[l][node]

            // Step 1: reverse-adjacency cleanup. Edges into this node can
            // exist outside `formerNeighbors` (one-sided pruning); `inEdges`
            // tracks exactly that set, so only the affected lists are touched
            // — O(in-degree) instead of a whole-layer sweep.
            for from in inEdges[l][node] {
                layers[l][from].removeAll { $0 == node }
            }
            inEdges[l][node] = []
            // Clear the node's own out-edges (and their reverse entries).
            setNeighbors([], node: node, layer: l)

            // Step 2: reconnect former neighbors among themselves.
            let maxConnections = (l == 0) ? config.mMax0 : config.m
            let liveNeighbors = formerNeighbors.filter { isLiveNode($0) }
            for f in liveNeighbors {
                var candidateNodes = Set(layers[l][f])
                for other in liveNeighbors where other != f {
                    candidateNodes.insert(other)
                }
                candidateNodes.remove(f)
                guard !candidateNodes.isEmpty else { continue }

                let fVector = vectorProvider.vector(at: f)
                let candidates = candidateNodes.map { candidate in
                    (node: candidate, distance: metric.distance(fVector, vectorProvider.vector(at: candidate)))
                }
                let selected = selectNeighborsHeuristic(
                    target: fVector,
                    candidates: candidates,
                    m: maxConnections
                )
                setNeighbors(selected.map(\.node), node: f, layer: l)
            }
        }

        // If we removed the entry point, find the node with the highest level.
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

    // ── Persistence ────────────────────────────────────────────────────

    /// Restores an HNSW index from a persistence snapshot.
    /// Directly sets all internal state — no re-insertion needed.
    public init(restoring snapshot: HNSWSnapshot) {
        self.dimension = snapshot.dimension
        self.metric = snapshot.metricType.makeMetric()
        self.config = snapshot.config
        self.levelMultiplier = 1.0 / log(Double(snapshot.config.m))
        // levelSeed is a build-time knob and is not persisted; a snapshot's
        // config carries nil, so restored indexes use the system RNG for any
        // post-load insertions.
        self.levelRNG = snapshot.config.levelSeed.map(SplitMix64.init(seed:))
        self.layers = snapshot.layers
        // Rebuild the reverse adjacency from the snapshot's layers. It is
        // derived state and deliberately NOT part of the snapshot, so the
        // on-disk format stays unchanged (ADR-010).
        var rebuiltInEdges: [[Set<Int>]] = snapshot.layers.map { layer in
            Array(repeating: Set<Int>(), count: layer.count)
        }
        for (l, layer) in snapshot.layers.enumerated() {
            for (from, neighbors) in layer.enumerated() {
                for to in neighbors {
                    rebuiltInEdges[l][to].insert(from)
                }
            }
        }
        self.inEdges = rebuiltInEdges
        self.nodeLevels = snapshot.nodeLevels
        self.vectorProvider = .resident(snapshot.vectors, dimension: snapshot.dimension)
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

    /// Restores an HNSW index in **paged** mode: identical to `init(restoring:)`
    /// except the vector section is served from `region` (a read-only file
    /// mapping) instead of a resident `[Vector]`. Post-snapshot adds (live adds
    /// and WAL replay) land in the provider's resident tail — node id
    /// `< region.count` maps into the file, otherwise into the tail. Used only
    /// by ``PersistenceEngine/loadHNSWPaged(from:)``; `snapshot.vectors` is
    /// ignored (empty by construction).
    internal init(restoringPaged snapshot: HNSWSnapshot, region: MappedVectorRegion) {
        self.dimension = snapshot.dimension
        self.metric = snapshot.metricType.makeMetric()
        self.config = snapshot.config
        self.levelMultiplier = 1.0 / log(Double(snapshot.config.m))
        self.levelRNG = snapshot.config.levelSeed.map(SplitMix64.init(seed:))
        self.layers = snapshot.layers
        var rebuiltInEdges: [[Set<Int>]] = snapshot.layers.map { layer in
            Array(repeating: Set<Int>(), count: layer.count)
        }
        for (l, layer) in snapshot.layers.enumerated() {
            for (from, neighbors) in layer.enumerated() {
                for to in neighbors {
                    rebuiltInEdges[l][to].insert(from)
                }
            }
        }
        self.inEdges = rebuiltInEdges
        self.nodeLevels = snapshot.nodeLevels
        self.vectorProvider = .paged(region: region, dimension: snapshot.dimension)
        self.isPagedMode = true
        self.metadata = snapshot.metadata
        self.nodeToUUID = snapshot.nodeToUUID
        self.entryPointNode = snapshot.entryPointNode
        self.maxLevel = snapshot.maxLevel
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
            vectors: vectorProvider.materialized(),
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

    /// Loads an HNSW index in the requested residency `mode`.
    ///
    /// `.resident` is byte-identical to ``load(from:)``; `.paged` serves the
    /// base vector section from a read-only file mapping (`MappedVectorRegion`)
    /// instead of copying it into `[Vector]`.
    ///
    /// - Throws: for `.paged`, a typed `PersistenceError` (never a trap) when
    ///   the base is not a padded v3 file. Checkpoint the index first to write
    ///   the paged-capable v3 base.
    public static func load(from url: URL, mode: IndexResidency = .resident) throws -> HNSWIndex {
        switch mode {
        case .resident:
            return try load(from: url)
        case .paged:
            return try PersistenceEngine.loadHNSWPaged(from: url)
        }
    }

    // ── Streaming persistence: WAL journaling (ADR-013, Stage 1, opt-in) ─
    //
    // Additive surface. `save(to:)`/`load(from:)` above are untouched (full
    // resident, atomic, v2). These methods layer a write-ahead log over a base
    // snapshot so saves become O(change) instead of O(corpus).

    /// Opens a journaled index: loads the base `.pxkt`, then replays the
    /// `.pxwal` sidecar (if present) over it, and attaches the WAL for further
    /// appends. Prefix semantics — a torn WAL tail is truncated to its longest
    /// intact record run; a stale WAL (parent generation ≠ base) throws
    /// `PersistenceError.walGenerationMismatch`.
    ///
    /// The base file must exist (create one first with
    /// ``checkpoint(baseURL:walURL:durability:)`` or a plain `save`).
    ///
    /// `mode` (ADR-013 Stage 2, additive — defaults to `.resident` so existing
    /// call sites are byte-identical):
    /// - `.resident` decodes the whole base into memory, exactly as before.
    /// - `.paged` serves the base's vector section from a read-only file
    ///   mapping instead of copying it resident, cutting cold-start and
    ///   residency to the graph + working-set of vector pages. Requires a
    ///   padded v3 base (any `checkpoint` writes one); an unpadded or non-v3
    ///   base throws a typed `PersistenceError`. Post-open adds and WAL
    ///   replay land in a resident tail. **Contract:** the mapping is opened
    ///   read-only and ProximaKit never truncates its own live files, but
    ///   truncating a mapped base from *outside* the library is out of
    ///   contract — a faulted page past a shrunken end-of-file raises SIGBUS,
    ///   which Swift cannot catch. See ``MappedVectorRegion`` and ADR-013 for
    ///   the full mmap-lifetime discussion.
    public static func open(
        baseURL: URL,
        walURL: URL,
        durability: WALDurability = .everyBatch,
        mode: IndexResidency = .resident
    ) async throws -> HNSWIndex {
        let index: HNSWIndex
        switch mode {
        case .resident:
            index = try PersistenceEngine.loadHNSW(from: baseURL)
        case .paged:
            index = try PersistenceEngine.loadHNSWPaged(from: baseURL)
        }
        let generation = try PersistenceEngine.readGeneration(from: baseURL)
        let baseBytes = (try? Data(contentsOf: baseURL, options: .mappedIfSafe).count) ?? 0
        try await index.attachJournal(
            baseURL: baseURL, walURL: walURL, generation: generation,
            baseByteCount: baseBytes, durability: durability
        )
        return index
    }

    /// Replays an existing WAL (validating its generation) and attaches it in
    /// append mode. Runs inside the actor.
    private func attachJournal(
        baseURL: URL,
        walURL: URL,
        generation: UInt64,
        baseByteCount: Int,
        durability: WALDurability
    ) throws {
        self.snapshotGeneration = generation
        self.baseByteCount = baseByteCount

        guard FileManager.default.fileExists(atPath: walURL.path) else {
            // No WAL yet — create a fresh one bound to the base generation.
            try startFreshJournal(walURL: walURL, durability: durability)
            return
        }

        let walData = try Data(contentsOf: walURL, options: .mappedIfSafe)
        // Throws for a damaged header or a stale generation; a torn record tail
        // is recovered as the longest valid prefix (no throw).
        let replay = try WALDecoder.decode(walData, expectedGeneration: generation)

        // The decoder validates the WAL's own framing (header CRC, parent
        // generation) but not that this sidecar was written for *this* base
        // index. Cross-check the header's dimension/metric against the base
        // before replaying: a crafted or mispaired WAL with a different
        // dimension would otherwise feed mismatched-length vectors straight into
        // `insertNode`, past the public `add(_:id:)` dimension guard.
        guard replay.dimension == dimension else {
            throw PersistenceError.walDimensionMismatch(expected: dimension, found: replay.dimension)
        }
        if let metricType = DistanceMetricType(metric: metric),
           replay.metricRaw != metricType.rawValue {
            throw PersistenceError.walMetricMismatch(expected: metricType.rawValue, found: replay.metricRaw)
        }

        applyReplay(replay.records)

        // Reopen in append mode. If the tail was torn, rewrite the file to the
        // valid prefix first so future appends extend clean bytes, not garbage.
        let validByteCount = walData.count - replay.trailingBytesDropped
        if replay.trailingBytesDropped > 0 {
            try walData.prefix(validByteCount).write(to: walURL, options: .atomic)
        }
        // Seed BOTH counters from the same valid prefix the decoder recovered:
        // `validByteCount` bytes carry exactly `replay.records.count` records, so
        // the byte- and op-count checkpoint rules stay consistent "since the last
        // checkpoint" from the very first post-reopen append (a torn tail dropped
        // its partial record and its bytes together).
        self.journal = try WALJournal(
            appendingTo: walURL,
            parentGeneration: generation,
            dimension: dimension,
            existingByteCount: validByteCount,
            existingRecordCount: replay.records.count,
            durability: durability
        )
    }

    private func startFreshJournal(walURL: URL, durability: WALDurability) throws {
        guard let metricType = DistanceMetricType(metric: metric) else {
            throw PersistenceError.unserializableMetric
        }
        journal?.close()
        journal = try WALJournal(
            creatingAt: walURL,
            parentGeneration: snapshotGeneration,
            dimension: dimension,
            metricRaw: metricType.rawValue,
            durability: durability
        )
    }

    /// Applies decoded WAL records to rebuild state. No journaling, no
    /// auto-compaction — records are replayed exactly as written, and the
    /// journaled level makes each insertion deterministic.
    private func applyReplay(_ records: [WALRecord]) {
        let wasSuppressed = suppressJournaling
        suppressJournaling = true
        defer { suppressJournaling = wasSuppressed }
        for record in records {
            switch record {
            case let .add(id, level, vector, metadata):
                insertNode(Vector(vector), id: id, metadata: metadata, level: level)
            case let .remove(id):
                _ = primitiveRemove(id: id)
            }
        }
    }

    /// Checkpoints: compacts, writes a fresh v3 base snapshot (generation
    /// bumped) atomically, `F_FULLFSYNC`, then resets the WAL to a new empty
    /// journal bound to the new generation. This is the supported way to both
    /// establish journaling on a fresh index and to fold an accumulated WAL
    /// back into the base.
    ///
    /// Crash-safety note (Stage 1): the base rename is the commit point. A
    /// crash after the new base is renamed but before the WAL is reset leaves a
    /// complete new base beside a stale WAL; the next ``open(baseURL:walURL:durability:mode:)``
    /// surfaces that as a typed `walGenerationMismatch` (no silent loss, no
    /// corruption) — the operator can delete the stale WAL to recover, since
    /// the new base already holds every committed record.
    ///
    /// Paged-mode remap (ADR-013 Stage 2): a checkpoint necessarily renumbers
    /// nodes (compaction) and writes an all-resident base, so the pre-checkpoint
    /// mapping can no longer serve the new node numbering. The commit therefore
    /// **re-maps**: after the padded v3 base is written and `F_FULLFSYNC`-ed,
    /// the index opens a fresh ``MappedVectorRegion`` over the *new* base inode
    /// and swaps the provider to paged again (`snapshotBoundary = count`,
    /// resident tail cleared). The whole method is a single synchronous,
    /// actor-isolated critical section — no `await` occurs between the compact,
    /// the write, and the swap — so a concurrent search can never observe a torn
    /// intermediate state. The old inode is released (unmapped) when the prior
    /// region is dropped. Peak residency during the checkpoint equals a resident
    /// checkpoint's (the snapshot is materialized to write it); steady-state
    /// paged residency is restored immediately by the re-map.
    public func checkpoint(
        baseURL: URL,
        walURL: URL,
        durability: WALDurability = .everyBatch
    ) throws {
        // Fold any deferred journal write error before we drop the old WAL.
        try journal?.sync()

        if liveCount < count {
            try compact()
        }
        guard let metricType = DistanceMetricType(metric: metric) else {
            throw PersistenceError.unserializableMetric
        }

        let newGeneration = snapshotGeneration &+ 1
        let snapshot = HNSWSnapshot(
            dimension: dimension, config: config, metricType: metricType,
            vectors: vectorProvider.materialized(), metadata: metadata, nodeToUUID: nodeToUUID,
            layers: layers, nodeLevels: nodeLevels,
            entryPointNode: entryPointNode, maxLevel: maxLevel
        )
        // Atomic base write (temp + rename) and force it to media.
        try PersistenceEngine.saveHNSW(snapshot, generation: newGeneration, to: baseURL)
        try fullSyncFile(baseURL)
        self.snapshotGeneration = newGeneration
        self.baseByteCount = (try? Data(contentsOf: baseURL, options: .mappedIfSafe).count) ?? 0

        // Paged-mode remap: re-establish the mapping over the freshly written
        // base so the index keeps serving vectors from the file rather than the
        // resident array `materialized()` just produced. Synchronous swap → no
        // torn state; dropping `vectorProvider` unmaps the previous inode.
        if isPagedMode {
            let region = try MappedVectorRegion(baseURL: baseURL)
            self.vectorProvider = .paged(region: region, dimension: dimension)
        }

        // Reset the WAL to a fresh empty journal bound to the new generation.
        try startFreshJournal(walURL: walURL, durability: durability)
    }

    /// Flushes the WAL per its durability policy (surfacing any deferred error).
    public func syncJournal() throws {
        try journal?.sync()
    }

    /// Whether the accumulated WAL warrants a checkpoint under `policy`.
    public func needsCheckpoint(policy: WALCheckpointPolicy = WALCheckpointPolicy()) -> Bool {
        guard let journal = journal else { return false }
        if journal.recordCount > policy.maxOps { return true }
        if baseByteCount > 0 {
            return Double(journal.byteCount) > Double(baseByteCount) * policy.walBytesFractionOfBase
        }
        return false
    }

    /// Current WAL size in bytes (header + records since last checkpoint).
    public var journalByteCount: Int { journal?.byteCount ?? 0 }

    /// Records appended to the WAL since the last checkpoint.
    public var journalRecordCount: Int { journal?.recordCount ?? 0 }

    /// Whether this index currently serves base vectors from a read-only file
    /// mapping (`.paged`) rather than keeping the full base vector payload in a
    /// resident array. Post-snapshot WAL replay and live adds may still occupy
    /// the resident tail until the next checkpoint.
    public var vectorsArePaged: Bool { vectorProvider.isPaged }

    /// The generation of the base this index is currently bound to.
    public var currentGeneration: UInt64 { snapshotGeneration }

    /// Detaches and closes the journal (further mutations are not journaled).
    public func closeJournal() {
        journal?.close()
        journal = nil
    }

    /// Every live (non-tombstoned) node's external id paired with its stored
    /// metadata, in internal node order.
    ///
    /// This is the projection a journaled store replays to rebuild derived
    /// sidecars — the document → UUID map, and (for a hybrid store) the whole
    /// sparse leg — from the dense index after WAL recovery, so those sidecars
    /// are never an independent on-disk source of truth that could diverge from
    /// the index on a crash (ADR-013 store-level journaling addendum). The
    /// liveness test (`uuidToNode[uuid] == node`) is the same identity check
    /// search and compaction use, so tombstoned and re-added slots are skipped.
    public func liveEntries() -> [(id: UUID, metadata: Data?)] {
        var out: [(id: UUID, metadata: Data?)] = []
        out.reserveCapacity(uuidToNode.count)
        for (node, uuid) in nodeToUUID.enumerated() where uuidToNode[uuid] == node {
            out.append((id: uuid, metadata: metadata[node]))
        }
        return out
    }

    private func fullSyncFile(_ url: URL) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        #if canImport(Darwin)
        if fcntl(handle.fileDescriptor, F_FULLFSYNC) == 0 { return }
        #endif
        _ = fsync(handle.fileDescriptor)
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
    /// try await index.compact()
    /// // Now count == liveCount
    /// ```
    public func compact() throws {
        // Compaction re-inserts live nodes through `add`, drawing fresh levels.
        // It is a base-rewrite (checkpoint) event, never a journaled one, so
        // suppress record emission for the duration.
        let wasSuppressed = suppressJournaling
        suppressJournaling = true
        defer { suppressJournaling = wasSuppressed }

        // Snapshot all live nodes before clearing state.
        var live: [(id: UUID, vector: Vector, metadata: Data?)] = []
        for (node, uuid) in nodeToUUID.enumerated() {
            guard uuidToNode[uuid] == node else { continue }
            live.append((id: uuid, vector: vectorProvider.vector(at: node), metadata: metadata[node]))
        }

        // Reset all storage. In paged mode this drops the mapping and returns
        // the provider to resident-empty; the re-inserted live nodes below land
        // in the resident tail. A following paged checkpoint re-establishes the
        // mapping over the freshly written base (see `checkpoint`).
        layers = []
        inEdges = []
        nodeLevels = []
        vectorProvider = .resident(dimension: dimension)
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
        let random: Double
        if var rng = levelRNG {
            random = Double.random(in: Double.leastNonzeroMagnitude...1.0, using: &rng)
            levelRNG = rng // write the advanced state back
        } else {
            random = Double.random(in: Double.leastNonzeroMagnitude...1.0)
        }
        return Int(floor(-log(random) * levelMultiplier))
    }

    /// Ensures the layers array has capacity through the given level.
    private func ensureLayer(_ level: Int) {
        while layers.count <= level {
            layers.append(Array(repeating: [], count: vectorProvider.count))
            inEdges.append(Array(repeating: Set<Int>(), count: vectorProvider.count))
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
        let epDist = metric.distance(query, vectorProvider.vector(at: entryPoint))

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

                let dist = metric.distance(query, vectorProvider.vector(at: neighbor))

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

    // ── Core: Graph-Aware Filtered Beam on Layer 0 (ADR-008 addendum) ─
    //
    // Differs from `searchLayer` in two ways:
    //   1. Result candidacy: only live, predicate-passing nodes enter the
    //      result heap. Rejected nodes (tombstoned or filtered out) still
    //      join the candidate frontier, so the beam routes THROUGH them
    //      toward matching regions instead of spending result slots on them.
    //   2. Adaptive widening: the effective ef is recomputed from the
    //      acceptance rate observed so far — roughly k / rate, add-one
    //      smoothed, clamped to [ef, efCap] — so selective predicates keep
    //      the beam exploring until k matching results are plausible. When
    //      (nearly) every node passes, the recomputed value stays at `ef`
    //      and the beam behaves like the unfiltered one.
    //
    // Termination: the beam stops early only once the result heap holds
    // `effectiveEf` ACCEPTED nodes and the nearest frontier candidate is
    // farther than the worst of them. With fewer matching nodes reachable
    // than `effectiveEf`, the beam therefore explores the whole connected
    // component — selective filters trade latency for filled, high-recall
    // results (worst case O(liveCount) distance evaluations; the same
    // trade hnswlib's filtered search makes). The unfiltered `searchLayer`
    // path is untouched.

    private func searchLayer0Filtered(
        query: Vector,
        entryPoint: Int,
        ef: Int,
        k: Int,
        filter: (UUID) -> Bool
    ) -> [(node: Int, distance: Float)] {
        // Cap on adaptive widening: bounds the result-heap size and how far
        // the early-exit check can be deferred once results ARE plentiful.
        // liveCount is the natural ceiling.
        let efCap = max(ef, min(liveCount, 16 * max(ef, k)))
        var effectiveEf = ef

        // Acceptance bookkeeping counts predicate *evaluations*, not visits:
        // far nodes that never pass frontier admission are never evaluated.
        var evaluated = 0
        var accepted = 0

        // Liveness first (identity-based tombstone check), then the user
        // predicate — the filter is never invoked for tombstoned slots,
        // matching the post-filter path's contract.
        func acceptsIntoResults(_ node: Int) -> Bool {
            evaluated += 1
            guard isLiveNode(node) else { return false }
            guard filter(nodeToUUID[node]) else { return false }
            accepted += 1
            return true
        }

        // effectiveEf ≈ k / acceptance-rate, add-one smoothed so an early
        // run of rejections widens gradually rather than jumping to the cap.
        // Recomputed (not monotonic): the estimate self-corrects as more
        // nodes are evaluated.
        func rewiden() {
            let rate = Double(accepted + 1) / Double(evaluated + 1)
            let needed = Int((Double(k) / rate).rounded(.up))
            effectiveEf = min(efCap, max(ef, needed))
        }

        let epDist = metric.distance(query, vectorProvider.vector(at: entryPoint))

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
               let furthest = results.peek(),
               nearest.distance > furthest.distance {
                break
            }

            for neighbor in layers[0][nearest.node] {
                guard !visited.contains(neighbor) else { continue }
                visited.insert(neighbor)

                let dist = metric.distance(query, vectorProvider.vector(at: neighbor))

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
            let candidateVector = vectorProvider.vector(at: candidate.node)
            var isGood = true

            for (selectedNode, _) in selected {
                let distToSelected = metric.distance(candidateVector, vectorProvider.vector(at: selectedNode))
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

        let nodeVector = vectorProvider.vector(at: node)
        let neighbors = layers[layer][node].map { neighbor in
            (node: neighbor, distance: metric.distance(nodeVector, vectorProvider.vector(at: neighbor)))
        }

        let selected = selectNeighborsHeuristic(
            target: nodeVector,
            candidates: neighbors,
            m: maxConnections
        )

        setNeighbors(selected.map(\.node), node: node, layer: layer)
    }

    // ── Reverse-Adjacency Maintenance ─────────────────────────────────
    //
    // Every mutation of `layers` flows through these two helpers (plus the
    // slot mirroring in add()/ensureLayer() and the wholesale resets in
    // compact()/init(restoring:)), so `inEdges` is always the exact
    // transpose of `layers`. Tests assert that invariant through
    // `reverseAdjacencyIsConsistent`.

    /// Appends the directed edge `from → to` on a layer, recording the
    /// reverse entry. Callers guarantee the edge is not already present
    /// (adjacency lists never hold duplicates).
    private func addEdge(from: Int, to: Int, layer: Int) {
        layers[layer][from].append(to)
        inEdges[layer][to].insert(from)
    }

    /// Replaces a node's out-edge list on a layer, diffing old vs. new to
    /// keep `inEdges` in sync. O(old + new).
    private func setNeighbors(_ newNeighbors: [Int], node: Int, layer: Int) {
        let oldSet = Set(layers[layer][node])
        let newSet = Set(newNeighbors)
        for dropped in oldSet.subtracting(newSet) {
            inEdges[layer][dropped].remove(node)
        }
        for added in newSet.subtracting(oldSet) {
            inEdges[layer][added].insert(node)
        }
        layers[layer][node] = newNeighbors
    }

    // ── Liveness ──────────────────────────────────────────────────────

    /// A slot is live iff its UUID still maps back to it.
    /// (After a re-add of the same UUID, the UUID maps to a newer node,
    /// tombstoning the old slot.)
    private func isLiveNode(_ node: Int) -> Bool {
        uuidToNode[nodeToUUID[node]] == node
    }

    // ── Test Hooks (internal) ─────────────────────────────────────────

    /// Whether any adjacency list still references a tombstoned slot.
    /// Used by tests to verify `remove()` leaves no dangling incoming edges.
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

    /// Structural invariant: `inEdges` must be exactly the transpose of
    /// `layers` — every edge u→v on layer l is mirrored by v's in-set
    /// containing u, and nothing else. O(total edges); used by tests to
    /// verify the map stays consistent through add/prune/remove/compact
    /// and is rebuilt correctly on restore.
    internal var reverseAdjacencyIsConsistent: Bool {
        guard inEdges.count == layers.count else { return false }
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

    /// Whether every live node is reachable from every other live node by
    /// traversing layer-0 edges (treated as undirected) between live nodes.
    /// Used by tests to verify `remove()` repairs connectivity.
    internal var isLayer0Connected: Bool {
        guard !layers.isEmpty else { return true }
        let liveNodes = (0..<vectorProvider.count).filter { isLiveNode($0) }
        guard let start = liveNodes.first else { return true }

        // Build an undirected view of layer 0 (pruning can leave edges
        // one-sided between live nodes; either direction is traversable
        // in practice because beam search discovers from both endpoints).
        var undirected: [Int: Set<Int>] = [:]
        for source in liveNodes {
            for neighbor in layers[0][source] where isLiveNode(neighbor) {
                undirected[source, default: []].insert(neighbor)
                undirected[neighbor, default: []].insert(source)
            }
        }

        var visited = Set<Int>([start])
        var frontier = [start]
        while let current = frontier.popLast() {
            for neighbor in undirected[current] ?? [] {
                if visited.insert(neighbor).inserted {
                    frontier.append(neighbor)
                }
            }
        }
        return visited.count == liveNodes.count
    }

    /// Full internal-state fingerprint for exact-equality assertions in WAL
    /// recovery tests. Two indexes with identical fingerprints are byte-for-byte
    /// the same graph — including tombstone slots, level assignments, adjacency,
    /// entry point, vectors, and metadata. Used to prove that a WAL replay
    /// reproduces the exact state that produced the log (ADR-013 acceptance 1).
    internal struct StructuralFingerprint: Equatable {
        let nodeToUUID: [UUID]
        let uuidToNode: [UUID: Int]
        let nodeLevels: [Int]
        let layers: [[[Int]]]
        let entryPointNode: Int?
        let maxLevel: Int
        let vectors: [Vector]
        let metadata: [Data?]
    }

    internal var structuralFingerprint: StructuralFingerprint {
        StructuralFingerprint(
            nodeToUUID: nodeToUUID,
            uuidToNode: uuidToNode,
            nodeLevels: nodeLevels,
            layers: layers,
            entryPointNode: entryPointNode,
            maxLevel: maxLevel,
            vectors: vectorProvider.materialized(),
            metadata: metadata
        )
    }
}
