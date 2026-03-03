**PRODUCT REQUIREMENTS DOCUMENT**

**ProximaKit**

A Swift-Native Semantic Search Engine

v1.0 \| February 2026

Author: Vivek \| Built with Claude Code

1\. Overview

  ---------------------- ---------------------------------------------------------------------------
  **Product Name**       ProximaKit

  **Type**               Open-source Swift Package + Demo iOS App

  **Target Platforms**   iOS 17+, macOS 14+, visionOS 1+

  **Swift Version**      5.9+ (strict concurrency)

  **Dependencies**       None (Apple frameworks only: Accelerate, CoreML, NaturalLanguage, Vision)

  **License**            MIT

  **Repository**         github.com/vivek/ProximaKit

  **Timeline**           5 weeks (Weeks 2--7 of prep plan)
  ---------------------- ---------------------------------------------------------------------------

1.1 Problem Statement

There is no production-quality, pure-Swift vector similarity search library. Developers who want semantic search on Apple platforms must wrap C++ (FAISS), use Python bridges, or rely on cloud APIs. This forces a choice between performance and platform-native development, and makes on-device semantic search inaccessible to most iOS/macOS developers.

1.2 Product Vision

ProximaKit is a zero-dependency Swift library that enables fast, on-device semantic search using Apple's Accelerate framework for SIMD math and a from-scratch HNSW (Hierarchical Navigable Small World) index for approximate nearest-neighbor search. It ships as a Swift Package with a protocol-oriented API that integrates naturally with Core ML, NaturalLanguage, and Vision frameworks.

1.3 Success Metrics

-   **Performance:** \< 50ms query latency for 10K vectors at 384 dimensions on iPhone 14

-   **Accuracy:** \> 95% recall@10 vs brute-force baseline

-   **API Quality:** Full DocC documentation, 90%+ test coverage on core algorithms

-   **Adoption Signal:** Published on GitHub with README, benchmarks, and demo app

1.4 Non-Goals (v1)

-   GPU-accelerated search via Metal (future v2)

-   Distributed/networked search

-   Training or fine-tuning embedding models

-   Android or cross-platform support

2\. Architecture

ProximaKit is organized into four layers. Each layer is a separate Swift module within the package, enabling consumers to import only what they need.

+--------------------------------------------------------------------------------------+
| **LAYER ARCHITECTURE**                                                               |
|                                                                                      |
| ProximaKit (core) → Vector math, index structures, query engine, persistence         |
|                                                                                      |
| ProximaEmbeddings → Protocol-based embedding providers (CoreML, NLEmbedding, Vision) |
|                                                                                      |
| ProximaKit Demo App → SwiftUI app demonstrating semantic photo + notes search        |
+--------------------------------------------------------------------------------------+

2.1 Core Module: ProximaKit

The heart of the library. Contains all vector math, index data structures, and the query engine. Zero external dependencies --- only Foundation and Accelerate.

Key Types

-   **Vector:** A value type wrapping \[Float\] with Accelerate-powered operations (dot product, cosine similarity, L2 distance, normalization). Uses vDSP under the hood. Conforms to Sendable.

-   **BruteForceIndex:** Exact nearest-neighbor search. O(n) per query. Used as the accuracy baseline and for small datasets (\< 1000 vectors).

-   **HNSWIndex:** Approximate nearest-neighbor search using Hierarchical Navigable Small World graphs. O(log n) per query. The primary index for production use.

-   **IndexConfiguration:** Value type configuring index parameters: dimension, distance metric, HNSW-specific params (M, efConstruction, efSearch).

-   **SearchResult:** Struct containing id (UUID), distance (Float), and optional metadata (any Codable).

-   **PersistenceEngine:** Saves/loads indices to disk using memory-mapped files for fast cold starts.

2.2 Embeddings Module: ProximaEmbeddings

Protocol-based embedding providers that convert raw content into vectors.

Protocols

-   **Embeddable:** Protocol that any content type can conform to. Requires func embed(using: some EmbeddingProvider) async throws -\> Vector

-   **EmbeddingProvider:** Protocol for embedding backends. Methods: embed(\_ text: String), embed(\_ image: CGImage), embed(batch: \[String\]).

Built-in Providers

-   **NLEmbeddingProvider:** Wraps Apple's NLEmbedding for text. Fastest option, lower quality. Good for prototyping.

-   **CoreMLEmbeddingProvider:** Wraps any Core ML model that outputs a float array. Supports custom SentenceTransformer ports.

-   **VisionEmbeddingProvider:** Uses VNFeaturePrintObservation to generate image embeddings. Enables semantic image search without a custom model.

2.3 API Design Principles

1.  **Swift-first:** Value types, structured concurrency, Sendable. No Objective-C bridging.

2.  **Protocol-oriented:** All major types are protocol-backed. Users can provide custom indices and embedding providers.

3.  **Progressive disclosure:** Simple things should be simple. let results = try await index.search(\"sunset\") should work in 3 lines of setup code.

4.  **Observable performance:** Built-in benchmarking utilities. Every query returns timing metadata.

3\. Technical Specifications

3.1 HNSW Algorithm Details

The HNSW implementation is the core of ProximaKit and the primary learning objective. It must be built from scratch --- no ports from C++ code.

Algorithm Parameters

  ---------------- ------------- ---------------------------------------------------------------------------------------
  **Parameter**    **Default**   **Description**

  M                16            Max connections per node per layer. Higher = better recall, more memory.

  efConstruction   200           Beam width during index building. Higher = better index quality, slower build.

  efSearch         50            Beam width during query. Higher = better recall, slower query. Tunable at query time.

  mL               1/ln(M)       Level generation factor. Controls the height of the graph.

  Metric           cosine        Distance function: cosine, L2 (euclidean), or dot product.
  ---------------- ------------- ---------------------------------------------------------------------------------------

Implementation Requirements

-   Layers stored as adjacency lists: \[\[Int\]\] per layer, where each inner array is the neighbor list for a node

-   Entry point maintained as the node with the highest layer assignment

-   Greedy search on upper layers (layer \> 0) to find entry point for layer 0 search

-   Beam search on layer 0 with efSearch candidates, returning top-k results

-   Neighbor selection uses the heuristic algorithm (not simple) for better graph connectivity

-   Thread-safe reads via actor isolation. Writes are serialized.

3.2 Accelerate Framework Usage

All vector math must use Accelerate/vDSP for SIMD performance. No manual loops over float arrays.

  ------------------------ --------------------- --------------------------
  **Operation**            **vDSP Function**     **Context**

  Dot product              vDSP_dotpr            Cosine sim numerator

  L2 norm                  vDSP_svesq + sqrt     Vector normalization

  Vector subtraction       vDSP_vsub             L2 distance computation

  Scalar multiply          vDSP_vsmul            Normalization division

  Batch distance           vDSP_mmul             Multi-query optimization
  ------------------------ --------------------- --------------------------

3.3 Persistence Format

Indices are saved as memory-mapped binary files for fast cold starts. The format:

-   **Header (64 bytes):** Magic number, version, dimension, vector count, distance metric, HNSW params

-   **Vector data:** Contiguous Float array, memory-mapped. vectors\[i\] starts at offset header_size + (i \* dimension \* 4)

-   **Graph data:** Adjacency lists serialized as \[layerCount, \[nodeCount, \[neighborCount, neighbors\...\]\]\]

-   **Metadata:** JSON-encoded dictionary keyed by vector ID. Stored separately for flexibility.

4\. Epic & Story Breakdown

Task breakdown follows industry-standard hierarchy: Epic → Story → Task. Points are Fibonacci (1, 2, 3, 5, 8). Each story has a learning objective so you're building skills alongside code.

+--------------------------------------------------------------------------------------------------------+
| **CLAUDE CODE WORKFLOW**                                                                               |
|                                                                                                        |
| For each task below, open Claude Code and give it the task description + acceptance criteria.          |
|                                                                                                        |
| Ask Claude Code to explain concepts before writing code. Use it as a tutor, not just a code generator. |
|                                                                                                        |
| After each story, review the generated code yourself. Understand every line. Refactor if needed.       |
|                                                                                                        |
| Run benchmarks after each epic. Track performance progression.                                         |
+--------------------------------------------------------------------------------------------------------+

  -----------------------------------------------------------------------
  **EPIC 1 Project Foundation & Vector Math** (Week 2, \~3 days)

  -----------------------------------------------------------------------

Set up the Swift Package, implement the Vector type, and build all distance functions using Accelerate. This is your foundation --- every other component depends on fast, correct vector math.

**PK-001: Swift Package Setup & CI** \[3 pts pts\]

+--------+------------+-------------------------------------------------------------------------------+-----------------------------------+
| **\#** | **Points** | **Task**                                                                      | **Learn / Notes**                 |
+--------+------------+-------------------------------------------------------------------------------+-----------------------------------+
| 1      | 1          | **Initialize Swift Package with Library + Test targets**                      | *Swift Package Manager structure* |
|        |            |                                                                               |                                   |
|        |            | > -- Package.swift with platforms: \[.iOS(.v17), .macOS(.v14)\]               |                                   |
|        |            | >                                                                             |                                   |
|        |            | > -- Two targets: ProximaKit (library), ProximaKitTests (test)                |                                   |
+--------+------------+-------------------------------------------------------------------------------+-----------------------------------+
| 2      | 1          | **Set up directory structure: Sources/ProximaKit, Sources/ProximaEmbeddings** | *Multi-module SPM packages*       |
|        |            |                                                                               |                                   |
|        |            | > -- Separate modules for core vs embeddings                                  |                                   |
|        |            | >                                                                             |                                   |
|        |            | > -- Public API surface in each module                                        |                                   |
+--------+------------+-------------------------------------------------------------------------------+-----------------------------------+
| 3      | 1          | **Add .gitignore, README skeleton, LICENSE (MIT)**                            | *OSS project structure*           |
|        |            |                                                                               |                                   |
|        |            | > -- Include badges: Swift version, platforms, license                        |                                   |
|        |            | >                                                                             |                                   |
|        |            | > -- Add basic usage example in README                                        |                                   |
+--------+------------+-------------------------------------------------------------------------------+-----------------------------------+

**PK-002: Vector Type with Accelerate Math** \[8 pts pts\]

*🎯 Learning goal: Accelerate framework, vDSP, unsafe pointer access in Swift*

+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| **\#** | **Points** | **Task**                                                                            | **Learn / Notes**                             |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| 1      | 2          | **Define Vector struct wrapping ContiguousArray\<Float\>**                          | *Value type design, ContiguousArray vs Array* |
|        |            |                                                                                     |                                               |
|        |            | > -- Conform to Sendable, Equatable, Hashable, Codable                              |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Initializers: from \[Float\], from UnsafeBufferPointer, dimension + fill value |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Property: dimension (Int), magnitude (Float)                                   |                                               |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| 2      | 2          | **Implement dot product using vDSP_dotpr**                                          | *vDSP fundamentals, UnsafeBufferPointer*      |
|        |            |                                                                                     |                                               |
|        |            | > -- Write the Accelerate call with proper pointer access                           |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Benchmark vs naive loop implementation (expect 5--10x speedup at 384 dims)     |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Add unit test comparing results to manual calculation                          |                                               |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| 3      | 2          | **Implement cosine similarity using vDSP**                                          | *vDSP_svesq, combining vDSP operations*       |
|        |            |                                                                                     |                                               |
|        |            | > -- cosine = dot(a,b) / (magnitude(a) \* magnitude(b))                             |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Use vDSP_svesq for magnitude calculation                                       |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Handle edge case: zero-magnitude vectors                                       |                                               |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| 4      | 1          | **Implement L2 (Euclidean) distance using vDSP_vsub + vDSP_svesq**                  | *vDSP_vsub, chaining operations*              |
|        |            |                                                                                     |                                               |
|        |            | > -- Subtract vectors, sum of squares, sqrt                                         |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Compare accuracy to manual implementation                                      |                                               |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+
| 5      | 1          | **Implement vector normalization using vDSP_vsdiv**                                 | *vDSP_vsdiv, immutable transforms*            |
|        |            |                                                                                     |                                               |
|        |            | > -- Divide all elements by magnitude                                               |                                               |
|        |            | >                                                                                   |                                               |
|        |            | > -- Return new Vector (immutable value type)                                       |                                               |
+--------+------------+-------------------------------------------------------------------------------------+-----------------------------------------------+

**PK-003: Distance Metric Protocol & Batch Operations** \[5 pts pts\]

*🎯 Learning goal: Batch SIMD operations, matrix layout for Accelerate*

+--------+------------+------------------------------------------------------------------------------------+----------------------------------------+
| **\#** | **Points** | **Task**                                                                           | **Learn / Notes**                      |
+--------+------------+------------------------------------------------------------------------------------+----------------------------------------+
| 1      | 1          | **Define DistanceMetric protocol with associated calculation**                     | *Protocol-oriented design*             |
|        |            |                                                                                    |                                        |
|        |            | > -- Protocol with func distance(\_ a: Vector, \_ b: Vector) -\> Float             |                                        |
|        |            | >                                                                                  |                                        |
|        |            | > -- Concrete types: CosineDistance, EuclideanDistance, DotProductDistance         |                                        |
+--------+------------+------------------------------------------------------------------------------------+----------------------------------------+
| 2      | 3          | **Implement batch distance computation using vDSP_mmul**                           | *Matrix multiplication with vDSP_mmul* |
|        |            |                                                                                    |                                        |
|        |            | > -- Given query vector + matrix of N vectors, compute all N distances in one call |                                        |
|        |            | >                                                                                  |                                        |
|        |            | > -- Reshape vectors into column-major matrix for vDSP_mmul                        |                                        |
|        |            | >                                                                                  |                                        |
|        |            | > -- This is the critical optimization for search performance                      |                                        |
+--------+------------+------------------------------------------------------------------------------------+----------------------------------------+
| 3      | 1          | **Write performance benchmark suite**                                              | *Swift performance benchmarking*       |
|        |            |                                                                                    |                                        |
|        |            | > -- Benchmark single-pair vs batch distance at 100, 1K, 10K vectors               |                                        |
|        |            | >                                                                                  |                                        |
|        |            | > -- Output results as formatted table                                             |                                        |
|        |            | >                                                                                  |                                        |
|        |            | > -- Use XCTest.measure for consistent timing                                      |                                        |
+--------+------------+------------------------------------------------------------------------------------+----------------------------------------+

  -----------------------------------------------------------------------
  **EPIC 2 Brute-Force Index** (Week 2--3, \~2 days)

  -----------------------------------------------------------------------

Implement the exact search index. This is simpler than HNSW but establishes the index protocol, query API, and testing infrastructure. It also serves as the accuracy baseline for HNSW.

**PK-004: Index Protocol & BruteForceIndex** \[5 pts pts\]

*🎯 Learning goal: Actor-based concurrency, protocol design for extensibility*

+--------+------------+-----------------------------------------------------------------------------------------------------+---------------------------------------------------+
| **\#** | **Points** | **Task**                                                                                            | **Learn / Notes**                                 |
+--------+------------+-----------------------------------------------------------------------------------------------------+---------------------------------------------------+
| 1      | 2          | **Define VectorIndex protocol**                                                                     | *Actor isolation, protocol with associated types* |
|        |            |                                                                                                     |                                                   |
|        |            | > -- Methods: add(\_ vector: Vector, id: UUID, metadata: (any Codable)?)                            |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- search(query: Vector, k: Int, ef: Int?, filter: ((UUID) -\> Bool)?) async -\> \[SearchResult\] |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- remove(id: UUID), count: Int, dimension: Int                                                   |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- Make it actor-isolated for thread safety                                                       |                                                   |
+--------+------------+-----------------------------------------------------------------------------------------------------+---------------------------------------------------+
| 2      | 2          | **Implement BruteForceIndex**                                                                       | *Cache-friendly data layout*                      |
|        |            |                                                                                                     |                                                   |
|        |            | > -- Store vectors in ContiguousArray for cache-friendly access                                     |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- Use batch distance computation from PK-003 for search                                          |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- Support metadata filtering via predicate closure                                               |                                                   |
+--------+------------+-----------------------------------------------------------------------------------------------------+---------------------------------------------------+
| 3      | 1          | **Write comprehensive test suite**                                                                  | *XCTest, property-based testing*                  |
|        |            |                                                                                                     |                                                   |
|        |            | > -- Test: add, search, remove, duplicate IDs, empty index, dimension mismatch                      |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- Test: search with filter, search k \> count, concurrent reads                                  |                                                   |
|        |            | >                                                                                                   |                                                   |
|        |            | > -- Property-based test: nearest neighbor is always correct for random data                        |                                                   |
+--------+------------+-----------------------------------------------------------------------------------------------------+---------------------------------------------------+

  -----------------------------------------------------------------------
  **EPIC 3 HNSW Index (Core)** (Week 3--4, \~6 days)

  -----------------------------------------------------------------------

This is the heart of ProximaKit and your primary learning objective. The HNSW algorithm is non-trivial --- expect to spend time reading the paper and understanding the algorithm before writing code.

+-----------------------------------------------------------------------------------------------------------------------------------------+
| **LEARNING APPROACH FOR HNSW**                                                                                                          |
|                                                                                                                                         |
| **Step 1:** Ask Claude Code to explain the HNSW paper section by section. Draw the graph structure on paper.                            |
|                                                                                                                                         |
| **Step 2:** Implement layer 0 only first (single-layer NSW). Get it working and tested.                                                 |
|                                                                                                                                         |
| **Step 3:** Add multi-layer support. The key insight: upper layers are \"express lanes\" to quickly navigate to the right neighborhood. |
|                                                                                                                                         |
| **Step 4:** Add the heuristic neighbor selection. This is what makes HNSW better than basic NSW.                                        |
+-----------------------------------------------------------------------------------------------------------------------------------------+

**PK-005: Single-Layer NSW (Navigable Small World)** \[8 pts pts\]

*🎯 Learning goal: HNSW paper section 3: Navigable Small World fundamentals*

+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+
| **\#** | **Points** | **Task**                                                                                  | **Learn / Notes**                                 |
+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+
| 1      | 2          | **Implement node storage: adjacency list representation**                                 | *Adjacency list graph representation*             |
|        |            |                                                                                           |                                                   |
|        |            | > -- Each node: id (Int), neighbors: \[Int\] (max M neighbors)                            |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Store node-to-UUID mapping separately                                                |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Vectors stored in contiguous array, indexed by node id                               |                                                   |
+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+
| 2      | 3          | **Implement greedy search on single layer**                                               | *Priority queue in Swift, greedy graph traversal* |
|        |            |                                                                                           |                                                   |
|        |            | > -- Start from entry point, greedily move to nearest unvisited neighbor                  |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Maintain visited set and candidate priority queue (min-heap)                         |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Return ef closest nodes (not just k --- ef \>= k, we trim later)                     |                                                   |
+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+
| 3      | 2          | **Implement node insertion with neighbor selection**                                      | *Graph construction algorithms*                   |
|        |            |                                                                                           |                                                   |
|        |            | > -- Find nearest neighbors via greedy search                                             |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Connect new node to found neighbors (bidirectional edges)                            |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- If any neighbor exceeds M connections, prune using simple selection (keep M nearest) |                                                   |
+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+
| 4      | 1          | **Test: verify recall vs brute-force at various dataset sizes**                           | *Recall metrics, benchmark methodology*           |
|        |            |                                                                                           |                                                   |
|        |            | > -- Generate random vectors (100, 1000, 5000)                                            |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Compare NSW top-10 results to BruteForceIndex top-10                                 |                                                   |
|        |            | >                                                                                         |                                                   |
|        |            | > -- Target: \> 90% recall@10 at ef=50                                                    |                                                   |
+--------+------------+-------------------------------------------------------------------------------------------+---------------------------------------------------+

**PK-006: Multi-Layer HNSW** \[8 pts pts\]

*🎯 Learning goal: HNSW paper sections 4--5: multi-layer construction and search*

+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+
| **\#** | **Points** | **Task**                                                                                            | **Learn / Notes**                              |
+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+
| 1      | 2          | **Implement layer assignment using exponential distribution**                                       | *Probability distributions, skip list analogy* |
|        |            |                                                                                                     |                                                |
|        |            | > -- Level for new node = floor(-ln(uniform_random) \* mL) where mL = 1/ln(M)                       |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- Most nodes exist only on layer 0. Few nodes reach higher layers.                               |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- Store max layer count and entry point (highest-layer node)                                     |                                                |
+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+
| 2      | 3          | **Implement multi-layer search: greedy descent + beam search**                                      | *Hierarchical search, beam search*             |
|        |            |                                                                                                     |                                                |
|        |            | > -- Start at entry point\'s layer. Greedy search (ef=1) to find best node per layer.               |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- Descend to next layer, using found node as new entry.                                          |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- At layer 0: beam search with ef=efSearch candidates.                                           |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- Return top-k from layer 0 results.                                                             |                                                |
+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+
| 3      | 2          | **Implement multi-layer insertion**                                                                 | *HNSW insertion algorithm (paper section 4)*   |
|        |            |                                                                                                     |                                                |
|        |            | > -- Assign layer level to new node                                                                 |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- Search from top to target layer (greedy)                                                       |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- At target layer and below: connect with neighbors                                              |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- If new node layer \> current max, update entry point                                           |                                                |
+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+
| 4      | 1          | **Implement heuristic neighbor selection (Algorithm 4 from paper)**                                 | *HNSW heuristic selection, graph quality*      |
|        |            |                                                                                                     |                                                |
|        |            | > -- Instead of keeping M nearest, keep M most diverse neighbors                                    |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- For each candidate: add only if it\'s closer to the node than to any already-selected neighbor |                                                |
|        |            | >                                                                                                   |                                                |
|        |            | > -- This creates longer-range edges and improves recall on clustered data                          |                                                |
+--------+------------+-----------------------------------------------------------------------------------------------------+------------------------------------------------+

**PK-007: HNSW Quality & Thread Safety** \[5 pts pts\]

*🎯 Learning goal: Swift concurrency patterns, performance benchmarking*

+--------+------------+--------------------------------------------------------------------------------------+------------------------------------------+
| **\#** | **Points** | **Task**                                                                             | **Learn / Notes**                        |
+--------+------------+--------------------------------------------------------------------------------------+------------------------------------------+
| 1      | 2          | **Make HNSWIndex an actor with read-write isolation**                                | *Actor re-entrancy, custom concurrency*  |
|        |            |                                                                                      |                                          |
|        |            | > -- Reads (search) can be concurrent --- use nonisolated(unsafe) for read-only data |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Writes (add/remove) must be serialized                                          |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Consider using a readers-writer lock if actor overhead is too high              |                                          |
+--------+------------+--------------------------------------------------------------------------------------+------------------------------------------+
| 2      | 1          | **Implement node deletion with lazy tombstoning**                                    | *Tombstone pattern in data structures*   |
|        |            |                                                                                      |                                          |
|        |            | > -- Mark deleted nodes, skip during search                                          |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Compact index periodically (rebuild without deleted nodes)                      |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Test: delete nodes and verify search results exclude them                       |                                          |
+--------+------------+--------------------------------------------------------------------------------------+------------------------------------------+
| 3      | 2          | **Comprehensive recall benchmarks**                                                  | *Performance profiling with Instruments* |
|        |            |                                                                                      |                                          |
|        |            | > -- Test at 1K, 5K, 10K, 50K vectors across 3 dimensions (128, 384, 768)            |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Vary efSearch: 10, 50, 100, 200. Plot recall@10 vs query time.                  |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Compare to BruteForceIndex as ground truth                                      |                                          |
|        |            | >                                                                                    |                                          |
|        |            | > -- Target: \> 95% recall@10 at efSearch=50 for 10K/384d                            |                                          |
+--------+------------+--------------------------------------------------------------------------------------+------------------------------------------+

  -----------------------------------------------------------------------
  **EPIC 4 Persistence & Index Serialization** (Week 4--5, \~2 days)

  -----------------------------------------------------------------------

**PK-008: Binary Persistence Engine** \[5 pts pts\]

*🎯 Learning goal: Binary I/O, memory-mapped files, unsafe Swift*

+--------+------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+
| **\#** | **Points** | **Task**                                                                                            | **Learn / Notes**                    |
+--------+------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+
| 1      | 1          | **Design binary file format with header**                                                           | *Binary file format design*          |
|        |            |                                                                                                     |                                      |
|        |            | > -- 64-byte header: magic (4), version (4), dimension (4), count (4), metric (4), HNSW params (44) |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Vector section: contiguous Float array                                                         |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Graph section: layer count + adjacency lists                                                   |                                      |
+--------+------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+
| 2      | 2          | **Implement save using UnsafeRawBufferPointer**                                                     | *UnsafeRawBufferPointer, binary I/O* |
|        |            |                                                                                                     |                                      |
|        |            | > -- Write header, then vectors as raw bytes, then graph structure                                  |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Use FileHandle for sequential writes                                                           |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Verify roundtrip: save then load, verify identical search results                              |                                      |
+--------+------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+
| 3      | 2          | **Implement load using memory-mapped files**                                                        | *Memory-mapped files, lazy loading*  |
|        |            |                                                                                                     |                                      |
|        |            | > -- Use mmap (via Data(contentsOf:options:.mappedIfSafe)) for vectors                              |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Only graph structure needs full deserialization                                                |                                      |
|        |            | >                                                                                                   |                                      |
|        |            | > -- Benchmark: cold start time for 10K, 50K, 100K vector indices                                   |                                      |
+--------+------------+-----------------------------------------------------------------------------------------------------+--------------------------------------+

  -----------------------------------------------------------------------
  **EPIC 5 Embedding Providers** (Week 5, \~2 days)

  -----------------------------------------------------------------------

**PK-009: Embedding Protocol & NLEmbedding Provider** \[3 pts pts\]

*🎯 Learning goal: NaturalLanguage + Vision frameworks*

+--------+------------+------------------------------------------------------------------------+---------------------------------------+
| **\#** | **Points** | **Task**                                                               | **Learn / Notes**                     |
+--------+------------+------------------------------------------------------------------------+---------------------------------------+
| 1      | 1          | **Define EmbeddingProvider protocol**                                  | *Protocol design for extensibility*   |
|        |            |                                                                        |                                       |
|        |            | > -- func embed(\_ text: String) async throws -\> Vector               |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- func embed(\_ image: CGImage) async throws -\> Vector             |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- func embedBatch(\_ texts: \[String\]) async throws -\> \[Vector\] |                                       |
+--------+------------+------------------------------------------------------------------------+---------------------------------------+
| 2      | 1          | **Implement NLEmbeddingProvider**                                      | *NaturalLanguage framework*           |
|        |            |                                                                        |                                       |
|        |            | > -- Wrap NLEmbedding.wordEmbedding(for: .english)                     |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- Handle missing embeddings gracefully (throw typed error)          |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- Support sentence-level by averaging word embeddings               |                                       |
+--------+------------+------------------------------------------------------------------------+---------------------------------------+
| 3      | 1          | **Implement VisionEmbeddingProvider**                                  | *Vision framework feature extraction* |
|        |            |                                                                        |                                       |
|        |            | > -- Use VNGenerateImageFeaturePrintRequest                            |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- Convert VNFeaturePrintObservation to Vector                       |                                       |
|        |            | >                                                                      |                                       |
|        |            | > -- Handle image preprocessing (resize, normalize)                    |                                       |
+--------+------------+------------------------------------------------------------------------+---------------------------------------+

**PK-010: CoreML Embedding Provider** \[5 pts pts\]

*🎯 Learning goal: Core ML deep internals*

+--------+------------+-------------------------------------------------------------------------------------+--------------------------------------------------+
| **\#** | **Points** | **Task**                                                                            | **Learn / Notes**                                |
+--------+------------+-------------------------------------------------------------------------------------+--------------------------------------------------+
| 1      | 3          | **Implement CoreMLEmbeddingProvider**                                               | *Core ML loading, MLMultiArray, batch inference* |
|        |            |                                                                                     |                                                  |
|        |            | > -- Load any .mlmodel that outputs a multi-array                                   |                                                  |
|        |            | >                                                                                   |                                                  |
|        |            | > -- Handle model loading (lazy vs eager), prediction configuration                 |                                                  |
|        |            | >                                                                                   |                                                  |
|        |            | > -- Support batch predictions using MLBatchProvider                                |                                                  |
|        |            | >                                                                                   |                                                  |
|        |            | > -- Map MLMultiArray output to Vector                                              |                                                  |
+--------+------------+-------------------------------------------------------------------------------------+--------------------------------------------------+
| 2      | 2          | **Write integration test with a real model**                                        | *Core ML model conversion pipeline*              |
|        |            |                                                                                     |                                                  |
|        |            | > -- Download a small SentenceTransformer model converted to Core ML                |                                                  |
|        |            | >                                                                                   |                                                  |
|        |            | > -- Verify embeddings are reasonable: similar sentences -\> high cosine similarity |                                                  |
|        |            | >                                                                                   |                                                  |
|        |            | > -- Benchmark: embeddings/second on device                                         |                                                  |
+--------+------------+-------------------------------------------------------------------------------------+--------------------------------------------------+

  --------------------------------------------------------------------------
  **EPIC 6 Demo App: Semantic Photo & Notes Search** (Week 5--7, \~5 days)

  --------------------------------------------------------------------------

The demo app is what makes ProximaKit tangible. It indexes your photo library and notes, then lets you search semantically. This is the piece that will be in your interview portfolio.

**PK-011: App Shell & Photo Library Access** \[5 pts pts\]

*🎯 Learning goal: PhotosUI framework, SwiftUI app architecture*

+--------+------------+--------------------------------------------------------------------------+--------------------------------------------------+
| **\#** | **Points** | **Task**                                                                 | **Learn / Notes**                                |
+--------+------------+--------------------------------------------------------------------------+--------------------------------------------------+
| 1      | 1          | **Create SwiftUI app target in the package workspace**                   | *SPM + app workspace setup*                      |
|        |            |                                                                          |                                                  |
|        |            | > -- Separate Xcode project or workspace that depends on ProximaKit SPM  |                                                  |
|        |            | >                                                                        |                                                  |
|        |            | > -- Set up navigation: search tab, library tab, settings tab            |                                                  |
+--------+------------+--------------------------------------------------------------------------+--------------------------------------------------+
| 2      | 2          | **Implement photo library access with PHPickerViewController**           | *PhotosUI, PHAsset, privacy permissions*         |
|        |            |                                                                          |                                                  |
|        |            | > -- Request limited photo access (privacy-first)                        |                                                  |
|        |            | >                                                                        |                                                  |
|        |            | > -- Enumerate photos using PHFetchResult                                |                                                  |
|        |            | >                                                                        |                                                  |
|        |            | > -- Display photo grid using LazyVGrid with thumbnail loading           |                                                  |
+--------+------------+--------------------------------------------------------------------------+--------------------------------------------------+
| 3      | 2          | **Build indexing pipeline: photos → embeddings → ProximaKit index**      | *TaskGroup for parallel work, progress tracking* |
|        |            |                                                                          |                                                  |
|        |            | > -- Use VisionEmbeddingProvider to embed each photo                     |                                                  |
|        |            | >                                                                        |                                                  |
|        |            | > -- Show progress UI during indexing (TaskGroup for parallel embedding) |                                                  |
|        |            | >                                                                        |                                                  |
|        |            | > -- Persist index to disk after indexing completes                      |                                                  |
+--------+------------+--------------------------------------------------------------------------+--------------------------------------------------+

**PK-012: Semantic Search UI** \[5 pts pts\]

*🎯 Learning goal: SwiftUI advanced patterns, async search*

+--------+------------+------------------------------------------------------------------------------+----------------------------------------+
| **\#** | **Points** | **Task**                                                                     | **Learn / Notes**                      |
+--------+------------+------------------------------------------------------------------------------+----------------------------------------+
| 1      | 2          | **Build search interface with text input**                                   | *SwiftUI search, async data flow*      |
|        |            |                                                                              |                                        |
|        |            | > -- Search bar at top, results grid below                                   |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- On text input: embed query text using NLEmbeddingProvider, search index |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- Display results ranked by similarity with distance score overlay        |                                        |
+--------+------------+------------------------------------------------------------------------------+----------------------------------------+
| 2      | 2          | **Add Notes indexing (optional second content type)**                        | *Multi-type index, metadata filtering* |
|        |            |                                                                              |                                        |
|        |            | > -- Access Notes via EventKit or simple text input                          |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- Index alongside photos in same index (with type metadata)               |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- Filter search results by content type                                   |                                        |
+--------+------------+------------------------------------------------------------------------------+----------------------------------------+
| 3      | 1          | **Add performance overlay and settings**                                     | *SwiftUI state management*             |
|        |            |                                                                              |                                        |
|        |            | > -- Show query latency, index size, vector count                            |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- Settings: efSearch slider, distance metric picker, re-index button      |                                        |
|        |            | >                                                                            |                                        |
|        |            | > -- This overlay becomes part of your demo/portfolio                        |                                        |
+--------+------------+------------------------------------------------------------------------------+----------------------------------------+

**PK-013: Polish & Open-Source Ship** \[5 pts pts\]

*🎯 Learning goal: OSS publishing, DocC, technical writing*

+--------+------------+----------------------------------------------------------------------------+---------------------------------------+
| **\#** | **Points** | **Task**                                                                   | **Learn / Notes**                     |
+--------+------------+----------------------------------------------------------------------------+---------------------------------------+
| 1      | 2          | **Write comprehensive DocC documentation**                                 | *DocC documentation system*           |
|        |            |                                                                            |                                       |
|        |            | > -- Document every public type and method                                 |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Include a Getting Started tutorial article                            |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Add code examples in documentation comments                           |                                       |
+--------+------------+----------------------------------------------------------------------------+---------------------------------------+
| 2      | 2          | **Write README with architecture diagram and benchmarks**                  | *Technical writing, OSS presentation* |
|        |            |                                                                            |                                       |
|        |            | > -- Include: what it is, why it exists, quick start, architecture diagram |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Performance benchmarks table: latency, recall, index size             |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Demo app screenshots / GIF                                            |                                       |
+--------+------------+----------------------------------------------------------------------------+---------------------------------------+
| 3      | 1          | **Final code review and cleanup**                                          | *Ship-quality code standards*         |
|        |            |                                                                            |                                       |
|        |            | > -- Remove all TODOs and commented code                                   |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Ensure consistent access control (internal default, public API)       |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Run SwiftLint, fix all warnings                                       |                                       |
|        |            | >                                                                          |                                       |
|        |            | > -- Tag v1.0.0 release                                                    |                                       |
+--------+------------+----------------------------------------------------------------------------+---------------------------------------+

5\. Acceptance Criteria Summary

  ------------ -------------------------------------------------------------------- ------------------
  **Story**    **Key Acceptance Criteria**                                          **Must Pass**

  PK-002       All vDSP operations return identical results to manual calculation   Unit tests

  PK-003       Batch distance 5x+ faster than sequential at 10K vectors             Benchmark

  PK-004       BruteForceIndex returns correct results for all query types          Unit tests

  PK-005       NSW recall@10 \> 90% vs brute-force at 5K vectors                    Recall test

  PK-006       Multi-layer HNSW recall@10 \> 95% at efSearch=50, 10K/384d           Recall test

  PK-007       No data races under concurrent search + insert load                  TSan

  PK-008       Index survives save/load roundtrip with identical search results     Integration test

  PK-010       Core ML embeddings produce sensible similarity scores                Integration test

  PK-012       Semantic photo search returns relevant results for text queries      Manual QA

  PK-013       README, DocC, benchmarks, v1.0.0 tag on GitHub                       Review
  ------------ -------------------------------------------------------------------- ------------------

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **FINAL NOTE**                                                                                                                                                                                          |
|                                                                                                                                                                                                         |
| This PRD is designed to be fed to Claude Code story by story. Each task has enough context for Claude Code to generate high-quality code, but you should always understand the output before moving on. |
|                                                                                                                                                                                                         |
| The learning goals in purple are the whole point. The code is a byproduct of the learning.                                                                                                              |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
