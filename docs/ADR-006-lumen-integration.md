# ADR-006: ProximaKit ↔ Lumen RAG Integration Design

**Status:** Draft — awaiting review by CTO-Lumen
**Date:** 2026-03-18
**Authors:** CTO – ProximaKit
**Reviewers:** CTO – Lumen, Eng – Lumen RAG

---

## Context

Lumen is building on-device document intelligence with RAG (Retrieval-Augmented Generation). ProximaKit is the natural vector search backend for the retrieval step. This document defines how Lumen document chunks flow into ProximaKit indices and what (if any) higher-level abstractions ProximaKit should expose.

### Current ProximaKit API (v1.1.0)

- `HNSWIndex` / `BruteForceIndex` — actor-isolated, single-index, `UUID`-keyed
- `EmbeddingProvider` protocol — text → `Vector`
- `PersistenceEngine` — binary save/load per index
- No collection abstraction, no auto-embedding, no document-level operations

### Lumen Requirements (Assumed)

1. Ingest documents → chunk into passages
2. Embed each chunk → store in vector index with metadata
3. Query: embed question → retrieve top-k chunks → feed to LLM context
4. Incremental: add/remove documents without full re-index
5. Multiple document collections (e.g., per-notebook or per-workspace)

---

## Decision

### 1. Integration Architecture

```
┌─────────────────────────────────────────────────┐
│                    Lumen App                     │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ Document  │──▶│ Chunker  │──▶│   RAG Engine │ │
│  │ Ingester  │   │          │   │              │ │
│  └──────────┘   └──────────┘   └──────┬───────┘ │
│                                       │         │
│                         ┌─────────────┼─────────┤
│                         ▼             ▼         │
│              ┌─────────────┐  ┌─────────────┐   │
│              │  Embedding  │  │  ProximaKit  │   │
│              │  Provider   │  │  VectorStore │   │
│              │ (ProximaEmb)│  │  (new layer) │   │
│              └─────────────┘  └─────────────┘   │
│                                       │         │
│                                       ▼         │
│                              ┌─────────────┐    │
│                              │  HNSWIndex  │    │
│                              │ (ProximaKit)│    │
│                              └─────────────┘    │
└─────────────────────────────────────────────────┘
```

### 2. New Abstraction: `VectorStore` (in ProximaKit)

A thin collection layer that bundles an index with an embedding provider, persistence location, and document-level operations.

```swift
/// A named, persistent vector collection with auto-embedding.
public actor VectorStore {
    /// The name of this collection (used for persistence).
    public let name: String

    /// The underlying vector index.
    public let index: HNSWIndex

    /// The embedding provider used for text operations.
    public let embedder: any EmbeddingProvider

    /// Directory where this store persists its index.
    public let storageURL: URL

    // MARK: - Initialization

    /// Creates or loads a named vector store.
    public init(
        name: String,
        embedder: any EmbeddingProvider,
        storageDirectory: URL,
        metric: any DistanceMetric = CosineDistance(),
        config: HNSWConfiguration = HNSWConfiguration()
    ) throws

    // MARK: - Document-Level Operations

    /// Adds text chunks with metadata, embedding them automatically.
    /// Returns the UUIDs assigned to each chunk.
    @discardableResult
    public func addChunks(
        _ chunks: [String],
        metadata: [ChunkMetadata]
    ) async throws -> [UUID]

    /// Queries the store with a text string.
    /// Embeds the query, searches the index, returns results with metadata.
    public func query(
        _ text: String,
        k: Int = 10,
        efSearch: Int? = nil,
        filter: (@Sendable (UUID) -> Bool)? = nil
    ) async throws -> [SearchResult]

    /// Removes all chunks associated with a document ID.
    public func removeDocument(id: String) async throws -> Int

    /// Persists the current state to disk.
    public func save() async throws

    /// Number of chunks in the store.
    public var count: Int { get async }
}
```

### 3. Chunk Metadata Schema

```swift
/// Metadata stored alongside each vector chunk.
public struct ChunkMetadata: Codable, Sendable {
    /// The source document identifier (Lumen's document ID).
    public let documentId: String

    /// Zero-based chunk index within the document.
    public let chunkIndex: Int

    /// The original text of this chunk (for retrieval display).
    public let text: String

    /// Optional additional metadata (title, page number, etc).
    public var extra: [String: String]?
}
```

### 4. Integration Flow (Lumen Side)

```swift
// 1. Setup — once per notebook/workspace
let embedder = try NLEmbeddingProvider()
let store = try VectorStore(
    name: "my-notebook",
    embedder: embedder,
    storageDirectory: appSupportURL
)

// 2. Ingest — when a document is added
let chunks = chunkDocument(document) // Lumen's chunking logic
let metadata = chunks.enumerated().map { i, text in
    ChunkMetadata(
        documentId: document.id,
        chunkIndex: i,
        text: text
    )
}
try await store.addChunks(chunks.map(\.text), metadata: metadata)
try await store.save()

// 3. Query — RAG retrieval
let results = try await store.query("What are the key findings?", k: 5)
let context = results.compactMap { result in
    result.decodeMetadata(as: ChunkMetadata.self)?.text
}
// Feed `context` array to LLM prompt
```

### 5. What Stays in Lumen (Not ProximaKit)

- **Document chunking logic** — domain-specific (paragraph splitting, overlap, size limits)
- **LLM prompt construction** — RAG prompt templates, context window management
- **Document lifecycle** — when to re-index, version tracking
- **UI concerns** — search results display, relevance highlighting

### 6. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Embed + add 1 chunk | < 50ms | NLEmbedding is fast; HNSW insert is O(log n) |
| Query (10K chunks, k=10) | < 200ms | HNSW O(log n) + embedding time |
| Cold start (load 10K index) | < 100ms | Memory-mapped persistence |
| Memory (10K × 512d) | < 25 MB | 10K × 512 × 4 bytes + graph overhead |

---

## Alternatives Considered

### A. No new abstraction — Lumen uses HNSWIndex directly

**Pros:** No ProximaKit changes needed, Lumen has full control.
**Cons:** Every consumer re-implements embedding + persistence + metadata wiring. Boilerplate-heavy.

**Rejected because:** The embedding+index+persist pattern will repeat across Lumen, TinyBrain, and future consumers. A thin wrapper pays for itself quickly.

### B. Full-featured document database (collections, schemas, migrations)

**Pros:** Feature-rich, handles complex use cases.
**Cons:** Massive scope creep. ProximaKit is a vector search library, not a database.

**Rejected because:** YAGNI. Start with `VectorStore`, add features as real consumers need them.

### C. VectorStore as a separate Swift package

**Pros:** Cleaner separation, optional dependency.
**Cons:** Adds package management overhead for a thin layer.

**Decision:** Ship `VectorStore` in the ProximaKit package for now. Extract if/when the abstraction grows complex enough to warrant it.

---

## Implementation Plan

### Phase 1: VectorStore (ProximaKit side)
1. Add `ChunkMetadata` type to ProximaKit
2. Implement `VectorStore` actor
3. Add persistence support (delegates to existing `PersistenceEngine`)
4. Add document-level remove (requires tracking documentId → [UUID] mapping)
5. Tests: add/query/remove/persist roundtrip
6. Benchmark: 10K chunks end-to-end

### Phase 2: Integration Prototype (Lumen side)
1. Add ProximaKit as SPM dependency in Lumen
2. Implement document chunker (paragraph-level, 200-500 token chunks)
3. Wire chunker → VectorStore → RAG query pipeline
4. Validate at 10K+ chunks with real documents

### Phase 3: Optimization
1. Batch embedding for bulk ingest
2. Background indexing (don't block UI thread)
3. Incremental persistence (save delta, not full index)

---

## Open Questions for CTO-Lumen

1. **Chunking strategy** — What chunk sizes and overlap does Lumen plan to use?
2. **Embedding model** — `NLEmbeddingProvider` (fast, 512d) vs CoreML sentence-transformer (better quality, slower)?
3. **Multi-collection** — Does Lumen need multiple independent indices (per notebook? per workspace?)?
4. **Document lifecycle** — When documents are edited, does Lumen re-chunk and re-embed, or append?
5. **Scale target** — What's the realistic upper bound? 10K chunks? 100K? 1M?

---

## Consequences

- ProximaKit gains a `VectorStore` type — small API surface addition
- Lumen gets a clean integration path without re-inventing embedding+persistence wiring
- Other Chakravyuha products (TinyBrain) can reuse the same pattern
- Performance targets are achievable with current HNSW + Accelerate architecture
