# ProximaKit Architecture

> Read this before modifying any core component. For decision rationale, see ADRs in `docs/adr/`.

## Module Diagram

```
┌──────────────────────────────────────────────────┐
│                   Consumer App                    │
├──────────────────────────────────────────────────┤
│              ProximaEmbeddings                     │
│  NLEmbeddingProvider | CoreMLProvider | Vision    │
│              ▼ EmbeddingProvider (protocol)        │
├──────────────────────────────────────────────────┤
│              ProximaKit (Core)                     │
│                                                    │
│  VectorIndex (protocol, actor-isolated)            │
│    ├── BruteForceIndex (exact, O(n))              │
│    └── HNSWIndex (approximate, O(log n))          │
│                                                    │
│  Vector (value type, vDSP math)                   │
│  DistanceMetric (protocol: Cosine, L2, Dot)       │
│  PersistenceEngine (mmap binary files)            │
│  SearchResult (id, distance, metadata)            │
│                                                    │
│  Imports: Foundation, Accelerate ONLY              │
└──────────────────────────────────────────────────┘
```

## Data Flow

**Index:** Content → EmbeddingProvider.embed() → Vector → VectorIndex.add()
**Query:** Text → embed() → Vector → VectorIndex.search() → [SearchResult]
**Persist:** VectorIndex → PersistenceEngine.save() → binary file → .load() → VectorIndex

## Module Rules (enforced by hooks)

| Module | Can Import | Cannot Import |
|--------|-----------|---------------|
| ProximaKit | Foundation, Accelerate | ProximaEmbeddings, UIKit, SwiftUI |
| ProximaEmbeddings | ProximaKit, CoreML, NaturalLanguage, Vision | UIKit, SwiftUI |
| Demo App | Everything | — |

## Concurrency Model
- Indices are `actor`-isolated. Reads queue, writes serialize.
- `Vector` and `SearchResult` are `Sendable` value types.
- Batch embedding uses `TaskGroup` for parallelism.

## Performance-Critical Paths
1. Distance computation → vDSP batch ops (ADR-001)
2. HNSW neighbor selection → heuristic algorithm (ADR-004)
3. Index loading → memory-mapped vectors (ADR-003)
4. Beam search → efficient min-heap on layer 0

## Extension Points

| Add a... | Location | Conform to |
|----------|----------|-----------|
| Distance metric | Sources/ProximaKit/Distance/ | `DistanceMetric` |
| Index type | Sources/ProximaKit/Index/ | `VectorIndex` (actor) |
| Embedding provider | Sources/ProximaEmbeddings/ | `EmbeddingProvider` |
