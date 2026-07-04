# ``ProximaKit``

Pure-Swift vector similarity search for Apple platforms.

## Overview

ProximaKit lets you search content by meaning, not keywords. It converts text and images into vectors (lists of numbers that represent meaning), stores them in a fast search index, and finds the closest matches to any query.

Everything runs on-device using Apple's Accelerate framework. No internet, no API keys, no external dependencies.

Beyond dense search, ProximaKit ships hybrid BM25 + dense retrieval with rank fusion, product and INT8 scalar quantization for memory-constrained deployments, filtered search, and versioned binary persistence.

### Three Steps to Semantic Search

1. **Embed**: Convert your content into a ``Vector`` using an embedding provider
2. **Index**: Store vectors in an ``HNSWIndex`` or ``BruteForceIndex``
3. **Search**: Query the index to find the closest matches

```swift
import ProximaKit

let index = HNSWIndex(dimension: 384, metric: CosineDistance())
try await index.add(vector, id: UUID())
let results = await index.search(query: queryVector, k: 10)
```

## Topics

### Essentials

- <doc:MeetProximaKit>
- <doc:GettingStarted>
- ``Vector``
- ``HNSWIndex``
- ``BruteForceIndex``
- ``SearchResult``

### Distance Metrics

- ``DistanceMetric``
- ``CosineDistance``
- ``EuclideanDistance``
- ``DotProductDistance``
- ``ManhattanDistance``
- ``HammingDistance``
- ``ChebyshevDistance``
- ``BrayCurtisDistance``
- ``JensenShannonDistance``
- ``MahalanobisDistance``

### Index Protocol

- ``VectorIndex``
- ``HNSWConfiguration``
- ``IndexError``

### Hybrid & Keyword Search

- ``HybridIndex``
- ``HybridFusionStrategy``
- ``SparseIndex``
- ``SparseVectorIndex``
- ``BM25Configuration``
- ``BM25Tokenizer``
- ``DefaultBM25Tokenizer``

### Quantization

- ``ProductQuantizer``
- ``PQConfiguration``
- ``QuantizedHNSWIndex``
- ``ScalarQuantizer``
- ``ScalarQuantizedHNSWIndex``
- ``ProductQuantizerError``
- ``QuantizedIndexError``

### Paged Originals for Quantized Reranking (ADR-014, opt-in)

- ``QuantizedHNSWIndex/load(from:mode:)``
- ``QuantizedHNSWIndex/save(to:layout:)``
- ``QuantizedHNSWIndex/upgradeToV3(at:)``
- ``PQHWOpenMode``
- ``PQHWSaveLayout``
- ``QuantizedHNSWIndex/originalsArePaged``
- ``QuantizedHNSWIndex/mappedOriginalStorageBytes``

### Document Stores

- ``VectorStore``
- ``HybridVectorStore``
- ``TextEmbedder``
- ``ChunkMetadata``
- ``VectorStoreError``

### GPU Acceleration

- ``MetalBatchDistance``

### Batch Operations

- ``batchDotProducts(query:matrix:vectorCount:dimension:)``
- ``batchL2Distances(query:matrix:vectorCount:dimension:)``
- ``batchL1Distances(query:matrix:vectorCount:dimension:)``
- ``batchDistances(query:vectors:metric:)``
- ``batchDistances(query:matrix:vectorCount:dimension:metric:)``

### Persistence

- ``PersistenceEngine``
- ``PersistenceError``
- ``HNSWSnapshot``
- ``BruteForceSnapshot``
- ``SparseIndexSnapshot``
- ``DistanceMetricType``
- ``PersistenceEngine/upgradeToV3(at:)``

### Streaming Persistence (WAL, opt-in)

- ``HNSWIndex/open(baseURL:walURL:durability:mode:)``
- ``HNSWIndex/checkpoint(baseURL:walURL:durability:)``
- ``HNSWIndex/syncJournal()``
- ``HNSWIndex/needsCheckpoint(policy:)``
- ``HNSWIndex/closeJournal()``
- ``HNSWIndex/journalByteCount``
- ``HNSWIndex/journalRecordCount``
- ``HNSWIndex/currentGeneration``
- ``HNSWIndex/liveEntries()``
- ``WALDurability``
- ``WALCheckpointPolicy``

### Store Journaling (opt-in)

- ``VectorStore/open(name:embedder:storageDirectory:metric:config:durability:)``
- ``VectorStore/checkpoint()``
- ``VectorStore/needsCheckpoint(policy:)``
- ``HybridVectorStore/open(name:embedder:storageDirectory:metric:hnswConfig:bm25Config:tokenizer:fusion:durability:)``
- ``HybridVectorStore/checkpoint()``
- ``HybridVectorStore/needsCheckpoint(policy:)``
