# ``ProximaKit``

Pure-Swift vector similarity search for Apple platforms.

## Overview

ProximaKit lets you search content by meaning, not keywords. It converts text and images into vectors (lists of numbers that represent meaning), stores them in a fast search index, and finds the closest matches to any query.

Everything runs on-device using Apple's Accelerate framework. No internet, no API keys, no external dependencies.

### Three Steps to Semantic Search

1. **Embed**: Convert your content into a ``Vector`` using an embedding provider
2. **Index**: Store vectors in an ``HNSWIndex`` or ``BruteForceIndex``
3. **Search**: Query the index to find the closest matches

```swift
import ProximaKit

let index = HNSWIndex(dimension: 384, metric: CosineDistance())
try await index.add(vector, id: UUID())
let results = try await index.search(query: queryVector, k: 10)
```

## Topics

### Essentials

- ``Vector``
- ``HNSWIndex``
- ``BruteForceIndex``
- ``SearchResult``

### Distance Metrics

- ``DistanceMetric``
- ``CosineDistance``
- ``EuclideanDistance``
- ``DotProductDistance``

### Index Protocol

- ``VectorIndex``
- ``HNSWConfiguration``
- ``IndexError``

### Batch Operations

- ``batchDotProducts(query:matrix:vectorCount:dimension:)``
- ``batchDistances(query:vectors:metric:)``
- ``batchDistances(query:matrix:vectorCount:dimension:metric:)``

### Persistence

- ``PersistenceEngine``
- ``PersistenceError``
- ``HNSWSnapshot``
- ``BruteForceSnapshot``
- ``DistanceMetricType``
