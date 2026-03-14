# ProximaKit

**Pure-Swift vector search, powered by Accelerate.**

[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2017%20%7C%20macOS%2014%20%7C%20visionOS%201-blue.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-117%20passing-brightgreen.svg)]()

ProximaKit is a zero-dependency semantic search library for Apple platforms. All vector math runs through Apple's Accelerate framework (vDSP/SIMD). HNSW is implemented from scratch in Swift — no C++ wrappers, no cloud APIs, no third-party dependencies.

## Quick Start

```swift
import ProximaKit

// Create an index
let index = HNSWIndex(dimension: 384, metric: CosineDistance())

// Add vectors
let vector = Vector([0.1, 0.2, 0.3, ...])  // 384 floats
try await index.add(vector, id: UUID())

// Search
let query = Vector([0.15, 0.18, 0.35, ...])
let results = try await index.search(query: query, k: 10)

for result in results {
    print("\(result.id): distance \(result.distance)")
}
```

## With Embeddings (Text Search)

```swift
import ProximaEmbeddings

// Convert text to vectors using Apple's NaturalLanguage framework
let provider = try NLEmbeddingProvider(language: .english)

let vector = try await provider.embed("sunset over the ocean")
try await index.add(vector, id: UUID())

// Search by meaning, not keywords
let query = try await provider.embed("beach at dusk")
let results = try await index.search(query: query, k: 5)
```

## With Images (Photo Search)

```swift
let vision = VisionEmbeddingProvider()
let imageVector = try await vision.embed(cgImage)
try await index.add(imageVector, id: photoID)
```

## Save & Load

```swift
// Save to disk (binary format, ~50ms cold start for 10K vectors)
try await index.save(to: fileURL)

// Load back
let loaded = try HNSWIndex.load(from: fileURL)
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Your App                              │
├──────────────────────────────────────────────────────────┤
│              ProximaEmbeddings                             │
│  NLEmbeddingProvider  │  VisionProvider  │  CoreMLProvider │
│              ▼ EmbeddingProvider (protocol)                │
├──────────────────────────────────────────────────────────┤
│              ProximaKit (Core)                             │
│                                                           │
│  VectorIndex (protocol, actor-isolated)                   │
│    ├── BruteForceIndex (exact, O(n))                     │
│    └── HNSWIndex (approximate, O(log n))                  │
│                                                           │
│  Vector ─── vDSP math (dot, cosine, L2, normalize)       │
│  DistanceMetric ─── Cosine, Euclidean, DotProduct         │
│  BatchDistance ─── vDSP_mmul (10K distances in 1 call)    │
│  PersistenceEngine ─── binary save/load with mmap         │
│                                                           │
│  Imports: Foundation + Accelerate ONLY                    │
└──────────────────────────────────────────────────────────┘
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Recall@10 (1K vectors, ef=50) | 98-99% | Euclidean, random data |
| Recall@10 (10K vectors, ef=100) | 87% | Random data; real embeddings achieve >95% |
| Query latency (1K/384d) | ~104ms | Single query, CosineDistance |
| Index build (1K vectors) | ~3s | M=16, efConstruction=200 |
| Save/Load roundtrip | Exact match | Binary format preserves full graph |

Full benchmark suite: `swift test --filter RecallBenchmark`

## Modules

| Module | What | Imports |
|--------|------|---------|
| **ProximaKit** | Vector math, indices, persistence | Foundation, Accelerate |
| **ProximaEmbeddings** | Text/image → Vector providers | CoreML, NaturalLanguage, Vision |
| **ProximaDemo** | SwiftUI semantic search app | SwiftUI + above |

## Demo App

```bash
swift run ProximaDemo
```

Launches a macOS SwiftUI app that indexes 50 sample sentences and lets you search by meaning. Type "animals" to find cat/dog sentences. Type "cooking" to find food sentences. Shows similarity distance and query latency.

## Distance Metrics

```swift
CosineDistance()       // Best for text embeddings (direction, not magnitude)
EuclideanDistance()    // Best for spatial data (straight-line distance)
DotProductDistance()   // Fast for pre-normalized vectors
```

## Configuration

```swift
let config = HNSWConfiguration(
    m: 16,              // Max connections per node (higher = better recall, more memory)
    efConstruction: 200, // Beam width during build (higher = better graph quality)
    efSearch: 50         // Beam width during query (higher = better recall, slower)
)
let index = HNSWIndex(dimension: 384, metric: CosineDistance(), config: config)
```

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/vivek/ProximaKit.git", from: "1.0.0")
]
```

Import only what you need:
```swift
import ProximaKit           // Core: vectors, indices, persistence
import ProximaEmbeddings    // Optional: NL, Vision, CoreML providers
```

## Requirements

Swift 5.9+ | iOS 17+ | macOS 14+ | visionOS 1+ | Xcode 15+

## Design Decisions

See [`docs/adr/`](docs/adr/) for Architecture Decision Records:
- [ADR-001](docs/adr/ADR-001-accelerate-for-math.md): Accelerate/vDSP for all vector math
- [ADR-002](docs/adr/ADR-002-actor-isolation.md): Actor isolation for index types
- [ADR-003](docs/adr/ADR-003-binary-persistence.md): Custom binary persistence format
- [ADR-004](docs/adr/ADR-004-hnsw-heuristic-selection.md): Heuristic neighbor selection

## License

MIT
