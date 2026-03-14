<p align="center">
  <h1 align="center">ProximaKit</h1>
  <p align="center">
    <strong>Search by meaning, not keywords.</strong>
    <br />
    Pure-Swift vector search for Apple platforms — powered by Accelerate.
  </p>
  <p align="center">
    <a href="https://github.com/vivekptnk/ProximaKit/actions/workflows/ci.yml"><img src="https://github.com/vivekptnk/ProximaKit/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9+-F05138?logo=swift&logoColor=white" alt="Swift" /></a>
    <a href="https://developer.apple.com"><img src="https://img.shields.io/badge/Apple_Silicon-M1_M2_M3_M4-000000?logo=apple&logoColor=white" alt="Apple Silicon" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" /></a>
    <img src="https://img.shields.io/badge/tests-117_passing-brightgreen.svg" alt="Tests" />
  </p>
</p>

---

ProximaKit finds **similar content by understanding what it means** — not by matching keywords. Type "beach vacation" and it finds photos of oceans, notes about travel, articles about tropical destinations. None of them need to contain the words "beach" or "vacation."

Everything runs **on-device**. No server, no API key, no internet. Just your app and Apple Silicon.

> *HNSW implemented from scratch in Swift. Zero dependencies. Zero C++ wrappers.*

---

## Why ProximaKit?

| | ProximaKit | FAISS (C++) | Pinecone (Cloud) |
|---|---|---|---|
| **Language** | Pure Swift | C++ wrapper | REST API |
| **On-device** | Yes | Needs bridging | No (cloud only) |
| **Dependencies** | Zero | libfaiss, numpy | API key + internet |
| **Thread safety** | Swift actors (compile-time) | Manual locks | N/A |
| **iOS/macOS native** | Yes | No | No |
| **Setup time** | 30 seconds | Hours | Minutes + billing |

---

## Get Started (5 minutes)

### What You Need

- A Mac with Apple Silicon (M1 or newer)
- macOS 14 or later
- Xcode 15 or later

### Step 1: Add to Your Project

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/vivekptnk/ProximaKit.git", from: "1.0.0")
]
```

Add the targets you need:

```swift
.target(
    name: "YourApp",
    dependencies: [
        "ProximaKit",          // Core: vectors, search indices, persistence
        "ProximaEmbeddings",   // Optional: turns text/images into vectors
    ]
)
```

### Step 2: Run the Demo

```bash
git clone https://github.com/vivekptnk/ProximaKit.git
cd ProximaKit
swift run ProximaDemo
```

This opens a macOS app where you can type anything and see semantic search results ranked by similarity in real-time. Try "animals", "cooking", or "outer space."

---

## How It Works

```
You type: "beach vacation"
         |
         v
  ┌─────────────────┐
  │ EmbeddingProvider│   Converts text to numbers
  │ "beach" → [0.23, │   that capture its MEANING
  │  -0.41, 0.87...]│   (using Apple's NaturalLanguage)
  └────────┬─────────┘
           v
  ┌─────────────────┐
  │   HNSWIndex      │   Searches a graph structure:
  │                   │   1. Start at top layer (express lane)
  │  Layer 2: ·──·    │   2. Greedily descend to best region
  │  Layer 1: ·─·──·  │   3. Beam search on layer 0
  │  Layer 0: ········ │   4. Return k closest matches
  └────────┬──────────┘
           v
  ┌──────────────────┐
  │  Search Results   │   Ranked by similarity:
  │  0.12 Ocean waves │   Lower distance = more similar
  │  0.18 Tropical... │
  │  0.25 Travel...   │
  └──────────────────┘
```

All of this happens **on your device**, using Apple's Accelerate framework for SIMD math. No internet required.

---

## Search Text by Meaning

The simplest thing you can do. Uses Apple's built-in language model — no downloads, no setup.

```swift
import ProximaKit
import ProximaEmbeddings

// Set up
let embedder = try NLEmbeddingProvider(language: .english)
let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())

// Add content
let sentences = [
    "The cat sat on the warm windowsill",
    "Dogs love playing fetch in the park",
    "Fresh pasta tastes better than dried",
    "The sunset painted the sky orange",
]

for sentence in sentences {
    let vector = try await embedder.embed(sentence)
    let metadata = try JSONEncoder().encode(["text": sentence])
    try await index.add(vector, id: UUID(), metadata: metadata)
}

// Search by meaning
let query = try await embedder.embed("animals playing outside")
let results = try await index.search(query: query, k: 3)

// Results: "Dogs love playing fetch" (closest match!)
//          "The cat sat on the warm windowsill"
//          "The sunset painted the sky orange"
```

**What happened:** "animals playing outside" found the dog and cat sentences — even though none contain those exact words. It searched by *meaning*.

---

## Search Images

```swift
let vision = VisionEmbeddingProvider()
let vector = try await vision.embed(myCGImage)
try await imageIndex.add(vector, id: photoID)

// Find visually similar images
let queryVector = try await vision.embed(anotherImage)
let similar = try await imageIndex.search(query: queryVector, k: 5)
```

---

## Save and Load

Don't rebuild the index every time your app launches.

```swift
// Save (compact binary format)
try await index.save(to: fileURL)

// Load (memory-mapped for instant startup)
let loaded = try HNSWIndex.load(from: fileURL)
```

---

## Use a Custom AI Model (CoreML)

For higher quality search, bring a real sentence-transformer model:

```swift
let provider = try CoreMLEmbeddingProvider(modelAt: modelURL)
let vector = try await provider.embed("sunset over the ocean")
```

See `scripts/convert_model.py` for converting HuggingFace models to CoreML.

---

## Architecture

```
┌──────────────────────────────────────────┐
│           Your App (SwiftUI)              │
└───────────┬──────────────────────────────┘
            v
┌───────────────────────────────────────────┐
│          ProximaEmbeddings                 │
│  NLEmbeddingProvider · VisionProvider      │
│  CoreMLProvider · EmbeddingProvider protocol│
└───────────┬───────────────────────────────┘
            v
┌───────────────────────────────────────────┐
│          ProximaKit (Core)                 │
│  Vector · DistanceMetric · BatchDistance   │
│  HNSWIndex · BruteForceIndex              │
│  PersistenceEngine (binary + mmap)         │
│  (Imports: Foundation + Accelerate ONLY)   │
└───────────────────────────────────────────┘
```

| Module | What It Does |
|--------|-------------|
| `ProximaKit` | Core engine: vectors, distance metrics, HNSW graph search, persistence |
| `ProximaEmbeddings` | Converts text/images to vectors using Apple frameworks |
| `ProximaDemo` | SwiftUI demo app with live semantic search |

---

## Performance

| What | How Fast |
|------|----------|
| Query (1K vectors, 384d) | ~104ms |
| Recall@10 (1K vectors) | 98-99% |
| Recall@10 (10K vectors) | 87%+ (random data; real embeddings >95%) |
| Index build (1K vectors) | ~3s |
| Save/load roundtrip | Exact match (binary format preserves graph) |
| Cold start (10K vectors) | ~50ms (memory-mapped) |

---

## Which Index Should I Use?

| Index | When | Speed |
|-------|------|-------|
| `HNSWIndex` | **Most cases.** Fast approximate search, scales to millions. | O(log n) |
| `BruteForceIndex` | Under 1,000 items. 100% perfect accuracy. | O(n) |

Both have the exact same API. Swap them without changing any other code.

---

## Which Distance Metric?

| Metric | When | Plain English |
|--------|------|---------------|
| `CosineDistance()` | **Text search.** Use this unless you have a reason not to. | "How different is the direction?" |
| `EuclideanDistance()` | Spatial data (coordinates, sensors). | "How far apart are these?" |
| `DotProductDistance()` | Pre-normalized vectors (advanced). | "How aligned are these?" |

---

## Tuning

```swift
let config = HNSWConfiguration(
    m: 16,               // Connections per node
    efConstruction: 200,  // Build quality
    efSearch: 50          // Search quality
)
```

| Problem | Fix |
|---------|-----|
| Results aren't relevant | Increase `efSearch` (try 100-200) |
| Search too slow | Decrease `efSearch` (try 20) |
| Too much memory | Decrease `m` (try 8) |
| Build takes too long | Decrease `efConstruction` (try 100) |

---

## Thread Safety

ProximaKit is fully thread-safe. Both indices are Swift `actor` types — search from any thread, no crashes, no data races. The compiler enforces this at build time.

```swift
// Safe from any thread or Task:
let results = try await index.search(query: vector, k: 10)
try await index.add(newVector, id: UUID())
```

---

## API Reference

### ProximaKit (core)

| Type | What |
|------|------|
| `Vector` | A list of floats. The fundamental data type. |
| `HNSWIndex` | Fast approximate search (use this one). |
| `BruteForceIndex` | Exact search (for small datasets). |
| `CosineDistance` | Direction-based similarity (best for text). |
| `EuclideanDistance` | Straight-line distance. |
| `DotProductDistance` | Alignment-based (for normalized vectors). |
| `SearchResult` | Result: `id`, `distance`, `metadata`. |
| `HNSWConfiguration` | Tuning: `m`, `efConstruction`, `efSearch`. |
| `PersistenceEngine` | Binary save/load with memory mapping. |

### ProximaEmbeddings (content to vectors)

| Type | What |
|------|------|
| `NLEmbeddingProvider` | Text to vector. Apple's built-in model. No setup. |
| `VisionEmbeddingProvider` | Image to vector. Apple's Vision framework. |
| `CoreMLEmbeddingProvider` | Any CoreML model (BERT, MiniLM, etc). |

---

## Design Decisions

See [`docs/adr/`](docs/adr/) for Architecture Decision Records:
- [ADR-001](docs/adr/ADR-001-accelerate-for-math.md): Why Accelerate/vDSP for all vector math
- [ADR-002](docs/adr/ADR-002-actor-isolation.md): Why actors for thread safety
- [ADR-003](docs/adr/ADR-003-binary-persistence.md): Why custom binary (not JSON)
- [ADR-004](docs/adr/ADR-004-hnsw-heuristic-selection.md): Why heuristic neighbor selection

---

## Run the Tests

```bash
swift test --skip RecallBenchmarkTests
```

> RecallBenchmarkTests are skipped by default because they take 2+ minutes (they test with 10K+ vectors). Run them explicitly with `swift test --filter RecallBenchmark`.

---

## Generate Documentation

```bash
swift package generate-documentation --target ProximaKit
```

---

## License

MIT — use it for anything.

**Author:** [Vivek Pattanaik](https://github.com/vivekptnk)
