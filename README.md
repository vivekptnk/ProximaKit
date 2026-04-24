<p align="center">

```
                                ·
                               /|\
                          · ─── ◆ ─── ·
                         /·\   /|\   /·\
                    · ──◆──── ◆───◆ ────◆── ·
                   / / /|\ \ /|\ /|\ / |\ \ \
                  · · · · · · · · · · · · · · ·

           ╔═══════════════════════════════════════════╗
           ║           P R O X I M A K I T             ║
           ║     Search by meaning, not keywords.      ║
           ╚═══════════════════════════════════════════╝
```

  <p align="center">
    Pure-Swift vector search for Apple platforms — powered by Accelerate.
  </p>
  <p align="center">
    <a href="https://github.com/vivekptnk/ProximaKit/actions/workflows/ci.yml"><img src="https://github.com/vivekptnk/ProximaKit/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
    <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-5.9+-F05138?logo=swift&logoColor=white" alt="Swift" /></a>
    <a href="https://developer.apple.com"><img src="https://img.shields.io/badge/Apple_Silicon-M1_M2_M3_M4-000000?logo=apple&logoColor=white" alt="Apple Silicon" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" /></a>
    <img src="https://img.shields.io/badge/tests-149_passing-brightgreen.svg" alt="Tests" />
  </p>
</p>

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

ProximaKit finds **similar content by understanding what it means** — not by matching keywords. Type "beach vacation" and it finds photos of oceans, notes about travel, articles about tropical destinations. None of them need to contain the words "beach" or "vacation."

Everything runs **on-device**. No server, no API key, no internet. Just your app and Apple Silicon.

> *HNSW implemented from scratch in Swift. Zero dependencies. Zero C++ wrappers.*

<table>
<tr>
<td align="center" width="33%">

```
  ┌─────────────┐
  │      ◆      │
  │    ╱   ╲    │
  │   ◆─────◆   │
  │  ON-DEVICE   │
  └─────────────┘
```

**No Cloud Required**<br/>
Runs entirely on Apple Silicon.<br/>
No server, no API key, no internet.

</td>
<td align="center" width="33%">

```
  ┌─────────────┐
  │    ┌───┐    │
  │    │ 0 │    │
  │    └───┘    │
  │  ZERO DEPS  │
  └─────────────┘
```

**Pure Swift**<br/>
Foundation + Accelerate only.<br/>
No C++ wrappers. No bridging.

</td>
<td align="center" width="33%">

```
  ┌─────────────┐
  │   L2: ·──·  │
  │   L1: ·─·─· │
  │   L0: ····· │
  │  HNSW BUILT │
  └─────────────┘
```

**From Scratch**<br/>
Full HNSW implementation.<br/>
Not a wrapper. Not a port.

</td>
</tr>
</table>

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Overview

ProximaKit is a pure-Swift approximate nearest-neighbour library built from scratch on Apple's Accelerate framework. It provides HNSW-based semantic search that runs entirely on-device — no server, no API key, no C++ wrapper required.

The library ships three targets: `ProximaKit` (core index + distance metrics + persistence), `ProximaEmbeddings` (text/image → vector converters using Apple's NaturalLanguage, Vision, and CoreML frameworks), and `ProximaDemo` (CLI) plus `ProximaDemoApp` (macOS SwiftUI app). All targets are distributed as a single Swift package.

ProximaKit is the foundation of the Chakravyuha stack and is used by TinyBrain (inference) and Lumen (knowledge retrieval) as their vector-search layer.

## Why ProximaKit?

| | ProximaKit | FAISS (C++) | Pinecone (Cloud) |
|---|---|---|---|
| **Language** | Pure Swift | C++ wrapper | REST API |
| **On-device** | Yes | Needs bridging | No (cloud only) |
| **Dependencies** | Zero | libfaiss, numpy | API key + internet |
| **Thread safety** | Swift actors (compile-time) | Manual locks | N/A |
| **iOS/macOS native** | Yes | No | No |
| **Setup time** | 30 seconds | Hours | Minutes + billing |

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Requirements

- macOS 14+ (macOS 15 recommended)
- Xcode 15+ / Swift 5.9+
- Apple Silicon (M1 or newer) — Accelerate SIMD paths are Apple Silicon–optimised

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/vivekptnk/ProximaKit.git", from: "1.0.0")
]
```

```swift
.target(
    name: "YourApp",
    dependencies: [
        "ProximaKit",          // Core: vectors, search indices, persistence
        "ProximaEmbeddings",   // Optional: turns text/images into vectors
    ]
)
```

## Quick Start

### Run the Demo

```bash
git clone https://github.com/vivekptnk/ProximaKit.git
cd ProximaKit
swift run ProximaDemo
```

Or open `Examples/ProximaDemoApp/ProximaDemoApp.xcodeproj` in Xcode for the full GUI experience.

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

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

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Demo

**ProximaDemoApp** is a macOS SwiftUI app that ships with the repo. It indexes 48 sample documents at startup and lets you search by meaning in real time, tune `efSearch` with a slider, add your own notes to the live index, and persist across app launches.

```
┌────────────────────────────────────────────────────────────────────────┐
│  ProximaDemoApp — semantic search over 48 sample documents              │
│                                                                          │
│  ┌────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │  efSearch  ─── 50  │  │  Query: "space exploration"              │  │
│  │  ▐██████████░░░░░░ │  │  ──────────────────────────────────────  │  │
│  │                    │  │  ●  0.41  Astronauts aboard the ISS...   │  │
│  │  Corpus: 48 docs   │  │  ●  0.44  NASA launched a new rover...  │  │
│  │  Dimension: 512d   │  │  ●  0.48  The moon landing changed...   │  │
│  │  Build:  ~0.9 s    │  │  ●  0.51  Scientists study black holes  │  │
│  │  Query:  ~104 ms   │  │  ●  0.55  The James Webb telescope...   │  │
│  │                    │  │                                          │  │
│  │  [  Add Note  ]    │  │  ●  dist < 0.55 — strong match          │  │
│  │  [  Add Image ]    │  │  ●  dist < 0.68 — partial match         │  │
│  │                    │  │  ●  dist ≥ 0.68 — weak match            │  │
│  └────────────────────┘  └──────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

Open in Xcode: `open Examples/ProximaDemoApp/ProximaDemoApp.xcodeproj`

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

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

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Search Images

```swift
let vision = VisionEmbeddingProvider()
let vector = try await vision.embed(myCGImage)
try await imageIndex.add(vector, id: photoID)

// Find visually similar images
let queryVector = try await vision.embed(anotherImage)
let similar = try await imageIndex.search(query: queryVector, k: 5)
```

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Save and Load

Don't rebuild the index every time your app launches.

```swift
// Save (compact binary format)
try await index.save(to: fileURL)

// Load (memory-mapped for instant startup)
let loaded = try HNSWIndex.load(from: fileURL)
```

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Use a Custom AI Model (CoreML)

For higher quality search, bring a real sentence-transformer model:

```swift
let provider = try CoreMLEmbeddingProvider(
    modelAt: modelURL,
    vocabURL: vocabURL   // WordPiece vocab for proper tokenization
)
let vector = try await provider.embed("sunset over the ocean")
```

To convert a HuggingFace model to CoreML, use [coremltools](https://github.com/apple/coremltools):

```bash
pip install coremltools transformers
python -c "
import coremltools as ct
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# Export to CoreML with coremltools.convert()
"
```

Place the exported `.mlmodelc` in `Models/` and ProximaKit will discover it automatically.

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Architecture

```
                ┌─────────────────────────────────────┐
                │         Y O U R   A P P             │
                │            (SwiftUI)                 │
                └──────────────┬──────────────────────┘
                               │
                     embed()   │   search()
                               │
           ┌───────────────────┼───────────────────────┐
           │                   │    ProximaEmbeddings   │
           │                   v                        │
           │   ┌──────────┐  ┌──────────┐  ┌────────┐ │
           │   │ NLEmbed  │  │ Vision   │  │ CoreML │ │
           │   │ Provider │  │ Provider │  │Provider│ │
           │   └─────┬────┘  └────┬─────┘  └───┬────┘ │
           │         └────────────┼─────────────┘      │
           │                      │                     │
           │        EmbeddingProvider protocol          │
           └──────────────────────┼────────────────────┘
                                  │
                        [Float] vectors
                                  │
           ┌──────────────────────┼────────────────────┐
           │                      v     ProximaKit     │
           │                                            │
           │   ┌────────────────────────────────────┐  │
           │   │  I N D E X   L A Y E R             │  │
           │   │                                     │  │
           │   │   HNSWIndex          BruteForce    │  │
           │   │   ◆──◆──◆             ◆ ◆ ◆ ◆     │  │
           │   │   │╲ │ ╱│             ◆ ◆ ◆ ◆     │  │
           │   │   ◆──◆──◆             ◆ ◆ ◆ ◆     │  │
           │   │   O(log n)            O(n)         │  │
           │   └──────────────┬─────────────────────┘  │
           │                  │                         │
           │   ┌──────────────┴─────────────────────┐  │
           │   │  D I S T A N C E   M E T R I C S   │  │
           │   │  cosine · euclidean · dot product   │  │
           │   │  manhattan · hamming                 │  │
           │   │       (vDSP / Accelerate)           │  │
           │   └──────────────┬─────────────────────┘  │
           │                  │                         │
           │   ┌──────────────┴─────────────────────┐  │
           │   │  P E R S I S T E N C E             │  │
           │   │  binary save · mmap load · compact  │  │
           │   └────────────────────────────────────┘  │
           │                                            │
           │   Foundation + Accelerate ONLY             │
           └────────────────────────────────────────────┘
```

| Module | What It Does |
|--------|-------------|
| `ProximaKit` | Core engine: vectors, distance metrics, HNSW graph search, persistence |
| `ProximaEmbeddings` | Converts text/images to vectors using Apple frameworks |
| `ProximaDemo` | Interactive demo app with live semantic search |

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Performance

```
 ╔══════════════════════════════════════════════════╗
 ║              P E R F O R M A N C E               ║
 ╠══════════════════════════════════════════════════╣
 ║                                                  ║
 ║  ⚡ Query          104 ms   ████████████░░░░░░  ║
 ║  ⚡ Cold start      50 ms   █████░░░░░░░░░░░░░  ║
 ║  ⚡ Build          ~3.0 s   ██████████████████░  ║
 ║                                                  ║
 ║  ◎ Recall@10 (1K)  98-99%  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ║
 ║  ◎ Recall@10 (10K)   87%+  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░ ║
 ║                                                  ║
 ║  ✓ Save/load roundtrip: exact binary match       ║
 ║  ✓ Memory-mapped I/O for instant startup         ║
 ║                                                  ║
 ╚══════════════════════════════════════════════════╝
```

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Which Index Should I Use?

| Index | When | Speed |
|-------|------|-------|
| `HNSWIndex` | **Most cases.** Fast approximate search, scales to millions. | O(log n) |
| `BruteForceIndex` | Under 1,000 items. 100% perfect accuracy. | O(n) |

Both have the exact same API. Swap them without changing any other code.

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Which Distance Metric?

| Metric | When | Plain English |
|--------|------|---------------|
| `CosineDistance()` | **Text search.** Use this unless you have a reason not to. | "How different is the direction?" |
| `EuclideanDistance()` | Spatial data (coordinates, sensors). | "How far apart are these?" |
| `DotProductDistance()` | Pre-normalized vectors (advanced). | "How aligned are these?" |
| `ManhattanDistance()` | Sparse data, grid-based problems. | "How many blocks apart?" |
| `HammingDistance()` | Binary/quantized vectors. | "How many bits differ?" |

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

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

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Thread Safety

ProximaKit is fully thread-safe. Both indices are Swift `actor` types — search from any thread, no crashes, no data races. The compiler enforces this at build time.

```swift
// Safe from any thread or Task:
let results = try await index.search(query: vector, k: 10)
try await index.add(newVector, id: UUID())
```

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

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
| `ManhattanDistance` | L1 / taxicab distance (sparse data). |
| `HammingDistance` | Count of differing positions (binary vectors). |
| `SearchResult` | Result: `id`, `distance`, `metadata`. |
| `HNSWConfiguration` | Tuning: `m`, `efConstruction`, `efSearch`. |
| `PersistenceEngine` | Binary save/load with memory mapping. |

### ProximaEmbeddings (content to vectors)

| Type | What |
|------|------|
| `NLEmbeddingProvider` | Text to vector. Apple's built-in model. No setup. |
| `VisionEmbeddingProvider` | Image to vector. Apple's Vision framework. |
| `CoreMLEmbeddingProvider` | Any CoreML model (BERT, MiniLM, etc). |
| `WordPieceTokenizer` | BERT-compatible tokenizer for CoreML models. |

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Design Decisions

See [`docs/adr/`](docs/adr/) for Architecture Decision Records:
- [ADR-001](docs/adr/ADR-001-accelerate-for-math.md): Why Accelerate/vDSP for all vector math
- [ADR-002](docs/adr/ADR-002-actor-isolation.md): Why actors for thread safety
- [ADR-003](docs/adr/ADR-003-binary-persistence.md): Why custom binary (not JSON)
- [ADR-004](docs/adr/ADR-004-hnsw-heuristic-selection.md): Why heuristic neighbor selection

<p align="center">◇ ── ◆ ── ◇ ── ◆ ── ◇</p>

## Building & Testing

```bash
# Build
swift build

# Unit + integration tests (fast)
swift test --skip RecallBenchmarkTests

# Full recall benchmarks (slow, needs Release mode)
swift test -c release --filter RecallBenchmarkTests

# Generate DocC documentation
swift package generate-documentation --target ProximaKit
```

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed plan. Highlights:

| Area | Status |
|------|--------|
| Additional distance metrics — Mahalanobis, Chebyshev, Bray-Curtis | Planned |
| GPU acceleration — Metal/MPSGraph backend for batch index builds | Planned |
| Binary quantization — INT8 scalar, product quantization (PQ) | Planned |
| Filtered search — pre-filter by metadata predicate before ANN | Planned |
| ADR backlog — quantization strategy, filtered search design | In progress |
| Demo app — iOS target, CoreML model download UI, result export | Planned |

Items flagged in the [documentation audit](../docs/DOCUMENTATION-AUDIT.md) (CONTRIBUTING.md polish, CHANGELOG.md, demo app README expansion) are tracked in the roadmap but are out of scope for this release.

<p align="center">◆ ─────── ◇ ─────── ◆ ─────── ◇ ─────── ◆</p>

## License

MIT — use it for anything.

**Author:** [Vivek Pattanaik](https://github.com/vivekptnk)
