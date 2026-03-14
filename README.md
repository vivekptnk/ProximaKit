# ProximaKit

**Search by meaning, not keywords. On-device. Pure Swift.**

[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2017%20%7C%20macOS%2014%20%7C%20visionOS%201-blue.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-117%20passing-brightgreen.svg)]()

---

## What does this do?

ProximaKit lets your app **understand the meaning** of text and images, then find similar content instantly.

**Example:** A user types "beach vacation" into your app. ProximaKit finds photos of oceans, notes about travel plans, and articles about tropical destinations — even if none of them contain the words "beach" or "vacation."

This all runs **on the device**. No internet needed. No API keys. No cloud costs.

---

## Who is this for?

You're building an iOS or macOS app and you want to add:

- **Smart search** — users type a description, you find matching content by meaning
- **Photo search** — "sunset photos" finds all your sunset pics without manual tags
- **Recommendations** — "show me more like this" for articles, products, or music
- **Duplicate detection** — find similar items even when they're worded differently

If you've ever wished `String.contains()` was smarter, this is for you.

---

## Install it (30 seconds)

Add this to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vivekptnk/ProximaKit.git", from: "1.0.0")
]
```

Then add the targets you need:

```swift
.target(
    name: "YourApp",
    dependencies: [
        "ProximaKit",          // Core search engine
        "ProximaEmbeddings",   // Turns text/images into searchable vectors (optional)
    ]
)
```

---

## How it works (the simple version)

ProximaKit works in 3 steps:

```
1. EMBED     "sunset on the beach"  -->  [0.23, -0.41, 0.87, ...]  (a list of numbers)
2. STORE     Put those numbers in a search index
3. SEARCH    Give it new numbers, it finds the closest matches
```

The list of numbers (called a "vector") captures the **meaning** of the text or image. Similar meanings produce similar numbers. ProximaKit finds which stored numbers are closest to your query numbers.

That's it. Everything below is just the code to do those 3 steps.

---

## Search text by meaning (copy-paste example)

This is the simplest thing you can do with ProximaKit. It uses Apple's built-in language model — no downloads, no setup.

```swift
import ProximaKit
import ProximaEmbeddings

// Step 1: Set up the embedding provider (turns text into numbers)
let embedder = try NLEmbeddingProvider(language: .english)

// Step 2: Create a search index
let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())

// Step 3: Add some content
let sentences = [
    "The cat sat on the warm windowsill",
    "Dogs love playing fetch in the park",
    "Fresh pasta tastes better than dried",
    "The sunset painted the sky orange",
    "Machine learning models recognize objects",
]

for sentence in sentences {
    let vector = try await embedder.embed(sentence)
    let metadata = try JSONEncoder().encode(["text": sentence])
    try await index.add(vector, id: UUID(), metadata: metadata)
}

// Step 4: Search!
let query = try await embedder.embed("animals playing outside")
let results = try await index.search(query: query, k: 3)

// Results: "Dogs love playing fetch" (closest match!)
//          "The cat sat on the warm windowsill"
//          "The sunset painted the sky orange"
for result in results {
    if let data = result.metadata,
       let info = try? JSONDecoder().decode([String: String].self, from: data) {
        print("\(info["text"] ?? "") — distance: \(result.distance)")
    }
}
```

**What just happened:** You typed "animals playing outside" and it found the dog and cat sentences — even though none of them contain the words "animals", "playing", or "outside." It searched by *meaning*.

---

## Search images (photo search)

```swift
import ProximaEmbeddings

// Turn an image into a searchable vector
let vision = VisionEmbeddingProvider()
let vector = try await vision.embed(myCGImage)

// Add it to the index (use a separate index — image vectors have different dimensions)
let imageIndex = HNSWIndex(dimension: vector.dimension, metric: CosineDistance())
try await imageIndex.add(vector, id: photoID)

// Later: find similar images
let queryVector = try await vision.embed(anotherImage)
let similar = try await imageIndex.search(query: queryVector, k: 5)
// Returns the 5 most visually similar images
```

---

## Save and load your index

Building an index takes time. Save it to disk so you don't have to rebuild it every time your app launches.

```swift
// Save (writes a binary file — fast and compact)
let fileURL = documentsDirectory.appendingPathComponent("my-index.proximakit")
try await index.save(to: fileURL)

// Load (memory-mapped for fast startup — loads in milliseconds, not seconds)
let loaded = try HNSWIndex.load(from: fileURL)

// Use it exactly like before
let results = try await loaded.search(query: queryVector, k: 10)
```

---

## Use a custom AI model (CoreML)

For higher quality search, use a real sentence-transformer model converted to CoreML:

```swift
let provider = try CoreMLEmbeddingProvider(modelAt: modelURL)
let vector = try await provider.embed("sunset over the ocean")
// Produces higher-quality embeddings than NLEmbeddingProvider
```

See `scripts/convert_model.py` for converting HuggingFace models to CoreML format.

---

## Which index should I use?

| Index | When to use | Speed |
|-------|------------|-------|
| `HNSWIndex` | **Most cases.** Fast approximate search. Works great up to millions of vectors. | Fast |
| `BruteForceIndex` | Tiny datasets (under 1,000 items) where you need 100% perfect results. | Slower as dataset grows |

```swift
// For most apps:
let index = HNSWIndex(dimension: 384, metric: CosineDistance())

// For tiny datasets where perfect accuracy matters:
let index = BruteForceIndex(dimension: 384, metric: CosineDistance())
```

Both have the exact same API — `add()`, `search()`, `remove()`. You can swap them without changing any other code.

---

## Which distance metric should I use?

| Metric | When to use | One-liner |
|--------|------------|-----------|
| `CosineDistance()` | **Text search.** This is the default. Use this unless you have a reason not to. | "How different is the direction?" |
| `EuclideanDistance()` | Spatial data (coordinates, sensor readings). | "How far apart are these points?" |
| `DotProductDistance()` | Pre-normalized vectors (advanced optimization). | "How aligned are these?" |

**If you're not sure, use `CosineDistance()`.** It's the standard for text and most AI embeddings.

---

## Tuning performance

If search is too slow or results aren't good enough, adjust these numbers:

```swift
let config = HNSWConfiguration(
    m: 16,               // Connections per node. Higher = better results, more memory.
    efConstruction: 200,  // Quality of index building. Higher = slower build, better search.
    efSearch: 50          // How hard to search. Higher = better results, slower queries.
)
let index = HNSWIndex(dimension: 384, metric: CosineDistance(), config: config)
```

| Problem | Fix |
|---------|-----|
| Search results aren't relevant enough | Increase `efSearch` (try 100 or 200) |
| Search is too slow | Decrease `efSearch` (try 20) |
| Index uses too much memory | Decrease `m` (try 8) |
| Building the index takes too long | Decrease `efConstruction` (try 100) |

---

## Try the demo app

```bash
git clone https://github.com/vivekptnk/ProximaKit.git
cd ProximaKit
swift run ProximaDemo
```

This opens a macOS app where you can type anything and see semantic search results ranked by similarity in real-time.

---

## Thread safety

ProximaKit is fully thread-safe. Both `HNSWIndex` and `BruteForceIndex` are Swift `actor` types — you can search from multiple threads at the same time without crashes or data corruption. Just use `await`:

```swift
// This is safe from any thread or Task:
let results = try await index.search(query: vector, k: 10)
try await index.add(newVector, id: UUID())
```

---

## What it's built with

- **Swift 5.9** with strict concurrency
- **Apple Accelerate** (vDSP) for SIMD-optimized vector math
- **Zero external dependencies** — only Apple's own frameworks
- **HNSW algorithm** implemented from scratch (not a wrapper around C++ code)

Runs on: **iOS 17+, macOS 14+, visionOS 1+**

---

## API Reference

### ProximaKit (core)

| Type | What it does |
|------|-------------|
| `Vector` | A list of floats. The fundamental data type. |
| `HNSWIndex` | Fast approximate search index (use this one). |
| `BruteForceIndex` | Exact search index (for small datasets). |
| `CosineDistance` | Measures similarity by direction (best for text). |
| `EuclideanDistance` | Measures similarity by straight-line distance. |
| `DotProductDistance` | Measures similarity by alignment (for normalized vectors). |
| `SearchResult` | What you get back from a search: `id`, `distance`, `metadata`. |
| `HNSWConfiguration` | Tuning knobs: `m`, `efConstruction`, `efSearch`. |

### ProximaEmbeddings (turns content into vectors)

| Type | What it does |
|------|-------------|
| `NLEmbeddingProvider` | Text to vector using Apple's built-in language model. No setup needed. |
| `VisionEmbeddingProvider` | Image to vector using Apple's Vision framework. |
| `CoreMLEmbeddingProvider` | Text/image to vector using any CoreML model you provide. |

---

## License

MIT — use it for anything.
