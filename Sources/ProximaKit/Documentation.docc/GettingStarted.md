# Getting Started with ProximaKit

Build your first semantic search in under 5 minutes.

## Overview

This tutorial walks you through adding ProximaKit to your project, indexing some text, and running your first semantic search query.

### Step 1: Add ProximaKit to Your Project

In your `Package.swift`, add ProximaKit as a dependency:

```swift
dependencies: [
    .package(url: "https://github.com/vivekptnk/ProximaKit.git", from: "1.0.0")
]
```

Then add it to your target:

```swift
.target(name: "YourApp", dependencies: ["ProximaKit", "ProximaEmbeddings"])
```

### Step 2: Create an Embedding Provider

An embedding provider converts text into vectors — lists of numbers that capture meaning. ProximaKit ships with `NLEmbeddingProvider`, which uses Apple's built-in language model.

```swift
import ProximaEmbeddings

let embedder = try NLEmbeddingProvider(language: .english)
```

No downloads, no API keys. It just works.

### Step 3: Build a Search Index

Create an index and add some content:

```swift
import ProximaKit

let index = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())

let sentences = [
    "The cat sat on the windowsill",
    "Dogs love playing fetch",
    "Fresh pasta tastes amazing",
    "The sunset was beautiful",
]

for sentence in sentences {
    let vector = try await embedder.embed(sentence)
    let metadata = try JSONEncoder().encode(["text": sentence])
    try await index.add(vector, id: UUID(), metadata: metadata)
}
```

### Step 4: Search by Meaning

```swift
let query = try await embedder.embed("pets and animals")
let results = try await index.search(query: query, k: 2)

for result in results {
    if let data = result.metadata,
       let info = try? JSONDecoder().decode([String: String].self, from: data) {
        print("\(info["text"] ?? "") — distance: \(result.distance)")
    }
}
// Output:
//   Dogs love playing fetch — distance: 0.15
//   The cat sat on the windowsill — distance: 0.22
```

The query "pets and animals" found the dog and cat sentences — even though none of them contain those exact words.

### Step 5: Save Your Index

Don't rebuild the index every time your app launches:

```swift
try await index.save(to: fileURL)

// Next launch:
let loaded = try HNSWIndex.load(from: fileURL)
```

### Next Steps

- Use ``VisionEmbeddingProvider`` for image search
- Use ``CoreMLEmbeddingProvider`` for higher quality with custom models
- Adjust ``HNSWConfiguration`` to tune recall vs speed
