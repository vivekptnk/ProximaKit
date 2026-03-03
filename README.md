# ProximaKit

**Pure-Swift vector search, powered by Accelerate.**

[![CI](https://github.com/vivek/ProximaKit/actions/workflows/ci.yml/badge.svg)](https://github.com/vivek/ProximaKit/actions)
[![Swift 5.9+](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-iOS%2017%20%7C%20macOS%2014-blue.svg)](https://developer.apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ProximaKit is a zero-dependency semantic search library for Apple platforms. All vector math runs through Apple's Accelerate framework. HNSW (Hierarchical Navigable Small World) is implemented from scratch in Swift — no C++ wrappers, no cloud APIs.

## Quick Start

```swift
import ProximaKit

let index = HNSWIndex(dimension: 384, metric: .cosine)
try await index.add(Vector([0.1, 0.2, ...]), id: UUID())
let results = try await index.search(query: queryVector, k: 10)
```

## Performance

| Metric | 10K/384d | Target |
|--------|---------|--------|
| HNSW Query p50 | ~5ms | <5ms |
| HNSW Query p99 | ~40ms | <50ms |
| Recall@10 (ef=50) | ~96% | >95% |
| Index Build | ~3s | <5s |
| Cold Start | ~90ms | <200ms |

Full benchmark suite: `swift test --filter Benchmark`. Strategy: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Architecture

```
ProximaKit (core)         ProximaEmbeddings
├── Vector (vDSP math)    ├── NLEmbeddingProvider
├── BruteForceIndex       ├── CoreMLProvider
├── HNSWIndex             └── VisionProvider
├── PersistenceEngine
└── DistanceMetric
```

Imports: Foundation + Accelerate (core), CoreML/NaturalLanguage/Vision (embeddings).

Full docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design decisions: [`docs/adr/`](docs/adr/)

## AI-Native Development

This repo is built for **harness-driven development**. Rules aren't documented and hoped-for — they're enforced by hooks that block your tool calls:

```
Hook                        What It Enforces
──────────────────────────────────────────────────────
gate-main-branch.sh         No edits on main — use feature branches
gate-module-bounds.py       No third-party imports, no boundary violations  
validate-swift-write.py     Build check + no force unwraps + no manual loops
auto-test.py                Auto-runs tests when test files change
stop-check.py               Can't stop with failing tests or dirty state
perf-guard.py               Reminds to benchmark perf-critical changes
prompt-context.py           Injects context + suggests the right model
```

### Model Routing

Different models for different jobs. The prompt hook auto-suggests:

| Model | Role | Use For |
|-------|------|---------|
| **Opus** | Architect | HNSW algorithm, ADRs, deep debugging, interviews |
| **Sonnet** | Builder | Daily features, tests, reviews, bug fixes |
| **Haiku** | Sprinter | Scaffolding, renaming, formatting, boilerplate |

Full strategy: [`docs/MODEL_GUIDE.md`](docs/MODEL_GUIDE.md)

## Contributing

```bash
git clone https://github.com/vivek/ProximaKit.git && cd ProximaKit
claude                   # Sonnet for most work
> /onboard               # Get oriented (any model)
> /fix-issue 42          # Pick up an issue (Sonnet)
> /review                # Self-check before PR (Sonnet)
```

Or for deep work:
```bash
claude --model opus      # Switch to Opus
> /explain HNSW          # Learn the algorithm
> /architect "topic"     # Make a design decision
> /interview HNSW        # Mock interview prep
```

Full guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Installation

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/vivek/ProximaKit.git", from: "1.0.0")
]
```

## Requirements

Swift 5.9+ · iOS 17+ / macOS 14+ / visionOS 1+ · Xcode 15+

## License

MIT
