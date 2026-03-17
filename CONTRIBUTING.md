# Contributing to ProximaKit

## Mission

Build the best pure-Swift vector search library for Apple platforms — correct, fast, and easy to use.

---

## Development Setup

### Prerequisites

- Mac with Apple Silicon (M1 or newer)
- macOS 14+
- Xcode 15+ (Swift 5.9+)

### Clone and Build

```bash
git clone https://github.com/vivekptnk/ProximaKit.git
cd ProximaKit
swift build
```

### Run Tests

```bash
# Standard test suite (fast, ~30s)
swift test --skip RecallBenchmarkTests

# Full suite including recall benchmarks (~2-5 min)
swift test
```

### Run the Demo

```bash
# CLI demo
swift run ProximaDemo

# GUI demo (Xcode)
open Examples/ProximaDemoApp/ProximaDemoApp.xcodeproj
```

### Generate Documentation

```bash
swift package generate-documentation --target ProximaKit
```

---

## Repository Structure

```
ProximaKit/
├── Sources/
│   ├── ProximaKit/             # Core: vectors, indices, persistence
│   │   ├── Distance/           # DistanceMetric protocol + implementations
│   │   ├── Index/              # VectorIndex protocol, HNSW, BruteForce
│   │   ├── Persistence/        # Binary save/load with mmap
│   │   ├── Query/              # SearchResult type
│   │   └── Documentation.docc/ # DocC catalog
│   ├── ProximaEmbeddings/      # Text/image → vector providers
│   └── ProximaDemo/            # CLI demo executable
├── Tests/
│   ├── ProximaKitTests/        # Core unit + benchmark tests
│   └── ProximaEmbeddingsTests/ # Embedding provider tests
├── Examples/
│   └── ProximaDemoApp/         # macOS SwiftUI demo app
├── docs/
│   ├── ARCHITECTURE.md         # System design document
│   ├── BENCHMARKS.md           # Performance methodology
│   └── adr/                    # Architecture Decision Records
├── Models/                     # CoreML model files (not tracked)
└── Package.swift
```

---

## Code Style

### Swift Conventions

- **Naming**: Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
- **Formatting**: 4-space indentation. No trailing whitespace.
- **Imports**: Group by (1) Foundation/Accelerate, (2) ProximaKit modules, (3) Apple frameworks, separated by blank lines.
- **Access control**: Default to `internal`. Use `public` only for the intended API surface. Use `private` for implementation details.

### Module Rules (Enforced)

| Module | May Import | Must Not Import |
|--------|------------|-----------------|
| `ProximaKit` | Foundation, Accelerate | UIKit, SwiftUI, CoreML, NaturalLanguage, Vision |
| `ProximaEmbeddings` | Foundation, ProximaKit, CoreML, NaturalLanguage, Vision | UIKit, SwiftUI |

### Concurrency

- Index types are `actor`-isolated. All access through `await`.
- Value types (`Vector`, `SearchResult`, `HNSWConfiguration`) must be `Sendable`.
- Use `nonisolated` for properties that are safe to read without await (e.g., `dimension`).

### Dependencies

Zero external dependencies for `ProximaKit`. Only Foundation and Accelerate. `ProximaEmbeddings` may use Apple first-party frameworks (CoreML, NaturalLanguage, Vision). No third-party packages in either module.

---

## Testing

### Naming Convention

```swift
func test{Behavior}_{Scenario}() { ... }

// Examples:
func testSearch_returnsKResults()
func testRecall_10K_128d()
func testBenchmarkBatchL2_1K_384d()
```

### Test Categories

| Category | Filter | Purpose |
|----------|--------|---------|
| Unit tests | `swift test --skip RecallBenchmarkTests --skip SIMDBenchmarkTests` | Core correctness |
| Recall benchmarks | `--filter RecallBenchmarkTests` | HNSW accuracy vs brute force |
| SIMD benchmarks | `--filter SIMDBenchmarkTests` | vDSP vs naive loop speedup |
| Embedding tests | `--filter ProximaEmbeddingsTests` | NL, Vision, CoreML providers |

### Coverage Expectations

- All public API methods must have at least one test.
- Edge cases: empty index, single element, dimension mismatch, duplicate IDs.
- Recall benchmarks: >90% recall@10 at efSearch=50 for 1K vectors.

---

## Documentation

### Doc Comment Requirements

All public types and methods must have `///` documentation:

```swift
/// Searches the index for the `k` nearest neighbors to the query vector.
///
/// Uses beam search across HNSW layers, starting from the top layer
/// and refining through progressively denser layers.
///
/// - Parameters:
///   - query: The query vector. Must match the index's dimension.
///   - k: Number of results to return. Clamped to the index size.
/// - Returns: An array of ``SearchResult`` sorted by ascending distance.
public func search(query: Vector, k: Int) -> [SearchResult]
```

- Explain **why**, not just **what**.
- Include code examples for complex APIs.
- Document parameters, returns, and throws.

### Architecture Docs

Update `docs/ARCHITECTURE.md` when changing module boundaries, concurrency model, or data flow. ADRs are append-only — supersede rather than edit.

---

## Architecture Decision Records (ADRs)

When making a significant technical decision, add an ADR to `docs/adr/`:

```
docs/adr/ADR-NNN-short-title.md
```

### Template

```markdown
# ADR-NNN: Title

**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-XXX
**Date:** YYYY-MM-DD

## Context
What problem or question prompted this decision?

## Decision
What was decided?

## Rationale
Why this approach over alternatives? Include trade-off analysis.

## Consequences
What becomes easier/harder?

## Benchmarks (when applicable)
| Approach | Metric | Result |
|----------|--------|--------|
```

Current ADRs:
- [ADR-001](docs/adr/ADR-001-accelerate-for-math.md): Accelerate/vDSP for all vector math
- [ADR-002](docs/adr/ADR-002-actor-isolation.md): Actor isolation for thread safety
- [ADR-003](docs/adr/ADR-003-binary-persistence.md): Custom binary persistence format
- [ADR-004](docs/adr/ADR-004-hnsw-heuristic-selection.md): Heuristic neighbor selection

---

## Pull Request Process

### Branch Naming

```
feature/{short-description}    # New functionality
fix/{short-description}        # Bug fixes
chore/{short-description}      # Maintenance, docs, CI
```

### Commit Messages

Use conventional format:

```
feat: add batch cosine distance computation
fix: correct layer selection in HNSW insert
chore: update CI to Xcode 16
docs: add ADR for memory-mapped persistence
```

### PR Checklist

Before submitting, verify:

- [ ] `swift build` succeeds with no warnings
- [ ] `swift test --skip RecallBenchmarkTests` passes
- [ ] New public APIs have `///` documentation
- [ ] New features have corresponding tests
- [ ] No new external dependencies added to `ProximaKit`
- [ ] ADR created for significant architectural decisions
- [ ] `docs/ARCHITECTURE.md` updated if module boundaries changed

### Review Focus Areas

Reviewers will check:

1. **Correctness** — Does it produce the right results?
2. **Performance** — Does it maintain or improve current benchmarks?
3. **API ergonomics** — Is it hard to misuse?
4. **Thread safety** — Is actor isolation maintained?

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
