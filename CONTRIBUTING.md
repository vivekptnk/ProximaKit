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

### Run Lint

CI pins an exact SwiftLint version — a floating `brew install swiftlint` tracks latest, and a future default-rule change could fail the strict gate with no code change on your end:

```bash
# Match CI's pin exactly (see .github/workflows/ci.yml, `lint` job):
gh release download 0.63.2 --repo realm/SwiftLint --pattern portable_swiftlint.zip --output /tmp/swiftlint.zip
unzip -o /tmp/swiftlint.zip -d /tmp/swiftlint
sudo cp /tmp/swiftlint/swiftlint /usr/local/bin/swiftlint

swiftlint lint --strict
```

[`.swiftlint.yml`](.swiftlint.yml) is a ratchet, not a cleanup mandate: it disables a fixed list of rules that the codebase violated at baseline (formatting, naming, force-unwrap, etc. — see the file's header comment), but every other default rule runs `--strict` in CI, so new code cannot introduce new classes of violation. Don't add new entries to `disabled_rules` to make an in-progress PR pass — fix the violation instead.

The same `lint` CI job also runs an import-boundary guard that SwiftLint can't express — it enforces the [Module Rules](#module-rules) allowlist below. Run it locally before pushing:

```bash
scripts/check-imports.sh
```

It exits `0` when clean and nonzero — listing every `file:line: forbidden import …` — when a module imports outside its allowed set. The script's header comment carries the authoritative allowlist and the justification for each allowed import.

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
│   │   ├── Index/              # VectorIndex protocol, HNSW, BruteForce, quantized/hybrid/sparse indexes
│   │   ├── Quantization/       # Product + scalar quantization codecs
│   │   ├── Persistence/        # Binary save/load with mmap, WAL sidecar journaling
│   │   ├── Query/              # SearchResult type
│   │   ├── Store/              # VectorStore / HybridVectorStore document-level API
│   │   └── Documentation.docc/ # DocC catalog + interactive tutorials
│   ├── ProximaEmbeddings/      # Text/image → vector providers
│   └── ProximaDemo/            # CLI demo executable
├── Tests/
│   ├── ProximaKitTests/        # Core unit + benchmark tests
│   └── ProximaEmbeddingsTests/ # Embedding provider tests
├── Benchmarks/                 # Separate SPM package: cross-library recall/latency harness
├── Examples/
│   ├── ProximaDemoApp/         # macOS/iOS/visionOS SwiftUI demo app
│   └── OnDeviceRAG/            # CLI RAG example (see docs/RAG-TUTORIAL.md)
├── docs/
│   ├── ARCHITECTURE.md         # System design document
│   ├── BENCHMARKS.md           # Performance methodology
│   ├── ROADMAP.md              # Planned work + ADR backlog
│   ├── HYBRID.md               # BM25 + dense fusion guide
│   ├── RAG-TUTORIAL.md         # On-device RAG walkthrough
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

### Module Rules

| Module | May Import | Must Not Import |
|--------|------------|-----------------|
| `ProximaKit` | Foundation, Accelerate | UIKit, SwiftUI, CoreML, NaturalLanguage, Vision |
| `ProximaEmbeddings` | Foundation, ProximaKit, CoreML, NaturalLanguage, Vision | UIKit, SwiftUI |

Enforcement is automated: [`scripts/check-imports.sh`](scripts/check-imports.sh) checks every import under `Sources/` against the per-module allowlist and runs in the `lint` job of [`.github/workflows/ci.yml`](.github/workflows/ci.yml), failing the build on any violation (run it locally with `scripts/check-imports.sh`). It backstops human review, because `import CoreML` inside `ProximaKit` would otherwise still compile — Apple's system frameworks are available to any target on the SDK, and `Package.swift` declares no per-target framework linkage that would reject it, so the build alone does not enforce this table. The script's header comment carries the authoritative allowlist and the justification for each allowed import; treat it as the source of truth rather than duplicating those details here.

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

### Prerequisite for Quantization, GPU, and Filtered-Search PRs

For these three areas specifically, an ADR with an **accepted** decision is a prerequisite for a PR — open the ADR (or point to an existing accepted one that already covers your change) before writing code, per [`docs/ROADMAP.md`](docs/ROADMAP.md#contributing). Other areas don't require an ADR up front, but still get one for any decision that would be expensive to reverse later.

Current ADRs (13 — see `docs/adr/` for full text):
- [ADR-001](docs/adr/ADR-001-accelerate-for-math.md): Accelerate/vDSP for all vector math
- [ADR-002](docs/adr/ADR-002-actor-isolation.md): Actor isolation for thread safety
- [ADR-003](docs/adr/ADR-003-binary-persistence.md): Custom binary persistence format
- [ADR-004](docs/adr/ADR-004-hnsw-heuristic-selection.md): Heuristic neighbor selection
- [ADR-005](docs/adr/ADR-005-benchmark-methodology.md): Cross-library benchmark methodology (FAISS/ScaNN comparison harness)
- [ADR-006](docs/adr/ADR-006-lumen-integration.md): ProximaKit ↔ Lumen RAG integration design (Draft — not yet accepted)
- [ADR-007](docs/adr/ADR-007-int8-scalar-quantization.md): INT8 scalar quantization (dequantization policy + codec format)
- [ADR-008](docs/adr/ADR-008-filtered-search.md): Post-filter strategy for filtered search, plus the graph-aware addendum
- [ADR-009](docs/adr/ADR-009-metal-backend.md): Metal backend for batch distance (v1 shipped a standalone build-phase utility; insert-loop integration measured and settled NO-GO)
- [ADR-010](docs/adr/ADR-010-format-evolution.md): Serialization format evolution policy
- [ADR-011](docs/adr/ADR-011-pq-codec.md): Product quantization codec format
- [ADR-012](docs/adr/ADR-012-pq-reranking.md): Full-precision reranking for quantized HNSW
- [ADR-013](docs/adr/ADR-013-streaming-persistence.md): Streaming persistence — WAL incremental saves (Stage 1 accepted and shipped); on-demand paged vectors (Stage 2) remains design-only

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
- [ ] `swiftlint lint --strict` passes (pinned to 0.63.2, matching CI — see [Run Lint](#run-lint))
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
