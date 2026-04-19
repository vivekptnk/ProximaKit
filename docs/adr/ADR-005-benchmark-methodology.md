# ADR-005: Cross-Library Benchmark Methodology

**Status**: Accepted — v1.4.0
**Deciders**: CTO, Perf Engineer
**Related**: [CHA-110](https://github.com/vivekptnk/Chakravyuha/issues) — ProximaKit v1.4b

## Context

ProximaKit ships a pure-Swift HNSW implementation. Users reasonably want to know how it stacks up against battle-tested C++/Python implementations (FAISS HNSW, ScaNN). We also need regression detection in CI so that a refactor that accidentally slows HNSW by 20 % does not ship silently.

There are three coarse strategies:

1. Bind FAISS as a SPM system library and call it in-process from Swift.
2. Re-implement FAISS's HNSW parameter sweep in Swift using numpy-style file loaders, and never run the original library.
3. Run each library in its native ecosystem and compare via a shared on-disk contract.

## Decision

We chose **strategy 3**: a tiny shared JSON schema, one harness per library, and an aggregator that globs the output directory.

Concretely:

- `Benchmarks/` is a **separate SPM package**, not a target of the main `Package.swift`. The core library keeps its `Foundation + Accelerate only` posture.
- All harnesses (Swift `ProximaBench`, Python `faiss_hnsw.py`, Python `scann_hnsw.py`) emit a single JSON document per run, schema v1 (see `Benchmarks/JSON_SCHEMA.md`).
- **Ground truth is computed once** by the Swift `BruteForceIndex` and checked into the output directory. Every library is evaluated against **that exact file** — nobody computes recall against their own approximate neighbors.
- Single-threaded search by default. FAISS is pinned to 1 OMP thread; the Swift harness runs queries sequentially. Multi-threaded numbers are a separate study.
- CI runs a SIFT1M 10K smoke slice on every PR that touches `Sources/ProximaKit/**`, and the full 100K slice nightly on `main`. Output JSON is published as a workflow artifact.

## Rationale

**Why not a FAISS SPM binding.** A SPM system library target would drag a C++ toolchain and a Python bridge into the library's dependency graph. That breaks the "zero external deps beyond Apple frameworks" story that ProximaKit actively sells. Benchmarking is not a good enough reason to compromise that.

**Why not reimplement FAISS in Swift.** A "Swift port of FAISS HNSW parameterization" is not FAISS. The whole point of the comparison is that FAISS is what everyone else runs in production.

**Why a shared JSON schema instead of CSV or in-memory.** JSON documents are diff-friendly, CI-artifact-friendly, and cheap to aggregate. They also make it straightforward to add a new library later (e.g., hnswlib) — write one more harness, drop its JSONs in the same directory, rerun `compare.py`.

**Why a separate SPM package.** Three reasons:

1. `ProximaBench` depends on `ProximaKit` via `path: ".."`, which is a one-way dependency. The main package stays exactly as dep-free as it is today.
2. `swift build` at the repo root is untouched — no accidental compile of the bench harness during the library's normal build.
3. The executable is opt-in: people who clone the repo to use ProximaKit don't pay for ProximaBench's compile time; people who want to reproduce the numbers pay for it once with `swift build --package-path Benchmarks`.

**Why ground truth from `BruteForceIndex`.** Because we already trust it: it is the accuracy baseline for HNSW's existing recall tests. Using `faiss.IndexFlatL2` as the ground truth would bake a FAISS-specific assumption into every library's scoring.

## Consequences

- Adding a new library is cheap: one new harness, one `requirements.txt` entry, one aggregator group.
- The numbers published in `docs/BENCHMARKS.md` are always reproducible because the JSON files carry platform, seed, library version, and index parameters.
- Cross-process overhead is real but constant across libraries: FAISS pays Python import cost, ProximaKit pays Swift process startup, ScaNN pays Python + TF cost. The interesting comparison is still the steady-state per-query latency and recall@k.
- We cannot do true in-process latency comparisons (e.g. amortized startup cost). That is out of scope for v1.4.0 and can be revisited with a SPM system library target if and when the need arises.

## Alternatives rejected

- **FAISS via SPM system library** — too invasive for a benchmark-only win. Revisit post-v1.4 only if we need apples-to-apples single-process timing.
- **Protobuf contract** — overkill for a flat record; pulls in protoc as a CI dep. JSON is fine.
- **Pickle/NumPy `.npy` result files** — not language-neutral. Swift would need a bespoke loader.
