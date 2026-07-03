# ADR-009: Metal Backend for Batch Distance Computation (v1: Build-Phase Utility)

## Status
Accepted (amended — see Correction, 2026-07, and Addendum: Insert-Loop Integration Benchmarked — NO-GO, 2026-07)

## Context
ADR-001 chose Accelerate/vDSP for all vector math and explicitly deferred Metal ("high dispatch overhead, overkill for < 100K vectors"). That caveat is the point: the roadmap's GPU item targets large index *builds*, where building a 100K-vector HNSW index is dominated by repeated one-query-to-N-candidates distance computation during insertion — exactly the shape where a single GPU dispatch amortizes its launch overhead over N parallel threads. The roadmap sketched a `DistanceBackend` protocol plus a `.metal` shader; this ADR scopes what v1 actually ships.

## Decision

### v1 scope: a standalone `MetalBatchDistance` utility
`MetalBatchDistance` computes one-query-to-N-vectors distances over the same flat row-major matrix layout as `batchDistances` (ADR-001):
- `batchSquaredL2(query:matrix:vectorCount:dimension:)` — **squared** L2, not Euclidean. Build-phase consumers only rank candidates, and ranking is invariant under `sqrt`; skipping it avoids N square roots and matches the asymmetric-distance convention already used by the PQ codec (ADR-011).
- `batchCosineDistances(query:matrix:vectorCount:dimension:)` — `1 − dot/(|q|·|v|)`, with the zero-vector neutral-distance semantics (`1.0`, never a perfect match) identical to the vDSP path in `BatchDistance.swift`.

One kernel thread per database vector; each thread loops over the dimension with `fma` accumulation. Numerical parity with the vDSP path is asserted in tests to 1e-4 relative tolerance (with a small absolute floor where the reference is ~0, where relative tolerance is meaningless).

### Availability and fallback
- The implementation is gated behind `#if canImport(Metal)`; a stub with the identical API compiles everywhere else, so call sites and tests never fork.
- `init?` returns `nil` when `MTLCreateSystemDefaultDevice()` does (CI runners, some simulators, older hardware). Tests gate on this with `XCTSkipIf`.
- Any runtime Metal failure (buffer allocation, command-buffer error, matrix exceeding `maxBufferLength`, shader compile failure) falls back automatically to the existing vDSP batch path, so results are always correct and the public methods never throw. The execution path (`gpu` vs `cpuFallback`) is observable internally so parity tests can assert the GPU actually ran — a parity test that silently compared vDSP to vDSP would prove nothing.
- Shader compilation failure is *our* bug (the MSL source is a fixed string); tests compile the pipelines explicitly and fail — not skip — if that throws.

### SPM constraint: inline MSL source, not a `.metal` file
`.metal` files in a Swift package are not first-class: `swift build` (the CI path) treats them as unhandled files requiring a `resources:`/`exclude:` declaration in `Package.swift`, and Metal-source compilation into a discoverable `default.metallib` behaves differently between the Xcode build system and the SwiftPM CLI. The portable route is an inline MSL source string compiled at first use via `device.makeLibrary(source:options:)` and cached (lock-protected) for the lifetime of the instance. Tradeoff: no compile-time MSL syntax checking and a one-time ~runtime-compile cost on first dispatch — accepted, covered by the fail-loudly pipeline-compilation test, and it keeps `Package.swift` untouched.

### Class with a lock, not an actor
ADR-002's actor isolation is for stores. This utility is called synchronously inside tight build loops; an actor would force `await` onto the distance path and add a hop per batch. `MTLDevice`, `MTLCommandQueue`, and `MTLComputePipelineState` are documented thread-safe; the only mutable state (the lazily compiled pipeline cache) is `NSLock`-protected, so the class is `@unchecked Sendable` with that justification written at the declaration.

## Not in scope (v1)
- **Per-query search latency.** At search-time candidate-set sizes (efSearch-scale N), kernel launch overhead dominates and vDSP wins; ADR-001's verdict stands for search.
- **Integration into the `HNSWIndex` insert loop.** Follow-up, gated on benchmarking the utility against the vDSP path at realistic build batch sizes (no speedup is claimed until measured).
- **The roadmap's `DistanceBackend` protocol.** Premature with a single GPU consumer; extract it when integration lands.
- **Zero-copy buffers.** `makeBuffer(bytes:)` copies the matrix per call; `makeBuffer(bytesNoCopy:)` needs page-aligned ownership and belongs in the integration follow-up, where the matrix can live in a persistent `MTLBuffer` across insertions.

## Consequences
- Squared L2 is not Euclidean: callers comparing against `batchL2Distances` must square the reference (tests do exactly this). The GPU kernel computes `Σ(q−v)²` directly, avoiding the cancellation in the vDSP path's `|a|²+|b|²−2·dot` identity.
- CI runners without a GPU skip the parity tests cleanly; the availability-probe test still runs everywhere, so the suite exercises the no-Metal code path too.
- Kernel indexing casts to `ulong` before the row-offset multiply, so matrices are limited by `maxBufferLength`, not 32-bit index arithmetic.
- No performance numbers are published here or in `docs/BENCHMARKS.md` yet. A timing comparison exists in the test suite but prints only (hardware-dependent; it must not assert). Measured build-speedup numbers are a prerequisite for the insert-loop integration follow-up, per the roadmap's "instrument first" plan.

## Correction (2026-07)

"The suite exercises the no-Metal code path too" (Consequences, above) overstates what runs on a non-Metal platform. The `#else` stub's `init?()` always returns `nil` there, and every test that could call the stub's `batchSquaredL2`/`batchCosineDistances` first constructs an instance through `requireMetal()` (or an equivalent nil-check) and skips via `XCTSkipIf` when that construction fails — so those tests skip on a non-Metal platform exactly as they do on a Metal platform with no device. The one test that always runs, `testAvailabilityProbeIsDeterministic`, only compares `MetalBatchDistance() != nil` against itself; it never calls a batch method. So the no-Metal code path actually exercised everywhere is `init?` only — the stub's fallback numerics (which delegate to the same `fallbackSquaredL2`/`fallbackCosineDistances` free functions the Metal-backed `allowGPU: false` path also uses) are compiled but never invoked by any test on a non-Metal platform. This has no runtime consequence: the stub's methods are unreachable through the public `init?` flow by construction, as the stub's own header comment already notes. The claim above is corrected to `init?` only; it is not a claim about the stub's batch methods.

## Addendum: Insert-Loop Integration Benchmarked — NO-GO (2026-07)

ADR-009 shipped `MetalBatchDistance` as a standalone utility and deferred
`HNSWIndex` insert-loop integration "gated on benchmarking the utility against
the vDSP path at realistic build batch sizes (no speedup is claimed until
measured)." This addendum records that benchmark and its decision: **NO-GO.**
vDSP stays the build-phase distance path; `MetalBatchDistance` remains a
correct, parity-tested standalone utility and is **not** integrated. The
deferral in "Not in scope (v1)" becomes permanent for this architecture,
reopenable only under the concrete conditions listed at the end.

The evidence is reproducible from the standalone `Benchmarks` package
(`ProximaBench`), two new subcommands — `insert-shape` (workload
characterization) and `distance-kernel` (GPU-vs-vDSP sweep). Both use seeded
data (SplitMix64, no system RNG) and a fixed `levelSeed` so topology — hence
eval counts — is byte-reproducible.

### Finding 1 — the premise does not survive contact with the real loop

ADR-009's Context claims the build is "dominated by repeated one-query-to-N-
candidates distance computation during insertion — exactly the shape where a
single GPU dispatch amortizes." The actual `HNSWIndex.add()` path does not
present that shape:

- **There is no batch-distance call anywhere in the insert path.** Every
  distance in `add()` flows through scalar `metric.distance(_:_:)` — in
  `searchLayer` (the beam), `selectNeighborsHeuristic` (diversity selection),
  and `pruneConnections` (over-capacity trim). `batchDistances` is used only by
  `BruteForceIndex.search`, never during an HNSW build.
- The work is **serial and data-dependent**: each `searchLayer` distance feeds
  an immediate heap-admission decision that determines which node is expanded
  next, gated by a `visited` set. There is no point at which a query is scored
  against a large contiguous candidate array in one shot.
- Instrumented distance-eval composition (`insert-shape`, `m=16`,
  `efConstruction=200`, `seed=42`, `levelSeed=0xB1115EED`, seeded uniform data):

  | metric | d | N | evals/insert | traversal (`searchLayer`) | pairwise (heuristic+prune) |
  |--------|---|---|--------------|---------------------------|----------------------------|
  | euclidean | 384 | 1000 | 9 809 | 7.3 % | 92.7 % |
  | euclidean | 384 | 5000 | 16 104 | 13.8 % | 86.2 % |
  | euclidean | 768 | 1000 | 10 198 | 7.1 % | 92.9 % |
  | euclidean | 768 | 5000 | 16 686 | 13.2 % | 86.8 % |
  | cosine | 384 | 1000 | 10 817 | 7.1 % | 92.9 % |
  | cosine | 384 | 5000 | 18 201 | 13.9 % | 86.1 % |
  | cosine | 768 | 1000 | 11 395 | 6.8 % | 93.2 % |
  | cosine | 768 | 5000 | 18 426 | 13.8 % | 86.2 % |

  Only the **traversal** slice (7–14 %, growing slowly with index size) is even
  *one-query-to-N* shaped. The dominant 86–93 % is **pairwise** —
  `distance(candidate, selected)` and `distance(node, neighbor)` — many
  distinct "queries", not one query against N vectors, so it is not GPU-batch
  shaped at all.
- **The largest batchable one-query-to-N unit is a single node expansion**,
  whose candidate list is a layer-0 adjacency list pruned to `≤ mMax0 = 2m`
  (`= 32` at the default `m=16`). No batchable unit anywhere in the build
  exceeds `mMax0`. Even that unit is serial (interleaved with heap admission),
  so realizing it as a GPU dispatch would require restructuring the beam.

### Finding 2 — no crossover exists at any realistic N

`distance-kernel` sweep, **Apple M4 Max (32-core GPU), macOS 26.0.1 (Darwin
25.0.0, arm64), release build (`swift build -c release`), 3 warmup + 7 timed
reps, median wall-clock of the public API, seed 42.** GPU parity vs vDSP was
verified per cell (max abs diff ≤ 1.0e-3 for squared-L2 where magnitudes are
large, ≤ 3e-7 for cosine), so the GPU path actually ran — these are not silent
CPU fallbacks. Latency is hardware-dependent and is recorded here / in the
emitted JSON only; **it is never asserted in CI** (see the portable assertion
below).

Median ms, GPU (`MetalBatchDistance`) vs vDSP flat batch (`BatchDistance`),
and the vDSP-faster factor (`vDSP× = GPUmed / vDSPmed`, i.e. how many times
faster vDSP is). Each ratio is computed from the **raw, unrounded medians** in
the emitted JSON, not from the rounded ms cells shown here:

| metric | d | N | GPU ms | vDSP ms | vDSP× faster |
|--------|---|-----|--------|---------|--------------|
| euclidean | 384 | 32 | 0.268 | 0.001 | ~215× |
| euclidean | 384 | 256 | 0.281 | 0.007 | ~38× |
| euclidean | 384 | 1 024 | 0.351 | 0.026 | ~14× |
| euclidean | 384 | 10 240 | 1.400 | 0.240 | ~5.8× |
| euclidean | 384 | 102 400 | 13.044 | 2.942 | ~4.4× |
| euclidean | 384 | 500 000 | 63.827 | 12.821 | ~5.0× |
| euclidean | 384 | 1 000 000 | 123.910 | 25.666 | ~4.8× |
| euclidean | 768 | 102 400 | 26.947 | 5.036 | ~5.4× |
| cosine | 384 | 32 | 0.664 | 0.001 | ~637× |
| cosine | 384 | 10 240 | 1.446 | 0.221 | ~6.6× |
| cosine | 384 | 102 400 | 12.622 | 2.499 | ~5.1× |
| cosine | 384 | 1 000 000 | 117.776 | 24.556 | ~4.8× |
| cosine | 768 | 102 400 | 26.596 | 4.834 | ~5.5× |

**vDSP wins at every measured N from 32 to 1 000 000, for both metrics and both
dimensions. There is no crossover.** The GPU/vDSP ratio improves with N (a fixed
~0.3–0.6 ms dispatch floor amortizes) but plateaus around 0.20 — GPU stays
~4.8× slower even at N = 1 000 000 and never approaches parity.

Why vDSP dominates on Apple Silicon for this shape: (1) `vDSP_mmul` runs the
query-to-N dot products on the **AMX matrix coprocessor**, which is extremely
efficient for tall-skinny `[N×d]·[d×1]` products; (2) the GPU path pays a
per-call **host→device copy** of the whole matrix (`makeBuffer(bytes:)` copies;
zero-copy buffers were themselves deferred in v1) plus dispatch plus a
synchronous `waitUntilCompleted`. At build-loop N (`≤ mMax0 = 32`) the GPU's
fixed ~0.3 ms floor is ~2–3 orders of magnitude above vDSP's ~0.001 ms, so the
loss is largest exactly where the build operates.

### Decision — NO-GO

- vDSP (`BatchDistance` / scalar `metric.distance`) remains the build-phase
  distance path. `HNSWIndex` is unchanged; no `DistanceBackend` protocol is
  extracted (still a single non-consumer).
- `MetalBatchDistance` stays as shipped — a correct, parity-tested standalone
  utility. Its value is optionality and the parity harness, not a build
  speedup: on this hardware it does not have one, at any N, for either metric.
- The "Integration into the `HNSWIndex` insert loop" item under **Not in scope
  (v1)** is now a settled **no**, not a pending follow-up.

### What measurement would reopen this

A GO would require re-running `ProximaBench distance-kernel` (and, for shape
claims, `insert-shape`) and observing **all** of:

1. **A build algorithm that actually presents a large one-query-to-N batch.**
   Today's beam scores `≤ mMax0 = 32` candidates per expansion, serially. Only
   a bulk-build variant that scores a query against `≥ N_crossover` candidates
   in one contiguous array would even reach the regime where a GPU could win —
   and Finding 2 shows `N_crossover > 1 000 000` on this hardware, i.e. it does
   not exist for the copy-per-call kernel. So this lever also requires (2).
2. **A zero-copy dispatch path.** The deferred `makeBuffer(bytesNoCopy:)` with a
   persistent `MTLBuffer` holding the vector region across insertions, plus
   command-buffer reuse and removal of the per-call `waitUntilCompleted` sync,
   to collapse the ~0.3 ms GPU fixed cost. Re-measure crossover after this; a
   GO needs `GPUmed < vDSPmed` at an N the build can actually produce.
3. **Hardware without AMX-class CPU matrix acceleration**, where the vDSP median
   rises enough that `GPUmed < vDSPmed` at `N ≤ mMax0`. (Weak lever: `mMax0=32`
   is so small that beating a ~0.3 ms GPU dispatch needs the CPU to spend
   >0.3 ms on 32 dot products — implausible on any current target.)

Concretely: reopen only if a future run records `vDSP× faster < 1.0` (GPU
faster) at some `N ≤ mMax0` **on the target hardware**, or a batched-build
design lifts the realizable batch N above a newly-measured crossover.

### Deviation from the Benchmarks JSON house schema (documented per repo rule)

`Benchmarks/JSON_SCHEMA.md` (v1) models an *ANN index run* — `recallAt10`,
`efSearch`, `buildTimeSeconds`, one run per file. A distance-kernel crossover
sweep has none of those fields, so `distance-kernel` and `insert-shape` emit
their **own** document shapes (`kind: "distance-kernel-sweep"` /
`"hnsw-insert-shape"`), each a single coherent experiment per file carrying the
full sweep as an array. They reuse the house `Platform` block, `seed`, and
`runStartedAt`/reproducibility fields. This is a deliberate, isolated deviation
from the one-run-per-file ANN contract; the ANN harnesses (`hnsw`,
`ground-truth`) are untouched and still follow `JSON_SCHEMA.md`.
