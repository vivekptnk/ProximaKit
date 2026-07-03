# ADR-009: Metal Backend for Batch Distance Computation (v1: Build-Phase Utility)

## Status
Accepted (amended — see Correction, 2026-07)

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
