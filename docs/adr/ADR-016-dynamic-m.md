# ADR-016: Dynamic-M HNSW — Per-Layer-Height Connection Schedules

## Status

**Proposed — design only. Recommendation: DEFER (measurement-gated), leaning NO-GO.**
**Date:** 2026-07-04
**Author:** Designer — ProximaKit
**HEAD:** `aaaab80` (v1.7.0). Every `file:line` citation below is at this commit
and was read, not inferred.
**Scope:** This ADR contains no code. It works out whether the roadmap's
"Hierarchical NSW variant with dynamic `M`" (Graph Improvements) is worth
building, decides the persistence and replay-determinism questions it raises,
declares the measurable recall threshold that would justify it, and recommends
**not building it now** — with a fully specified measurement gate that would
convert the decision to a GO. A benchmarked reject is a valid outcome here, per
the **ADR-009** Metal insert-loop precedent ("instrument first," then NO-GO with
the exact conditions that reopen it). All performance figures are **arithmetic**;
every measured claim is deferred to **ADR-005** methodology at implementation
time.

---

## Context

The roadmap item, verbatim (`docs/ROADMAP.md:115`):

> **Hierarchical NSW variant with dynamic `M`:** vary the number of connections
> per layer based on layer height to improve recall at low `efSearch` values.

The premise needs sharpening against what the code already does, because the
"2× factor" the classic HNSW paper prescribes is **not** a future feature — it
is shipped and hard-wired.

### What the code does today (code facts)

**The paper's `M_max0 = 2·M` heuristic is already the schedule, and it is
derived, not configurable.** `HNSWConfiguration` carries two separate stored
`let`s — `m` (upper-layer cap) at `HNSWIndex.swift:26` and `mMax0` (layer-0 cap)
at `:29` — but `mMax0` is **not** an init parameter: the initializer *derives*
it, `self.mMax0 = 2 * m` (`:71`), with the doc at `:28` naming it "(2 * m per the
paper)." Default `m = 16` (`:55`) ⇒ `mMax0 = 32`. A caller cannot set `mMax0`
independently. So ProximaKit already ships the paper's **constant-above-0**
schedule; "dynamic M" as a *new* feature can only mean varying M across the
**upper** layers (ℓ ≥ 1), which today are uniformly `m`.

**The per-layer cap is chosen at exactly the insert/reconnect sites, by a single
ternary.** `insertNode` (begins `HNSWIndex.swift:298`) picks the new node's cap
`let maxConnections = (level == 0) ? config.mMax0 : config.m` (`:350`), passes it
to the diversity heuristic as `m: maxConnections` (`:354`), and picks each
reciprocal neighbor's prune cap `let neighborMax = (level == 0) ? config.mMax0 :
config.m` (`:363`), passed to `pruneConnections(node:layer:maxConnections:)`
(`:365`). The remove-time reconnection path mirrors it: `let maxConnections = (l
== 0) ? config.mMax0 : config.m` (`:548`). `selectNeighborsHeuristic(target:
candidates:m:)` (`:1243`) takes the cap purely as an argument — the diversity
algorithm (ADR-004) is cap-agnostic — and `pruneConnections` (`:1292`,
guard `layers[layer][node].count > maxConnections` at `:1293`) likewise takes it
as an argument. **A dynamic-M schedule is therefore a localized change: replace
the value computed by those three-to-four ternaries; the heuristic, the beam,
and the prune primitive are untouched.**

**Level assignment depends only on `m`, not on the per-layer cap.** The level
multiplier `mL = 1/ln(M)` is `self.levelMultiplier = 1.0 / log(Double(config.m))`
(`:262`, formula doc `:1019`), and `assignLevel()` returns `Int(floor(-log(random)
* levelMultiplier))` (`:1037`). With `mL = 1/ln(m)`, the fraction of nodes that
reach layer ≥ 1 is `exp(-1/mL) = 1/m` — **6.25%** at `m = 16` (arithmetic; the
standard HNSW level geometry). This is the lever that makes an upper-layer-only M
boost nearly free (below), and it is *orthogonal* to the M cap: changing the
upper-layer cap does not change `mL`.

**Search floors `ef` to `k`.** `search(query:k:efSearch:filter:)` (`:392`)
computes `let ef = max(efSearch ?? config.efSearch, k)` (`:402`). **Consequence
for the hypothesis:** the roadmap's illustrative "low `efSearch`" and the
mission brief's `{8, 16, 32}` include `ef = 8`, which for `k = 10` is clamped to
`ef = 10` — `ef = 8` and `ef = 10` are the *same* search. The valid low-ef regime
for `k = 10` is `ef ∈ {16, 32, 64}` (or a smaller `k`). Any measurement must use
a regime where `ef > k`, or the low end is degenerate.

**Adjacency is variable-length, count-prefixed, in all three index families —
never padded to `mMax0`.** The `.pxkt` writer emits, per node,
`appendUInt32(&data, UInt32(neighbors.count))` then exactly that many `Int32`
ids (`PersistenceEngine.swift:268-276`, v2; identical loop in the v3 padded
writer `:593-600`); the decoder reads the count then that many ids
(`:464-482`, bounds-checked `:473`). The scalar- and product-quantized siblings
use the same count-prefixed scheme (`ScalarQuantizedHNSWIndexPersistence.swift:154-155`
/`:364-366`; `QuantizedHNSWIndexPersistence.swift:303-304`/`:706-708`). **There
is no fixed-width padding to `m` or `mMax0` anywhere** — the on-disk row already
encodes each node's true degree, whatever the layer cap was.

**Config travels with the file; `mMax0 = 2·m` is a load-time invariant.** The
64-byte header persists the config: `m`, `mMax0`, `efConstruction`, `efSearch`
are written (`PersistenceEngine.swift:231-234` v2, `:553-556` v3; header struct
fields `:925-928`; read at fixed offsets 24/28/32/36, `:981-984`), and on load
the config is **reconstructed from the header** — `HNSWConfiguration(m:
Int(header.m), efConstruction: …, efSearch: …)` (`:506-511`), with `mMax0`
re-derived from `m`, not read back. Both the resident and paged loaders enforce
`guard UInt64(header.mMax0) == 2 * UInt64(header.m)` (`:317-319`, `:402-404`) and
reject a violation as typed corruption. **So the `2·m` rule is asserted in two
places — the init derivation (`:71`) and the load guard (`:317`/`:402`) — and a
schedule that changed the *layer-0* cap would trip the guard.** `HNSWConfiguration`
is `Sendable` but **not `Codable`** (`:24`); persistence is hand-rolled binary,
so any new persisted field is manual header/trailer bytes, not a free `Codable`
addition. `levelSeed` is deliberately **not persisted** (`:50-52`, restore
comment `:596-599`).

**WAL replay preserves per-node levels (deterministic M); compaction re-draws
them (a pre-existing reshuffle).** The `add` WAL record carries the *drawn* level
(`WALFormat.swift:74` record, encode `:111`, decode `:236`); `add` journals
`newLevel = assignLevel()` (`HNSWIndex.swift:278`, appended `:286-288`); and
replay re-inserts with the stored level, bypassing the RNG —
`case let .add(id, level, vector, metadata): insertNode(…, level: level)`
(`:835-836`, journaling suppressed `:831`). ADR-013's byte-exact-replay
guarantee rests on this. **But `compact()` re-inserts every live node through
`add`** (`:1011-1014`), which calls `assignLevel()` afresh, and `checkpoint()`
compacts before writing the base (`:869`, `:877-879`). Because `levelSeed` is not
persisted, a post-reload compaction draws from the system RNG and produces a
**new** topology. This is today's behavior for the *existing* graph; dynamic-M
would ride it unchanged (below).

**Recall is measured against exact `BruteForceIndex` ground truth.**
`RecallBenchmarkTests.measureRecall` builds a parallel `BruteForceIndex`
(`:127`) and scores `bruteIDs.intersection(hnswIDs).count / 10.0` (`:146-148`),
averaged over `queries = 20` (`:123`), on **seeded synthetic uniform** vectors
(`seed 0xCA11…`, `:134`) with **unseeded** graph topology (`:129-133`). Its only
in-process ef sweep is `testEfSearchSweep` over `[10, 50, 100, 200]` (`:73`,
`count = 1_000`, `dim = 32`). `FilteredSearchSelectivityTests` is the
**topology-seeded** rig (`levelSeed: 0xF117…`, `:53-58`; `2000 × 32d`,
`k = 10`, `ef = 50`), so its A/B is deterministic. The `ProximaBench` package
(`BenchRunner.swift:50`, `:91-100`) computes recall@k against a precomputed
exact-GT JSON over **real `.fvecs` datasets** (SIFT-style), with `efSearch` a
single CLI knob (`ProximaBenchCLI.swift:62`, default 50) — swept only by invoking
the CLI repeatedly, not in-process. `SearchProviderBench` is latency-only.

### The sharpened subject

Given the above, "dynamic M" reduces to one honest question: **does varying the
connection cap across the upper layers (ℓ ≥ 1) — beyond the shipped
constant-`m` — buy enough recall@10 at low `ef` to justify permanent config
surface, when the library already exposes two proven levers for the same end
(raise `m`, raise `efSearch`) and the classic 2·m-at-layer-0 heuristic is
already in place?** The layer-0 cap is deliberately left out of scope: touching
it means reworking the two-place `2·m` invariant for no clear win (layer-0 edges
are already the most-optimized part of the graph), so the cheap, interesting
variant is **upper-layers-only**.

---

## What exactly varies — the M(layer) schedule candidates

Let `M(ℓ)` be the connection cap at layer `ℓ`. Today (S0), hard-wired at
`:350`/`:363`/`:548`:

- **S0 — constant-above-0 (shipped):** `M(0) = 2m`, `M(ℓ≥1) = m`.

Candidate schedules "dynamic M" could mean, all **leaving `M(0) = 2m`
untouched** (so the `mMax0 == 2m` guard at `:317`/`:402` and the init derivation
at `:71` are undisturbed):

- **S1 — geometric decay to `m`:** `M(0) = 2m`, `M(1) = ⌈c·m⌉`, decaying by a
  ratio toward `m` at the top (more edges on the lower upper-layers, which hold
  the most upper-layer nodes and carry the bulk of long-range navigation).
- **S2 — linear-in-height:** `M(ℓ) = m + ⌈a·(L_max − ℓ)⌉` — decreasing with
  height, a schedule shape the roadmap's "based on layer height" phrasing most
  directly names.
- **S3 — uniform upper boost:** `M(0) = 2m`, `M(ℓ≥1) = ⌈c·m⌉`, `c > 1` — the
  simplest, and the cleanest to reason about for both memory and the "did it
  help" question.

All three are pure functions of `m` plus a small parameter (`c`, `a`, or a
ratio). That matters for persistence (below): a schedule **derivable from the
already-persisted `m`** plus a **library-fixed** choice needs *zero* new
persisted state; a **per-index** choice needs one descriptor field.

**Why upper-only is near-free (arithmetic).** With `mL = 1/ln(m)` (`:262`),
`1/m ≈ 6.25%` of nodes reach layer ≥ 1 (`m = 16`), and total edges above layer 0
are `≈ N·m/(m−1) ≈ 1.07·N`, versus a layer-0 budget of up to `N·2m = 32·N`. So
**upper-layer edges are ≈ 3.3% of the layer-0 edge budget**; doubling them (S3
with `c = 2`) adds `≈ 1.07·N` `Int32` edges ≈ **4 B/node averaged over all N**,
against ≈ 200 B/node adjacency (ADR-013) and a ≥ 1.75 KB/node file — **< 0.25% of
resident**. The free-memory upper boost is the *only* thing that distinguishes
dynamic-M from "just raise `m`," which raises the 94%-of-nodes layer-0 cap too.

---

## The measurable hypothesis and the declared threshold

**Mechanism (the recall bet).** At low `ef`, HNSW recall is bounded by (1) the
quality of the entry point the greedy upper-layer descent hands to the layer-0
beam, and (2) the beam width `ef` itself. Denser upper layers sharpen (1): a
better entry point lets a narrow layer-0 beam start closer to the true kNN. The
2·m-at-0 heuristic targets layer-0 connectivity (which shapes the *final*
neighborhood), **not** upper-layer navigation — so an upper-layer boost is a
mechanistically distinct lever, and low `ef` is exactly where entry-point
quality dominates. That is the honest case *for* the feature.

**The honest prior against it.** The classic HNSW paper settled on uniform
`M_max` for upper layers as near-optimal after exploring `M_max`; the dominant
practitioner lever for low-ef recall is raising `M` globally or raising `ef`;
and the most-optimized libraries (FAISS, hnswlib) expose `M`/`efConstruction`/
`efSearch` but **not** a per-layer `M` — a conspicuous absence that is itself
evidence the marginal win is small. Upper layers are so sparse (`1/m²  ≈ 0.4%` of
nodes reach layer ≥ 2) that the cap barely binds above layer 1. So the expected
uplift at `ef ∈ {16, 32}` is plausibly **~1 pp or less**, and the cheaper, shipped,
already-benchmarked lever (raise `m`/`ef`) likely dominates.

**The declared GO threshold.** Measured offline under ADR-005, on the
**topology-seeded** rig (levelSeed set, so S0-vs-Sx is apples-to-apples on
identical `m`, `efConstruction`, seed, and query set), against exact
`BruteForceIndex` ground truth, on a **clustered** fixture (ADR-004: clustered
data is where graph quality separates — 88% vs 96% recall — while uniform data
barely distinguishes schedules), with `Q ≥ 1000` queries (see noise floor), a
GO requires **all** of:

1. **Recall uplift:** `Δ recall@10 ≥ +2.0 pp at ef = 16` **and** `≥ +1.0 pp at
   ef = 32`, versus S0 at identical `m`/`efConstruction`/`levelSeed`.
2. **No high-ef regression:** `recall@10 at ef ≥ 64` within **−0.5 pp** of S0
   (the schedule must not trade top-end recall for low-end).
3. **Memory:** resident adjacency growth **< 5%** (upper-only schedules clear
   this by ~20×; a schedule that fails it is really "raise `m`" in disguise).
4. **Build time:** **< 10%** increase (the heuristic is O(M²) per node per layer,
   incurred on only the ~6% of nodes above layer 0).
5. **The Pareto gate — the sharpest, and the one the prior says it fails:** the
   uplift must **exceed what raising uniform `m` to the same total edge budget
   delivers** on the same fixture. Dynamic-M's entire justification is
   "recall per byte spent on *upper* edges." If uniform-`m` at equal resident
   cost matches or beats it, the existing knob wins and the answer is **REJECT**.

**Noise floor (ADR-005 discipline).** For recall@10 over `Q` queries, the
standard error is `≈ √(p(1−p)/Q)`; at `p ≈ 0.9`, `Q = 1000` gives `≈ 0.95 pp`
(1σ). The current harness averages **20 queries** (`RecallBenchmarkTests:123`) —
a ~6.7 pp 1σ, far too coarse to resolve a 2 pp effect. **A valid measurement must
raise `Q` to ≥ 1000**; the +2 pp threshold is then ~2σ above noise, and any
reported sub-1 pp "win" is indistinguishable from zero and reads as a fail.

---

## The cost side (each question settled)

**1. Config surface — additive, cheap.** One optional field on
`HNSWConfiguration` — a schedule selector (enum) plus at most one scalar (`c`/`a`
/ratio) — defaulting to S0 so existing construction is byte-identical. The struct
is `Sendable` (`:24`); the change is one stored `let` and one branch at the
`:350`/`:363`/`:548` ternaries. **Additive, opt-in, default-transparent.**

**2. Persistence — no adjacency-format change; the schedule descriptor is the
only question, and it can be zero-cost.** Confirmed: adjacency is count-prefixed
in all three families (`PersistenceEngine.swift:268-276`/`:593-600`/`:464-482`;
the quantized siblings likewise), never padded to a cap — so **variable per-layer
M needs no on-disk format change**, exactly as the mission brief hypothesized.
The *only* persistence question is whether a loaded file must be
self-describing about which schedule built it:

- If the schedule is **library-fixed** and **derivable from the persisted `m`**
  (e.g. S3 with a single compiled-in `c`), the file needs **no new field** — `m`
  is already in the header (`:231-234`), and every loader/replayer reconstructs
  the schedule from it deterministically. **Zero format change.**
- If the schedule is **per-index configurable**, its descriptor must be
  persisted so `load` rebuilds the correct config. The 64-byte header has **no
  spare slots** — the four config words are `m`/`mMax0`/`efConstruction`/
  `efSearch` (`:231-234`), and ADR-010 records that v2 consumed the last reserved
  bytes — so a per-index descriptor would ride the **v3 trailer** (the existing
  extension point, ADR-013), a *localized* format touch, **not** an
  adjacency-format change and **not** a version bump beyond the shipped v3
  machinery. Recommendation: **prefer the library-fixed, derive-from-`m` form**
  and keep dynamic-M zero-format-change; escalate to a trailer field only if
  measurement shows different schedules win on different corpora (unlikely, and a
  YAGNI red flag).
- **The `mMax0 == 2m` guard (`:317`/`:402`) stays untouched** because every
  candidate leaves `M(0) = 2m`. This is a further reason to fence dynamic-M to
  the upper layers: it costs *nothing* in the format-invariant surface.

**3. Replay determinism — "same file + different config on open" is a
non-issue.** Config is reconstructed **from the file header** (`:506-511`), not
supplied by the caller, so a base built with schedule Sx *carries* Sx (via `m`
for the derive-from-`m` form, or the trailer descriptor for the per-index form)
and reopens with Sx by construction — there is no "open the same file with a
different config" path for the persisted fields. WAL replay re-runs `insertNode`
with the **restored** config and the **journaled** level (`:835-836`), so
ADR-013's byte-exact-replay guarantee holds unchanged: the M schedule is simply
more restored config on the same deterministic path. **Dynamic-M introduces no
new class of nondeterminism.** The one nuance — `compact()`/`checkpoint()`
re-draws levels (`:1011-1014`, `:877-879`) and `levelSeed` is unpersisted, so
per-node M reshuffles across a compaction cycle — is a **pre-existing** property:
the *entire* topology already reshuffles on post-reload compaction today.
Dynamic-M rides it identically (same-schedule, new levels ⇒ still a valid graph
of the same recall class); it neither fixes nor worsens it. (If reproducible
post-compaction topology is ever wanted, the fix is persisting `levelSeed` — a
separate, pre-existing item, not this ADR's.)

**4. Benchmark burden — real, and it is most of Stage 1.** Per ADR-005, no
recall claim ships without measurement, and unlike ADR-009 (where the utility
under test already existed) **there is no way to measure candidate schedules
without first prototyping per-upper-layer construction.** The harness gaps are
concrete: the in-process sweep stops at `ef = 10` (`:73`) and must extend to
`{16, 32, 64}` for `k = 10`; `Q` must rise from 20 to ≥ 1000; the A/B must be
topology-seeded (the `FilteredSearchSelectivityTests` `levelSeed` precedent, not
the `RecallBenchmarkTests` unseeded one); and a **clustered** fixture is needed
(ADR-004) since uniform data barely separates schedules. This is the Stage-1
deliverable, and it is why the measurement is not free.

---

## Options

### Option A — Defer now; ship nothing (recommended)

Do not build, and do not yet run the experiment. Convert the open-ended roadmap
item into a **precisely gated** one with a pre-committed NO-GO default. Rationale
below. Cost: zero. Risk: a real (if likely small) recall win stays unrealized —
mitigated by the fully specified gate that can be picked up on demand.

### Option B — Build behind a measurement gate, staged

Stage 1 (experimental construction + harness) measures S0 vs {S1/S3} and vs
uniform-`m`-at-equal-memory; Stage 2 (public surface) ships **only** if the
threshold clears. This is the "instrument first" shape of ADR-009. It is the
right plan *if* the item is picked up — but picking it up now spends
engineering weeks against no consumer pull (below), so it is the *contingency*,
not the recommendation.

### Option C — Ship nothing new; document the existing levers (the null lever)

Point consumers who want more low-ef recall at raising `m` or `efSearch` — both
shipped, both benchmarked, both directly understood — and at building with a
higher `m` when memory allows. This is what the library already supports; it is
the baseline every dynamic-M measurement must beat (threshold gate 5). If Stage-1
measurement ever lands NO-GO, this *is* the permanent answer, documented as such
(the ADR-009 register).

### Option D — Ship dynamic-M now, unconditionally — rejected

Ship a public schedule knob without measurement. Rejected outright: it violates
ADR-005 ("no performance claim until measured"), adds permanent config surface
and a per-consumer "which schedule?" cognitive tax on a weak prior, and risks
a persisted descriptor (format touch) for a win that may not exist. The house
does not ship recall features on intuition (ADR-004 benchmarked both
distributions before accepting the heuristic; ADR-009 benchmarked before
rejecting Metal).

---

## Recommendation

**DEFER (Option A), leaning NO-GO — with the measurement gate of Option B fully
specified and pre-committed.** Do not implement now; do not yet spend the Stage-1
cycles. Reasons, in priority order:

1. **No named consumer need.** ADR-013/014/015 each serve a concrete consumer
   (tinybrain's continuous ingest, bounded memory, exact rerank). Dynamic-M
   serves *no requested need* — nobody has asked for "more low-ef recall at fixed
   memory." YAGNI and ADR-006's "add features as real consumers need them"
   discipline apply directly. Its opportunity cost is real: the same weeks buy
   consumer-driven memory/durability work instead.
2. **A proven, shipped alternative lever already exists.** Raising `m` or
   `efSearch` trades directly for low-ef recall, is documented, and is already
   benchmarked (`RecallBenchmarkTests.testEfSearchSweep`). Dynamic-M's *only*
   distinct value — the near-free upper-layer boost — is precisely what the
   literature and FAISS/hnswlib's non-exposure suggest is marginal, and it must
   clear the Pareto gate against exactly this lever to matter.
3. **The paper's dynamic-M is already shipped** (`mMax0 = 2m`, `:71`). The
   further per-upper-layer refinement is a small, speculative delta on top of an
   already-good heuristic.
4. **But the item is cheap to *settle* definitively**, so rather than reject it
   forever, this ADR specifies the exact experiment and threshold and
   pre-commits the default — the ADR-009 pattern — so it can be closed with
   evidence the moment a consumer signal or slack appears.

### If picked up: the staged plan (contingency, not a commitment)

- **Stage 1 — experimental schedule + measurement harness (≈ 1–1.5 wk).**
  Per-upper-layer M in construction behind an **internal / bench-only** flag
  (NOT public `HNSWConfiguration`, NOT persisted) — the four ternaries at
  `:350`/`:363`/`:548` become a schedule call. Extend the recall harness: ef
  sweep `{16, 32, 64}` (k = 10), `Q ≥ 1000`, topology-seeded A/B (the
  `FilteredSearchSelectivityTests` `levelSeed` precedent), a **clustered**
  fixture plus the existing uniform one, and the **uniform-`m`-at-equal-memory**
  comparison arm. Emit under the ADR-005 JSON contract (or a documented
  sibling shape, per the ADR-009 addendum's deviation note). **Throwaway if
  NO-GO** — the public surface and format are never touched.
- **Stage 2 — public surface + acceptance (≈ 1–1.5 wk), *only if Stage 1 clears
  all five gates*.** Promote the schedule to an additive `HNSWConfiguration`
  field defaulting to S0 (byte-identical default); keep it **derive-from-`m`,
  library-fixed** for zero format change (escalate to a v3-trailer descriptor
  only if per-corpus schedules provably win — they should not); acceptance tests
  for the recall floor, the no-high-ef-regression bound, the memory bound, a
  **replay round-trip parity** check (build Sx → journal adds → reopen → replay
  reproduces the Sx graph, reusing the ADR-013 exact-equality rig), and an
  assertion that the `mMax0 == 2m` guard (`:317`/`:402`) is untouched (upper-only
  by construction); docs.
- **Total ≈ 2.5–3 engineering weeks *if built*** — but the recommendation is
  to spend **zero** now.

### Pre-committed default

**NO-GO unless Stage 1 clears every gate, including the Pareto gate against
uniform-`m`.** Given the prior, NO-GO is the expected and fully acceptable
outcome: it closes the roadmap item with evidence and makes Option C
(raise `m`/`ef`) the documented permanent answer — exactly what ADR-009 did for
the Metal insert loop.

---

## What this would unlock (honestly small)

If — and only if — measurement clears the gate: a modest low-ef recall lift at
near-zero memory cost, useful to a latency-bound consumer who must keep `ef`
small (e.g. an agent doing many recalls per turn) and cannot afford the resident
cost of a higher global `m`. That is a real niche, but it is narrow, unrequested,
and dominated in the common case by the shipped levers. Nothing downstream
depends on this; no ADR-014/015 unlock is gated on it.

## Consequences

- If deferred (recommended): the roadmap item is converted from vague to
  precisely gated; `docs/ROADMAP.md:115` points here; no code, config, or format
  changes.
- If eventually built: `HNSWConfiguration` gains one optional additive field
  (default S0, byte-identical); **no adjacency-format change** (count-prefixed
  rows already absorb variable M); **no version bump** if the schedule stays
  derive-from-`m` and library-fixed; the `mMax0 == 2m` invariant and its two
  enforcement sites are untouched (upper-only scope); replay determinism is
  unchanged (config travels with the file; journaled levels unchanged).
- Until measured, every recall figure here is arithmetic or a declared
  acceptance gate under ADR-005 — this document makes **no** performance claim.

## Open questions

Left to whoever picks up the gate:

1. **Which schedule shape** (S1 geometric / S2 linear / S3 uniform-boost) and
   which parameter value, if any clears the gate — a Stage-1 measurement output,
   not a design choice to pre-make.
2. **Library-fixed vs per-index schedule.** Zero-format-change (derive-from-`m`,
   fixed) is strongly preferred; only per-corpus measurement divergence would
   justify a persisted v3-trailer descriptor. Default: fixed.
3. **`levelSeed` persistence.** Reproducible *post-compaction* topology (hence
   stable per-node M across a compaction/reload) would need `levelSeed`
   persisted — a pre-existing gap (`:596-599`), orthogonal to dynamic-M, raised
   here only because dynamic-M makes the per-node-M reshuffle more visible.
4. **Whether the quantized families follow.** `QuantizedHNSWIndex` /
   `ScalarQuantizedHNSWIndex` build a transient full-precision `HNSWIndex`
   internally, so a schedule there would ride the same construction change — in
   scope only if the HNSW measurement clears first.
