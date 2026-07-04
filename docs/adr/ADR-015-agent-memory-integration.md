# ADR-015: Agent-Memory Integration — ProximaKit as the Substrate for On-Device Agents

## Status

**Proposed — design only.**
**Date:** 2026-07-04
**Author:** Designer — ProximaKit
**HEAD:** `aaaab80` (v1.7.0). Every `file:line` citation below is at this commit and was read, not inferred.
**Scope:** This ADR contains no code. It maps an agentic on-device consumer's
memory lifecycle onto the already-shipped ProximaKit surface (ADR-006, ADR-011/012,
ADR-013, ADR-014), adjudicates the three open API-ergonomics findings the mission-5
audit confirmed (`api-ergo-01/02/03`), settles the M5-F49 store-level paging
question, and draws the minimalism line between what ProximaKit ships and what the
consumer builds. All performance figures are **arithmetic**; every measured claim is
deferred to ADR-005 methodology at implementation time.

---

## Context

The consumer is **tinybrain** — an agentic, on-device assistant (the maintainer's
parallel project) using ProximaKit as its *memory substrate* rather than as a
document search index. ADR-006 already named it a reuse consumer ("Other Chakravyuha
products (TinyBrain) can reuse the same pattern") and its CHA-107 addendum set the
"drop-in shape-compatible sibling / one-line construction-site swap" precedent that
governs how new store surface is added here. The `OnDeviceRAG` example is the existing
starter kit; its `LanguageModel` protocol (`Examples/OnDeviceRAG/LanguageModel.swift:15`)
is the deliberate seam that keeps *generation* consumer-side and model-agnostic
(`FoundationModels`, MLX, `llama.cpp`). This ADR keeps that seam and adds the mirror
for *memory*: retrieval and durability are ProximaKit's; cognition is tinybrain's.

An agent's memory loop is not a RAG corpus. It has three phases with different shapes:

| Phase | Frequency | Shape | Natural ProximaKit primitive |
|---|---|---|---|
| **Ingest** a turn | every turn (write-hot) | append a few small memory chunks, durably, cheaply | journaled `HybridVectorStore` — `save()` is an O(1) WAL flush (ADR-013 store journaling) |
| **Recall** for the next turn | every turn (read-hot) | hybrid dense+sparse top-k, filtered by memory-type / recency / salience | `query(_:k:filter:)` + graph-aware `@Sendable` predicate (ADR-008) |
| **Distill** old turns | periodic (cold) | summarize/compact aged memories into a compact, exact-rerankable long-tail | build-once retaining `QuantizedHNSWIndex` served `.paged` (ADR-014) |

The first two phases already have first-class support. The third is where the
design question lives, and where three low-severity ergonomics gaps become
consumer-fit pressure at agent cadence.

### What the code does today (code facts)

**Journaled store, O(1) ingest.** `HybridVectorStore.open(...)` is the journaled
factory (`HybridVectorStore.swift:232-242`, the only path that sets `journaling = true`
at `:279`); it is explicitly documented as the continuous-mutation entry point
("Choose this factory for continuous-mutation (agentic) workloads", the `open()` doc
mirrored on `VectorStore.swift:182`). Under journaling, `save()` (`:433`) takes the
`performSave` journaled branch (`:448-455`) that does **not** rewrite the corpus — it
is a bare `_dense.syncJournal()` fsync of the WAL tail (`:450`) plus a generation bump.
The full base is only rewritten by `checkpoint()` (`:495`), the periodic O(corpus) fold.
So per-turn ingest is genuinely O(change): `addChunks` (`:319-337`, batch-only —
a single memory is `addChunks([text], metadata: [meta])`) appends one WAL record per
chunk (`~1.6 KB/add` at 384d, ADR-013), and `save()` flushes it.

**But the fold is a manual poll the canonical example omits (`api-ergo-01`, confirmed
low).** Nothing folds automatically. `needsCheckpoint(policy:)` (`:503`) is a
caller-polled query; `checkpoint()` (`:495`) is a separate manual call; `WALCheckpointPolicy`
(`WALJournal.swift:52-63`, `walBytesFractionOfBase = 0.10`, `maxOps = 10_000`) is only a
*threshold descriptor* — no library code fires a checkpoint from it. The README store
snippet (`README.md:383-390`) shows `open() + addChunks() + save()` and **omits** the
`needsCheckpoint → checkpoint` step, so the documented happy path silently accrues an
unbounded `.pxwal`, and every restart replays the entire journal — inverting the exact
O(change)-vs-O(corpus) win the feature promises. At agent cadence (below) this is not a
corner case; it is the default trajectory.

**The audit quantified the minimal *correct* loop today** (recon `api-ergonomics`,
verbatim):

```
let store = try await VectorStore.open(name:embedder:storageDirectory:metric:config:durability:)  // 1 async factory, 6 params
// per ingest batch:
_ = try await store.addChunks(texts, metadata: metas)   // (2) awaited
try await store.save()                                  // (3) WAL fsync flush — does NOT fold
if await store.needsCheckpoint() {                      // (4) manual poll
    try await store.checkpoint()                        // (5) manual O(corpus) fold
}
// recall:
let hits = try await store.query(text, k: 10)           // (6)
=> 6 awaited calls per loop; (4)+(5) are a manual fold dance the consumer must remember.
```

**Recall is UUID-filtered, not attribute-filtered.** `query(_:k:candidatePoolK:filter:)`
(`:372-386`) takes `filter: (@Sendable (UUID) -> Bool)?` (`:376`) — a predicate over the
chunk **identity**, not over `ChunkMetadata`. There is no metadata-typed filter overload
anywhere (`HybridIndex.search` `:196`, `HNSWIndex.search` `:396`, `QuantizedHNSWIndex.search`
`:508/:562`). The predicate is applied graph-aware during the layer-0 beam with adaptive
`ef` widening (`QuantizedHNSWIndex.swift:543-549`; ADR-008), so selective filters still
fill `k` — the mechanism is sound; only the *typing* is identity-shaped.

**`ChunkMetadata` has no time and only string extras.** `ChunkMetadata`
(`ChunkMetadata.swift:21-45`) carries `documentId: String` (`:23`), `chunkIndex: Int`
(`:26`), `text: String` (`:29`), `extra: [String: String]?` (`:32`); it is `Codable`,
`Sendable`, `Equatable`. There is **no timestamp/date field** and `extra` values are
`String` only. Temporal ordering — which every forgetting policy needs — is not modeled.

**Two paged families, non-mirrored entry points (`api-ergo-02`, confirmed low).**
Paged full-precision HNSW is reachable **only** through the journaled `open(...mode:)`
(`HNSWIndex.swift:730-735`, `HNSWOpenMode {resident, paged}` at `:101`); `HNSWIndex.load(from:)`
(`:697`) is resident-only with no `mode:` overload, and the paged loader `loadHNSWPaged`
is `internal` (`PersistenceEngine.swift:373`). The quantized sibling is the mirror image:
`QuantizedHNSWIndex.load(from:mode:)` (`QuantizedHNSWIndexPersistence.swift:374`,
`PQHWOpenMode {resident, paged}` at `:104`, whose doc literally says "Mirrors
`HNSWOpenMode`" at `:102-103`) is a **WAL-free** paged loader, and the quantized family has
**no journaled `open`/checkpoint at all**. So a consumer who learns
`QuantizedHNSWIndex.load(from:url, mode:.paged)` and reaches for the identical
`HNSWIndex.load(from:url, mode:.paged)` finds it does not exist — the only paged HNSW
path forces a `.pxwal` sidecar they never asked for.

**Divergent open-mode names (`api-ergo-03`, confirmed low).** The two enums are
structurally and semantically identical (`{resident, paged}`) but named on opposite
conventions: `HNSWOpenMode` after the public class, `PQHWOpenMode` after the on-disk
format magic `PQHW` — a token in no public class name. The class a consumer holds is
`QuantizedHNSWIndex`; to name its residency mode they must first learn its file format's
magic. The save-layout enum `PQHWSaveLayout {resident, pagedV3}` (`:90`) shares the leak.

**The cold tier is unreachable through the store.** `VectorStore.index` and
`HybridVectorStore.index` are hardcoded `HNSWIndex`/`HybridIndex` (`VectorStore.swift:35`,
`HybridVectorStore.swift:48`); nothing in the store layer touches a quantized index. And
the journaled store opens its dense leg **resident** — `HybridVectorStore.open` calls
`HNSWIndex.open(baseURL:walURL:durability:)` with **no `mode:` argument** (`:249`), so it
defaults to `.resident` (`HNSWIndex.swift:734`). This is exactly **M5-F49** ("Paged dense
leg inside journaled stores (compose ADR-013 stages; additive param)", `pending`): the
index layer supports `.paged`, the store never plumbs it through. The mission-5 recon
named the store-layer HNSW-hardcoding "the single biggest consumer-fit pressure the lens
surfaces."

**PQHW is remove-only.** `QuantizedHNSWIndex` exposes `search`/rerank (`:504`, `:557`) and
`remove(id:)` (`:705`) but **no `add`** — it is build-once. This is not incidental; it *forces*
the hot/cold split at the index layer: continuous ingest cannot land in a quantized index,
so any quantized long-tail must be **rebuilt** from distilled memories, never appended.

### Problem quantification: a plausible agent profile

**Labeled assumptions** (arithmetic only; none from code):
- **A1** — 50 turns/day (the stated profile).
- **A2** — 2 memory chunks ingested/turn (e.g. one distilled user-intent memory + one
  agent-outcome memory) → 100 chunks/day.
- **A3** — horizons: 1 yr (365 d) = **36,500** chunks; 3 yr = **109,500**; 5 yr = **182,500**.
- **A4** — 384d `Float32` embedding = 384 × 4 = **1,536 B/vector** (ADR-007).
- **A5** — per-node non-vector resident overhead: adjacency ≈ 200 B (m = 16, ADR-011/013)
  + node level 4 B + UUID 16 B = **220 B**; encoded `ChunkMetadata` ≈ **260 B** (a ~200 B
  distilled memory line in `text` + `documentId` + `chunkIndex` + `extra`) — an
  **estimate**, and the load-bearing sensitivity knob (see below).

Per-vector resident cost, three storage modes (arithmetic):

| Mode | Resident/vector | Flash/vector | Notes |
|---|---|---|---|
| Full-precision **resident** dense leg | 1,536 + 220 + 260 = **2,016 B** | 0 | today's journaled store (`:249`) |
| **Paged** dense leg (M5-F49) | 220 + 260 = **480 B** | 1,536 B (mapped) | vector payload on flash, exact search |
| **PQHW paged** cold (M = 48) | 48 + 220 + 260 = **528 B** | 1,536 B (originals) | codes resident, originals mapped, exact rerank |

Resident totals across horizons (arithmetic):

| Horizon | Chunks | FP-resident (2,016 B) | Paged-dense (480 B) | Flash if paged (1,536 B) |
|---|---|---|---|---|
| 1 yr | 36,500 | **73.6 MB** | **17.5 MB** | 56.1 MB |
| 3 yr | 109,500 | **220.8 MB** | **52.6 MB** | 168.2 MB |
| 5 yr | 182,500 | **367.9 MB** | **87.6 MB** | 280.3 MB |

*(Excludes the sparse BM25 leg, which a `HybridVectorStore` rebuilds resident on open —
a fraction of the dense leg for short memory lines — and the doc map. These add
overhead, not a new order of magnitude.)*

**The crux this arithmetic settles.** Paged full-precision (480 B) and PQHW-paged
(528 B) are within **~10% resident** — the codes tax (48 B) almost exactly offsets the
payload the paged-dense leg already moved to flash. **So resident RAM is *not* the
axis that chooses between them, and neither bites the iPhone budget at the stated scale**
(87.6 MB resident at five years, paged). What actually differs:

1. **Per-query fault profile / latency.** In a paged **dense** leg, the beam reads
   vectors *through the mapping*, so cold faults land inside the traversal loop —
   ≈ 1 fault per ⌊16,384 / 1,536⌋ = **10.67 vectors/page** across the ≈ efSearch
   candidates a query visits (e.g. C = 200 fully-cold candidates ⇒ ≈ 19 faults ≈
   300 KiB paged in, on the critical path; warm working set amortizes). In a
   **PQHW-paged** tier the ADC beam runs **entirely on resident codes and distance
   tables**; the *only* originals read is the post-beam rerank, bounded by
   `rerankDepth` ⇒ ≤ 4·k = **40 faults ≈ 640 KiB, off the traversal critical path**
   (ADR-014's exact result). As the cold corpus grows, PQHW-paged's bounded,
   off-path fault budget is the latency win — but that only matters well past the
   stated scale.
2. **Mutation.** PQHW has no `add` (only `remove(id:)` at `:705`) — continuous ingest
   **cannot** live in it.
3. **Text stays resident either way.** Paging pages the *vector section* only
   (ADR-013 Stage 2); `ChunkMetadata.text` (A5's 260 B) is resident in both paged
   modes. If memory chunks are long (say 1 KB text), paged-dense resident rises to
   ≈ 220 + 1,024 = 1,244 B/vector → ≈ **227 MB at 5 yr** — so **summarization is not
   only a recall lever, it is the resident-memory lever.** This is a tinybrain
   responsibility (below), and the reason distillation belongs in the consumer.

---

## Options (the tiering shape)

### Option A — Single **resident** journaled store (status-quo shape)

One `HybridVectorStore.open(...)`, dense leg resident (today's `:249`). Ingest is O(1);
recall is hybrid+filtered. Rejected as the *target*: resident climbs to **367.9 MB at
5 yr** (arithmetic) — jetsam-bait for a foreground iPhone app — with the full vector
payload paying RAM it never needs to. It remains the correct *default* for small,
short-lived agents and is byte-identical to ship untouched.

### Option B — Single journaled store with a **paged dense leg** (M5-F49) — *recommended, near-term*

Plumb `HNSWOpenMode.paged` through the store's dense `open` (the missing `mode:` at
`:249`). Everything stays in one journaled hybrid store; the vector payload moves to
flash via the already-shipped `MappedVectorRegion`; ingest stays O(1); recall stays
hybrid+filtered and byte-identical to resident (ADR-013 Stage-2 parity). Resident holds
at **17.5 MB (1 yr) → 87.6 MB (5 yr)** (arithmetic) — comfortably in budget — with no
distillation machinery, no second index family, no rebuild. This is the smallest change
that meets the memory bound, and it is *exactly* M5-F49's framing ("compose ADR-013
stages; additive param"). Cost: query-path faults scale with efSearch on cold pages
(point 1 above) — acceptable at this scale, and the precise thing Option C later bounds.

### Option C — **Two-tier**: journaled HNSW hot leg + PQHW-paged cold leg, consumer-composed distillation — *recommended, eventual, as a pattern*

Recent turns live in a small journaled HNSW hot leg (bounded working set, O(1) ingest);
periodically, tinybrain distills aged memories — summarize/compact/score — and **rebuilds**
them into a retaining `QuantizedHNSWIndex` saved `.pagedV3` and opened `.paged` (ADR-014):
codes resident, originals on flash, exact rerank with a bounded off-path fault budget.
The hot/cold split is **already forced by PQHW's no-`add` constraint** (`:705` is `remove`,
there is no `add`), so this
is the natural end-state, not an imposition. Its distinctive payoff — bounded rerank
latency as the tail grows, and lossy compression of the cold tail — arrives *after* the
stated scale. Crucially, every mechanical piece it needs already exists at the **index**
layer (journaled HNSW + PQHW `.pagedV3` + `.paged` load + `upgradeToV3`); what it *lacks*
is policy (what to age, how to summarize, when to fold), which is not ProximaKit's to
own. **Recommended as a documented consumer pattern, not new store API.**

### Option D — Ship a `QuantizedVectorStore` sibling now (store-level cold tier) — *rejected / deferred*

Make the cold PQHW tier a first-class store (`index.pqhw` beside `index.pxkt`, the CHA-107
"drop-in sibling" shape) so tinybrain never drops to a raw `QuantizedHNSWIndex` and
re-implements docmap/embedding/persistence. This is the genuine consumer-fit pull the
recon flagged. Rejected **for now**: ADR-014 already scoped the quantized-backed store
sibling as "the unlock, not a commitment"; the tiering *cadence and policy* that would
define its lifecycle (`save()`/`checkpoint()` semantics for a build-once cold leg, when
the hot leg folds into it) are unsettled consumer-land; and shipping a store shape before
tinybrain's distillation policy stabilizes risks the wrong contract. Held as an open
question, not a decision — see below.

---

## Recommendation

**Near-term: Option B**, and make continuous ingest correct-by-default — i.e. ship the
**paged dense leg (M5-F49)** plus the **store-level auto-checkpoint hook** (`api-ergo-01`)
plus the **`HNSWIndex.load(from:mode:)` mirror** (`api-ergo-02`). **Eventual: Option C**,
endorsed and documented as a *consumer-composed* pattern over already-shipped index
primitives. **Defer Option D** (the `QuantizedVectorStore` sibling) as an open question.

This answers Design Question 1 crisply: **one journaled store with a paged dense leg
suffices at the stated scale** (memory bound met at 87.6 MB / 5 yr, arithmetic); the
two-tier shape is right *eventually*, but its driver is mutation semantics + latency-
bounding + text/flash compression, **not** resident RAM (paged-dense and PQHW-paged are
within ~10%), and none of those bite until well past the profile. Ship the single paged
store now; grow into two tiers as a pattern.

### The minimal additive surface (and the line)

Four candidates, each adjudicated; every inclusion is additive and opt-in, default
behavior byte-identical.

| Candidate | Verdict | Why |
|---|---|---|
| **1. Store auto-checkpoint hook** — `open(..., checkpointAutomatically: WALCheckpointPolicy? = nil)` on `VectorStore`/`HybridVectorStore` | **INCLUDE** | Highest leverage. The store is *documented* for agentic workloads (`:182`) yet the canonical example (`README.md:383-390`) omits the fold, so an agent's default trajectory is an unbounded `.pxwal` and O(total-mutations) cold-open (`api-ergo-01`'s failure scenario). When non-nil, fold **inside** the serialized mutation chain once the policy trips, reusing the existing `needsCheckpoint(policy:)` (`:503`) + `checkpoint()` (`:495`). Default `nil` = today's manual behavior, untouched. Turns the 6-await ceremony into `addChunks + save`. |
| **2. `HNSWIndex.load(from:mode:)` mirror** (`api-ergo-02`) | **INCLUDE (small)** | Closes a real asymmetry: the quantized family has a WAL-free paged `load(from:mode:)` (`:374`); HNSW's only paged path forces a journal sidecar (`:730`). A consumer-composed cold HNSW tier wants WAL-free paged loads. Wraps the **already-internal** `loadHNSWPaged` (`PersistenceEngine.swift:373`) — near-zero implementation risk. Makes both families' paged story learnable once. |
| **3. Store-level paged dense option** (M5-F49) — `open(..., dense: HNSWOpenMode = .resident)` plumbed to the dense `HNSWIndex.open` at `:249` | **INCLUDE (the tiering answer)** | The single change that makes Option B real: a bigger-than-RAM full-precision memory with the payload on flash. Composes ADR-013 Stage 1 (journaling) + Stage 2 (paging); "additive param" exactly as M5-F49 frames it. |
| **4. `distill(into:)` helper** | **EXCLUDE (consumer-land)** | Distillation *is* policy — which memories to age, how to summarize (needs an LLM ProximaKit does not and should not have — cf. the `LanguageModel` seam), how to score salience, when to fold. A helper would either be a trivial wrapper over existing primitives (build a retaining `QuantizedHNSWIndex`, `save(to:layout:.pagedV3)`, `load(mode:.paged)`) — premature — or bake a forgetting/summarization policy into the wrong layer — scope creep. Named an explicit non-goal. |

**The line:** ProximaKit ships **mechanism** — durable O(1) ingest, hybrid+filtered
recall, paged bigger-than-RAM storage, auto-bounded WAL, exact-rerankable cold format,
and the metadata channel a policy keys on. tinybrain owns **policy** — what to remember,
how to summarize, how to score salience, when to forget, how to tier. This is the ADR-006
"add features as real consumers need them / start with a thin wrapper" discipline held.

### Naming adjudication (`api-ergo-03`, Design Question 3)

**Decision: unify to a single shared `IndexResidency` enum now, alias the two existing
names, stage the deprecation for the next minor, remove at 2.0.** Concretely: introduce
`public enum IndexResidency: Sendable { case resident, paged }` as the canonical type
consumed by every `mode:` parameter; add `public typealias HNSWOpenMode = IndexResidency`
and `public typealias PQHWOpenMode = IndexResidency` for source compatibility
(zero breakage today); annotate the two old names `@available(*, deprecated, renamed:
"IndexResidency")` at the next **minor** to steer new code; drop the aliases at **2.0**.

Reasoning: the two enums are *literally the same type modulo name* (`:101` vs `:104`,
the doc at `:103` concedes "Mirrors `HNSWOpenMode`"), and the whole two-tier story
treats the two index families as siblings — a consumer should learn **one** residency
vocabulary, not two, and certainly should not have to discover the internal format magic
`PQHW` to name a public, hot construction-site parameter. Unifying beats per-class
renaming (option (b) `QuantizedHNSWOpenMode`) because renaming preserves *two* parallel
names for *one* concept — the very redundancy that confuses. Document-and-hold (option
(c)) is rejected: it is the cheapest but leaves the format-magic leak in the public
surface indefinitely, and the sibling-vocabulary confusion is precisely what an agent
consumer composing hot+cold tiers hits first. The severity is low, so the *breaking*
step (alias removal) correctly waits for the deliberate major (the ADR-013 deviation-1
"keep it until a major" precedent), but the *canonical* name lands now so all new code
(the M5-F49 store param, the `load(from:mode:)` mirror) is written against it from the
start. `PQHWSaveLayout` (`:90`) rides the same cleanup, but more lightly — its `.pagedV3`
case name legitimately encodes a *format version*, not a residency mode, so it is renamed
(`IndexSaveLayout`) for consistency but its case names are not a leak of the same kind.

### Metadata patterns for agent memory (Design Question 4) — pattern, not API

Agent memory wants recall filtered by **memory-type** (episodic / semantic / task),
**recency**, and **salience**. The mechanism already exists; only the ergonomics are
identity-shaped. The canonical pattern:

- Encode policy keys into `ChunkMetadata.extra` (`:32`) at ingest — e.g.
  `["kind": "episodic", "ts": "<epoch-ms>", "salience": "0.62"]` (string-typed, since
  `extra` is `[String:String]`).
- Snapshot a lightweight `[UUID: Attributes]` the consumer already builds from its own
  ingest, and **capture it in the `@Sendable` filter closure**:
  `query(text, k: 10, filter: { uuid in attrs[uuid].map { $0.recencyOK && $0.kind == .episodic } ?? false })`.
- The graph-aware beam applies the predicate during layer-0 with adaptive `ef` widening
  (`QuantizedHNSWIndex.swift:543-549`, ADR-008), so even selective recency/salience
  filters still fill `k` — no post-filter under-fill.

This needs **no new API** and is the recommendation. Two candidate additions are named
as **open questions and deliberately held**, not shipped: (a) a
`filter: (@Sendable (ChunkMetadata) -> Bool)` recall overload so the consumer need not
thread its own map; and (b) a native optional `timestamp: Date?` on `ChunkMetadata`
(`:32` has no time field) — recency is universal to *every* forgetting policy, which is
the strongest case for baking it in, but `ChunkMetadata` is a persisted `Codable` type
so adding a field is a format-compatibility concern (additive-optional decodes old files,
but the semantics of "missing timestamp" must be defined). Both wait for tinybrain's
policy to prove the friction is real — YAGNI over speculative surface, per ADR-006.

### Non-goals — what tinybrain builds itself (Design Question 5)

Explicitly **out of scope** for ProximaKit; the consumer owns these:

- **Summarization / compaction of old turns** — needs a language model; the
  `LanguageModel` seam (`Examples/OnDeviceRAG/LanguageModel.swift:15`) is the established
  consumer-side boundary. ProximaKit has no LLM and adds none.
- **Salience / importance scoring** — domain policy (what matters to *this* agent).
- **Forgetting policy** — TTL, decay curves, what to drop vs. archive, when to age.
- **Tiering cadence** — hot-window size, when to distill hot→cold, rebuild frequency.
- **Turn chunking** — how a turn becomes memory items (ADR-006 already placed chunking
  consumer-side).
- **Embedding-model choice** — already consumer-side via `TextEmbedder` / `EmbeddingProvider`.
- **Generation** — the answer/action seam stays model-agnostic and consumer-plugged.

ProximaKit's half: durable O(1) ingest, hybrid+filtered recall, exact-rerankable paged
cold storage, memory-bound paging, the metadata channel, and the in-place `upgradeToV3`
path so an existing on-device base adopts paging without a rebuild. tinybrain's half: the
cognition.

### Staged recommendation and honest cost

Stages are shippable alone, in ADR-013/014's register. There is **no new on-disk
format** here — this is store-layer wiring, one param, and an enum alias over
already-shipped index primitives — which is why it is cheaper than either predecessor.

- **Stage A — auto-checkpoint hook (`api-ergo-01`) — ≈ 1–1.5 weeks.** Add
  `checkpointAutomatically: WALCheckpointPolicy? = nil` to `VectorStore.open` /
  `HybridVectorStore.open`; when non-nil, fold inside the serialized mutation chain
  (`serialized(_:)`) after a batch once `needsCheckpoint(policy:)` trips; fix the README
  store example to show bounded WAL; WAL-bound + recovery tests. **Ships alone** — it
  fixes the documented footgun and is the single biggest agent-cadence correctness win.
- **Stage B — paged dense leg (M5-F49) + `HNSWIndex.load(from:mode:)` mirror
  (`api-ergo-02`) — ≈ 1.5–2 weeks.** Plumb `dense: HNSWOpenMode` from the store `open`
  through the dense `HNSWIndex.open` (`:249`); surface `HNSWIndex.load(from:mode:)`
  wrapping the internal `loadHNSWPaged` (`PersistenceEngine.swift:373`); `phys_footprint`
  memory acceptance + paged-vs-resident parity, reusing the ADR-013 Stage-2 rigs
  (`PagedVectorMemoryTests` precedent). **Ships alone** — the memory-bound win.
- **Stage C — naming unification (`api-ergo-03`) + agent-memory docs — ≈ 0.5–1 week.**
  Introduce `IndexResidency`, alias `HNSWOpenMode`/`PQHWOpenMode`, stage deprecations;
  document the metadata pattern, the two-tier consumer pattern (Option C), and the
  non-goals boundary. Mostly mechanical + docs. **Ships alone.**
- **Total ≈ 3–4.5 engineering weeks**, each stage independently shippable. **Primary
  risk:** the auto-checkpoint fold must run *inside* the serialized mutation chain
  without reentrancy/deadlock (the store serializes mutations; `checkpoint()` folds the
  dense WAL and refreshes the sparse+map projections) — mitigated by reusing the shipped
  `checkpoint()` path and the store-journaling recovery-test precedent
  (`StoreJournalRecoveryTests`). Secondary risk: scope pressure to ship Option D's store
  sibling — fenced by this ADR's line.

### Acceptance criteria

Measurable; arithmetic until measured under ADR-005.

1. **Ceremony reduction (measurable line/await count).** The canonical ingest+recall
   loop's **per-turn** awaited store calls drop from **4** (`addChunks`, `save`,
   `needsCheckpoint`, `checkpoint`) to **2** (`addChunks`, `save`), and the "manual fold"
   calls from **2 to 0**, with `checkpointAutomatically:` set once at `open`. The whole
   canonical loop (recon's numbering) drops from **6 awaited calls to 4**. The README
   store example is updated so the documented happy path no longer accrues an unbounded
   WAL. Asserted as a doc/example await-count, not a benchmark.
2. **Memory bound (env-gated `phys_footprint`).** The single journaled store opened with
   a **paged** dense leg holds resident within the arithmetic envelope — target **≤ ~18 MB
   at 1 yr (36,500 chunks)** and **≤ ~90 MB at 5 yr (182,500 chunks)**, versus 73.6 /
   367.9 MB resident — with the vector payload demonstrably *not* resident (resident-vs-
   paged delta ≈ the payload), and flat after a warm-recall sweep bounded by the working
   set, not the corpus. The exact threshold is set from a measured baseline at
   implementation, not invented now (ADR-013/014 discipline).
3. **Recall parity + latency envelope.** Paged-dense recall is **byte-identical** to
   resident (same ids and bit-equal distances; ADR-013 Stage-2 parity), across seeded
   queries × {filtered, post-remove}. The per-query cold-fault budget is the measured
   quantity: for the paged dense leg, ≈ efSearch-candidates / 10.67 faults on cold pages
   (on the beam critical path); for the consumer-composed PQHW-paged cold tier,
   ≤ `rerankDepth` faults (≤ 40 at 4·k, k = 10) **off** the critical path (ADR-014).
   Envelope reported, not a fixed SLA, until measured on-device under ADR-005.
4. **Auto-checkpoint bound + correctness.** With `checkpointAutomatically: policy`, the
   `.pxwal` byte-count stays within the policy across a continuous-ingest run (e.g. 10K
   adds), and cold-open replay cost stays O(recent-change), not O(total-mutations-since-
   first-open) — the `api-ergo-01` inversion, gone. Folds occur inside the serialized
   mutation chain (no interleave with `addChunks`/`removeDocument`); the default-`nil`
   path is byte-identical to today, proven red-green.
5. **Additive-only.** Existing `open` / `save` / `query` / `load` are byte-identical in
   behavior and on-disk output when the new params are omitted; `checkpointAutomatically:`,
   `dense:`, `HNSWIndex.load(from:mode:)`, and `IndexResidency` are new opt-in surface.
   No format bump (this rides `.pxkt` v3 / `PQHW` v3 unchanged).
6. **No performance claim until measured** under the existing harness; this document
   contains arithmetic only.

---

## What this unlocks

Option B is the piece that makes ProximaKit a *memory* substrate rather than a *search*
index for on-device agents: continuous ingest that stays O(change) *and* bounded in RAM,
recall that stays hybrid+filtered and exact, on a corpus that outgrows the resident
budget — without the consumer touching a second index family. It also makes Option C
tractable as a **pattern**: with `HNSWIndex.load(from:mode:)` mirroring the quantized
sibling and one `IndexResidency` vocabulary, a consumer can compose a journaled HNSW hot
leg and a rebuilt PQHW-paged cold leg using only documented, already-shipped primitives.
The `QuantizedVectorStore` store sibling (Option D) remains the deferred unlock — this
ADR designs the store-layer paging + convenience surface that *precedes* it, not the
sibling itself.

## Consequences

- The store `open` factories gain two optional parameters (`checkpointAutomatically:`,
  `dense:`); the surface grows by one param each plus one `load(from:mode:)` overload and
  one shared enum — small, additive, default-transparent.
- The auto-checkpoint fold introduces a code path where a `save()` can *become* an
  O(corpus) checkpoint when the policy trips — documented as the deliberate cost of
  bounded WAL; latency-sensitive callers keep the manual (`nil`) mode.
- `text`-in-metadata stays resident under paging; long memory chunks re-inflate resident,
  which is now the explicit reason summarization lives in the consumer — a design pressure
  named, not solved here.
- Two open-mode names become deprecated aliases; a source-compatible today, a rename at
  2.0. One more format family (`PQHWSaveLayout`) is renamed for consistency.
- Until implemented, every number here is arithmetic or an acceptance gate to be measured
  under ADR-005.

## Open questions

Left to the implementer and to tinybrain's stabilizing policy:

1. **The `QuantizedVectorStore` sibling (Option D).** When tinybrain's tiering cadence
   settles, does the cold PQHW tier become a first-class store (`index.pqhw` beside
   `index.pxkt`, CHA-107 sibling shape) — with defined `save()`/`checkpoint()` semantics
   for a build-once leg — or stay a raw-index composition? Deferred, not decided.
2. **Metadata-typed recall filter** — add `filter: (@Sendable (ChunkMetadata) -> Bool)`
   so the consumer need not thread its own `[UUID: Attributes]` map, or hold the pattern?
3. **Native `ChunkMetadata.timestamp: Date?`** — recency is universal to forgetting, but
   it touches a persisted `Codable` type; ship it, or keep recency in `extra`?
4. **Auto-checkpoint policy shape** — is `WALCheckpointPolicy` (bytes-fraction + maxOps)
   the right trigger for an agent's bursty ingest, or does agent cadence want a
   time-based or idle-triggered fold that folds off the ingest path entirely?
5. **`IndexResidency` canonical vs. alias** — reuse `HNSWOpenMode` as canonical
   (lowest churn, but "HNSW" misnames the quantized family) or introduce the neutral
   `IndexResidency` (chosen here)? Confirm at implementation against source-compat cost.

## Implementation notes (Stages A+B)

Shipped in this implementation:

- **Stage A** — `VectorStore.open` and `HybridVectorStore.open` now accept
  `checkpointAutomatically: WALCheckpointPolicy? = nil`. When non-nil, the store
  checks the policy after each serialized mutation batch and calls the existing
  checkpoint fold inside that same serialized chain. Automatic fold errors are
  rethrown by the mutation call that triggered the fold; the default `nil` path
  preserves the manual `needsCheckpoint` / `checkpoint` lifecycle.
- **Stage B** — `HNSWIndex.load(from:mode:)` mirrors the quantized loader and wraps
  the existing paged HNSW loader, while store opens now accept `dense:
  IndexResidency = .resident` and plumb it to the dense HNSW open path.
- **Naming foundation** — `IndexResidency` is the canonical residency enum, with
  `HNSWOpenMode` and `PQHWOpenMode` retained as source-compatible typealiases.

Still pending:

- **Stage C save-layout rename** — the `PQHWSaveLayout` -> `IndexSaveLayout`
  rename is explicitly deferred to Stage C. Stages A+B did not fold it into the
  residency-alias work.
- **Stage C docs/pattern work** beyond the shared residency naming foundation:
  broader agent-memory documentation, metadata-filter pattern docs, and any future
  staged deprecation annotations for old residency spellings.
