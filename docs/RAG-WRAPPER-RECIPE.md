# Build Your Own RAG Index on `HNSWIndex`

*The recipe for consumers who own their chunk pipeline and wrap the raw index directly — crash-safe chunk records, one file, no sidecar to keep in sync.*

ProximaKit ships two layers you can build RAG on. `VectorStore` / `HybridVectorStore` (see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) → Store Layer) manage the document→chunk map for you and are the right default when you want text in, answers out. But plenty of real consumers already own a chunking pipeline — sentence splitters, overlap windows, per-source provenance, their own IDs — and want to drop embeddings straight into the graph and get their own records back on search. **Wrapping `HNSWIndex` directly is a first-class path, not an advanced hack.** This recipe is how you do it without giving up crash safety.

The mechanism is proven in this repository twice over: it is the same per-vector-metadata pattern ProximaKit's own store layer is built on (see the derivation design in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)), and the full wrapper lifecycle — including crash recovery of chunk records through the WAL — is CI-verified by the recipe's companion test ([`CustomRAGWrapperRecipeTests`](../Tests/ProximaKitTests/CustomRAGWrapperRecipeTests.swift)). It is also shipping in practice: tinybrain — the agent-memory consumer [ADR-015](adr/ADR-015-agent-memory-integration.md) was designed around — persists its RAG index exactly this way (in-index chunk metadata, fingerprint-keyed cache acceptance) on its main branch.

Every snippet below is extracted or adapted from a compiled, CI-verified template: [`Tests/ProximaKitTests/CustomRAGWrapperRecipeTests.swift`](../Tests/ProximaKitTests/CustomRAGWrapperRecipeTests.swift) — the `RecipeRAGIndex` actor and its tests (snapshot round-trip, journaled WAL recovery, checkpoint-then-reopen, torn-WAL-tail prefix recovery, first-open-establishes-base, missing-base error, remove durability, and WAL-growth observability under a custom checkpoint policy). If a line here compiles in your head but not on your machine, the test file is the source of truth.

---

## The one idea to take away

`HNSWIndex.add` has a metadata slot, and **that slot is WAL-journaled**:

```swift
public func add(_ vector: Vector, id: UUID, metadata: Data? = nil) throws
```

The obvious use is what the [RAG tutorial](RAG-TUTORIAL.md) does: stash the chunk text there so a search result hands back the passage without a side lookup. The non-obvious, load-bearing fact this recipe teaches on top: the `metadata` bytes are not just held in memory and written into `.pxkt` snapshots — they are **recorded in every WAL `add` record**. The on-disk add-record layout is literally `UUID + level + vector + metadataLength + metadata bytes` (`WALFormat.swift`), and replay reinserts each node *with its metadata* (`WALRecord.add(id:level:vector:metadata:)`).

That single fact is what makes a journaled wrapper crash-safe with **zero sidecar files**: if your chunk record *is* the vector's metadata, then recovering the index recovers your chunk catalog too. Vectors and chunk records advance and roll back together, atomically, because they are the same bytes in the same log. There is nothing to keep in sync because there is nothing separate to sync.

Two strategies follow from this:

- **Strategy A — in-index metadata (primary).** Your `ChunkRecord` is JSON in the metadata slot. Crash consistency is free.
- **Strategy B — owned sidecar (alternative).** For records that outgrow a `Data` blob, you keep your own store and *reconcile it from the recovered index* on open. More power, and an obligation you now own.

---

## Strategy A — in-index metadata (start here)

### The record and the write

Define a `Codable` record and encode it into the metadata slot on `add`:

```swift
struct ChunkRecord: Codable, Sendable {
    let sourcePath: String
    let byteOffset: Int64
    let text: String
}

// One add carries the vector AND the chunk record.
let record = ChunkRecord(sourcePath: path, byteOffset: offset, text: chunkText)
try await index.add(embedding, id: id, metadata: JSONEncoder().encode(record))
```

`HNSWIndex` is an actor ([ADR-002](adr/ADR-002-actor-isolation.md)), so `add` is `try await` at the call site. You choose the `id` (a `UUID`) — hold onto it if you need to `remove` the chunk later, or let it default. The record can be anything `Encodable`; keep it small (it rides in memory, in snapshots, and in every WAL record — see [When metadata outgrows the slot](#strategy-b--owned-sidecar-for-when-records-outgrow-a-blob)).

### The search and the decode

`search` is non-throwing and returns `[SearchResult]`; each result carries `id`, `distance`, and the `metadata: Data?` you stored. Decode it back into your record:

```swift
func search(query: Vector, k: Int) async throws -> [(ChunkRecord, Float)] {
    let results = await index.search(query: query, k: k)
    return try results.map { result in
        guard let metadata = result.metadata else {
            throw RecipeRAGIndexError.missingChunkMetadata(result.id)
        }
        return (try JSONDecoder().decode(ChunkRecord.self, from: metadata), result.distance)
    }
}
```

This explicit form treats *missing* metadata as an error worth surfacing — appropriate when every vector must have a record. If you'd rather skip undecodable results silently, `SearchResult` also has a convenience that returns `nil` on absent-or-invalid metadata instead of throwing (this is what the DocC tutorial uses):

```swift
public func decodeMetadata<T: Decodable>(as type: T.Type) -> T?   // nil if no metadata or decode fails
```

Pick per site: throw when a record must exist, `decodeMetadata` when a best-effort filter is fine.

### The wrapper

Put it behind a small actor so callers never touch raw vectors. Adapted from `RecipeRAGIndex` in the test file (the `load` / `openJournaled` / `checkpoint` factories come in the next section):

```swift
actor RecipeRAGIndex {
    private let index: HNSWIndex
    private let baseURL: URL?            // set for journaled builds; nil for plain-file / in-memory
    private let walURL: URL?
    private let durability: WALDurability
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(dimension: Int,
         metric: any DistanceMetric = EuclideanDistance(),
         config: HNSWConfiguration = HNSWConfiguration(),
         baseURL: URL? = nil,
         walURL: URL? = nil,
         durability: WALDurability = .everyRecord) {
        self.index = HNSWIndex(dimension: dimension, metric: metric, config: config)
        self.baseURL = baseURL
        self.walURL = walURL
        self.durability = durability
    }

    // Used by the load / openJournaled factories to wrap an already-built index.
    private init(index: HNSWIndex, baseURL: URL?, walURL: URL?, durability: WALDurability) {
        self.index = index
        self.baseURL = baseURL
        self.walURL = walURL
        self.durability = durability
    }

    @discardableResult
    func addChunk(text: String, sourcePath: String, offset: Int64,
                  embedding: Vector, id: UUID = UUID()) async throws -> UUID {
        let record = ChunkRecord(sourcePath: sourcePath, byteOffset: offset, text: text)
        try await index.add(embedding, id: id, metadata: encoder.encode(record))
        return id
    }

    func search(query: Vector, k: Int) async throws -> [(ChunkRecord, Float)] {
        let results = await index.search(query: query, k: k)
        return try results.map { result in
            guard let metadata = result.metadata else {
                throw RecipeRAGIndexError.missingChunkMetadata(result.id)
            }
            return (try decoder.decode(ChunkRecord.self, from: metadata), result.distance)
        }
    }
}
```

`addChunk` and `search` are the whole retrieval half; the `baseURL` / `walURL` / `durability` fields and the private init wire up the persistence half, which is where we turn next.

---

## Persistence lifecycle

Three levels, in increasing order of durability guarantee and moving-parts. Pick the lowest one that meets your needs.

### Choosing static vs journaled persistence

The decision rule, before the details below: a **static or rebuilt-rarely corpus wants plain `save`/`load`** (level 1) — one file, no lifecycle overhead; a **corpus that grows or mutates incrementally wants journaled `open` + a checkpoint policy** (level 2). The WAL earns its keep only when there is a delta between saves to protect: its entire value is turning incremental writes into O(change) appends instead of O(corpus) rewrites, and keeping that delta crash-safe until the next fold. A corpus that is built once and never mutated again has no inter-save delta — so a journal buys you a base file *plus* a `.pxwal` sidecar *plus* checkpoint bookkeeping (generation/dimension/metric cross-checks on open, a `needsCheckpoint()` policy, a checkpoint cadence to plan) for **zero durability or cost win over the single plain `save` you were always going to do**. Build-once, load-many: level 1. Grow-over-time: level 2. (This is the rule one consumer integration validated in practice.)

### 1. Plain save / load — one call each

The default persistence path is untouched by any of the journaling machinery: `save(to:)` serializes vectors *and* their metadata into one atomically-rewritten `.pxkt` file; `load(from:)` restores both.

```swift
try await rag.save(to: url)                     // vectors + chunk records → one file
let loaded = try RecipeRAGIndex.load(from: url) // records come back decoded on search
```

```swift
static func load(from url: URL) throws -> RecipeRAGIndex {
    RecipeRAGIndex(index: try HNSWIndex.load(from: url),
                   baseURL: nil, walURL: nil, durability: .everyRecord)
}
```

`HNSWIndex.load(from:)` is a static, non-async `throws` (no actor to await — it builds a fresh one), so no `await`. The CI test `testSnapshotRoundTripReturnsIdenticalChunkRecords` asserts the loaded index returns *identical* `ChunkRecord`s, in the same neighbour order, as before the save. This is the right level when you rebuild or re-save in bulk and a full rewrite per save is acceptable.

**Cost model.** Every `save(to:)` rewrites the entire snapshot — O(corpus), not O(change). ADR-013 works the arithmetic: ≈1.76 GB per save for a 1M × 384d index, regardless of how many vectors actually changed. If your ingest loop saves often, graduate to level 2.

### 2. Journaled open + checkpoint — O(change) incremental durability

Opt into the write-ahead log and each mutation appends one record (~1.6 KB per `add` at 384d) instead of rewriting the snapshot. Because the metadata rides that record, **your chunk records get the same incremental, crash-safe persistence the vectors do.** Bind the wrapper to a base file plus a `.pxwal` sidecar and reopen with `open`:

```swift
static func openJournaled(baseURL: URL, walURL: URL,
                          durability: WALDurability = .everyRecord) async throws -> RecipeRAGIndex {
    let index = try await HNSWIndex.open(baseURL: baseURL, walURL: walURL, durability: durability)
    return RecipeRAGIndex(index: index, baseURL: baseURL, walURL: walURL, durability: durability)
}
```

`open` loads the base, replays the WAL to its longest valid prefix, and attaches the journal for future appends. (It cross-checks the sidecar against the base — generation, dimension, and metric — and throws `PersistenceError.walGenerationMismatch` / `.walDimensionMismatch` / `.walMetricMismatch` on a stale or mispaired WAL before replaying a single record.) The CI test `testJournaledRecoveryReplaysChunkMetadataFromWAL` adds chunks *after* a checkpoint, then reopens and asserts every chunk record comes back from WAL replay — proving the metadata survived in the log, not just in a snapshot.

**First open — there is no base yet.** `open` (and `openJournaled` over it) *requires the base `.pxkt` to already exist* — it loads the base with `Data(contentsOf:)` before it touches the WAL, so a first launch with no base on disk does not create one. It throws Foundation's file-not-found (`NSCocoaErrorDomain` / `NSFileReadNoSuchFileError`), **not** a typed `PersistenceError` you can pattern-match. The supported way to create the base is a single `checkpoint`: it is the one call that both *establishes* a journaled base on a fresh index and later *folds* an accumulated WAL back into it. So the first-launch shape is open-if-present-else-establish:

```swift
let rag: RecipeRAGIndex
if FileManager.default.fileExists(atPath: baseURL.path) {
    rag = try await RecipeRAGIndex.openJournaled(baseURL: baseURL, walURL: walURL, durability: durability)
} else {
    // First launch: no base on disk. Bind the journal paths to a fresh
    // in-memory index, then checkpoint ONCE to write the generation-1 base
    // (plus a fresh empty WAL) so every later launch can open it.
    rag = RecipeRAGIndex(dimension: dimension, baseURL: baseURL, walURL: walURL, durability: durability)
    try await rag.checkpoint()
}
```

The `else` branch is exactly what the journaled CI tests do before adding a single chunk (`RecipeRAGIndex(...baseURL:walURL:...)` then `checkpoint()`). Two tests pin this end to end: `testFirstOpenEstablishesBaseViaCheckpointThenReopens` walks the create-checkpoint-add-reopen path and asserts every chunk returns, and `testOpenJournaledWithoutBaseFileThrowsFoundationFileNotFound` pins that a raw open of a missing base throws the Foundation file error, not a `PersistenceError` — so don't reach for a typed catch to detect "no base yet"; check `fileExists` (or the Cocoa error) instead.

**Folding the WAL — checkpoint.** The WAL grows with every mutation; periodically fold it back into a fresh base with `checkpoint`, which compacts pending tombstones, writes a new generation-bumped base, `F_FULLFSYNC`s it, and resets the WAL to empty. Ask `needsCheckpoint()` when to do it rather than guessing:

```swift
try await index.add(embedding, id: id, metadata: encoder.encode(record))
if await index.needsCheckpoint() {                       // policy-driven, see below
    try await index.checkpoint(baseURL: base, walURL: wal, durability: durability)
}
```

The default policy is `WALCheckpointPolicy(walBytesFractionOfBase: 0.10, maxOps: 10_000)` — either bound being exceeded trips a checkpoint; both are configurable. (The store layer exposes this as an opt-in `checkpointAutomatically:` that folds inside the mutation chain; at the raw-index level you drive it yourself, exactly as above.)

**Testing against the WAL.** If a consumer test wants to assert "the WAL grew by exactly one record per add," it must pass a **custom policy with the byte-fraction arm disabled**, because the default 0.10 arm fires far sooner than a naive reading suggests. `checkpoint` writes a page-padded v3 base (its vector section aligned to a 16 KiB boundary so it can be mapped — see [When to graduate](#when-to-graduate)), so `baseByteCount` starts near 16 KiB even for a near-empty corpus; the 10% arm then trips once the WAL passes ~1.6 KiB. At the recipe's 384d add cost (~1.6 KB per record, from level 2 above) that is only one or two adds — correct production behavior (fold early while the base is small), but it means a byte-arm-driven checkpoint can land in the middle of the growth you were trying to observe. Set `walBytesFractionOfBase: .infinity` to isolate the op-count arm:

```swift
// Byte-fraction arm off; op arm left high so it doesn't trip during the test.
let policy = WALCheckpointPolicy(walBytesFractionOfBase: .infinity, maxOps: 10_000)
XCTAssertEqual(await index.journalRecordCount, addedCount)     // WAL grew by exactly N records
XCTAssertFalse(await index.needsCheckpoint(policy: policy))    // custom policy tolerates the growth
```

`journalRecordCount` and `journalByteCount` are the observability hooks (both `public`, both actor-isolated so both take `await`): `journalRecordCount` is records appended since the last checkpoint, `journalByteCount` is header + record bytes since the last checkpoint — `checkpoint` resets the record count to 0 and the byte count to the fresh WAL's 32-byte header. `testCustomPolicyDisablingByteFractionExposesWALRecordGrowth` uses exactly this policy to prove the count advances one-per-add, and checks that lowering `maxOps` below the record count still trips `needsCheckpoint` — so the custom policy disabled only the byte arm, not checkpointing itself.

**The do-not-retry contract — quote it, restate it, obey it.** Verbatim from the store layer, which builds on this same mechanism (`VectorStore.open`'s doc-comment, mirrored in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)):

> A failed automatic fold is rethrown by the mutation call that triggered it, but the triggering mutation has already been applied and made durable — to the extent of the active `WALDurability` dial (see the `checkpointAutomatically` parameter for the `.manual` caveat). Do not retry that mutation: `addChunks` assigns fresh UUIDs, so retrying would duplicate chunks. The store remains consistent, and the next mutation, `save()`, or `checkpoint()` re-attempts the fold.

Restated at the raw-index level (from the test file's own header): if a journaled `add` or a checkpoint throws, **do not blindly retry it.** At the index level the vector may already be present in memory or durable in the WAL; retrying with a fresh `UUID` can duplicate the chunk. There is no failure latch — the index stays consistent and the next mutation or checkpoint re-attempts the fold. If you must retry, retry only under an idempotency policy *you* own (e.g. a stable, content-derived `id` you can dedupe on), or reopen and reconcile from the recovered metadata.

**Remove is durable only after the next throwing op — `remove` returning `true` is not a durable delete.** `add` throws, so its WAL append surfaces any write error on the same call. `HNSWIndex.remove(id:)` is **non-throwing** (`@discardableResult ... -> Bool`), so it *can't* surface a failed append the same way: a failed `appendRemove` is **deferred** into the journal's pending error and re-raised by the **next throwing journaled op** — another `add`, an explicit `syncJournal()`, or a `checkpoint()`. So `remove` returning `true` means only "gone from the in-memory graph," not "durably journaled." If the process crashes before any throwing op runs, replay omits the removal on recovery and the "removed" vector reappears. (This durability asymmetry is documented in `HNSWIndex.remove`'s own doc-comment; the mechanism is `WALJournal.appendRemove` capturing the error and the next `surfacePending()` re-throwing it.)

The practical rule for remove-driven replacement — delete a re-indexed file's old chunks, add the new ones — is to **call `syncJournal()` (or `checkpoint()`) after a batch of removals** to surface that deferred path before you treat the delete as committed:

```swift
for id in staleChunkIDs { await index.remove(id: id) }   // non-throwing; a WAL error would be deferred
try await index.syncJournal()                            // surface any deferred WAL error now
// …now add the replacement chunks.
```

`testRemoveThenSyncJournalIsCleanAndRemovalSurvivesReopen` pins the honestly-testable depth: removals followed by `syncJournal()` throw nothing on the clean path, and a reopen replays the removals — removed chunks gone, survivors present, `liveCount` matching. (Forcing the deferred *error* to fire needs a WAL write to actually fail, which isn't reachable hermetically without fault injection, so the test pins the clean contract and the removal's durability across a reopen rather than faking a write fault.)

### 3. Durability dial — Darwin honesty in one paragraph

`WALDurability` chooses how hard each append pushes to disk. Mirror of the source doc (`WALJournal.swift`, and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)):

> `WALDurability` offers `.everyRecord`, `.everyBatch` (default, one `fsync` per mutation's record), and `.manual` (no `fsync` on append; a power loss before the next checkpoint can lose the unsynced tail). On Darwin, `fsync(2)` only pushes writes to the **drive cache**, not the physical media — these levels do **not** guarantee media durability. Only `fcntl(_:F_FULLFSYNC)`, which `checkpoint` always calls on the base file, forces a media write. Choose the durability level knowing which guarantee it does and doesn't give.

The template wrapper defaults to `.everyRecord` (strictest per-append flush); the underlying `HNSWIndex.open`/`checkpoint` default is `.everyBatch`. Either way, the *only* media-durable commit point is a `checkpoint` — plan your checkpoint cadence, not just your fsync level.

### Crash recovery is prefix-tolerant, not silently lossy

WAL records are CRC-framed. On reopen, the decoder stops at the first torn or bit-damaged record and returns the **longest intact prefix** — a tail torn off by a crash mid-append is *expected*, recovers cleanly, and never throws. Only a damaged WAL *header* (or a stale generation) throws a typed error. The CI test `testTornWALTailRecoversLongestValidChunkMetadataPrefix` proves both halves: it truncates the WAL mid-record and asserts exactly the pre-truncation chunks recover; then it truncates below the header and asserts a typed `PersistenceError.fileTooSmall` — never a trap, never silent corruption. Your recovery code should expect "some suffix of my most recent chunks may be gone" and never "the file is unreadable so I lost everything".

### Recovered ranking equals a rebuild's top hit and set, not its exact order

WAL replay is deterministic: it re-inserts each node with the *journaled* level (not a fresh RNG draw), so a reopened index is the same graph, node-for-node, as the one that wrote the log — `WALRecoveryTests` asserts exactly that through a full structural fingerprint (adjacency, levels, entry point, tombstones, vectors, metadata). What a recovered index does **not** equal is a *from-scratch rebuild* of the same vectors. A journaled index is grown incrementally — a checkpoint compacts and renumbers its live nodes into the base, then later adds append on top — and HNSW topology is a function of that insertion history. A single from-scratch build inserts the same vectors in a different order, so it builds a *different but equally valid* graph.

Two different-but-valid HNSW graphs agree on the nearest neighbour and, for well-separated data, on the top-`k` result *set* — but the exact ordering of near-ties at ranks 2..k can differ between them. So a consumer that validates a recovered or incrementally-grown index by rebuilding from source and diffing the two should assert **top-result and result-set equivalence, not positional or byte order.** That is the shape of every assertion in this recipe's tests: `results.first` for the top hit, `Set(results.map(\.0))` for membership — never a full ordered-array compare. (Those two, in turn, are recall properties of ANN over your data's separation, not hard guarantees — gate on the equivalence your corpus actually earns.)

### Cache acceptance: validating a loaded index

A successful `load(from:)` or `open(baseURL:walURL:...)` proves only that the file *parsed* — it does not prove the index is the one *this* run wants. A cache left by an earlier build can carry a different embedding dimension, a different vector count, or records your current `ChunkRecord` can no longer decode. **Before you trust a loaded index, validate it against what this run expects, and rebuild from source on any mismatch.** Three checks:

- **(a) Dimension.** `index.dimension == expectedDimension`. `dimension` is a `nonisolated let` on the actor (`HNSWIndex.swift`), so it reads with **no `await`** even though `HNSWIndex` is an actor.
- **(b) Live count.** `await index.liveCount == expectedCount`. `liveCount` *is* actor-isolated, so it takes `await` — and it counts live nodes only, unlike `count`, which still includes tombstones.
- **(c) Every live record decodes.** Not just the records a query happens to surface — *every* id: `for (id, metadata) in await index.liveEntries()`, decoding each and treating missing or undecodable metadata as a validation failure. `liveEntries()` (`await` — actor-isolated) is the right hook here precisely because it is a **complete enumeration of every live `(id, metadata)` pair**, not the top-k neighbours of some probe query; sampling via `search` would leave the long tail of never-retrieved records unchecked. It is the same enumeration `VectorStore.open` leans on to rebuild its sidecar (Strategy B, below).

```swift
func accepts(_ index: HNSWIndex, dimension: Int, count: Int) async -> Bool {
    guard index.dimension == dimension else { return false }        // nonisolated — no await
    guard await index.liveCount == count else { return false }      // actor-isolated — await
    for (_, metadata) in await index.liveEntries() {                // every live id, not top-k
        guard let metadata,
              (try? JSONDecoder().decode(ChunkRecord.self, from: metadata)) != nil
        else { return false }
    }
    return true
}

func loadOrRebuild(url: URL, dimension: Int, count: Int,
                   rebuild: () async throws -> HNSWIndex) async throws -> HNSWIndex {
    if let index = try? HNSWIndex.load(from: url),                  // a throw is treated as a miss
       await accepts(index, dimension: dimension, count: count) {
        return index
    }
    try? FileManager.default.removeItem(at: url)                    // drop the cache…
    return try await rebuild()                                      // …and rebuild from source
}
```

Delete-and-rebuild is a **safe, total fallback for every failure mode**, not only the three checks above — because, as established under [Crash recovery is prefix-tolerant, not silently lossy](#crash-recovery-is-prefix-tolerant-not-silently-lossy), a corrupt or truncated load throws a typed `PersistenceError` rather than trapping or returning a half-decoded index. The `try?` on `load` folds that throw into the same *miss → rebuild* path as a failed check, so no failure mode slips past both the checks and the throw: whatever is wrong with the cache, you delete it and rebuild.

*A pattern one consumer integration uses:* name the cache file after a hash of everything that would invalidate it — embedder identity, dimension, chunking config, corpus identity — e.g. `"\(fingerprint).pxkt"` where `fingerprint` is a hash over those four inputs. Any change to those inputs becomes a *different filename*, so a stale cache is never opened at all; the acceptance checks above then only have to catch corruption of a cache that already claims to match. It is a cheap first filter, not a full answer to cache naming and versioning.

*Optional optimization — skip the per-add dedup scan when open-time validation already covers it.* If your open path already runs check (c) — enumerate `liveEntries()` and force a full rebuild on *any* divergence from your manifest — then, once open returns, the index's live ids provably equal your manifest. A further `O(liveEntries)` scan of `liveEntries()` before each `add` to guard against re-adding an already-indexed file is then largely redundant: it re-checks an invariant open already established. Keep it as cheap insurance if you like, but it is not a correctness requirement, and either way it is dwarfed by the embedding call each add already pays — so it is never the first thing to optimize. (This only applies to consumers who validate against a manifest on open; without that open-time guarantee, keep whatever dedup you rely on.)

---

## Strategy B — owned sidecar, for when records outgrow a blob

Strategy A is right until your per-chunk state stops fitting comfortably in a `Data` blob that rides every WAL record and every snapshot: large payloads, cross-chunk structures, or state you want to query relationally (by source, by tag, by recency) rather than only by vector neighbourhood. Then you keep your own store keyed by the vector `UUID` and put only the embedding in the index:

```swift
actor SidecarRAGIndex {
    private let index: HNSWIndex
    private var chunksByID: [UUID: ChunkRecord]   // your own store: a DB, a file, whatever

    func addChunk(/* ... */) async throws -> UUID {
        let id = UUID()
        try await index.add(embedding, id: id)     // no metadata — sidecar owns the record
        chunksByID[id] = ChunkRecord(/* ... */)
        try saveSidecar(chunksByID)
        return id
    }
}
```

**The obligation you just took on.** You now have two things that can crash out of step. The HNSW WAL can replay an `id` your sidecar write never recorded (index ahead of sidecar), or your sidecar can hold an `id` that never reached the index (sidecar ahead of index). Ordering tricks do not fully close this — a crash can land between any two writes.

The robust fix is not tighter ordering, it is **derivation**: on open, rebuild (or reconcile) your sidecar *from the recovered index*, treating the index + its WAL as the single source of truth. This is exactly how ProximaKit's own store layer stays consistent — `VectorStore.open` does not trust `docmap.json` on disk; it calls `rebuildDocumentMap(from: await index.liveEntries())` after WAL replay, so a doc-map entry exists **iff** a live vector exists, and any stale, absent, or hand-corrupted sidecar is simply ignored (`VectorStore.swift`; design in [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) → "solved by derivation, not ordering", and [ADR-013](adr/ADR-013-streaming-persistence.md)). `HNSWIndex.liveEntries()` hands you `[(id: UUID, metadata: Data?)]` for exactly this — enumerate it on open and make your sidecar agree with it.

The practical consequence: Strategy A gets this property for free (the record *is* the metadata, so it can never diverge); Strategy B gets it only if you write the reconcile-on-open step. If you skip it, you have re-introduced the split-brain the store layer exists to prevent.

---

## When to graduate

The recipe above scales a long way as-is. Two thresholds are worth planning for.

**Memory footprint → page the vectors.** As the corpus grows into the tens of thousands of vectors and up, the resident Float32 vectors become the dominant cost. You don't change the wrapper — you change how you open it. `HNSWIndex.load(from:mode: .paged)` and journaled `HNSWIndex.open(baseURL:walURL:durability:mode: .paged)` serve the vector section from a read-only file mapping instead of decoding it resident, with search results byte-identical to resident mode:

```swift
let index = try HNSWIndex.load(from: url, mode: .paged)                       // snapshot
// or, journaled:
let index = try await HNSWIndex.open(baseURL: base, walURL: wal,
                                     durability: durability, mode: .paged)
```

Paging requires a padded v3 base (checkpoint once to write one; `.paged` on a non-v3 base throws a typed `PersistenceError`, never a trap). Measured on an Apple M4 Max (release, 100,000 × 384d, 146.5 MB vector payload): a paged open sits at **18.1 MB** resident versus **112.3 MB** for the same base opened `.resident` — 64% of the payload never resident. `.resident` stays the default. Design and full numbers: [ADR-013](adr/ADR-013-streaming-persistence.md) and [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

**Compression → quantize, and page the originals too.** When you want 32× smaller vectors, move the embedding store to `QuantizedHNSWIndex` (product quantization; L2-only search path, one k-means training pass — trade-offs in [ADR-011](adr/ADR-011-pq-codec.md)). Your metadata pattern is unchanged: same `add(_:id:metadata:)`, same decode on search. If you also need exact-distance reranking without paying resident memory for the retained originals, open `QuantizedHNSWIndex.load(from:mode: .paged)` on a base saved `save(to:layout: .pagedV3)` — the retained originals map from flash, restoring the full 32× story (measured 8.0 MB paged vs 43.1 MB resident at 100K × 384d). Design: [ADR-012](adr/ADR-012-pq-reranking.md) (reranking) and [ADR-014](adr/ADR-014-paged-originals.md) (paged originals). tinybrain's two-tier hot/cold shape — journaled HNSW for recent memory, PQHW-paged for cold — is a consumer-composed pattern over exactly these primitives ([ADR-015](adr/ADR-015-agent-memory-integration.md)).

---

## Where next

- [`docs/RAG-TUTORIAL.md`](RAG-TUTORIAL.md) — the end-to-end on-device RAG walkthrough (embed → retrieve → augment → answer) if you don't yet own a chunk pipeline
- [`Tests/ProximaKitTests/CustomRAGWrapperRecipeTests.swift`](../Tests/ProximaKitTests/CustomRAGWrapperRecipeTests.swift) — the compiled `RecipeRAGIndex` template every snippet here comes from
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — streaming persistence, derivation-based crash consistency, the store layer
- [ADR-013](adr/ADR-013-streaming-persistence.md) · [ADR-014](adr/ADR-014-paged-originals.md) · [ADR-015](adr/ADR-015-agent-memory-integration.md) — WAL + paging, paged originals, and the agent-memory consumer this recipe generalizes
- [`docs/HYBRID.md`](HYBRID.md) — add BM25 keyword recall via `HybridIndex` when dense retrieval alone misses exact terms
