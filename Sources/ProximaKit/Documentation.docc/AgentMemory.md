# Building Agent Memory

Use ProximaKit as the durable retrieval substrate for an on-device agent while keeping memory policy in your application.

## Overview

For most agents, start with one journaled ``HybridVectorStore``. It keeps semantic and keyword recall together, persists dense mutations through a write-ahead log, reconstructs the sparse leg from recovered dense entries, and can page the dense vector payload from disk. Use ``VectorStore`` instead when dense-only recall is intentional.

```swift
let memory = try await HybridVectorStore.open(
    name: "agent-memory",
    embedder: embedder,
    storageDirectory: applicationSupport,
    checkpointAutomatically: WALCheckpointPolicy(),
    dense: .paged
)
```

The store factory creates and checkpoints a missing base before opening it, so this is also the recommended first-launch path. ``IndexResidency/paged`` requires a paging-capable v3 base; store factories establish one automatically.

## Ingest with bounded WAL growth

Write each memory as a chunk and call ``HybridVectorStore/save()`` when the active ``WALDurability`` policy requires an explicit flush:

```swift
let metadata = ChunkMetadata(
    documentId: conversationID.uuidString,
    chunkIndex: turnIndex,
    text: memoryText,
    extra: [
        "kind": "episodic",
        "timestamp": String(timestamp.timeIntervalSince1970),
        "salience": "0.72"
    ]
)

let ids = try await memory.addChunks([memoryText], metadata: [metadata])
try await memory.save()
```

With `checkpointAutomatically` set, the store checks the policy after each serialized `addChunks` or `removeDocument` mutation and folds the WAL when either threshold is strictly exceeded (`>`). The fold can therefore make the triggering mutation slower than a normal append.

An automatic fold can fail after the mutation is already applied and durable, including during post-commit cache refresh. Do not retry `addChunks`: it assigns new UUIDs, so retrying can duplicate memory. The store remains consistent. Retry an explicit ``HybridVectorStore/checkpoint()`` when appropriate, or let a later mutation attempt the automatic fold again. ``HybridVectorStore/save()`` only synchronizes the journaled WAL; it does not retry a failed automatic fold.

The same lifecycle is available on ``VectorStore`` for dense-only memory.

## Recall semantically and lexically

``HybridVectorStore/query(_:k:candidatePoolK:filter:)`` embeds once, searches dense and BM25 legs, and fuses the rankings. Reciprocal Rank Fusion is the default. `candidatePoolK` controls the pre-fusion candidate pool.

```swift
let results = try await memory.query(
    "What did the user decide about offline storage?",
    k: 8,
    candidatePoolK: 32,
    filter: allowedMemory
)
```

A hybrid result's `distance` is the negative fused score, so lower values still rank first; it is not a dense-vector distance. If the query embedding dimension does not match the dense leg, hybrid retrieval can continue with sparse-only recall rather than failing the whole query.

## Map domain metadata to UUID filters

``ChunkMetadata`` stores `documentId`, `chunkIndex`, `text`, and optional string-valued `extra`. Persist policy attributes such as memory kind, timestamp, and salience in `extra`, but keep typed policy state in the consumer.

The exact filter type is `(@Sendable (UUID) -> Bool)?`. Capture an immutable, consumer-owned snapshot whose values are `Sendable`:

```swift
struct MemoryAttributes: Sendable {
    let kind: String
    let timestamp: TimeInterval
    let salience: Double
}

let attributesByID: [UUID: MemoryAttributes] = currentAttributes
let cutoff = Date.now.addingTimeInterval(-30 * 24 * 60 * 60).timeIntervalSince1970

let allowedMemory: @Sendable (UUID) -> Bool = { id in
    guard let attributes = attributesByID[id] else { return false }
    return attributes.kind == "episodic"
        && attributes.timestamp >= cutoff
        && attributes.salience >= 0.5
}
```

Keep the UUID-to-domain mapping when `addChunks` returns its IDs. Graph-backed dense filtering occurs during layer-0 traversal with adaptive widening, rather than as a final post-filter.

## Use raw HNSW only when you own the wrapper

The store layer is the default because it manages chunk metadata, journaling, recovery, and first-launch base creation. If your application already owns chunk records and wraps ``HNSWIndex`` directly, follow [docs/RAG-WRAPPER-RECIPE.md](https://github.com/vivekptnk/ProximaKit/blob/main/docs/RAG-WRAPPER-RECIPE.md) rather than duplicating that lifecycle here.

One raw-index rule is essential: ``HNSWIndex/open(baseURL:walURL:durability:mode:)`` requires an existing base file. For a fresh index, create the base before the first open:

```swift
let fresh = HNSWIndex(dimension: embedder.dimension, metric: CosineDistance())
try await fresh.checkpoint(
    baseURL: baseURL,
    walURL: walURL,
    durability: .everyBatch
)

let index = try await HNSWIndex.open(
    baseURL: baseURL,
    walURL: walURL,
    durability: .everyBatch,
    mode: .paged
)
```

A paged raw open requires a padded v3 base. ``HNSWIndex/checkpoint(baseURL:walURL:durability:)`` writes that layout.

## Add a hot/cold tier only when needed

A consumer may later compose two tiers:

- A small journaled ``HybridVectorStore`` or journaled ``HNSWIndex`` for recent, mutable memories.
- A rebuilt ``QuantizedHNSWIndex`` for distilled cold memory, saved with ``IndexSaveLayout/pagedV3`` and loaded with ``IndexResidency/paged`` so retained originals are mapped for exact reranking.

This is an optional consumer architecture, not a second store API. Product-quantized indexes are build-once rather than append-oriented, so promotion into the cold tier means rebuilding it on the consumer's cadence.

## Keep mechanism and policy separate

ProximaKit owns mechanism:

- durable journaled mutation and checkpointing
- dense, sparse, and hybrid recall
- UUID filtering
- resident or paged index storage
- product quantization and exact reranking

The consumer owns policy:

- summarization and compaction of old turns
- salience or importance scoring
- forgetting and retention rules
- promotion between hot and cold tiers
- tier rebuild and checkpoint cadence
- distillation and the language model that performs it

This boundary keeps retrieval and persistence reusable without embedding one agent's cognition or lifecycle rules into the library.
