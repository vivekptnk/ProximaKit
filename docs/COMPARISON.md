# ProximaKit and the Swift Vector-Search Ecosystem

An honest positioning of ProximaKit against the libraries a Swift developer actually shortlists for on-device work. Every claim below is checked against each project's own README; where the source does not state something, it is left out rather than guessed. Sources and the fetch date are at the bottom.

## The short version

ProximaKit is a pure-Swift, from-scratch HNSW **search engine** for Apple platforms, with two quantization tiers, hybrid BM25 + dense retrieval, and crash-safe incremental persistence. It ships an optional embeddings module (Apple NaturalLanguage / Vision / CoreML), but the core is engine-first: you bring vectors, it indexes and searches them.

The other names you will weigh:

- **SimilaritySearchKit** — embedding-first: on-device text embeddings plus semantic search, with models bundled in.
- **VecturaKit** — an on-device vector database with pluggable embedders and the broadest Apple-platform reach.
- **USearch** — a compact, cross-platform C++ HNSW engine with a Swift binding.

## Side by side

| Dimension | ProximaKit | SimilaritySearchKit | VecturaKit | USearch |
|---|---|---|---|---|
| **Core language** | Pure Swift | Pure Swift | Pure Swift | C++11 single-header (Swift binding) |
| **Primary focus** | Search engine + optional embeddings module | Embeddings + semantic search, batteries-included | On-device vector database, pluggable embedders | Bare similarity-search & clustering engine |
| **Search algorithm** | HNSW, implemented from scratch | Brute force (HNSW / Annoy listed as future work) | Vector similarity + BM25 hybrid (underlying ANN method not specified) | HNSW |
| **Hybrid dense + keyword** | Yes (BM25 + dense, RRF or weighted sum) | Not documented | Yes (BM25) | No (bare engine) |
| **Quantization** | Product (32×) and INT8 scalar (~4×) | Not documented | Not documented | i8, f16, bf16 |
| **Platform reach** | iOS 17+ · macOS 14+ · visionOS 1+ | iOS · macOS (iOS 16+ / macOS 13+ per its examples) | iOS 18+ · macOS 15+ · tvOS 18+ · visionOS 2+ · watchOS 11+ | Cross-platform: Linux, macOS, Windows, iOS, Android, WebAssembly |

*"Not documented" means the project's README does not describe the feature. It is not a claim that the feature is absent.* *One cell is an inference rather than a README quote: SimilaritySearchKit's "Pure Swift" comes from its repository's language breakdown, not an explicit README statement.*

## Choose X when

- **Choose ProximaKit** when you want a pure-Swift, from-scratch HNSW engine for Apple platforms — with INT8 and product quantization, hybrid BM25 + dense search, graph-aware filtered search, and crash-safe write-ahead-log persistence — and you would rather not bridge a C++ library or call a server.
- **Choose SimilaritySearchKit** when you want the fastest path to on-device semantic search with embedding models already bundled, and your corpus is small enough that exhaustive (brute-force) search is fine. Approximate indexing is on its roadmap, not shipped.
- **Choose VecturaKit** when you want a batteries-included on-device vector database with pluggable embedders — Apple NaturalLanguage, OpenAI-compatible endpoints, MLX — and the broadest Apple-platform coverage, including watchOS and tvOS.
- **Choose USearch** when you need a battle-tested, cross-platform HNSW engine and want the same index to run across iOS, Android, servers, and the browser — and you are comfortable consuming a C++ core through its Swift binding.

## A different category: FAISS and Pinecone

FAISS and Pinecone show up in a lot of comparisons, but neither is really on a Swift developer's on-device shortlist:

- **FAISS** is a C++ similarity-search library aimed at server and desktop. Running it on an Apple device means bridging C++ and shipping the native library — a different integration story from a Swift package.
- **Pinecone** is a hosted cloud vector database. It is a managed service: your vectors leave the device and you pay per usage. ProximaKit's whole premise is the opposite — everything stays on the device.

They remain useful reference points for scale and maturity, which is why ProximaKit's cross-library benchmark harness measures against FAISS (and ScaNN); see [BENCHMARKS.md](BENCHMARKS.md).

## Sources

Verified against each project's public README on 2026-07-10:

- SimilaritySearchKit — <https://github.com/ZachNagengast/similarity-search-kit>
- VecturaKit — <https://github.com/rryam/VecturaKit>
- USearch — <https://github.com/unum-cloud/usearch>

Positioning, platform support, core language, and the embedding-first vs engine-first distinction are taken directly from those READMEs (one exception, noted in the table footnote: SimilaritySearchKit's core language is inferred from its repository language breakdown). Where a detail was not stated in the source — for example VecturaKit's underlying approximate-search method, or whether SimilaritySearchKit and VecturaKit ship vector quantization — it is marked "not documented" or "not specified" above rather than inferred.
