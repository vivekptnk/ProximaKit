# Benchmarks — Cross-Library Comparison Harness

ProximaKit is honest about how it compares to FAISS and ScaNN. This directory
contains the full harness that produces the numbers published in
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md#cross-library-comparison).

Design constraints:

1. **The core `ProximaKit` library has no dependency on FAISS or ScaNN.**
   Baselines are separate Python scripts. The Swift harness reads the
   ground-truth file they produced (or produces its own via
   `BruteForceIndex`), then writes its own result JSON.
2. **All harnesses agree on a single JSON schema** — see
   [`JSON_SCHEMA.md`](./JSON_SCHEMA.md). The aggregator just globs the output
   directory and emits a Markdown table.
3. **Reproducibility over winning.** CPU model, OS version, library versions
   and seeds are all in every JSON document. When ProximaKit loses on an
   axis we publish that.

## Layout

```
Benchmarks/
  Package.swift              # standalone SPM package, not a target of the library
  Sources/ProximaBench/      # Swift harness (hnsw + ground-truth subcommands)
  python/
    common.py                # shared schema + dataset loaders + memory probe
    faiss_hnsw.py            # FAISS HNSW baseline
    scann_hnsw.py            # ScaNN baseline (auto-skip on unsupported platforms)
    compare.py               # aggregator → markdown table
    requirements.txt
  datasets/
    download_sift1m.sh       # SIFT1M from INRIA TEXMEX
    download_msmarco.sh      # MS MARCO passages + MiniLM embeddings
    embed_tsv.py             # helper for download_msmarco.sh
  JSON_SCHEMA.md
```

## Quickstart (SIFT1M 100K slice)

```bash
# 1. Download SIFT1M once (~500 MB).
./datasets/download_sift1m.sh

# 2. Build the Swift harness in release mode.
swift build -c release --package-path Benchmarks
BIN="Benchmarks/.build/release/ProximaBench"

mkdir -p out

# 3. Compute exact ground truth for the 100K slice once.
"$BIN" ground-truth \
    --base    datasets/sift-1m/sift_base.fvecs \
    --queries datasets/sift-1m/sift_query.fvecs \
    --size    100000 --query-count 1000 --k 10 --metric l2 \
    --dataset sift-1m-100k \
    --out     out/GroundTruth__sift-1m-100k__k10.json

# 4. Run ProximaKit HNSW at efSearch=50.
"$BIN" hnsw \
    --base    datasets/sift-1m/sift_base.fvecs \
    --queries datasets/sift-1m/sift_query.fvecs \
    --gt      out/GroundTruth__sift-1m-100k__k10.json \
    --size    100000 --query-count 1000 \
    --dataset sift-1m-100k \
    --k 10 --m 16 --efc 200 --ef 50 --metric l2 \
    --out     out/ProximaKit__sift-1m-100k__hnsw__ef50.json

# 5. Run FAISS HNSW at the same config.
python -m pip install -r Benchmarks/python/requirements.txt
python Benchmarks/python/faiss_hnsw.py \
    --base datasets/sift-1m/sift_base.fvecs \
    --queries datasets/sift-1m/sift_query.fvecs \
    --gt   out/GroundTruth__sift-1m-100k__k10.json \
    --size 100000 --query-count 1000 \
    --dataset sift-1m-100k \
    --k 10 --m 16 --efc 200 --ef 50 --metric l2 \
    --out out/FAISS__sift-1m-100k__hnsw__ef50.json

# 6. (Optional) ScaNN — only runs on supported platforms.
python Benchmarks/python/scann_hnsw.py \
    --base datasets/sift-1m/sift_base.fvecs \
    --queries datasets/sift-1m/sift_query.fvecs \
    --gt   out/GroundTruth__sift-1m-100k__k10.json \
    --size 100000 --query-count 1000 \
    --dataset sift-1m-100k \
    --k 10 --num-leaves 100 --num-leaves-to-search 10 --reorder 100 \
    --metric cosine --skip-if-unavailable \
    --out out/ScaNN__sift-1m-100k__scann__leaves100.json

# 7. Combine into a Markdown table.
python Benchmarks/python/compare.py --in out/ --out out/compare.md
cat out/compare.md
```

## MS MARCO passages (semantic)

```bash
BASE_LIMIT=50000 QUERY_LIMIT=1000 ./datasets/download_msmarco.sh
# Then use datasets/ms-marco/msmarco_base.fvecs and msmarco_query.fvecs in
# place of the SIFT paths above. Use --metric cosine for semantic distance.
```

## CI smoke slice

Every PR that touches `Sources/ProximaKit/**` runs the SIFT1M **10K slice**
(`--size 10000 --query-count 200`) via
[`.github/workflows/benchmark.yml`](../.github/workflows/benchmark.yml).
Output JSON is published as a workflow artifact. The nightly job runs the
full 100K slice.

## Why a separate SPM package?

ProximaKit's `Package.swift` lists only `Foundation + Accelerate` as
imports. Adding a benchmark-only executable target would force the library
package to pull in `Benchmarks/` every time SPM resolves. Keeping
`Benchmarks/Package.swift` separate means `swift build` at the repo root
stays as clean as it always was, and `swift build --package-path Benchmarks`
is an opt-in call for people who actually want to run the harness.
