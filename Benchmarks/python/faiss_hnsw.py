"""FAISS HNSW baseline for cross-library comparison with ProximaKit.

Emits a JSON doc matching Benchmarks/JSON_SCHEMA.md v1.

Usage:
    python faiss_hnsw.py \\
        --base datasets/sift-1m/sift_base.fvecs \\
        --queries datasets/sift-1m/sift_query.fvecs \\
        --gt datasets/sift-1m/GroundTruth__sift-1m-100k__k10.json \\
        --dataset sift-1m-100k \\
        --size 100000 --query-count 1000 \\
        --k 10 --m 16 --efc 200 --ef 50 \\
        --metric l2 \\
        --out out/FAISS__sift-1m-100k__hnsw__ef50.json
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import (
    compute_recall,
    ground_truth_matrix,
    load_fvecs,
    load_ground_truth,
    now_seconds,
    resident_memory_mb,
    timed_block,
    write_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAISS HNSW baseline")
    p.add_argument("--base", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--query-count", type=int, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--efc", type=int, default=200)
    p.add_argument("--ef", type=int, default=50)
    p.add_argument("--metric", choices=["l2", "ip", "cosine"], default="l2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    p.add_argument("--version", default=None, help="override FAISS version string")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = now_seconds()

    import faiss  # imported lazily so the module is importable without faiss

    base = load_fvecs(args.base, limit=args.size).astype("float32", copy=False)
    queries = load_fvecs(args.queries, limit=args.query_count).astype("float32", copy=False)

    # Cosine is implemented as IP on unit-normalized vectors.
    if args.metric == "cosine":
        base = base / np.linalg.norm(base, axis=1, keepdims=True).clip(min=1e-12)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif args.metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    else:
        faiss_metric = faiss.METRIC_L2

    dim = base.shape[1]
    index = faiss.IndexHNSWFlat(dim, args.m, faiss_metric)
    index.hnsw.efConstruction = args.efc
    index.hnsw.efSearch = args.ef
    # Single-threaded search to match the Swift harness.
    faiss.omp_set_num_threads(1)

    with timed_block() as build:
        index.add(base)
    rss_after_build = resident_memory_mb()

    gt = load_ground_truth(args.gt)
    truth = ground_truth_matrix(gt, args.k)
    if truth.shape[0] != queries.shape[0]:
        raise SystemExit(
            f"ground truth has {truth.shape[0]} queries, harness loaded {queries.shape[0]}"
        )

    latencies_ms: list[float] = []
    predicted = np.empty((queries.shape[0], args.k), dtype="int64")
    for q in range(queries.shape[0]):
        t0 = time.perf_counter()
        _, idx = index.search(queries[q : q + 1], args.k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        predicted[q] = idx[0]

    recall = compute_recall(predicted, truth)

    write_result(
        library="FAISS",
        library_version=args.version or getattr(faiss, "__version__", "unknown"),
        dataset=args.dataset,
        dataset_size=int(base.shape[0]),
        dimension=int(dim),
        metric=args.metric,
        index_params={
            "type": "hnsw",
            "M": args.m,
            "efConstruction": args.efc,
            "efSearch": args.ef,
        },
        k=args.k,
        query_count=int(queries.shape[0]),
        build_time_seconds=build.elapsed,
        latencies_ms=latencies_ms,
        recall_at_10=recall,
        resident_memory_mb=rss_after_build,
        seed=args.seed,
        run_started_at=started,
        run_duration_seconds=now_seconds() - started,
        notes=args.notes,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
