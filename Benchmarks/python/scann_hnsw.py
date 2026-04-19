"""ScaNN baseline for cross-library comparison with ProximaKit.

ScaNN (Google) is not a pure HNSW library — it uses a tree-AH hybrid that is
usually tuned for dot-product recall on dense embeddings. We report it
alongside FAISS HNSW + ProximaKit HNSW as a separate datapoint with its own
`indexParams.type = "scann"`.

ScaNN only supports linux+x86_64 and macos+arm64 (via prebuilt wheels for
specific Python versions). On unsupported platforms this script exits with
code 0 after writing a 'skipped' marker file, so CI doesn't fail the job just
because the runner can't install ScaNN.

Emits a JSON doc matching Benchmarks/JSON_SCHEMA.md v1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    p = argparse.ArgumentParser(description="ScaNN baseline")
    p.add_argument("--base", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--query-count", type=int, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--metric", choices=["l2", "ip", "cosine"], default="cosine")
    # ScaNN tuning knobs
    p.add_argument("--num-leaves", type=int, default=100)
    p.add_argument("--num-leaves-to-search", type=int, default=10)
    p.add_argument("--reorder", type=int, default=100)
    p.add_argument("--training-sample-size", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--notes", default="")
    p.add_argument("--skip-if-unavailable", action="store_true",
                   help="write a 'skipped' marker and exit 0 if scann is not importable")
    p.add_argument("--version", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = now_seconds()

    try:
        import scann  # type: ignore
    except Exception as exc:
        msg = f"scann unavailable on this platform: {exc}"
        sys.stderr.write(f"[ScaNN] {msg}\n")
        if args.skip_if_unavailable:
            _write_skipped(args.out, msg)
            return
        raise SystemExit(2)

    base = load_fvecs(args.base, limit=args.size).astype("float32", copy=False)
    queries = load_fvecs(args.queries, limit=args.query_count).astype("float32", copy=False)

    # ScaNN is best-suited to normalized embeddings + dot product. Match cosine
    # by unit-normalizing; match L2 by a conversion (||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # monotone in dot when ||a||==||b||, so for unit-norm this is exactly cosine).
    if args.metric in ("cosine", "l2"):
        base = base / np.linalg.norm(base, axis=1, keepdims=True).clip(min=1e-12)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True).clip(min=1e-12)
    distance_measure = "dot_product"

    dim = base.shape[1]

    with timed_block() as build:
        builder = scann.scann_ops_pybind.builder(base, args.k, distance_measure)
        builder = builder.tree(
            num_leaves=args.num_leaves,
            num_leaves_to_search=args.num_leaves_to_search,
            training_sample_size=min(args.training_sample_size, base.shape[0]),
        )
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        if args.reorder > 0:
            builder = builder.reorder(args.reorder)
        searcher = builder.build()
    rss_after_build = resident_memory_mb()

    gt = load_ground_truth(args.gt)
    truth = ground_truth_matrix(gt, args.k)

    latencies_ms: list[float] = []
    predicted = np.empty((queries.shape[0], args.k), dtype="int64")
    for q in range(queries.shape[0]):
        t0 = time.perf_counter()
        idx, _ = searcher.search(queries[q], final_num_neighbors=args.k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        predicted[q] = idx

    recall = compute_recall(predicted, truth)

    write_result(
        library="ScaNN",
        library_version=args.version or getattr(scann, "__version__", "unknown"),
        dataset=args.dataset,
        dataset_size=int(base.shape[0]),
        dimension=int(dim),
        metric=args.metric,
        index_params={
            "type": "scann",
            "numLeaves": args.num_leaves,
            "numLeavesToSearch": args.num_leaves_to_search,
            "reorder": args.reorder,
            "trainingSampleSize": args.training_sample_size,
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


def _write_skipped(path: str, reason: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"library": "ScaNN", "skipped": True, "reason": reason}, fh, indent=2)
    sys.stderr.write(f"[ScaNN] wrote skipped marker to {path}\n")


if __name__ == "__main__":
    main()
