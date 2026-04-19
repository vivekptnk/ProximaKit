"""Shared helpers for the Python baseline harnesses.

Contract: every baseline (faiss_hnsw.py, scann_hnsw.py) emits a single JSON
document matching Benchmarks/JSON_SCHEMA.md v1.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- datasets


def load_fvecs(path: str, limit: int | None = None) -> np.ndarray:
    """Load a SIFT-style .fvecs file. Returns (N, d) float32 array."""
    raw = np.fromfile(path, dtype="int32")
    if raw.size == 0:
        return np.empty((0, 0), dtype="float32")
    dim = int(raw[0])
    stride = dim + 1
    count = raw.size // stride
    if limit is not None:
        count = min(count, limit)
    # View as (count, stride) then strip the leading dim column.
    view = raw[: count * stride].reshape(count, stride)
    if not np.all(view[:, 0] == dim):
        raise ValueError(f"inconsistent dim in {path}")
    return view[:, 1:].copy().view("float32")


def load_ivecs(path: str, limit: int | None = None) -> np.ndarray:
    """Load a SIFT-style .ivecs file. Returns (N, d) int32 array."""
    raw = np.fromfile(path, dtype="int32")
    if raw.size == 0:
        return np.empty((0, 0), dtype="int32")
    dim = int(raw[0])
    stride = dim + 1
    count = raw.size // stride
    if limit is not None:
        count = min(count, limit)
    view = raw[: count * stride].reshape(count, stride)
    if not np.all(view[:, 0] == dim):
        raise ValueError(f"inconsistent dim in {path}")
    return view[:, 1:].copy()


def load_ground_truth(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        gt = json.load(fh)
    if gt.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError(f"ground truth schemaVersion != {SCHEMA_VERSION}")
    return gt


def ground_truth_matrix(gt: dict[str, Any], k: int) -> np.ndarray:
    """Return (queryCount, k) int array of ground-truth neighbor indices."""
    stored_k = gt["k"]
    if stored_k < k:
        raise ValueError(f"ground truth has k={stored_k}, requested k={k}")
    flat = np.asarray(gt["neighbors"], dtype="int64")
    mat = flat.reshape(gt["queryCount"], stored_k)
    return mat[:, :k]


# ---------------------------------------------------------------- metrics


def compute_recall(predicted: np.ndarray, truth: np.ndarray) -> float:
    """predicted/truth are (Q, k) int arrays of dataset indices."""
    assert predicted.shape == truth.shape, (predicted.shape, truth.shape)
    k = predicted.shape[1]
    hits = 0
    for p_row, t_row in zip(predicted, truth):
        hits += len(set(int(x) for x in p_row) & set(int(x) for x in t_row))
    return hits / (predicted.shape[0] * k)


def percentiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    """Return (mean, p50, p95) over a list of per-query latencies in ms."""
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    arr = np.asarray(latencies_ms, dtype="float64")
    return (
        float(arr.mean()),
        float(np.percentile(arr, 50, method="higher")),
        float(np.percentile(arr, 95, method="higher")),
    )


# ---------------------------------------------------------------- memory


def resident_memory_mb() -> float:
    """Steady-state RSS in MiB. Uses psutil when available, else /proc fallback.

    Returns 0.0 on failure — the benchmark still succeeds, the memory column
    is just missing for that run.
    """
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    # Fallback for Linux-only envs without psutil.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------- platform


@dataclass
class PlatformInfo:
    os: str
    kernel: str
    arch: str
    cpuModel: str
    swiftVersion: str | None
    pythonVersion: str | None


def probe_platform() -> PlatformInfo:
    return PlatformInfo(
        os=platform.system().lower(),
        kernel=platform.release(),
        arch=platform.machine(),
        cpuModel=_cpu_model(),
        swiftVersion=None,
        pythonVersion=platform.python_version(),
    )


def _cpu_model() -> str:
    sysname = platform.system().lower()
    try:
        if sysname == "darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            return out or "unknown"
        if sysname == "linux":
            with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or socket.gethostname()


# ---------------------------------------------------------------- writing


def write_result(
    *,
    library: str,
    library_version: str,
    dataset: str,
    dataset_size: int,
    dimension: int,
    metric: str,
    index_params: dict[str, Any],
    k: int,
    query_count: int,
    build_time_seconds: float,
    latencies_ms: list[float],
    recall_at_10: float,
    resident_memory_mb: float,
    seed: int,
    run_started_at: float,
    run_duration_seconds: float,
    notes: str,
    output_path: str,
) -> None:
    mean_ms, p50_ms, p95_ms = percentiles(latencies_ms)
    qps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0
    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "library": library,
        "libraryVersion": library_version,
        "dataset": dataset,
        "datasetSize": dataset_size,
        "dimension": dimension,
        "metric": metric,
        "indexParams": index_params,
        "k": k,
        "queryCount": query_count,
        "buildTimeSeconds": build_time_seconds,
        "searchLatencyMeanMs": mean_ms,
        "searchLatencyP50Ms": p50_ms,
        "searchLatencyP95Ms": p95_ms,
        "queriesPerSecond": qps,
        "recallAt10": recall_at_10,
        "residentMemoryMb": resident_memory_mb,
        "platform": asdict(probe_platform()),
        "seed": seed,
        "runStartedAt": dt.datetime.fromtimestamp(
            run_started_at, tz=dt.timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "runDurationSeconds": run_duration_seconds,
        "notes": notes,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    sys.stderr.write(
        f"[{library}] wrote {output_path}\n"
        f"  dataset={dataset} size={dataset_size} dim={dimension}\n"
        f"  build={build_time_seconds:.2f}s  p50={p50_ms:.2f}ms  p95={p95_ms:.2f}ms\n"
        f"  recall@{k}={recall_at_10:.3f}  rss={resident_memory_mb:.1f}MB\n"
    )


def now_seconds() -> float:
    return time.time()


def timed_block() -> "Timer":
    return Timer()


class Timer:
    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self.t0
