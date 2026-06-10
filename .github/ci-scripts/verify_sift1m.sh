#!/usr/bin/env bash
# Integrity check for the SIFT1M dataset used by .github/workflows/benchmark-core.yml (workflow_call core shared by benchmark.yml jobs).
#
# The upstream archive is served from a plain-FTP mirror with no published
# checksum, so a truncated or corrupted transfer would otherwise flow silently
# into benchmark numbers. This script hard-fails on:
#   - exact byte sizes (fvecs/ivecs records are fixed-width, so the full-file
#     sizes are fully determined by the documented vector counts/dims),
#   - the per-record dimension headers in the first record of each file, and
#   - SHA-256 digests pinned from a trusted CI run (see SPECS below).
# It also logs the SHA-256 digests for the job log.
#
# Usage: .github/ci-scripts/verify_sift1m.sh [dataset-dir]   (default: Benchmarks/datasets/sift-1m)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${1:-$ROOT/Benchmarks/datasets/sift-1m}"

# name : expected bytes : expected leading dim header (uint32 LE) : pinned SHA-256
# sift_base.fvecs       1,000,000 x (4 + 128*4) bytes
# sift_query.fvecs         10,000 x (4 + 128*4) bytes
# sift_groundtruth.ivecs   10,000 x (4 + 100*4) bytes
#
# SHA-256 digests pinned from the successful nightly Cross-Library Benchmark
# run 27271620693 (event: schedule, started 2026-06-10T11:00Z), the trusted
# run whose logs first recorded them:
#   https://github.com/vivekptnk/ProximaKit/actions/runs/27271620693
# A digest mismatch means the file differs byte-for-byte from the copy that
# run benchmarked. Re-pin ONLY from logs of a trusted CI run, never from a
# local download.
SPECS=(
    "sift_base.fvecs:516000000:128:21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816"
    "sift_query.fvecs:5160000:128:f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc"
    "sift_groundtruth.ivecs:4040000:100:2b71de0a8d5a83e6a84eec3e23fb8b611d8801dd9b3a6cd62f070ab65ea65f4f"
)

fail=0
for spec in "${SPECS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    expected_size="${rest%%:*}"
    rest="${rest#*:}"
    expected_dim="${rest%%:*}"
    expected_sha="${rest#*:}"
    path="$DIR/$name"

    if [[ ! -f "$path" ]]; then
        echo "::error::missing dataset file: $path" >&2
        fail=1
        continue
    fi

    actual_size="$(wc -c < "$path" | tr -d ' ')"
    actual_dim="$(od -An -t u4 -N 4 "$path" | tr -d ' ')"
    sha256="$(shasum -a 256 "$path" 2>/dev/null | cut -d' ' -f1 \
        || sha256sum "$path" | cut -d' ' -f1)"

    echo "[sift1m-verify] $name size=$actual_size dim=$actual_dim sha256=$sha256"

    if [[ "$actual_size" != "$expected_size" ]]; then
        echo "::error::$name: size $actual_size != expected $expected_size (truncated/corrupt download?)" >&2
        fail=1
    fi
    if [[ "$actual_dim" != "$expected_dim" ]]; then
        echo "::error::$name: leading dim header $actual_dim != expected $expected_dim" >&2
        fail=1
    fi
    if [[ "$sha256" != "$expected_sha" ]]; then
        echo "::error::$name: sha256 $sha256 != pinned $expected_sha (content drift — see pinning comment above SPECS)" >&2
        fail=1
    fi
done

if [[ "$fail" -ne 0 ]]; then
    echo "::error::SIFT1M integrity check FAILED — delete the dataset cache and re-download" >&2
    exit 1
fi
echo "[sift1m-verify] OK"
