#!/usr/bin/env bash
# Integrity check for the SIFT1M dataset used by .github/workflows/benchmark.yml.
#
# The upstream archive is served from a plain-FTP mirror with no published
# checksum, so a truncated or corrupted transfer would otherwise flow silently
# into benchmark numbers. This script hard-fails on structural mismatches:
#   - exact byte sizes (fvecs/ivecs records are fixed-width, so the full-file
#     sizes are fully determined by the documented vector counts/dims), and
#   - the per-record dimension headers in the first record of each file.
# It also logs SHA-256 digests for the job log.
#
# TODO(audit/ci): pin the logged SHA-256 values here (and compare against
# them) once a trusted CI run has recorded them; sizes+headers already catch
# truncation and decompression corruption.
#
# Usage: .github/ci-scripts/verify_sift1m.sh [dataset-dir]   (default: Benchmarks/datasets/sift-1m)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="${1:-$ROOT/Benchmarks/datasets/sift-1m}"

# name : expected bytes : expected leading dim header (uint32 LE)
# sift_base.fvecs       1,000,000 x (4 + 128*4) bytes
# sift_query.fvecs         10,000 x (4 + 128*4) bytes
# sift_groundtruth.ivecs   10,000 x (4 + 100*4) bytes
SPECS=(
    "sift_base.fvecs:516000000:128"
    "sift_query.fvecs:5160000:128"
    "sift_groundtruth.ivecs:4040000:100"
)

fail=0
for spec in "${SPECS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    expected_size="${rest%%:*}"
    expected_dim="${rest#*:}"
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
done

if [[ "$fail" -ne 0 ]]; then
    echo "::error::SIFT1M integrity check FAILED — delete the dataset cache and re-download" >&2
    exit 1
fi
echo "[sift1m-verify] OK"
