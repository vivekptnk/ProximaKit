#!/usr/bin/env bash
# Download the SIFT1M dataset from INRIA TEXMEX and extract the .fvecs files.
# Idempotent: safe to re-run, noop if files already exist.
#
# Files produced (under datasets/sift-1m/):
#   sift_base.fvecs       1,000,000 vectors × 128d  (488 MB)
#   sift_query.fvecs         10,000 vectors × 128d  (4.9 MB)
#   sift_groundtruth.ivecs   10,000 × 100 neighbors (4.0 MB)
#
# License: research-only (see INRIA page). Not redistributed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/sift-1m"
ARCHIVE="$HERE/sift.tar.gz"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

mkdir -p "$DEST"

if [[ -f "$DEST/sift_base.fvecs" && -f "$DEST/sift_query.fvecs" && -f "$DEST/sift_groundtruth.ivecs" ]]; then
    echo "[sift1m] already present at $DEST"
    exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "[sift1m] downloading $URL"
    curl --fail --silent --show-error --location --output "$ARCHIVE" "$URL"
fi

echo "[sift1m] extracting $ARCHIVE"
tar -xzf "$ARCHIVE" -C "$HERE"

# Upstream archive extracts to ./sift/
if [[ -d "$HERE/sift" ]]; then
    mv "$HERE/sift/"* "$DEST/"
    rmdir "$HERE/sift"
fi

rm -f "$ARCHIVE"
echo "[sift1m] ready at $DEST"
