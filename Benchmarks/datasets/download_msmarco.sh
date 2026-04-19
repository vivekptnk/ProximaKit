#!/usr/bin/env bash
# Download the MS MARCO passages (TREC-approved subset) and embed them with
# sentence-transformers. Produces .fvecs-format vectors that the harness can
# consume without any further preprocessing.
#
# Idempotent: reuses cached archives and embedded vectors.
#
# Files produced (under datasets/ms-marco/):
#   msmarco_base.fvecs   N passages × 384d  (MiniLM-L6-v2)
#   msmarco_query.fvecs  Q queries  × 384d
#   passages.tsv         raw passages (id \t text)
#   queries.tsv          raw queries  (id \t text)
#
# The --size flag in the Swift/Python harnesses decides how many of these to
# index, so we embed the full base set once and slice at bench time.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/ms-marco"
mkdir -p "$DEST"

PASSAGES_TSV="$DEST/passages.tsv"
QUERIES_TSV="$DEST/queries.tsv"
BASE_VEC="$DEST/msmarco_base.fvecs"
QUERY_VEC="$DEST/msmarco_query.fvecs"

# Tunables — override via env.
BASE_LIMIT="${BASE_LIMIT:-50000}"
QUERY_LIMIT="${QUERY_LIMIT:-1000}"
MODEL="${MSMARCO_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

if [[ -f "$BASE_VEC" && -f "$QUERY_VEC" ]]; then
    echo "[msmarco] already embedded at $DEST"
    exit 0
fi

# Fetch the TREC DL 2019 passage collection + queries.
fetch() {
    local url="$1"
    local out="$2"
    if [[ -f "$out" ]]; then return; fi
    echo "[msmarco] fetch $url"
    curl --fail --silent --show-error --location --output "$out" "$url"
}

fetch "https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz" "$DEST/collection.tar.gz"
fetch "https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv" "$DEST/queries.dev.small.tsv"

if [[ ! -f "$DEST/collection.tsv" ]]; then
    echo "[msmarco] extracting collection.tar.gz"
    tar -xzf "$DEST/collection.tar.gz" -C "$DEST"
fi

# Subset to the first N rows (deterministic, stable across runs).
head -n "$BASE_LIMIT" "$DEST/collection.tsv" > "$PASSAGES_TSV"
head -n "$QUERY_LIMIT" "$DEST/queries.dev.small.tsv" > "$QUERIES_TSV"

# Embed with sentence-transformers → .fvecs.
python3 "$HERE/embed_tsv.py" \
    --model "$MODEL" \
    --tsv "$PASSAGES_TSV" --out "$BASE_VEC"
python3 "$HERE/embed_tsv.py" \
    --model "$MODEL" \
    --tsv "$QUERIES_TSV" --out "$QUERY_VEC"

echo "[msmarco] ready at $DEST"
