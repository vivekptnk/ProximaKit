"""Embed a TSV file into a SIFT-style .fvecs file.

Second column of each row is the text; first column is the id, discarded
because the harness tracks positional indices, not dataset ids.
"""

from __future__ import annotations

import argparse
import struct
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        sys.stderr.write(
            "[embed_tsv] sentence-transformers missing; "
            "pip install sentence-transformers to use MS MARCO bench\n"
        )
        return 2

    texts: list[str] = []
    with open(args.tsv, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                texts.append(parts[1])

    if not texts:
        sys.stderr.write(f"[embed_tsv] no rows in {args.tsv}\n")
        return 2

    model = SentenceTransformer(args.model)
    vecs = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32", copy=False)

    dim = int(vecs.shape[1])
    with open(args.out, "wb") as fh:
        for row in vecs:
            fh.write(struct.pack("<i", dim))
            fh.write(row.tobytes())

    sys.stderr.write(
        f"[embed_tsv] wrote {args.out}: {vecs.shape[0]} vectors × {dim}d\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
