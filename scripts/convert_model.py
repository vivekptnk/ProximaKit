#!/usr/bin/env python3
"""
Converts all-MiniLM-L6-v2 sentence-transformer to CoreML.

Usage:
    python3 -m venv venv && source venv/bin/activate
    pip install coremltools==8.1 transformers==4.46.0 torch==2.5.0 numpy sentence-transformers
    python3 scripts/convert_model.py

Output:
    Models/MiniLM-L6-v2.mlpackage (384-dim sentence embeddings)
"""

import os
import sys

def main():
    try:
        import torch
        import numpy as np
        import coremltools as ct
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing: {e}")
        print("pip install coremltools transformers torch numpy sentence-transformers")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models")
    output_path = os.path.join(output_dir, "MiniLM-L6-v2.mlpackage")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return

    max_length = 128

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    transformer = st_model[0].auto_model
    transformer.eval()
    # Force CPU to avoid MPS device conflicts during tracing
    transformer = transformer.cpu()

    class MeanPoolWrapper(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            tokens = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
            return torch.sum(tokens * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    wrapper = MeanPoolWrapper(transformer).cpu().eval()

    # Trace with dummy input on CPU
    dummy_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long, device="cpu")
    dummy_mask = torch.ones(1, max_length, dtype=torch.long, device="cpu")

    print("Tracing (torch.jit.trace)...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embeddings")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    os.makedirs(output_dir, exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved: {output_path}")

    # Verify
    tokenizer = st_model[0].tokenizer
    text = "This is a test sentence."
    tokens = tokenizer(text, padding="max_length", max_length=max_length,
                       truncation=True, return_tensors="np")

    result = mlmodel.predict({
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
    })
    emb = np.array(result["embeddings"]).flatten()
    ref = st_model.encode([text])[0]

    cos = np.dot(ref, emb) / (np.linalg.norm(ref) * np.linalg.norm(emb))
    print(f"Dim: {len(emb)}, Cosine sim vs reference: {cos:.6f}")
    print("Done!" if cos > 0.99 else f"WARNING: similarity {cos:.4f} is low")


if __name__ == "__main__":
    main()
