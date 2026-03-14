#!/usr/bin/env python3
"""
Converts all-MiniLM-L6-v2 to CoreML using torch.jit.trace.

Tested with: coremltools==7.2, torch==2.5.0, transformers==4.36.2
Use Python 3.11 (not 3.14).

Usage:
    /opt/homebrew/bin/python3.11 -m venv venv
    source venv/bin/activate
    pip install "coremltools==7.2" "torch==2.5.0" "transformers==4.36.2" "sentence-transformers==3.3.1" numpy
    python3 scripts/convert_model.py

Output: Models/MiniLM-L6-v2.mlpackage (384-dim sentence embeddings)
"""

import os, sys
import numpy as np

def main():
    try:
        import torch
        import coremltools as ct
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        print(f"Missing: {e}")
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, "Models")
    output_path = os.path.join(output_dir, "MiniLM-L6-v2.mlpackage")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return

    max_len = 128
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).cpu().eval()

    # Wrapper: BERT + mean pooling → 384-dim embedding
    class Wrapper(torch.nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
        def forward(self, ids, mask):
            out = self.bert(input_ids=ids, attention_mask=mask)
            tok = out.last_hidden_state
            m = mask.unsqueeze(-1).expand(tok.size()).float()
            return torch.sum(tok * m, 1) / torch.clamp(m.sum(1), min=1e-9)

    wrapper = Wrapper(bert).eval()

    print("Tracing...")
    dummy_ids = torch.randint(0, 1000, (1, max_len), dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))

    print("Converting to CoreML...")
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType("input_ids", shape=(1, max_len), dtype=np.int32),
            ct.TensorType("attention_mask", shape=(1, max_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType("embeddings")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )

    os.makedirs(output_dir, exist_ok=True)
    ml.save(output_path)
    print(f"Saved: {output_path}")

    # Verify
    text = "This is a test sentence."
    tok = tokenizer(text, padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="np")
    out = ml.predict({
        "input_ids": tok["input_ids"].astype(np.int32),
        "attention_mask": tok["attention_mask"].astype(np.int32),
    })
    emb = np.array(list(out.values())[0]).flatten()
    print(f"Output dim: {len(emb)}, norm: {np.linalg.norm(emb):.4f}")

    # Compare to PyTorch
    with torch.no_grad():
        ids = torch.tensor(tok["input_ids"], dtype=torch.long)
        mask = torch.tensor(tok["attention_mask"], dtype=torch.long)
        ref = wrapper(ids, mask).numpy().flatten()
    cos = np.dot(ref, emb) / (np.linalg.norm(ref) * np.linalg.norm(emb) + 1e-9)
    print(f"Cosine similarity vs PyTorch: {cos:.6f}")
    print("Done!" if cos > 0.99 else f"WARNING: low similarity {cos:.4f}")

if __name__ == "__main__":
    main()
