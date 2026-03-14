#!/usr/bin/env python3
"""
Converts a HuggingFace sentence-transformer to CoreML format.

Usage:
    pip install coremltools transformers torch numpy
    python3 scripts/convert_model.py

Output:
    Models/MiniLM-L6-v2.mlpackage

The converted model takes:
    - input_ids: int32[1, 128]       (tokenized text)
    - attention_mask: int32[1, 128]  (1 for real tokens, 0 for padding)

And outputs:
    - embeddings: float32[1, 384]    (sentence embedding via mean pooling)
"""

import os
import sys
import numpy as np

def main():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        import coremltools as ct
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install coremltools transformers torch numpy")
        sys.exit(1)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
    output_path = os.path.join(output_dir, "MiniLM-L6-v2.mlpackage")

    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return

    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Create a wrapper that does mean pooling (sentence-transformers style)
    class MeanPoolingModel(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, input_ids, attention_mask):
            output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = output.last_hidden_state  # [batch, seq, dim]
            # Mean pooling: average token embeddings, weighted by attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask  # [batch, dim]

    wrapped = MeanPoolingModel(model)
    wrapped.eval()

    # Trace with dummy input
    max_length = 128
    dummy_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.int32)
    dummy_mask = torch.ones(1, max_length, dtype=torch.int32)

    print("Tracing model...")
    traced = torch.jit.trace(wrapped, (dummy_ids, dummy_mask))

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="embeddings"),
        ],
        minimum_deployment_target=ct.target.macOS14,
    )

    os.makedirs(output_dir, exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Verify
    print("\nVerifying...")
    text = "This is a test sentence."
    tokens = tokenizer(text, padding="max_length", max_length=max_length,
                       truncation=True, return_tensors="np")

    result = mlmodel.predict({
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
    })

    embedding = result["embeddings"]
    print(f"Output shape: {embedding.shape}")
    print(f"First 5 values: {embedding[0][:5]}")
    print(f"Norm: {np.linalg.norm(embedding):.4f}")
    print("\nConversion successful!")


if __name__ == "__main__":
    main()
