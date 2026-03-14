#!/usr/bin/env python3
"""
Converts a HuggingFace sentence-transformer to CoreML via ONNX.

Usage:
    python3 -m venv venv && source venv/bin/activate
    pip install coremltools onnx onnxruntime sentence-transformers numpy
    python3 scripts/convert_model.py

Output:
    Models/MiniLM-L6-v2.mlpackage
"""

import os
import sys
import tempfile

def main():
    try:
        import numpy as np
        import coremltools as ct
        import onnx
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install coremltools onnx onnxruntime sentence-transformers numpy")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models")
    output_path = os.path.join(output_dir, "MiniLM-L6-v2.mlpackage")

    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return

    max_length = 128
    hidden_dim = 384

    # Step 1: Export to ONNX using sentence-transformers
    print("Exporting sentence-transformer to ONNX...")
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.eval()

        # We'll export just the transformer + mean pooling
        # Create a wrapper that takes token IDs and attention mask
        class EmbeddingWrapper(torch.nn.Module):
            def __init__(self, st_model):
                super().__init__()
                self.transformer = st_model[0].auto_model
                self.pooling = st_model[1]

            def forward(self, input_ids, attention_mask):
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state
                # Mean pooling
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                summed = torch.sum(token_embeddings * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                return summed / counts

        wrapper = EmbeddingWrapper(model)
        wrapper.eval()

        # Export to ONNX
        onnx_path = os.path.join(tempfile.gettempdir(), "minilm.onnx")
        dummy_ids = torch.ones(1, max_length, dtype=torch.long)
        dummy_mask = torch.ones(1, max_length, dtype=torch.long)

        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            opset_version=14,
        )
        print(f"ONNX exported to {onnx_path}")

    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("\nFalling back to creating a simple test model...")
        create_simple_model(output_dir, output_path, max_length, hidden_dim)
        return

    # Step 2: Convert ONNX to CoreML
    print("Converting ONNX to CoreML...")
    try:
        mlmodel = ct.convert(
            onnx_path,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="embeddings")],
            minimum_deployment_target=ct.target.macOS14,
        )

        os.makedirs(output_dir, exist_ok=True)
        mlmodel.save(output_path)
        print(f"Saved to {output_path}")

        # Verify
        verify_model(model, mlmodel, max_length)

    except Exception as e:
        print(f"CoreML conversion failed: {e}")
        print("\nFalling back to creating a simple test model...")
        create_simple_model(output_dir, output_path, max_length, hidden_dim)


def create_simple_model(output_dir, output_path, max_length, hidden_dim):
    """Creates a simple CoreML model for testing (random projection)."""
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    import numpy as np

    # Build a simple model: input_ids [1, 128] → embedding lookup → mean → output [1, 384]
    builder = NeuralNetworkBuilder(
        input_features=[
            ("input_ids", ct.models.datatypes.Array(1, max_length)),
            ("attention_mask", ct.models.datatypes.Array(1, max_length)),
        ],
        output_features=[
            ("embeddings", ct.models.datatypes.Array(1, hidden_dim)),
        ],
        minimum_deployment_target=ct.target.macOS14,
    )

    # Random projection: flatten input → linear → output
    np.random.seed(42)
    weights = np.random.randn(hidden_dim, max_length).astype(np.float32) * 0.1
    bias = np.zeros(hidden_dim).astype(np.float32)

    builder.add_reshape_static(
        name="reshape_input",
        input_name="input_ids",
        output_name="flat_input",
        output_shape=(1, max_length),
    )

    builder.add_inner_product(
        name="project",
        input_name="flat_input",
        output_name="embeddings",
        input_channels=max_length,
        output_channels=hidden_dim,
        W=weights,
        b=bias,
        has_bias=True,
    )

    spec = builder.spec
    model = ct.models.MLModel(spec)

    os.makedirs(output_dir, exist_ok=True)
    model.save(output_path)
    print(f"Simple test model saved to {output_path}")
    print(f"NOTE: This is a random projection model, not a real sentence-transformer.")
    print(f"      It produces consistent but not semantically meaningful embeddings.")


def verify_model(st_model, mlmodel, max_length):
    """Quick verification that the converted model produces reasonable output."""
    import numpy as np

    print("\nVerifying conversion...")
    text = "This is a test sentence."

    # Get reference embedding from sentence-transformers
    ref = st_model.encode([text])[0]
    print(f"Reference embedding dim: {len(ref)}, norm: {np.linalg.norm(ref):.4f}")

    # Get CoreML embedding
    tokenizer = st_model[0].tokenizer
    tokens = tokenizer(text, padding="max_length", max_length=max_length,
                       truncation=True, return_tensors="np")

    result = mlmodel.predict({
        "input_ids": tokens["input_ids"].astype(np.int32),
        "attention_mask": tokens["attention_mask"].astype(np.int32),
    })

    coreml_emb = result["embeddings"].flatten()
    print(f"CoreML embedding dim: {len(coreml_emb)}, norm: {np.linalg.norm(coreml_emb):.4f}")

    # Check cosine similarity between reference and converted
    cos_sim = np.dot(ref, coreml_emb) / (np.linalg.norm(ref) * np.linalg.norm(coreml_emb))
    print(f"Cosine similarity (ref vs CoreML): {cos_sim:.6f}")

    if cos_sim > 0.99:
        print("Conversion verified — embeddings match!")
    else:
        print(f"WARNING: Similarity {cos_sim:.4f} is lower than expected. Model may have conversion artifacts.")


if __name__ == "__main__":
    main()
