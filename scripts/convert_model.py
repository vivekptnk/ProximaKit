#!/usr/bin/env python3
"""
Downloads the pre-exported ONNX model for all-MiniLM-L6-v2
and converts it to CoreML.

Usage:
    python3 -m venv venv && source venv/bin/activate
    pip install coremltools onnx numpy
    python3 scripts/convert_model.py

Output:
    Models/MiniLM-L6-v2.mlpackage (384-dim sentence embeddings)
"""

import os
import sys

def main():
    try:
        import numpy as np
        import coremltools as ct
        import onnx
    except ImportError as e:
        print(f"Missing: {e}")
        print("pip install coremltools onnx numpy")
        sys.exit(1)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Models")
    output_path = os.path.join(output_dir, "MiniLM-L6-v2.mlpackage")

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}")
        return

    max_length = 128

    # Step 1: Download pre-exported ONNX from HuggingFace
    onnx_path = download_onnx_model()

    # Step 2: Convert ONNX to CoreML
    print("Converting ONNX to CoreML...")
    mlmodel = ct.convert(
        onnx_path,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="token_type_ids", shape=(1, max_length), dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )

    os.makedirs(output_dir, exist_ok=True)
    mlmodel.save(output_path)
    print(f"Saved: {output_path}")

    # Step 3: Quick verification
    verify(mlmodel, max_length)


def download_onnx_model():
    """Downloads the ONNX model from HuggingFace Hub."""
    import tempfile
    onnx_dir = os.path.join(tempfile.gettempdir(), "minilm-onnx")

    # Check if already downloaded
    onnx_path = os.path.join(onnx_dir, "model.onnx")
    if os.path.exists(onnx_path):
        print(f"ONNX model cached at {onnx_path}")
        return onnx_path

    print("Downloading ONNX model from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        onnx_path = hf_hub_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            filename="onnx/model.onnx",
            local_dir=onnx_dir,
        )
        print(f"Downloaded to {onnx_path}")
        return onnx_path
    except ImportError:
        # Fallback: direct URL download
        import urllib.request
        url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
        os.makedirs(onnx_dir, exist_ok=True)
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, onnx_path)
        print(f"Downloaded to {onnx_path}")
        return onnx_path


def verify(mlmodel, max_length):
    """Quick check that the model produces output."""
    import numpy as np

    print("\nVerifying...")
    # Create dummy tokenized input
    input_ids = np.array([[101, 2023, 2003, 1037, 3231, 6251, 1012, 102] +
                           [0] * (max_length - 8)], dtype=np.int32)
    attention_mask = np.array([[1] * 8 + [0] * (max_length - 8)], dtype=np.int32)
    token_type_ids = np.zeros((1, max_length), dtype=np.int32)

    result = mlmodel.predict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # Find the embedding output (might have different names)
    for key, value in result.items():
        arr = np.array(value)
        if arr.size > 10:  # embedding output, not a scalar
            print(f"Output '{key}': shape={arr.shape}, norm={np.linalg.norm(arr.flatten()):.4f}")

    print("Conversion successful!")


if __name__ == "__main__":
    main()
