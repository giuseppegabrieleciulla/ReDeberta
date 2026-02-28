"""
extract_layer0.py
-----------------
Extracts Layer 0 weights from the DeBERTa-v3-Large discriminator checkpoint
and saves them to a file that can be used to warm-start the recurrent model.

Usage (run from the /content/ReDeberta dir on Colab):
    python experiments/language_model/extract_layer0.py \
        --output /tmp/layer0_discriminator.bin

The script will automatically download the checkpoint from HuggingFace if
it is not already cached locally.
"""

import argparse
import os
import torch
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(
        description="Extract Layer 0 weights from DeBERTa-v3-Large discriminator."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="microsoft/deberta-v3-large",
        help="HuggingFace model ID to download the checkpoint from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/layer0_discriminator.bin",
        help="Where to save the extracted Layer 0 weights.",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from HuggingFace: {args.model_id} ...")
    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=args.model_id, filename="pytorch_model.bin")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download checkpoint from HuggingFace. "
            f"Make sure 'huggingface_hub' is installed (`pip install huggingface_hub`). "
            f"Error: {e}"
        )

    print(f"Loading state dict from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    print("Extracting Layer 0 weights...")
    layer0_state = OrderedDict()

    # Typically HuggingFace keys are like: deberta.encoder.layer.0.xxx
    # We want to produce keys that match the shape of a single RecurrentBertEncoder layer:
    # encoder.layer.xxx  (without the layer index)
    SRC_PREFIX = "deberta.encoder.layer.0."
    DST_PREFIX = "deberta.encoder.layer."

    # Also copy rel_embeddings and LayerNorm which are at the encoder level
    ENCODER_LEVEL_KEYS = [
        "deberta.encoder.rel_embeddings.weight",
        "deberta.encoder.LayerNorm.weight",
        "deberta.encoder.LayerNorm.bias",
        "deberta.encoder.conv.conv.weight",
        "deberta.encoder.conv.conv.bias",
        "deberta.encoder.conv.LayerNorm.weight",
        "deberta.encoder.conv.LayerNorm.bias",
    ]

    # Copy all embeddings and other top-level keys (embeddings, etc.)
    for k, v in state_dict.items():
        if k.startswith(SRC_PREFIX):
            # Remap layer.0.xxx -> layer.xxx
            new_key = DST_PREFIX + k[len(SRC_PREFIX):]
            layer0_state[new_key] = v
            print(f"  Mapped: {k} -> {new_key}")
        elif k in ENCODER_LEVEL_KEYS or not k.startswith("deberta.encoder.layer."):
            # Copy embeddings and encoder-level params verbatim
            layer0_state[k] = v

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(layer0_state, args.output)
    print(f"\nSaved {len(layer0_state)} tensors to: {args.output}")
    print("Done! You can now pass this file to --init_discriminator in rtd.sh.")


if __name__ == "__main__":
    main()
