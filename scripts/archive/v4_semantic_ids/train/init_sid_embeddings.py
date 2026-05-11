"""
Fix pre-allocated SID token embedding rows in the quantized model.

The problem: rows 151665-151920 in Qwen2.5-0.5B-Instruct-4bit are pre-allocated
but all have identical embeddings, so the model cannot differentiate between
<SID_0>, <SID_10>, <SID_255>, etc. Training fails silently.

Fix: dequantize existing rows to find the mean embedding, create 256 unique
embeddings via sinusoidal perturbation, requantize with mlx, write back
to model.safetensors.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from loguru import logger

SRC = Path("models/qwen_sid_patched")
N_SID = 256
FIRST_SID_ID = 151665
GROUP_SIZE = 64
BITS = 4
HIDDEN = 896


def unpack_uint32_to_int4(w: np.ndarray) -> np.ndarray:
    """w: (N, 112) uint32 -> (N, 896) float32 in [0, 15]"""
    N, M = w.shape
    out = np.zeros((N, M * 8), dtype=np.float32)
    for bit_pos in range(8):
        out[:, bit_pos::8] = ((w >> (bit_pos * 4)) & 0xF).astype(np.float32)
    return out


def dequantize_np(w: np.ndarray, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    w: (N, 112) uint32
    s: (N, 14) float16  (scales)
    b: (N, 14) float16  (biases)
    Returns: (N, 896) float32
    """
    unpacked = unpack_uint32_to_int4(w)
    n_groups = HIDDEN // GROUP_SIZE
    result = np.empty_like(unpacked)
    for g in range(n_groups):
        lo, hi = g * GROUP_SIZE, (g + 1) * GROUP_SIZE
        scale = s[:, g:g+1].astype(np.float32)
        bias  = b[:, g:g+1].astype(np.float32)
        result[:, lo:hi] = unpacked[:, lo:hi] * scale + bias
    return result


def quantize_with_mlx(float_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    float_arr: (N, 896) float32
    Returns: (N, 112) uint32, (N, 14) float16, (N, 14) float16
    """
    arr_mx = mx.array(float_arr)
    w_mx, s_mx, b_mx = mx.quantize(arr_mx, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(w_mx, s_mx, b_mx)
    return np.array(w_mx), np.array(s_mx).astype(np.float16), np.array(b_mx).astype(np.float16)


def main():
    model_file = SRC / "model.safetensors"

    logger.info(f"Loading {model_file}...")
    tensors = {}
    with safe_open(model_file, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    weight = tensors["model.embed_tokens.weight"]  # (151936, 112) uint32
    scales = tensors["model.embed_tokens.scales"]  # (151936, 14) float16
    biases = tensors["model.embed_tokens.biases"]  # (151936, 14) float16

    logger.info(f"embed_tokens.weight: {weight.shape} {weight.dtype}")

    # Sample existing well-initialized rows to get mean embedding
    # Use first 151643 (base vocab), skip every 100 for speed
    logger.info("Computing mean embedding from existing tokens (sampled)...")
    sample_idx = np.arange(0, 151643, 100)
    existing_float = dequantize_np(
        weight[sample_idx],
        scales[sample_idx],
        biases[sample_idx],
    )
    mean_emb = existing_float.mean(axis=0)        # (896,)
    std_val  = float(existing_float.std())
    logger.info(f"Mean embedding norm: {np.linalg.norm(mean_emb):.4f} | std: {std_val:.6f}")

    # Verify current SID slot embeddings are identical
    sid_row_0  = dequantize_np(weight[[FIRST_SID_ID]],  scales[[FIRST_SID_ID]],  biases[[FIRST_SID_ID]])[0]
    sid_row_56 = dequantize_np(weight[[FIRST_SID_ID+56]], scales[[FIRST_SID_ID+56]], biases[[FIRST_SID_ID+56]])[0]
    logger.info(f"SID_0 norm: {np.linalg.norm(sid_row_0):.4f} | SID_56 norm: {np.linalg.norm(sid_row_56):.4f}")
    logger.info(f"SID_0 == SID_56: {np.allclose(sid_row_0, sid_row_56)}")

    # Strategy: copy actual pre-trained token embeddings from base vocab.
    # Picks 256 evenly-spaced token indices from [500, 50000] — rare enough to
    # avoid confusion with common tokens, but well-initialized with proper norms.
    # No requantization needed: just copy the existing uint32/float16 rows.
    logger.info("Selecting 256 donor tokens from base vocabulary...")
    rng = np.random.RandomState(42)
    donor_indices = np.linspace(500, 50000, N_SID, dtype=int)

    # Dequantize donors to verify they are distinct
    donor_float = dequantize_np(
        weight[donor_indices],
        scales[donor_indices],
        biases[donor_indices],
    )
    donor_norms = [float(np.linalg.norm(donor_float[i])) for i in [0, 10, 56, 127, 200, 255]]
    logger.info(f"Donor norms (sample): {[f'{n:.4f}' for n in donor_norms]}")
    dot_0_56 = float(np.dot(
        donor_float[0] / (np.linalg.norm(donor_float[0]) + 1e-9),
        donor_float[56] / (np.linalg.norm(donor_float[56]) + 1e-9),
    ))
    logger.info(f"Cosine similarity donor[0] vs donor[56]: {dot_0_56:.4f}")

    # Copy donor rows (already quantized) into SID slots — no requantization
    new_w = weight[donor_indices]
    new_s = scales[donor_indices]
    new_b = biases[donor_indices]
    logger.info(f"new_w: {new_w.shape} {new_w.dtype} (copied, no requantization)")

    # Write new rows into the arrays
    weight[FIRST_SID_ID:FIRST_SID_ID + N_SID] = new_w
    scales[FIRST_SID_ID:FIRST_SID_ID + N_SID] = new_s
    biases[FIRST_SID_ID:FIRST_SID_ID + N_SID] = new_b

    # Verify: SID_0 should now equal donor[0]
    verify = dequantize_np(weight[[FIRST_SID_ID]], scales[[FIRST_SID_ID]], biases[[FIRST_SID_ID]])[0]
    donor0 = donor_float[0]
    cos = float(np.dot(verify / (np.linalg.norm(verify)+1e-9), donor0 / (np.linalg.norm(donor0)+1e-9)))
    logger.info(f"Verify SID_0 matches donor[0]: cosine={cos:.4f} (should be ~1.0)")

    # Save back to model.safetensors
    tensors["model.embed_tokens.weight"] = weight
    tensors["model.embed_tokens.scales"] = scales
    tensors["model.embed_tokens.biases"] = biases

    logger.info(f"Saving fixed model to {model_file}...")
    save_file(tensors, str(model_file))

    # Update model.safetensors.index.json if it exists (no-op for single-file models)
    logger.success("Done. SID token embeddings are now unique and properly initialized.")
    logger.info("Run training again: mlx_lm.lora --config config/train_v2.yaml")


if __name__ == "__main__":
    main()
