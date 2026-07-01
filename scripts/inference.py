"""
Batch inference: enhance all audio files in a folder.
Reads low-resolution wav files, runs UniverSR, writes 48 kHz outputs.

Usage:
    python scripts/inference.py \
        --input_dir  data/lr_8k/ \
        --output_dir results/enhanced/ \
        --model woongzip1/universr-audio \
        --input-sr 8000 \
        --ode-steps 5
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from glob import glob
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from universr import UniverSR


def parse_args():
    parser = argparse.ArgumentParser(description="UniverSR batch inference")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory of input LR wav files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save enhanced wav files.")

    # Model source (choose one)
    parser.add_argument('--model', type=str, default=None,
                        help="HuggingFace repo ID or local directory (e.g. woongzip1/universr-audio).")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Path to local training checkpoint (.pth). Requires --config.")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML config (required when using --ckpt).")

    # Inference options
    parser.add_argument('--input-sr', type=int, required=True,
                        choices=[8000, 12000, 16000, 24000], help="Input sample rate in Hz.")
    parser.add_argument('--ode-steps', type=int, default=4, help="Number of ODE integration steps.")
    parser.add_argument('--ode-method', type=str, default='midpoint',
                        choices=['euler', 'midpoint', 'rk4'], help="ODE solver method.")
    parser.add_argument('--guidance-scale', type=float, default=1.5, help="CFG guidance scale (None=disabled).")
    parser.add_argument('--reconstruction-method', type=str, default='original_signal',
                        choices=['original', 'original_signal'],
                        help="Spectrum assembly method. original keeps legacy behavior; original_signal preserves original low-frequency bins.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def load_model(args):
    """Load UniverSR from HuggingFace or local checkpoint."""
    if args.model is not None:
        return UniverSR.from_pretrained(args.model, device=args.device)
    elif args.ckpt is not None:
        if args.config is None:
            raise ValueError("--config is required when using --ckpt.")
        return UniverSR.from_local(args.ckpt, args.config, device=args.device)
    else:
        raise ValueError("Provide either --model (HuggingFace) or --ckpt + --config (local).")


def main():
    args = parse_args()
    model = load_model(args)

    input_files = sorted(glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True))
    if not input_files:
        print(f"No wav files found in {args.input_dir}")
        return
    print(f"Found {len(input_files)} files. Enhancing to {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    for filepath in tqdm(input_files, desc="Enhancing"):
        # Preserve relative directory structure
        rel_path = os.path.relpath(filepath, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        output = model.enhance(
            filepath,
            input_sr=args.input_sr,
            ode_method=args.ode_method,
            ode_steps=args.ode_steps,
            guidance_scale=args.guidance_scale,
            reconstruction_method=args.reconstruction_method,
        )
        torchaudio.save(out_path, output.cpu(), 48000)

    print(f"Done. Enhanced files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
