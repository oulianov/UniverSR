"""
Compute LSD metrics between generated and reference (ground truth) audio folders.

Typical workflow:
    1. Prepare GT folder (48 kHz reference wavs)
    2. Create LR folder (downsample GT to target input rate)
    3. Run inference:  python scripts/inference.py --input_dir lr/ --output_dir gen/ ...
    4. Compute metrics: python scripts/compute_metrics.py --reference_dir gt/ --output_dir gen/ --cutoff-sr 8

Usage:
    python scripts/compute_metrics.py \
        --reference_dir data/gt_48k/ \
        --output_dir    results/enhanced/ \
        --cutoff-sr 8
"""

import argparse
import json
import os
from glob import glob

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Must match model.sr_to_lr_bins in config (n_fft=1024, sr=48kHz)
SR_TO_CUTOFF_BIN = {8: 80, 12: 128, 16: 170, 24: 256}
TARGET_SR = 48000

def parse_args():
    parser = argparse.ArgumentParser(description="Compute LSD metrics: generated vs reference")
    parser.add_argument('--reference_dir', type=str, required=True, help="Directory of reference (GT) wav files.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory of generated wav files.")
    parser.add_argument('--cutoff-sr', type=int, required=True, choices=[8, 12, 16, 24],
                        help="Input sample rate in kHz (determines LSD-high cutoff bin).")
    parser.add_argument('--output_json', type=str, default=None, help="Path to save results as JSON.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def find_file_pairs(reference_dir, output_dir):
    """Match files by relative path between reference and output directories."""
    ref_files = sorted(glob(os.path.join(reference_dir, '**', '*.wav'), recursive=True))
    pairs = []
    for ref_path in ref_files:
        rel_path = os.path.relpath(ref_path, reference_dir)
        out_path = os.path.join(output_dir, rel_path)
        if os.path.exists(out_path):
            pairs.append((ref_path, out_path))
    return pairs


def load_audio(path, target_sr):
    """Load and preprocess audio to mono at target sample rate."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T.copy())
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav


def stft_magnitude(audio, n_fft=1024, hop_length=256):
    """Compute STFT magnitude spectrum."""
    window = torch.hann_window(n_fft).to(audio.device)
    return torch.abs(torch.stft(audio, n_fft, hop_length, window=window, return_complex=True))


def compute_lsd(pred, target, cutoff_bin):
    """
    Compute Log-Spectral Distance with frequency band separation.
    Returns: dict with lsd_total, lsd_high, lsd_low.
    """
    sp = torch.log10(stft_magnitude(pred).square().clamp(min=1e-6))
    st = torch.log10(stft_magnitude(target).square().clamp(min=1e-6))

    def _lsd(a, b):
        return (a - b).square().mean(dim=1).sqrt().mean()

    return {
        'lsd_total': float(_lsd(sp, st)),
        'lsd_high': float(_lsd(sp[..., cutoff_bin:, :], st[..., cutoff_bin:, :])),
        'lsd_low': float(_lsd(sp[..., :cutoff_bin, :], st[..., :cutoff_bin, :])),
    }


def main():
    args = parse_args()
    cutoff_bin = SR_TO_CUTOFF_BIN[args.cutoff_sr]

    pairs = find_file_pairs(args.reference_dir, args.output_dir)
    if not pairs:
        print(f"No matching file pairs found between {args.reference_dir} and {args.output_dir}")
        return
    print(f"Found {len(pairs)} file pairs. Cutoff: {args.cutoff_sr} kHz (bin {cutoff_bin})")

    totals = {'lsd_total': 0.0, 'lsd_high': 0.0, 'lsd_low': 0.0}
    count = 0

    for ref_path, out_path in tqdm(pairs, desc="Computing metrics"):
        ref_wav = load_audio(ref_path, TARGET_SR).to(args.device)
        out_wav = load_audio(out_path, TARGET_SR).to(args.device)

        # Align lengths
        min_len = min(ref_wav.shape[-1], out_wav.shape[-1])
        ref_wav = ref_wav[..., :min_len]
        out_wav = out_wav[..., :min_len]

        metrics = compute_lsd(out_wav, ref_wav, cutoff_bin)
        for k in totals:
            totals[k] += metrics[k]
        count += 1

    avg = {k: v / count for k, v in totals.items()}
    avg['num_files'] = count

    print(f"\n{'='*40}")
    print(f"Results ({count} files, cutoff={args.cutoff_sr} kHz)")
    print(f"{'='*40}")
    print(f"  LSD Total: {avg['lsd_total']:.4f}")
    print(f"  LSD High:  {avg['lsd_high']:.4f}")
    print(f"  LSD Low:   {avg['lsd_low']:.4f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(avg, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
