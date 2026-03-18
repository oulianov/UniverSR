"""
Prepare a file list (.txt) for UniverSR training or evaluation.

Recursively finds all .wav files under the given directories and writes
their absolute paths to a text file (one path per line).

Usage:
    python scripts/prepare_file_list.py \
        --dirs /path/to/dataset1 /path/to/dataset2 \
        --output data/train.txt
"""

import argparse
import os
from glob import glob
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a file list of .wav paths for UniverSR training / evaluation."
    )
    parser.add_argument(
        "--dirs", type=str, nargs="+", required=True,
        help="One or more directories to scan for .wav files.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output text file path (e.g. data/train.txt).",
    )
    parser.add_argument(
        "--ext", type=str, default="wav",
        help="Audio file extension to search for (default: wav).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_files = []
    for d in args.dirs:
        pattern = os.path.join(d, "**", f"*.{args.ext}")
        found = sorted(glob(pattern, recursive=True))
        all_files.extend(found)

    # Resolve to absolute paths and deduplicate
    seen = set()
    unique = []
    for f in all_files:
        abspath = str(Path(f).resolve())
        if abspath not in seen:
            seen.add(abspath)
            unique.append(abspath)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(unique) + "\n")

    print(f"Wrote {len(unique)} file paths to {args.output}")


if __name__ == "__main__":
    main()