"""
SIV-AI Phase-4 Batch Inference

Usage:
    python inference.py --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR>

Processes every supported image / NPY file in the input directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PHASE4_SCRIPT = PROJECT_ROOT / "inference" / "infer_phase4.py"

SUPPORTED = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".npy",
}


def main():
    parser = argparse.ArgumentParser(
        description="SIV-AI Phase-4 Batch Image Restoration"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded input files",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored outputs will be saved",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_dir}"
        )

    if not PHASE4_SCRIPT.exists():
        raise FileNotFoundError(
            f"Phase-4 inference script not found:\n{PHASE4_SCRIPT}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not files:
        raise FileNotFoundError(
            f"No supported input files found in:\n{input_dir}"
        )

    print("=" * 70)
    print("SIV-AI PHASE-4 BATCH INFERENCE")
    print("=" * 70)
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files found     : {len(files)}")
    print("=" * 70)

    failed = []

    for index, input_file in enumerate(files, start=1):

        print()
        print(f"[{index}/{len(files)}] Processing: {input_file.name}")

        # Tell the Phase-4 script where the output should go.
        command = [
            sys.executable,
            str(PHASE4_SCRIPT),
            str(input_file),
            "--output_dir",
            str(output_dir),
        ]

        result = subprocess.run(command)

        if result.returncode != 0:
            failed.append(input_file.name)
            print(f"FAILED: {input_file.name}")
        else:
            print(f"SUCCESS: {input_file.name}")

    print()
    print("=" * 70)
    print("BATCH INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total files : {len(files)}")
    print(f"Successful  : {len(files) - len(failed)}")
    print(f"Failed      : {len(failed)}")

    if failed:
        print()
        print("Failed files:")
        for name in failed:
            print(f"  - {name}")

    print("=" * 70)


if __name__ == "__main__":
    main()