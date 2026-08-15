"""
SIV-AI Inference Launcher

Root-level wrapper for the Phase-4 inference pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PHASE4_SCRIPT = PROJECT_ROOT / "inference" / "infer_phase4.py"


def main():
    parser = argparse.ArgumentParser(
        description="SIV-AI Phase-4 Image Restoration"
    )

    parser.add_argument(
        "--input_dir",
        default="inputs",
        help="Directory containing input images"
    )

    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory for restored images"
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Optional single input file"
    )

    args = parser.parse_args()

    if not PHASE4_SCRIPT.exists():
        raise FileNotFoundError(
            f"Phase-4 inference script not found:\n{PHASE4_SCRIPT}"
        )

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # If a specific input was supplied, run Phase-4 directly on it.
    if args.input:
        input_path = Path(args.input).resolve()

        command = [
            sys.executable,
            str(PHASE4_SCRIPT),
            str(input_path),
        ]

    else:
        # Find usable inputs.
        supported = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
            ".npy",
        }

        files = sorted(
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        )

        if not files:
            raise FileNotFoundError(
                f"No supported input files found in:\n{input_dir}"
            )

        print("=" * 70)
        print("SIV-AI INFERENCE LAUNCHER")
        print("=" * 70)
        print(f"Input directory : {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Files found     : {len(files)}")
        print()

        # Run Phase-4 once for each input.
        for input_file in files:
            print("=" * 70)
            print(f"Processing: {input_file.name}")
            print("=" * 70)

            command = [
                sys.executable,
                str(PHASE4_SCRIPT),
                str(input_file),
            ]

            result = subprocess.run(command)

            if result.returncode != 0:
                print(
                    f"\nWARNING: Phase-4 failed for {input_file.name}"
                )

        print("\nInference processing completed.")
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()