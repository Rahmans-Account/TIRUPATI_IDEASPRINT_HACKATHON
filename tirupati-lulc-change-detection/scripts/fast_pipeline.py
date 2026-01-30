"""Fast pipeline for < 1 min runs (clip-only + baseline + fast export)."""
import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], label: str) -> None:
    print(f"\n>> {label}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast LULC pipeline")
    parser.add_argument("--years", nargs=2, type=int, help="Two years to compare")
    parser.add_argument("--skip-export", action="store_true", help="Skip PNG export step")
    parser.add_argument("--enhanced-visuals", action="store_true", help="Generate enhanced visualizations")
    args = parser.parse_args()
    python = sys.executable

    # Step 1: Clip-only preprocess (fast)
    preprocess_cmd = [python, "scripts/preprocess_all.py", "--clip-only"]
    if args.years:
        preprocess_cmd.extend(["--years", str(args.years[0]), str(args.years[1])])
    _run(preprocess_cmd, "Clip rasters to boundary (fast mode)")

    # Step 2: Baseline inference + change detection
    inference_cmd = [python, "scripts/run_inference.py", "--model", "baseline", "--detect-changes"]
    if args.years:
        inference_cmd.extend(["--years", str(args.years[0]), str(args.years[1])])
    _run(inference_cmd, "Baseline classification + change detection")

    # Step 3: Export visuals
    if not args.skip_export:
        if args.enhanced_visuals:
            # Use enhanced visualization script
            enhanced_cmd = [python, "scripts/generate_enhanced_visuals.py"]
            if args.years:
                enhanced_cmd.extend(["--years", str(args.years[0]), str(args.years[1])])
            _run(enhanced_cmd, "Generate enhanced visualizations")
        else:
            # Use fast export
            _run(
                [python, "quick_export.py", "--fast"],
                "Fast export visuals",
            )

    print("\nFast pipeline complete. Open http://localhost:3000")


if __name__ == "__main__":
    main()
