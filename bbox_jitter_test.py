#!/usr/bin/env python3
"""
Evaluate a trained SGNetPose model across multiple bounding-box jitter tiers.

Loads a saved checkpoint once, then evaluates against each relative jitter tier
and prints a comparison table.

Usage:
    python PNP-Benchmark/bbox_filter_test.py \
        --checkpoint tools/jaad/checkpoints/SGNet_CVAE/0/best.pth \
        --jitter-scales 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
"""

import argparse
import json
import os.path as osp
import sys

# Ensure SGNetPose repo root (parent of PNP-Benchmark/) is importable
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
_SGNET_ROOT = osp.abspath(osp.join(_SCRIPT_DIR, ".."))
for _p in (_SGNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imports below depend on sys.path; kept after path setup (Ruff E402).
import lib.utils as utl  # noqa: E402
import torch  # noqa: E402
from configs.jaad import parse_sgnet_args as parse_args  # noqa: E402
from lib.losses import rmse_loss  # noqa: E402
from lib.models import build_model  # noqa: E402
from lib.utils.jaadpie_train_utils_cvae import test  # noqa: E402
from torch import nn  # noqa: E402


def _format_scale_label(scale):
    return f"scale_{scale:.3f}".rstrip("0").rstrip(".")


def main():
    # Jitter CLI flags are parsed first; remaining argv is handled by parse_sgnet_args().
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--jitter-scales", nargs="+", type=float)
    temp_parser.add_argument("--output-json", type=str)
    temp_parser.add_argument("--jitter-seed", type=int)
    temp_args, remaining = temp_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    try:
        sgnet_args = parse_args()
    finally:
        sys.argv = old_argv

    jitter_scales = (
        temp_args.jitter_scales
        if temp_args.jitter_scales
        else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    )
    output_json = temp_args.output_json if temp_args.output_json else ""
    jitter_seed = (
        temp_args.jitter_seed
        if temp_args.jitter_seed is not None
        else int(sgnet_args.seed)
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    utl.set_seed(int(sgnet_args.seed))

    model = build_model(sgnet_args)
    if device.type == "cuda":
        model = nn.DataParallel(model)
    model = model.to(device)

    if osp.isfile(sgnet_args.checkpoint):
        checkpoint = torch.load(
            sgnet_args.checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded checkpoint: {sgnet_args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint found at {sgnet_args.checkpoint}")
        print("Running with randomly initialized weights.")

    criterion = rmse_loss().to(device)

    results = {}
    print(f"\n{'=' * 80}")
    print(f"  Evaluating Gaussian bbox jitter across {len(jitter_scales)} tiers")
    print(f"{'=' * 80}\n")

    # One eval pass per tierx
    for scale in jitter_scales:
        label = _format_scale_label(scale)
        print(f"\n--- {label} ---")

        sgnet_args.bbox_jitter_scale = float(scale)
        sgnet_args.bbox_jitter_distribution = "gaussian"
        sgnet_args.bbox_jitter_seed = jitter_seed
        sgnet_args.bbox_jitter_apply_split = "test"

        test_gen = utl.build_data_loader(sgnet_args, "test")
        print(f"  Test samples: {len(test_gen.dataset)}")

        if len(test_gen.dataset) == 0:
            print(f"  SKIP: no test samples for {label}")
            del test_gen
            continue

        test_loss, MSE_15, MSE_05, MSE_10, FMSE, AIOU, FIOU, CMSE, CFMSE, ADE, FDE = (
            test(model, test_gen, criterion, device)
        )

        del test_gen
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        import gc

        gc.collect()

        results[label] = {
            "bbox_jitter_scale": float(scale),
            "bbox_jitter_distribution": "gaussian",
            "test_loss": float(test_loss),
            "ADE": float(ADE),
            "FDE": float(FDE),
            "MSE_05": float(MSE_05),
            "MSE_10": float(MSE_10),
            "MSE_15": float(MSE_15),
            "FMSE": float(FMSE),
        }

        print(
            "  ADE: %.4f  FDE: %.4f  MSE_05: %.4f  MSE_10: %.4f  MSE_15: %.4f  FMSE: %.4f"
            % (ADE, FDE, MSE_05, MSE_10, MSE_15, FMSE)
        )

    print(f"\n\n{'=' * 80}")
    print("  SUMMARY: Trajectory Prediction Quality vs. Bounding Box Jitter (Gaussian)")
    print(f"{'=' * 80}")
    header = f"{'Tier':<14} {'ADE':>10} {'FDE':>10} {'MSE_05':>10} {'MSE_10':>10} {'MSE_15':>10} {'FMSE':>10}"
    print(header)
    print("-" * len(header))
    for scale in jitter_scales:
        label = _format_scale_label(scale)
        if label in results:
            r = results[label]
            print(
                f"{label:<14} {r['ADE']:>10.4f} {r['FDE']:>10.4f} {r['MSE_05']:>10.4f} "
                f"{r['MSE_10']:>10.4f} {r['MSE_15']:>10.4f} {r['FMSE']:>10.4f}"
            )

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_json}")

    print("\nDone!")


if __name__ == "__main__":
    main()
