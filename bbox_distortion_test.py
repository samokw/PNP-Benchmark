"""
Evaluate a trained SGNetPose model across multiple bbox distortion tiers.

Loads a saved checkpoint once, then evaluates against each distortion tier
and prints a comparison table.

Usage:
    python SGNetPose/PNP-Benchmark/bbox_distortion_test.py \
      --checkpoint SGNetPose/tools/jaad/checkpoints/SGNet_CVAE/0/best.pth \
      --distortion-mode uniform_scale \
      --distortion-levels 0.0 0.05 0.10 0.15 0.20
"""

import argparse
import json
import os.path as osp
import sys

import torch
from torch import nn

# Ensure SGNetPose repo root (parent of PNP-Benchmark/) and this package dir are importable
_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
_SGNET_ROOT = osp.abspath(osp.join(_SCRIPT_DIR, ".."))
for _p in (_SGNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib.utils as utl
from configs.jaad import parse_sgnet_args as parse_args
from distortion_utils import DEFAULT_LEVELS, DEFAULT_MODE, VALID_MODES
from lib.losses import rmse_loss
from lib.models import build_model
from lib.utils.jaadpie_train_utils_cvae import test


def _format_level_label(level):
    return f"level_{level:.3f}".rstrip("0").rstrip(".")


def main():
    # Distortion CLI flags are parsed first; remaining argv is handled by parse_sgnet_args().
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument(
        "--distortion-mode", choices=VALID_MODES, default=DEFAULT_MODE, type=str
    )
    temp_parser.add_argument("--distortion-levels", nargs="+", type=float)
    temp_parser.add_argument("--output-json", type=str)
    temp_parser.add_argument("--distortion-seed", type=int)
    temp_args, remaining = temp_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    try:
        sgnet_args = parse_args()
    finally:
        sys.argv = old_argv

    distortion_mode = temp_args.distortion_mode
    distortion_levels = (
        temp_args.distortion_levels if temp_args.distortion_levels else DEFAULT_LEVELS
    )
    output_json = temp_args.output_json if temp_args.output_json else ""
    distortion_seed = (
        temp_args.distortion_seed
        if temp_args.distortion_seed is not None
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
    print(
        f"  Evaluating {distortion_mode} with Gaussian sampling across "
        f"{len(distortion_levels)} bbox distortion tiers"
    )
    print(f"{'=' * 80}\n")

    # One eval pass per tier; JAADDataLayer applies bbox_distortion_* on the test loader.
    for level in distortion_levels:
        label = _format_level_label(level)
        print(f"\n--- {label} ---")

        sgnet_args.bbox_jitter_scale = 0.0
        sgnet_args.bbox_distortion_mode = distortion_mode
        sgnet_args.bbox_distortion_scale = float(level)
        sgnet_args.bbox_distortion_distribution = "gaussian"
        sgnet_args.bbox_distortion_seed = distortion_seed
        sgnet_args.bbox_distortion_apply_split = "test"

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
            "bbox_distortion_mode": distortion_mode,
            "bbox_distortion_scale": float(level),
            "bbox_distortion_distribution": "gaussian",
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
    print(
        f"  SUMMARY: Trajectory Prediction Quality vs. {distortion_mode} Distortion (Gaussian)"
    )
    print(f"{'=' * 80}")
    header = f"{'Tier':<14} {'ADE':>10} {'FDE':>10} {'MSE_05':>10} {'MSE_10':>10} {'MSE_15':>10} {'FMSE':>10}"
    print(header)
    print("-" * len(header))
    for level in distortion_levels:
        label = _format_level_label(level)
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
