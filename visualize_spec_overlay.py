"""
Specular Prior Green Overlay Visualizer.

Loads GT images and spec_prior maps from output_test results,
then overlays green color on specular-capable regions indicated by the prior.

Usage:
  # All scenes in output_test/
  python visualize_spec_overlay.py -d output_test

  # Single scene
  python visualize_spec_overlay.py -m output_test/cecum_t1_b

  # Custom threshold and alpha
  python visualize_spec_overlay.py -d output_test --threshold 0.5 --alpha 0.5

Output:
  Saves to <scene>/test/ours_<iter>/spec_green_overlay/  (one PNG per frame)
"""

import os
import sys
import glob
import torch
import torchvision
from argparse import ArgumentParser


def load_image(path):
    """Load a PNG image as [3, H, W] float tensor in [0, 1]."""
    return torchvision.io.read_image(path).float() / 255.0


def create_green_overlay(gt_rgb, spec_prior, threshold=0.5, alpha=0.5):
    """
    Overlay green tint on GT image where spec_prior >= threshold.

    Args:
        gt_rgb:     [3, H, W] in [0, 1]
        spec_prior: [C, H, W] in [0, 1] (grayscale, C=1 or 3)
        threshold:  binarization threshold for the prior
        alpha:      blending strength of the green overlay (0=no overlay, 1=solid green)

    Returns:
        [3, H, W] overlaid image
    """
    # Convert spec_prior to single-channel if needed
    if spec_prior.shape[0] == 3:
        sp = spec_prior.mean(dim=0)  # [H, W]
    else:
        sp = spec_prior.squeeze(0)   # [H, W]

    mask = (sp >= threshold).float()  # [H, W]

    # Green color: (0, 1, 0)
    green = torch.zeros_like(gt_rgb)
    green[1] = 1.0

    # Blend: where mask=1, mix green into GT
    overlay = gt_rgb.clone()
    for c in range(3):
        overlay[c] = torch.where(
            mask > 0.5,
            gt_rgb[c] * (1.0 - alpha) + green[c] * alpha,
            gt_rgb[c],
        )
    return overlay.clamp(0, 1)


def find_iteration(model_path, split="test"):
    """Find the latest iteration directory."""
    split_dir = os.path.join(model_path, split)
    if not os.path.isdir(split_dir):
        return None
    iterations = []
    for d in os.listdir(split_dir):
        if d.startswith("ours_"):
            try:
                iterations.append(int(d.split("_")[1]))
            except (ValueError, IndexError):
                pass
    return max(iterations) if iterations else None


def process_scene(model_path, split="test", iteration=-1,
                  threshold=0.5, alpha=0.5):
    """Process one scene: overlay green on GT at specular regions."""
    if iteration == -1:
        iteration = find_iteration(model_path, split)
        if iteration is None:
            print("  SKIP %s: no ours_* found" % model_path)
            return

    scene_dir = os.path.join(model_path, split, "ours_%d" % iteration)
    gt_dir = os.path.join(scene_dir, "gt")
    prior_dir = os.path.join(scene_dir, "spec_prior")
    out_dir = os.path.join(scene_dir, "spec_green_overlay")

    if not os.path.isdir(gt_dir):
        print("  SKIP: gt/ not found in %s" % scene_dir)
        return
    if not os.path.isdir(prior_dir):
        print("  SKIP: spec_prior/ not found in %s" % scene_dir)
        return

    os.makedirs(out_dir, exist_ok=True)

    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    prior_paths = sorted(glob.glob(os.path.join(prior_dir, "*.png")))

    assert len(gt_paths) == len(prior_paths), \
        "Mismatch: %d gt vs %d spec_prior in %s" % (len(gt_paths), len(prior_paths), scene_dir)

    for gt_path, prior_path in zip(gt_paths, prior_paths):
        fname = os.path.basename(gt_path)
        gt_rgb = load_image(gt_path)
        spec_prior = load_image(prior_path)

        overlay = create_green_overlay(gt_rgb, spec_prior, threshold, alpha)
        torchvision.utils.save_image(overlay, os.path.join(out_dir, fname))

    print("  [%s] %d images -> %s" % (os.path.basename(model_path), len(gt_paths), out_dir))


def process_all(base_dir, split="test", iteration=-1,
                threshold=0.5, alpha=0.5):
    """Process all scene subdirectories under base_dir."""
    scenes = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.isdir(os.path.join(base_dir, d, split))
    ])
    if not scenes:
        print("No scenes found in %s" % base_dir)
        return

    print("=" * 60)
    print("[SpecGreenOverlay] %d scenes | threshold=%.2f | alpha=%.2f" % (
        len(scenes), threshold, alpha))
    print("=" * 60)

    for scene_name in scenes:
        model_path = os.path.join(base_dir, scene_name)
        process_scene(model_path, split, iteration, threshold, alpha)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Overlay green on GT at specular prior regions")
    parser.add_argument("-m", "--model_path", default=None, type=str,
                        help="Single scene model path")
    parser.add_argument("-d", "--base_dir", default=None, type=str,
                        help="Directory containing multiple scenes (batch mode)")
    parser.add_argument("--split", default="test", type=str, choices=["train", "test"])
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration (-1 = latest)")
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Spec prior threshold for green overlay")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Blending strength (0=transparent, 1=solid green)")
    args = parser.parse_args()

    if args.base_dir is not None:
        process_all(args.base_dir, args.split, args.iteration,
                    args.threshold, args.alpha)
    elif args.model_path is not None:
        process_scene(args.model_path, args.split, args.iteration,
                      args.threshold, args.alpha)
    else:
        parser.print_help()
        sys.exit(1)
