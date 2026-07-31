"""
Standalone Specular Masking Evaluation Module.

Uses THREE inputs:
  - gt/         : Ground truth images
  - <pred_dir>/ : Predicted images (default: "base_rgb", changeable via --pred_dir)
  - spec_prior  : Loaded from spec_prior/ folder if it exists, otherwise computed
                  from GT as (1-S)*V  (HSV weak cue)

The pred_dir name is configurable so this works for any method:
  - Ours (specular branch): --pred_dir base_rgb
  - Vanilla 3DGS:           --pred_dir renders

Usage:
  # Batch: all scenes under output_test/
  python specular_eval.py -d output_test --pred_dir base_rgb

  # Vanilla 3DGS comparison
  CUDA_VISIBLE_DEVICES=2 python specular_eval.py -d output_test-3m --pred_dir renders

  # Single scene
  python specular_eval.py -m output_test/cecum_t1_b --pred_dir base_rgb
  
  
  
  # gs_endo/output_test-0327-2의 spec_prior를 사용해서 다른 output 평가 (base_rgb 기준)
CUDA_VISIBLE_DEVICES=2 python ../gs_endo_origin/specular_eval.py \
    -d output_test-0327-2 --pred_dir base_rgb

# renders 기준 (vanilla 3DGS)
CUDA_VISIBLE_DEVICES=2 python specular_eval.py \
    -d output_test-3m --pred_dir renders

# spec_prior 비활성화 (GT 계산으로 폴백)
CUDA_VISIBLE_DEVICES=0 python gs_endo_origin/specular_eval.py \
    -d gs_endo_origin/output_test-3m --pred_dir renders \
    --spec_prior_base_dir none
"""

import os
import sys
import glob
import json
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from argparse import ArgumentParser
from math import exp

try:
    import lpips as lpips_module
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from pytorch_msssim import ms_ssim as compute_ms_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False


# ============================================================================
# 1. Specular Prior (HSV-based weak cue)
# ============================================================================

def compute_specular_prior(rgb, gamma=1.0):
    """
    prior = (1 - S) * V   (bright, low-saturation -> specular-capable)

    Args:
        rgb:   [3, H, W] in [0, 1]
        gamma: sharpening exponent
    Returns:
        [1, H, W] soft map in [0, 1]
    """
    maxc = torch.max(rgb, dim=0).values
    minc = torch.min(rgb, dim=0).values
    s = (maxc - minc) / (maxc + 1e-8)
    v = maxc
    prior = (1.0 - s) * v
    if gamma != 1.0:
        prior = prior.clamp(0, 1).pow(gamma)
    return prior.unsqueeze(0)


# ============================================================================
# 2. Masking
# ============================================================================

def create_specular_mask(spec_prior, threshold=0.5):
    """[1,H,W] binary: 1 = specular, 0 = tissue"""
    return (spec_prior >= threshold).float()


def create_tissue_mask(spec_prior, threshold=0.5):
    """[1,H,W] binary: 1 = tissue, 0 = specular"""
    return (spec_prior < threshold).float()


# ============================================================================
# 3. Masked Metrics (pred vs gt, tissue pixels only)
# ============================================================================

def masked_psnr(pred, gt, tissue_mask):
    """PSNR on tissue pixels only."""
    sq_err = (pred - gt) ** 2
    masked_err = sq_err * tissue_mask
    n_tissue = tissue_mask.sum() * 3
    if n_tissue < 1:
        return torch.tensor(0.0)
    mse = masked_err.sum() / n_tissue
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def _create_ssim_window(window_size, channel, device):
    sigma = 1.5
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)],
        dtype=torch.float32, device=device
    )
    gauss = gauss / gauss.sum()
    window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()


def masked_ssim(pred, gt, tissue_mask, window_size=11):
    """SSIM map averaged over tissue pixels only."""
    channel = pred.shape[0]
    img1 = pred.unsqueeze(0)
    img2 = gt.unsqueeze(0)
    window = _create_ssim_window(window_size, channel, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1 * mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map_avg = ssim_map.mean(dim=1, keepdim=True).squeeze(0)  # [1, H, W]

    n_tissue = tissue_mask.sum()
    if n_tissue < 1:
        return torch.tensor(0.0)
    return (ssim_map_avg * tissue_mask).sum() / n_tissue


def masked_lpips(pred, gt, tissue_mask, lpips_fn):
    """LPIPS with specular pixels neutralised (replaced with GT)."""
    pred_neutral = pred * tissue_mask + gt * (1.0 - tissue_mask)
    return lpips_fn(pred_neutral.unsqueeze(0), gt.unsqueeze(0)).squeeze()


def masked_msssim(pred, gt, tissue_mask):
    """MS-SSIM with specular pixels neutralised."""
    pred_neutral = pred * tissue_mask + gt * (1.0 - tissue_mask)
    return compute_ms_ssim(
        pred_neutral.unsqueeze(0), gt.unsqueeze(0),
        data_range=1.0, size_average=True,
    ).squeeze()


def compute_all_masked_metrics(pred, gt, spec_prior, lpips_fn=None, threshold=0.5):
    """All tissue-only metrics for one image pair."""
    tissue_mask = create_tissue_mask(spec_prior, threshold)
    results = {
        "psnr": masked_psnr(pred, gt, tissue_mask).item(),
        "ssim": masked_ssim(pred, gt, tissue_mask).item(),
        "tissue_ratio": tissue_mask.mean().item(),
    }
    if lpips_fn is not None and LPIPS_AVAILABLE:
        results["lpips"] = masked_lpips(pred, gt, tissue_mask, lpips_fn).item()
    if MSSSIM_AVAILABLE:
        results["msssim"] = masked_msssim(pred, gt, tissue_mask).item()
    return results


# ============================================================================
# 4. Visualisation (gt + pred + spec_prior only)
# ============================================================================

def visualize_specular_detection(save_dir, tag, gt_rgb, pred_rgb, spec_prior, threshold=0.5):
    """
    Save diagnostic images using only gt, pred, spec_prior.

    Outputs:
      01_gt, 02_pred, 03_spec_prior, 04_spec_mask,
      05_gt_overlay (red tint on specular), 06_tissue_only_gt,
      07_tissue_only_pred, 08_error_tissue, 09_comparison (2x4 grid)
    """
    os.makedirs(save_dir, exist_ok=True)
    spec_mask = create_specular_mask(spec_prior, threshold)
    tissue_mask = 1.0 - spec_mask

    def _save(t, name):
        t = t.detach().cpu().clamp(0, 1)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        torchvision.utils.save_image(t, os.path.join(save_dir, name))

    _save(gt_rgb, "%s_01_gt.png" % tag)
    _save(pred_rgb, "%s_02_pred.png" % tag)
    _save(spec_prior.squeeze(0), "%s_03_spec_prior.png" % tag)
    _save(spec_mask.squeeze(0), "%s_04_spec_mask.png" % tag)

    # GT with red tint on specular regions
    gt_overlay = gt_rgb.clone()
    sm = spec_mask.squeeze(0)
    gt_overlay[0] = torch.where(sm > 0.5, torch.ones_like(gt_overlay[0]) * 0.8, gt_overlay[0])
    gt_overlay[1] = torch.where(sm > 0.5, gt_overlay[1] * 0.3, gt_overlay[1])
    gt_overlay[2] = torch.where(sm > 0.5, gt_overlay[2] * 0.3, gt_overlay[2])
    _save(gt_overlay, "%s_05_gt_overlay.png" % tag)

    _save(gt_rgb * tissue_mask, "%s_06_tissue_only_gt.png" % tag)
    _save(pred_rgb * tissue_mask, "%s_07_tissue_only_pred.png" % tag)

    error_vis = ((pred_rgb - gt_rgb).abs() * tissue_mask * 5.0).clamp(0, 1)
    _save(error_vis, "%s_08_error_tissue.png" % tag)

    grid_images = [
        gt_rgb.cpu().clamp(0, 1),
        pred_rgb.cpu().clamp(0, 1),
        spec_prior.squeeze(0).cpu().repeat(3, 1, 1).clamp(0, 1),
        spec_mask.squeeze(0).cpu().repeat(3, 1, 1),
        gt_overlay.cpu().clamp(0, 1),
        (gt_rgb * tissue_mask).cpu().clamp(0, 1),
        (pred_rgb * tissue_mask).cpu().clamp(0, 1),
        error_vis.cpu(),
    ]
    grid = torchvision.utils.make_grid(grid_images, nrow=4, padding=4, pad_value=1.0)
    torchvision.utils.save_image(grid, os.path.join(save_dir, "%s_09_comparison.png" % tag))


# ============================================================================
# 5. Image loading
# ============================================================================

def load_images_from_dir(dirpath):
    """Load all PNG images from directory, sorted by filename."""
    paths = sorted(glob.glob(os.path.join(dirpath, "*.png")))
    images = []
    for p in paths:
        img = torchvision.io.read_image(p).float() / 255.0
        images.append(img)
    return images, paths


def find_iteration(model_path, split):
    """Find the latest iteration in a model's split directory."""
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


# ============================================================================
# 6. Single scene evaluation
# ============================================================================

def load_spec_prior_from_dir(spec_prior_dir, n_images, device, gamma=1.0):
    """Load spec_prior images from folder. Returns list of [1,H,W] tensors."""
    paths = sorted(glob.glob(os.path.join(spec_prior_dir, "*.png")))
    assert len(paths) == n_images, (
        "spec_prior/ has %d images but gt/ has %d" % (len(paths), n_images))
    priors = []
    for p in paths:
        img = torchvision.io.read_image(p).float() / 255.0  # [C, H, W]
        if img.shape[0] == 3:
            prior = img.mean(dim=0, keepdim=True)           # [1, H, W]
        else:
            prior = img[:1]
        if gamma != 1.0:
            prior = prior.clamp(0, 1).pow(gamma)
        priors.append(prior.to(device))
    return priors


SPEC_PRIOR_FIXED_ITERATION = 40000  # spec_prior는 항상 이 iteration에서 로드


def resolve_spec_prior_dir(scene_dir, scene_name, split,
                           spec_prior_base_dir=None):
    """
    Determine which spec_prior/ directory to use.

    Priority:
      1. spec_prior_base_dir/<scene_name>/<split>/ours_40000/spec_prior/
         iteration은 SPEC_PRIOR_FIXED_ITERATION(=40000)으로 고정.
         (external reference dir = /data/jyeon/endo/gs_endo/output_test-0327-2)
      2. <scene_dir>/spec_prior/  (same output folder as pred)
      3. None  -> fall back to computing from GT
    """
    if spec_prior_base_dir is not None:
        ref_dir = os.path.join(
            spec_prior_base_dir, scene_name, split,
            "ours_%d" % SPEC_PRIOR_FIXED_ITERATION, "spec_prior"
        )
        if os.path.isdir(ref_dir):
            return ref_dir, "external"
        else:
            print("[SpecularEval] WARNING: external spec_prior not found: %s" % ref_dir)

    local_dir = os.path.join(scene_dir, "spec_prior")
    if os.path.isdir(local_dir):
        return local_dir, "local"

    return None, "computed_from_gt"


def evaluate_directory(model_path, split, iteration, pred_dir="base_rgb",
                       spec_prior_base_dir=None,
                       threshold=0.5, gamma=1.0, device="cuda"):
    """
    Evaluate one scene. Uses gt/, <pred_dir>/, and spec_prior/.

    Args:
        pred_dir:             name of prediction folder (e.g. "base_rgb" or "renders")
        spec_prior_base_dir:  root dir containing reference spec_prior results, e.g.
                              /data/jyeon/endo/gs_endo/output_test-0327-2
                              Expected layout:
                                <spec_prior_base_dir>/<scene_name>/<split>/ours_<iter>/spec_prior/
                              If None, falls back to local spec_prior/ or GT computation.
    """
    scene_name = os.path.basename(model_path)
    scene_dir = os.path.join(model_path, split, "ours_%d" % iteration)
    gt_dir = os.path.join(scene_dir, "gt")
    pred_dir_path = os.path.join(scene_dir, pred_dir)

    if not os.path.isdir(gt_dir):
        print("[SpecularEval] ERROR: gt/ not found: %s" % gt_dir)
        return None
    if not os.path.isdir(pred_dir_path):
        print("[SpecularEval] ERROR: %s/ not found: %s" % (pred_dir, pred_dir_path))
        return None

    eval_dir = os.path.join(scene_dir, "spec_eval")
    vis_dir = os.path.join(eval_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    gts, _ = load_images_from_dir(gt_dir)
    preds, _ = load_images_from_dir(pred_dir_path)
    n_images = len(gts)
    assert len(preds) == n_images, "Mismatch: %d gt vs %d pred" % (n_images, len(preds))

    spec_prior_dir, prior_source = resolve_spec_prior_dir(
        scene_dir, scene_name, split, spec_prior_base_dir)

    if spec_prior_dir is not None:
        print("[SpecularEval] spec_prior source [%s]: %s" % (prior_source, spec_prior_dir))
        spec_priors = load_spec_prior_from_dir(spec_prior_dir, n_images, device, gamma)
    else:
        print("[SpecularEval] spec_prior/ not found, computing from GT")

    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips_module.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    metrics = {"psnr": [], "ssim": [], "lpips": [], "msssim": [], "tissue_ratio": []}

    for idx in range(n_images):
        gt = gts[idx].to(device)
        pred = preds[idx].to(device)
        spec_prior = spec_priors[idx] if spec_prior_dir else compute_specular_prior(gt, gamma=gamma)

        result = compute_all_masked_metrics(pred, gt, spec_prior, lpips_fn, threshold)
        for k in result:
            metrics[k].append(result[k])

        visualize_specular_detection(vis_dir, "view_%05d" % idx, gt, pred, spec_prior, threshold)

    def _stats(vals):
        if not vals:
            return 0.0, 0.0
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std())

    json_data = {
        "model_path": model_path,
        "split": split,
        "iteration": iteration,
        "pred_dir": pred_dir,
        "spec_prior_source": prior_source,
        "spec_prior_dir": spec_prior_dir or "computed_from_gt",
        "threshold": threshold,
        "gamma": gamma,
        "n_images": n_images,
        "tissue_only": {},
    }
    for k in ["psnr", "ssim", "lpips", "msssim", "tissue_ratio"]:
        if metrics[k]:
            m, s = _stats(metrics[k])
            json_data["tissue_only"][k] = {"mean": m, "std": s}

    json_path = os.path.join(eval_dir, "tissue_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print("[SpecularEval] %s: PSNR=%.2f  SSIM=%.4f  tissue_ratio=%.3f" % (
        model_path,
        json_data["tissue_only"].get("psnr", {}).get("mean", 0),
        json_data["tissue_only"].get("ssim", {}).get("mean", 0),
        json_data["tissue_only"].get("tissue_ratio", {}).get("mean", 0),
    ))
    return json_data


# ============================================================================
# 7. Batch evaluation: all scenes under a base directory
# ============================================================================

def evaluate_all(base_dir, split="test", iteration=-1, pred_dir="base_rgb",
                 spec_prior_base_dir=None,
                 threshold=0.5, gamma=1.0, device="cuda"):
    """
    Evaluate ALL scene subdirectories under base_dir.

    Args:
        base_dir:             root dir with scene subdirectories to evaluate
        pred_dir:             prediction folder name ("base_rgb" or "renders")
        spec_prior_base_dir:  root dir containing reference spec_prior results, e.g.
                              /data/jyeon/endo/gs_endo/output_test-0327-2
                              Expected layout:
                                <spec_prior_base_dir>/<scene_name>/<split>/ours_<iter>/spec_prior/
                              If None, falls back to local spec_prior/ or GT computation.

    Output:
      base_dir/spec_eval_summary.txt
      base_dir/spec_eval_summary.json
    """
    scenes = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.isdir(os.path.join(base_dir, d, split))
    ])
    if not scenes:
        print("[SpecularEval] No scenes found in %s" % base_dir)
        return

    print("=" * 70)
    print("[SpecularEval] Batch: %d scenes | pred_dir=%s | threshold=%.2f" % (
        len(scenes), pred_dir, threshold))
    if spec_prior_base_dir:
        print("[SpecularEval] spec_prior_base_dir: %s" % spec_prior_base_dir)
    else:
        print("[SpecularEval] spec_prior_base_dir: None (using local or GT fallback)")
    print("=" * 70)

    # Load LPIPS once
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips_module.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    all_scene_results = {}
    overall = {"psnr": [], "ssim": [], "lpips": [], "msssim": [], "tissue_ratio": []}

    for si, scene_name in enumerate(scenes):
        model_path = os.path.join(base_dir, scene_name)
        it = iteration
        if it == -1:
            it = find_iteration(model_path, split)
            if it is None:
                print("  SKIP %s: no ours_* found" % scene_name)
                continue

        scene_dir = os.path.join(model_path, split, "ours_%d" % it)
        gt_dir = os.path.join(scene_dir, "gt")
        pred_dir_path = os.path.join(scene_dir, pred_dir)

        if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir_path):
            print("  SKIP %s: missing gt/ or %s/" % (scene_name, pred_dir))
            continue

        eval_dir = os.path.join(scene_dir, "spec_eval")
        vis_dir = os.path.join(eval_dir, "visualisations")
        os.makedirs(vis_dir, exist_ok=True)

        spec_prior_dir, prior_source = resolve_spec_prior_dir(
            scene_dir, scene_name, split, spec_prior_base_dir)

        print("\n[%d/%d] %s (iter=%d) | spec_prior=[%s] %s" % (
            si + 1, len(scenes), scene_name, it,
            prior_source, spec_prior_dir or ""))

        gts, _ = load_images_from_dir(gt_dir)
        preds, _ = load_images_from_dir(pred_dir_path)
        n_images = len(gts)
        assert len(preds) == n_images, "Mismatch in %s" % scene_name

        if spec_prior_dir is not None:
            scene_spec_priors = load_spec_prior_from_dir(spec_prior_dir, n_images, device, gamma)

        scene_metrics = {"psnr": [], "ssim": [], "lpips": [], "msssim": [], "tissue_ratio": []}

        for idx in range(n_images):
            gt = gts[idx].to(device)
            pred = preds[idx].to(device)
            spec_prior = scene_spec_priors[idx] if spec_prior_dir else compute_specular_prior(gt, gamma=gamma)

            result = compute_all_masked_metrics(pred, gt, spec_prior, lpips_fn, threshold)
            for k in result:
                scene_metrics[k].append(result[k])

            visualize_specular_detection(
                vis_dir, "view_%05d" % idx, gt, pred, spec_prior, threshold)

        # Scene average
        def _stats(vals):
            if not vals:
                return 0.0, 0.0
            arr = np.array(vals)
            return float(arr.mean()), float(arr.std())

        scene_result = {"scene": scene_name, "iteration": it, "n_images": n_images,
                        "spec_prior_source": prior_source,
                        "spec_prior_dir": spec_prior_dir or "computed_from_gt",
                        "tissue_only": {}}
        for k in ["psnr", "ssim", "lpips", "msssim", "tissue_ratio"]:
            if scene_metrics[k]:
                m, s = _stats(scene_metrics[k])
                scene_result["tissue_only"][k] = {"mean": m, "std": s}

        all_scene_results[scene_name] = scene_result

        # Save per-scene JSON
        with open(os.path.join(eval_dir, "tissue_metrics.json"), "w") as f:
            json.dump(scene_result, f, indent=2)

        # Accumulate per-image values for overall mean
        for k in ["psnr", "ssim", "lpips", "msssim", "tissue_ratio"]:
            overall[k].extend(scene_metrics[k])

        # Print scene summary
        pm = scene_result["tissue_only"].get("psnr", {}).get("mean", 0)
        sm = scene_result["tissue_only"].get("ssim", {}).get("mean", 0)
        tr = scene_result["tissue_only"].get("tissue_ratio", {}).get("mean", 0)
        print("  -> n=%d | Tissue PSNR=%.2f  SSIM=%.4f  ratio=%.3f" % (n_images, pm, sm, tr))

    # ========================================================================
    # Overall summary
    # ========================================================================
    def _stats(vals):
        if not vals:
            return 0.0, 0.0
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std())

    print("\n" + "=" * 70)
    print("[SpecularEval] OVERALL (%d scenes, pred_dir=%s, threshold=%.2f)" % (
        len(all_scene_results), pred_dir, threshold))
    print("=" * 70)

    # --- Build text output ---
    lines = []
    lines.append("Specular Evaluation - Tissue-Only Metrics")
    lines.append("base_dir: %s" % base_dir)
    lines.append("pred_dir: %s  |  split: %s  |  threshold: %.2f  |  gamma: %.1f" % (
        pred_dir, split, threshold, gamma))
    lines.append("scenes: %s" % ", ".join(sorted(all_scene_results.keys())))
    lines.append("total images: %d" % len(overall["psnr"]))
    lines.append("")

    # Per-scene table
    lines.append("=" * 70)
    lines.append("PER-SCENE RESULTS (tissue-only)")
    lines.append("=" * 70)
    metric_keys = ["psnr", "ssim"]
    if overall["lpips"]:
        metric_keys.append("lpips")
    if overall["msssim"]:
        metric_keys.append("msssim")

    table_header = "%-18s  %5s  %8s" % ("Scene", "N", "T-RATIO")
    for mk in metric_keys:
        table_header += "  %10s" % mk.upper()
    lines.append(table_header)
    lines.append("-" * len(table_header))

    for sn in sorted(all_scene_results.keys()):
        sr = all_scene_results[sn]
        row = "%-18s  %5d  %8.4f" % (
            sn, sr["n_images"],
            sr["tissue_only"].get("tissue_ratio", {}).get("mean", 0))
        for mk in metric_keys:
            row += "  %10.4f" % sr["tissue_only"].get(mk, {}).get("mean", 0)
        lines.append(row)

    # Overall mean row
    lines.append("-" * len(table_header))
    row = "%-18s  %5d  %8.4f" % ("OVERALL", len(overall["psnr"]), _stats(overall["tissue_ratio"])[0])
    for mk in metric_keys:
        row += "  %10.4f" % _stats(overall[mk])[0]
    lines.append(row)

    lines.append("")
    lines.append("=" * 70)
    lines.append("OVERALL MEAN +/- STD")
    lines.append("=" * 70)
    for k in metric_keys + ["tissue_ratio"]:
        if overall[k]:
            m, s = _stats(overall[k])
            lines.append("%-15s  %10.4f  +/- %10.4f" % (k.upper(), m, s))

    # Print to console
    print("")
    for line in lines[-len(metric_keys) - 4:]:
        print("  %s" % line)
    print("=" * 70)

    # Save text
    summary_path = os.path.join(base_dir, "spec_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[SpecularEval] Summary -> %s" % summary_path)

    # Save JSON
    summary_json = {
        "base_dir": base_dir,
        "pred_dir": pred_dir,
        "spec_prior_base_dir": spec_prior_base_dir or "local_or_computed_from_gt",
        "split": split,
        "threshold": threshold,
        "gamma": gamma,
        "n_scenes": len(all_scene_results),
        "total_images": len(overall["psnr"]),
        "per_scene": all_scene_results,
        "overall": {},
    }
    for k in metric_keys + ["tissue_ratio"]:
        if overall[k]:
            m, s = _stats(overall[k])
            summary_json["overall"][k] = {"mean": m, "std": s}

    with open(os.path.join(base_dir, "spec_eval_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    return summary_json


# ============================================================================
# 8. CLI entry point
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # DEFAULT SPEC_PRIOR_BASE_DIR                                         #
    # Change this path to point to the output directory whose spec_prior/ #
    # folders should be used as the reference mask.                       #
    # Set to None to fall back to local spec_prior/ or GT computation.   #
    # ------------------------------------------------------------------ #
    DEFAULT_SPEC_PRIOR_BASE_DIR = "/data/jyeon/endo/gs_endo/output_test-0327-2"

    parser = ArgumentParser(description="Tissue-only specular masking evaluation")
    parser.add_argument("-m", "--model_path", default=None, type=str,
                        help="Single scene model path")
    parser.add_argument("-d", "--base_dir", default=None, type=str,
                        help="Directory containing multiple scenes (batch mode)")
    parser.add_argument("--pred_dir", default="base_rgb", type=str,
                        help="Name of prediction image folder (default: base_rgb, use 'renders' for vanilla 3DGS)")
    parser.add_argument("--spec_prior_base_dir", default=DEFAULT_SPEC_PRIOR_BASE_DIR, type=str,
                        help=(
                            "Root directory containing reference spec_prior results "
                            "(default: %s). "
                            "Expected layout: <spec_prior_base_dir>/<scene>/<split>/ours_<iter>/spec_prior/. "
                            "Pass 'none' to disable and use local spec_prior/ or GT computation."
                            % DEFAULT_SPEC_PRIOR_BASE_DIR
                        ))
    parser.add_argument("--iteration", default=-1, type=int,
                        help="Iteration (-1 = latest)")
    parser.add_argument("--split", default="test", type=str, choices=["train", "test"])
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Spec prior threshold")
    parser.add_argument("--gamma", default=1.0, type=float,
                        help="Spec prior gamma")
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    spec_prior_base_dir = (
        None if args.spec_prior_base_dir.lower() == "none"
        else args.spec_prior_base_dir
    )

    if args.base_dir is not None:
        evaluate_all(
            base_dir=args.base_dir,
            split=args.split,
            iteration=args.iteration,
            pred_dir=args.pred_dir,
            spec_prior_base_dir=spec_prior_base_dir,
            threshold=args.threshold,
            gamma=args.gamma,
            device=args.device,
        )
    elif args.model_path is not None:
        iteration = args.iteration
        if iteration == -1:
            iteration = find_iteration(args.model_path, args.split)
            if iteration is None:
                print("[SpecularEval] ERROR: no ours_* found")
                sys.exit(1)
        evaluate_directory(
            model_path=args.model_path,
            split=args.split,
            iteration=iteration,
            pred_dir=args.pred_dir,
            spec_prior_base_dir=spec_prior_base_dir,
            threshold=args.threshold,
            gamma=args.gamma,
            device=args.device,
        )
    else:
        parser.print_help()
        sys.exit(1)
