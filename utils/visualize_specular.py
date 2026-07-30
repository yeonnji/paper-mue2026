"""
Specular branch visualisation utilities.

Saves side-by-side diagnostic images and logs scalar stats to tensorboard.
"""

import os
import torch
import torchvision


def save_specular_debug_images(
    save_dir: str,
    iteration: int,
    gt_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    spec_prior: torch.Tensor,
    base_mask_prior: torch.Tensor,
    spec_map: torch.Tensor,
    spec_rgb: torch.Tensor,
    final_rgb: torch.Tensor,
):
    """
    Save diagnostic images for a single training sample.

    All tensors are expected in [C, H, W] or [1, H, W], range [0,1] or signed.
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = f"iter_{iteration:06d}"

    def _save(tensor, name, normalize=False):
        t = tensor.detach().cpu().clamp(0, 1) if not normalize else tensor.detach().cpu()
        if normalize:
            # map [-max, max] -> [0, 1] for signed images
            absmax = t.abs().max().clamp(min=1e-6)
            t = (t / absmax) * 0.5 + 0.5
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        torchvision.utils.save_image(t, os.path.join(save_dir, f"{prefix}_{name}.png"))

    _save(gt_rgb, "gt")
    _save(base_rgb, "base_rgb")
    _save(base_mask_prior.squeeze(0) if base_mask_prior.dim() == 3 else base_mask_prior, "base_mask_prior")
    _save(spec_prior.squeeze(0) if spec_prior.dim() == 3 else spec_prior, "spec_prior")
    _save(spec_map.squeeze(0) if spec_map.dim() == 3 else spec_map, "spec_map")
    _save(spec_rgb, "spec_rgb", normalize=True)
    _save(final_rgb, "final_rgb")

    # Difference maps
    diff_gt_base = gt_rgb - base_rgb
    diff_gt_final = gt_rgb - final_rgb
    spec_contribution = final_rgb - base_rgb

    _save(diff_gt_base, "diff_gt_minus_base", normalize=True)
    _save(diff_gt_final, "diff_gt_minus_final", normalize=True)
    _save(spec_contribution, "spec_contribution", normalize=True)


def log_specular_scalars(
    tb_writer,
    iteration: int,
    loss_dict: dict,
    spec_map: torch.Tensor,
    spec_rgb: torch.Tensor,
    light_latent: torch.Tensor,
):
    """Log specular-related scalars to tensorboard."""
    if tb_writer is None:
        return

    for key, val in loss_dict.items():
        tb_writer.add_scalar(f"specular_loss/{key}", val, iteration)

    with torch.no_grad():
        tb_writer.add_scalar("specular_stats/spec_map_mean", spec_map.mean().item(), iteration)
        tb_writer.add_scalar("specular_stats/spec_map_max", spec_map.max().item(), iteration)
        tb_writer.add_scalar("specular_stats/spec_rgb_mean", spec_rgb.abs().mean().item(), iteration)
        tb_writer.add_scalar("specular_stats/spec_rgb_max", spec_rgb.abs().max().item(), iteration)
        tb_writer.add_scalar("specular_stats/latent_norm", light_latent.norm().item(), iteration)


def log_specular_images(
    tb_writer,
    iteration: int,
    gt_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    spec_prior: torch.Tensor,
    base_mask_prior: torch.Tensor,
    spec_map: torch.Tensor,
    spec_rgb: torch.Tensor,
    final_rgb: torch.Tensor,
):
    """Log specular images to tensorboard (called less frequently)."""
    if tb_writer is None:
        return

    def _to_3ch(t):
        t = t.detach().cpu()
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    def _norm_signed(t):
        t = t.detach().cpu()
        absmax = t.abs().max().clamp(min=1e-6)
        return (t / absmax) * 0.5 + 0.5

    tb_writer.add_image("specular/gt", gt_rgb.clamp(0, 1).cpu(), iteration)
    tb_writer.add_image("specular/base_rgb", base_rgb.clamp(0, 1).cpu(), iteration)
    tb_writer.add_image("specular/final_rgb", final_rgb.clamp(0, 1).cpu(), iteration)
    tb_writer.add_image("specular/base_mask_prior", _to_3ch(base_mask_prior.squeeze(0)), iteration)
    tb_writer.add_image("specular/spec_prior", _to_3ch(spec_prior.squeeze(0)), iteration)
    tb_writer.add_image("specular/spec_map", _to_3ch(spec_map.squeeze(0)), iteration)
    tb_writer.add_image("specular/spec_rgb", _norm_signed(spec_rgb), iteration)
    tb_writer.add_image("specular/diff_gt_base", _norm_signed(gt_rgb - base_rgb), iteration)
    tb_writer.add_image("specular/diff_gt_final", _norm_signed(gt_rgb - final_rgb), iteration)
