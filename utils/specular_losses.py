"""
Loss functions for the specular branch.

Key losses:
  - L_res:    direct residual-target loss (spec_rgb vs GT-base residual)
  - L_prior:  weak alignment of spec_map with HSV prior
  - L_sparse: spec_map mean penalty
  - L_tv:     spec_map total variation
  - L_z:      latent L2 regularisation
"""

import torch
import torch.nn.functional as F


def residual_target_loss(
    spec_rgb: torch.Tensor,
    base_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    spec_prior: torch.Tensor,
    prior_alpha: float = 2.0,
) -> torch.Tensor:
    """
    Direct residual supervision: spec_rgb should approximate GT - base_rgb,
    weighted more heavily in specular-prone regions.

    weight_map = 1 + alpha * spec_prior
    L_res = mean(weight_map * |spec_rgb - (gt - base_rgb)|)

    Args:
        spec_rgb:   [3, H, W] or [B, 3, H, W]
        base_rgb:   [3, H, W] (may be non-detached for coupled optimization)
        gt_image:   [3, H, W]
        spec_prior: [1, H, W]
        prior_alpha: emphasis on specular regions (0 = uniform)
    Returns:
        scalar loss
    """
    residual_gt = gt_image - base_rgb  # [3, H, W] — gradient flows to base if non-detached
    weight_map = 1.0 + prior_alpha * spec_prior  # [1, H, W] broadcasts over 3ch
    weighted_err = weight_map * torch.abs(spec_rgb - residual_gt)
    return weighted_err.mean()


def prior_guidance_loss(spec_map: torch.Tensor, spec_prior: torch.Tensor) -> torch.Tensor:
    """
    Weak alignment between learned spec_map and HSV prior.
    L_prior = |spec_map - stopgrad(spec_prior)|_1
    """
    return F.l1_loss(spec_map, spec_prior.detach())


def sparsity_loss(spec_map: torch.Tensor) -> torch.Tensor:
    """L_sparse = mean(spec_map)"""
    return spec_map.mean()


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """L_tv = mean(|∇x_h|) + mean(|∇x_w|)"""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return diff_h + diff_w


def latent_reg_loss(light_latent: torch.Tensor) -> torch.Tensor:
    """L_z = ||z_t||_2^2"""
    return (light_latent ** 2).mean()


def final_rgb_loss(
    final_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """
    Combined L1 + SSIM loss on final_rgb (base + spec) vs GT.
    Provides direct reconstruction gradient to the specular branch.
    """
    l1 = F.l1_loss(final_rgb, gt_image)
    # SSIM on [B,C,H,W]
    fr = final_rgb.unsqueeze(0) if final_rgb.dim() == 3 else final_rgb
    gt = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
    ssim_val = 1.0 - _ssim(fr, gt)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * ssim_val


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple SSIM (single-scale) for loss computation."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ch = img1.shape[1]
    # Gaussian window
    gauss = torch.arange(window_size, dtype=img1.dtype, device=img1.device).float()
    gauss = gauss - window_size // 2
    gauss = torch.exp(-gauss.pow(2) / (2 * 1.5 ** 2))
    gauss = gauss / gauss.sum()
    win_1d = gauss.unsqueeze(0).unsqueeze(0)  # [1,1,K]
    win_2d = (win_1d.unsqueeze(-1) @ win_1d.unsqueeze(-2))  # [1,1,K,K]
    win_2d = win_2d.expand(ch, 1, -1, -1)

    pad = window_size // 2
    mu1 = F.conv2d(img1, win_2d, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, win_2d, padding=pad, groups=ch)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, win_2d, padding=pad, groups=ch) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, win_2d, padding=pad, groups=ch) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, win_2d, padding=pad, groups=ch) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_specular_losses(
    spec_rgb: torch.Tensor,
    spec_map: torch.Tensor,
    spec_prior: torch.Tensor,
    base_rgb: torch.Tensor,
    gt_image: torch.Tensor,
    light_latent: torch.Tensor,
    lambda_spec_res: float = 1.0,
    spec_res_prior_alpha: float = 2.0,
    lambda_prior: float = 0.01,
    lambda_sparse: float = 0.005,
    lambda_tv: float = 0.005,
    lambda_z: float = 1e-4,
    lambda_final: float = 1.0,
    lambda_dssim: float = 0.2,
):
    """
    Compute all specular losses and return (total, dict).

    Args:
        base_rgb: may be non-detached for coupled Gaussian+spec optimization.

    Returns:
        spec_loss_total: scalar tensor
        loss_dict:       dict of individual scalar values for logging
    """
    l_res = residual_target_loss(spec_rgb, base_rgb, gt_image, spec_prior, spec_res_prior_alpha)
    l_prior = prior_guidance_loss(spec_map, spec_prior)
    l_sparse = sparsity_loss(spec_map)
    l_tv = total_variation_loss(spec_map)
    l_z = latent_reg_loss(light_latent)

    # Direct reconstruction loss on final_rgb
    final_rgb = (base_rgb + spec_rgb).clamp(0, 1)
    l_final = final_rgb_loss(final_rgb, gt_image, lambda_dssim)

    spec_loss_total = (
        lambda_spec_res * l_res
        + lambda_prior * l_prior
        + lambda_sparse * l_sparse
        + lambda_tv * l_tv
        + lambda_z * l_z
        + lambda_final * l_final
    )

    loss_dict = {
        "spec/res_target": l_res.item(),
        "spec/prior": l_prior.item(),
        "spec/sparse": l_sparse.item(),
        "spec/tv": l_tv.item(),
        "spec/latent_reg": l_z.item(),
        "spec/final_rgb": l_final.item(),
        "spec/total": spec_loss_total.item(),
    }

    return spec_loss_total, loss_dict
