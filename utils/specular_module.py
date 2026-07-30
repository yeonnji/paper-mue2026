"""
Illumination-Dependent Specular Modeling for Endoscopic 3DGS.

Components:
  1. compute_specular_prior  - HSV-based weak prior  (1-S)*V  (GT-only, no base_rgb dependency)
  2. PoseEmbedding           - camera-pose-based latent z_t via MLP
  3. SpecularBranch           - unified spec_map + residual head → spec_rgb

Design philosophy:
  - Specular highlights are NOT artifacts to suppress.
  - We model them as an illumination-dependent additive component.
  - final_rgb = base_rgb + spec_rgb
  - The endoscope rig prior justifies a low-dimensional pose-based latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# 1. HSV weak prior  (GT-only — no circular dependency with base_rgb)
# ---------------------------------------------------------------------------

def _rgb_to_hsv_3d(rgb: torch.Tensor) -> torch.Tensor:
    """[3,H,W] -> [3,H,W]  (H in [0,1], S in [0,1], V in [0,1])."""
    maxc = torch.max(rgb, dim=0).values
    minc = torch.min(rgb, dim=0).values
    v = maxc
    s = (maxc - minc) / (maxc + 1e-8)
    h = torch.zeros_like(v)
    return torch.stack([h, s, v], dim=0)


def compute_specular_prior(
    rgb: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Compute specular prior from GT image only.

    prior = (1 - S) * V   (bright, low-saturation → high)

    This is a weak spatial cue for specular-capable regions.
    It does NOT depend on base_rgb, avoiding the circular dependency
    where a well-fitted base collapses the prior to zero.

    Args:
        rgb:   [3, H, W] GT image tensor in [0, 1]
        gamma: optional gamma for prior sharpening (>1 sharpens, <1 softens)

    Returns:
        prior: [1, H, W] soft map in [0, 1]
    """
    hsv = _rgb_to_hsv_3d(rgb)
    s = hsv[1]  # saturation
    v = hsv[2]  # value
    prior = (1.0 - s) * v  # bright, low-saturation → high

    if gamma != 1.0:
        prior = prior.clamp(0, 1).pow(gamma)

    return prior.unsqueeze(0)  # [1, H, W]


def compute_residual_specular_prior(
    gt_image: torch.Tensor,
    base_rgb: torch.Tensor,
    hsv_prior: torch.Tensor,
) -> torch.Tensor:
    """
    Residual-weighted specular prior.

    Combines the positive reconstruction residual (GT - base) with the
    HSV-based spatial cue to focus on regions where the Gaussian actually
    under-reconstructs AND that look specular-capable.

    Args:
        gt_image:  [3, H, W] GT image
        base_rgb:  [3, H, W] Gaussian rendering (must be detached by caller)
        hsv_prior: [1, H, W] output of compute_specular_prior

    Returns:
        prior: [1, H, W] in [0, 1]
    """
    residual_pos = (gt_image - base_rgb).clamp(min=0)       # [3, H, W]
    residual_mag = residual_pos.mean(dim=0, keepdim=True)   # [1, H, W]
    prior = residual_mag * hsv_prior                         # [1, H, W]
    prior = prior / (prior.max() + 1e-8)                     # normalize to [0, 1]
    return prior


# ---------------------------------------------------------------------------
# 2. Pose-based light latent (replaces frame-index embedding)
# ---------------------------------------------------------------------------

class PoseEmbedding(nn.Module):
    """
    Camera-pose-based latent vector z_t via MLP.

    In endoscopy, the camera and light source are mounted on the same rig,
    so camera pose directly determines illumination state. This replaces
    frame-index-based embedding with a physically motivated pose mapping.

    Input:  6D pose vector = [camera_position (3), viewing_direction (3)]
    Output: z_t [1, D]

    Normalization statistics are stored as buffers so they persist in checkpoints.
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Normalization buffers — saved/loaded with state_dict
        self.register_buffer('pos_mean', torch.zeros(3))
        self.register_buffer('pos_std', torch.ones(3))
        self.register_buffer('dir_mean', torch.zeros(3))
        self.register_buffer('dir_std', torch.ones(3))

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Last layer: small output so z_t starts near zero
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def set_normalization_stats(self, cameras):
        """
        Compute and store normalization statistics from training cameras.
        Must be called once before training starts.

        Args:
            cameras: list of Camera objects with .R and .T attributes (numpy arrays)
        """
        positions = []
        directions = []
        for cam in cameras:
            R = np.array(cam.R).reshape(3, 3)
            T = np.array(cam.T).flatten()        # ensure (3,) even if stored as (1,3)
            pos = -R.T @ T                       # world-space camera position
            d = R.T @ np.array([0.0, 0.0, 1.0])  # viewing direction in world
            positions.append(pos)
            directions.append(d)

        positions = torch.tensor(np.stack(positions), dtype=torch.float32)
        directions = torch.tensor(np.stack(directions), dtype=torch.float32)

        self.pos_mean.copy_(positions.mean(dim=0))
        self.pos_std.copy_(positions.std(dim=0).clamp(min=1e-6))
        self.dir_mean.copy_(directions.mean(dim=0))
        self.dir_std.copy_(directions.std(dim=0).clamp(min=1e-6))

    def _extract_pose_input(self, R, T) -> torch.Tensor:
        """
        Extract normalized 6D pose vector from camera R, T.

        Args:
            R: [3,3] rotation matrix (numpy or tensor)
            T: [3] translation vector (numpy or tensor)

        Returns:
            [1, 6] normalized pose input tensor on same device as buffers
        """
        R_np = np.array(R).reshape(3, 3)
        T_np = np.array(T).flatten()  # ensure (3,) even if stored as (1,3)

        pos = torch.tensor(-R_np.T @ T_np, dtype=torch.float32, device=self.pos_mean.device)
        d = torch.tensor(R_np.T @ np.array([0.0, 0.0, 1.0]), dtype=torch.float32, device=self.pos_mean.device)

        pos_norm = (pos - self.pos_mean) / self.pos_std
        dir_norm = (d - self.dir_mean) / self.dir_std

        return torch.cat([pos_norm, dir_norm]).unsqueeze(0)  # [1, 6]

    def forward(self, R, T) -> torch.Tensor:
        """
        Compute latent z_t from camera pose.

        Args:
            R: [3,3] rotation matrix
            T: [3] translation vector

        Returns:
            z_t: [1, D]
        """
        pose_input = self._extract_pose_input(R, T)
        return self.mlp(pose_input)

    @property
    def latent_dim(self):
        return self.mlp[-1].out_features


# ---------------------------------------------------------------------------
# 3. Specular Branch  (spec_map head + residual head, unified)
# ---------------------------------------------------------------------------

class SpecularBranch(nn.Module):
    """
    Image-space specular branch with FiLM-conditioned latent.

    Pipeline:
        shared_feat = encoder(base_rgb)                          → [B, H_dim, H, W]
        gamma, beta = FiLM(light_latent)                         → [B, H_dim]
        modulated   = gamma * shared_feat + beta                 → [B, H_dim, H, W]
        spec_map    = sigmoid(map_head(modulated))               → [B, 1, H, W]
        residual    = tanh(res_head(modulated)) * gain           → [B, 3, H, W]
        spec_rgb    = spec_map * residual                        → [B, 3, H, W]
        final_rgb   = base_rgb + spec_rgb

    Note: spec_prior is NOT used as network input (to avoid GT dependency at eval).
    It is only used as a loss signal during training.
    """

    def __init__(self, latent_dim: int = 8, hidden_dim: int = 32, gain: float = 0.5):
        super().__init__()
        self.gain = gain

        # Shared encoder: base_rgb → spatial features
        # dilation 1→2→4→8 gives effective receptive field ~33px
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )

        # FiLM: latent → per-channel gamma, beta
        self.film = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        # Lightweight heads on modulated features
        self.map_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

        self.res_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Bias the map head's final layer so sigmoid starts near 0.12
        nn.init.constant_(self.map_head[-1].bias, -2.0)
        # Init FiLM output so gamma≈1, beta≈0 (identity modulation at start)
        nn.init.zeros_(self.film[-1].weight)
        with torch.no_grad():
            bias = self.film[-1].bias
            hd = bias.shape[0] // 2
            bias[:hd].fill_(1.0)   # gamma init = 1
            bias[hd:].fill_(0.0)   # beta init = 0

    def forward(
        self,
        base_rgb: torch.Tensor,
        light_latent: torch.Tensor,
    ):
        """
        Args:
            base_rgb:      [1, 3, H, W]  (or [3, H, W], will be unsqueezed)
            light_latent:  [1, D]

        Returns:
            spec_rgb:  [1, 3, H, W]
            spec_map:  [1, 1, H, W]
            residual:  [1, 3, H, W]
        """
        if base_rgb.dim() == 3:
            base_rgb = base_rgb.unsqueeze(0)

        # Shared spatial features from base_rgb
        feat = self.encoder(base_rgb)  # [B, H_dim, H, W]

        # FiLM modulation: latent controls how spatial features are scaled/shifted
        film_params = self.film(light_latent)  # [B, H_dim*2]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [B, H_dim]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, H_dim, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        feat = gamma * feat + beta

        spec_map = torch.sigmoid(self.map_head(feat))
        residual = torch.tanh(self.res_head(feat)) * self.gain
        spec_rgb = spec_map * residual

        return spec_rgb, spec_map, residual
