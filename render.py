#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, SpecularParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.specular_module import compute_specular_prior, compute_residual_specular_prior, PoseEmbedding, SpecularBranch
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,
               train_test_exp, separate_sh, spec_branch=None, pose_embed=None,
               spec_args=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    use_spec = spec_branch is not None and pose_embed is not None
    if use_spec:
        base_path = os.path.join(model_path, name, "ours_{}".format(iteration), "base_rgb")
        # spec_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "spec_map")
        # spec_rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "spec_rgb")
        spec_prior_path = os.path.join(model_path, name, "ours_{}".format(iteration), "spec_prior")
        for p in [base_path, spec_prior_path]:
            makedirs(p, exist_ok=True)
        print(f"[Specular] Rendering with pose-based latent")

    views = sorted(views, key=lambda obj: obj.colmap_id)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        base_rgb = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if use_spec:
            z_t = pose_embed(view.R, view.T)
            s_rgb, s_map, _ = spec_branch(base_rgb, z_t)
            s_rgb = s_rgb.squeeze(0)
            s_map = s_map.squeeze(0)
            rendering = (base_rgb + s_rgb).clamp(0, 1)
        else:
            rendering = base_rgb

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if use_spec:
            torchvision.utils.save_image(base_rgb.clamp(0, 1), os.path.join(base_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(s_map.repeat(3, 1, 1), os.path.join(spec_map_path, '{0:05d}'.format(idx) + ".png"))
            # s_vis = s_rgb.detach()
            # absmax = s_vis.abs().max().clamp(min=1e-6)
            # s_vis = (s_vis / absmax) * 0.5 + 0.5
            # torchvision.utils.save_image(s_vis, os.path.join(spec_rgb_path, '{0:05d}'.format(idx) + ".png"))
            # Residual-weighted spec_prior (same as training)
            gamma = spec_args.spec_prior_gamma if spec_args else 1.0
            hsv_prior = compute_specular_prior(gt, gamma=gamma)
            sp = compute_residual_specular_prior(gt, base_rgb.detach(), hsv_prior)
            torchvision.utils.save_image(sp.repeat(3, 1, 1), os.path.join(spec_prior_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, separate_sh: bool, spec_args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Load specular branch if available
        spec_branch = None
        pose_embed = None

        if spec_args is not None and spec_args.use_specular_branch:
            loaded_iter = scene.loaded_iter
            spec_ckpt_path = os.path.join(dataset.model_path, f"spec_chkpnt{loaded_iter}.pth")
            if os.path.exists(spec_ckpt_path):
                pose_embed = PoseEmbedding(spec_args.spec_latent_dim).cuda()
                spec_branch = SpecularBranch(
                    latent_dim=spec_args.spec_latent_dim,
                    hidden_dim=spec_args.spec_hidden_dim,
                    gain=spec_args.spec_gain,
                ).cuda()
                ckpt = torch.load(spec_ckpt_path)
                spec_branch.load_state_dict(ckpt["spec_branch"])
                pose_embed.load_state_dict(ckpt["pose_embed"])
                spec_branch.eval()
                pose_embed.eval()
                print(f"[Specular] Loaded checkpoint from {spec_ckpt_path}")
            else:
                print(f"[Specular] WARNING: No spec checkpoint at {spec_ckpt_path}")
                print(f"[Specular] Falling back to base-only rendering.")
                print(f"[Specular] Ensure training was run with --use_specular_branch and checkpoint was saved.")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       spec_branch, pose_embed, spec_args)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(),
                       gaussians, pipeline, background, dataset.train_test_exp, separate_sh,
                       spec_branch, pose_embed, spec_args)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    sp = SpecularParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                SPARSE_ADAM_AVAILABLE, spec_args=sp.extract(args))
