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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SpecularParams
import json

from utils.specular_module import compute_specular_prior, compute_residual_specular_prior, PoseEmbedding, SpecularBranch
from utils.specular_losses import compute_specular_losses
from utils.visualize_specular import save_specular_debug_images, log_specular_scalars, log_specular_images

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _save_spec_checkpoint(scene, spec_branch, pose_embed, spec_optimizer, iteration):
    """Save specular branch state to disk."""
    spec_ckpt = {
        "spec_branch": spec_branch.state_dict(),
        "pose_embed": pose_embed.state_dict(),
        "spec_optimizer": spec_optimizer.state_dict(),
    }
    path = os.path.join(scene.model_path, f"spec_chkpnt{iteration}.pth")
    torch.save(spec_ckpt, path)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, spec_args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iteration)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iteration), desc="Training progress")
    first_iter += 1

    log_file_path = os.path.join(scene.model_path, "results.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    with open(log_file_path, 'w') as file:
        file.write(f"results for {scene.model_path}\n")

    best_model = {'iter': 0, 'test_avg_psnr_base': 0, 'test_avg_psnr_final': 0}

    # --- Specular branch setup ---
    use_spec = spec_args is not None and spec_args.use_specular_branch
    spec_branch = None
    pose_embed = None
    spec_optimizer = None
    staged_applied = False
    if use_spec:
        train_cameras = scene.getTrainCameras()
        pose_embed = PoseEmbedding(spec_args.spec_latent_dim).cuda()
        pose_embed.set_normalization_stats(train_cameras)
        spec_branch = SpecularBranch(
            latent_dim=spec_args.spec_latent_dim,
            hidden_dim=spec_args.spec_hidden_dim,
            gain=spec_args.spec_gain,
        ).cuda()
        spec_optimizer = torch.optim.Adam(
            [
                {"params": spec_branch.parameters(), "lr": spec_args.spec_lr},
                {"params": pose_embed.parameters(), "lr": spec_args.spec_pose_lr},
            ]
        )
        spec_vis_dir = os.path.join(scene.model_path, "specular_vis")
        print(f"[Specular] Branch ON  |  latent_dim={spec_args.spec_latent_dim}  "
              f"|  schedule={spec_args.spec_schedule}  |  start_iter={spec_args.spec_start_iter}  "
              f"|  lambda_spec_res={spec_args.lambda_spec_res}  "
              f"|  mask_alpha={spec_args.base_mask_alpha}  |  mask_clamp_min={spec_args.base_mask_clamp_min}"
              f"\n[Specular] PoseEmbedding initialized with {len(train_cameras)} training cameras")

    for iteration in range(first_iter, opt.iteration + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iteration)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # --- Specular prior & loss masking (from iter 0 to prevent baked-in) ---
        spec_active = use_spec and iteration >= spec_args.spec_start_iter
        spec_loss_dict = {}

        # Shield: GT-only HSV cue for Gaussian loss masking (no base_rgb dependency)
        if use_spec:
            base_mask_prior = compute_specular_prior(gt_image, gamma=spec_args.spec_prior_gamma)  # [1, H, W]

        # Staged schedule: reduce Gaussian LR after densification ends (decoupled from spec_start)
        if use_spec and spec_args.spec_schedule == "staged" and not staged_applied and iteration >= opt.densify_until_iter:
            scale = spec_args.staged_gauss_lr_scale
            for pg in gaussians.optimizer.param_groups:
                pg['lr'] *= scale
            for pg in gaussians.exposure_optimizer.param_groups:
                pg['lr'] *= scale
            staged_applied = True
            print(f"\n[Specular] Staged: Gaussian LR scaled by {scale} at iter {iteration}")

        # --- Specular branch forward (only after spec_start_iter) ---
        if spec_active:
            # Spear: residual-weighted prior for spec branch guidance
            spec_prior = compute_residual_specular_prior(
                gt_image, image.detach(), base_mask_prior
            )  # [1, H, W]

            z_t = pose_embed(viewpoint_cam.R, viewpoint_cam.T)  # [1, D]
            spec_rgb, spec_map, spec_residual = spec_branch(image.detach(), z_t)
            spec_rgb = spec_rgb.squeeze(0)      # [3, H, W]
            spec_map = spec_map.squeeze(0)      # [1, H, W]
            # Decoupled: Gaussian only receives gradient from masked Ll1_base
            final_rgb = (image.detach() + spec_rgb).clamp(0, 1)
        else:
            final_rgb = image

        # ---- Base reconstruction loss (Gaussian gradient) ----
        if use_spec:
            # Masked L1: w→0 in highlight regions, w→1 in tissue regions
            # Full masking from iter 0 (clamp_min ensures gradient floor, no ramp needed)
            weight_map = (1.0 - spec_args.base_mask_alpha * base_mask_prior).clamp(min=spec_args.base_mask_clamp_min)  # [1,H,W]
            Ll1_base = (weight_map * torch.abs(image - gt_image)).sum() / (weight_map.sum() + 1e-8)
            # Masked SSIM: apply weight_map to both images so SSIM focuses on tissue regions
            weight_3ch = weight_map.expand(3, -1, -1)  # [3, H, W]
            image_w = image * weight_3ch
            gt_w = gt_image * weight_3ch
            if FUSED_SSIM_AVAILABLE:
                ssim_value_base = fused_ssim(image_w.unsqueeze(0), gt_w.unsqueeze(0))
            else:
                ssim_value_base = ssim(image_w, gt_w)
            loss = (1.0 - opt.lambda_dssim) * Ll1_base + opt.lambda_dssim * (1.0 - ssim_value_base)
        else:
            Ll1_base = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value_base = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value_base = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1_base + opt.lambda_dssim * (1.0 - ssim_value_base)

        # ---- Specular losses (coupled: gradients flow to both spec branch and Gaussians) ----
        if spec_active:
            spec_loss, spec_loss_dict = compute_specular_losses(
                spec_rgb=spec_rgb,
                spec_map=spec_map,
                spec_prior=spec_prior,
                base_rgb=image.detach(),  # detached: spec losses do not push Gaussian to learn highlights
                gt_image=gt_image,
                light_latent=z_t,
                lambda_spec_res=spec_args.lambda_spec_res,
                spec_res_prior_alpha=spec_args.spec_res_prior_alpha,
                lambda_prior=spec_args.lambda_prior,
                lambda_sparse=spec_args.lambda_sparse,
                lambda_tv=spec_args.lambda_tv,
                lambda_z=spec_args.lambda_z,
                lambda_final=spec_args.lambda_final,
                lambda_dssim=spec_args.lambda_dssim_spec,
            )
            loss = loss + spec_loss

        # For progress bar logging
        Ll1 = l1_loss(final_rgb, gt_image) if spec_active else Ll1_base

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            total_points = gaussians.get_xyz.shape[0]

            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "Points": f"{total_points}"
                }
                if spec_active:
                    postfix["SpecMap"] = f"{spec_map.mean().item():.4f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iteration:
                progress_bar.close()

            # Spec collapse warning
            if spec_active and iteration % 1000 == 0:
                sm_mean = spec_map.mean().item()
                sr_mean = spec_rgb.abs().mean().item()
                if sm_mean < 0.01 and sr_mean < 0.001:
                    print(f"\n[Specular WARNING iter {iteration}] spec_map_mean={sm_mean:.5f}, spec_rgb_abs_mean={sr_mean:.6f} — branch may be collapsing")

            # -----------------------------------------------------------
            # Log and save
            # -----------------------------------------------------------
            current_test_psnr_base, current_test_psnr_final = training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end), testing_iterations, scene, render,
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp, log_file_path=log_file_path,
                spec_branch=spec_branch if spec_active else None,
                pose_embed=pose_embed if spec_active else None,
                spec_args=spec_args,
            )

            if iteration in testing_iterations:
                if current_test_psnr_base >= best_model['test_avg_psnr_base']:
                    best_model['test_avg_psnr_base'] = current_test_psnr_base
                if current_test_psnr_final >= best_model['test_avg_psnr_final']:
                    best_model['test_avg_psnr_final'] = current_test_psnr_final
                    best_model['iter'] = iteration

            # Save Gaussians + specular checkpoint at saving_iterations
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if use_spec:
                    _save_spec_checkpoint(scene, spec_branch, pose_embed, spec_optimizer, iteration)
                    print(f"[ITER {iteration}] Saved spec checkpoint")

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Specular logging & visualisation
            if spec_active:
                log_specular_scalars(tb_writer, iteration, spec_loss_dict, spec_map, spec_rgb, z_t)
                if spec_args.spec_vis_interval > 0 and iteration % spec_args.spec_vis_interval == 0:
                    log_specular_images(tb_writer, iteration, gt_image, image, spec_prior, base_mask_prior, spec_map.squeeze(0), spec_rgb, final_rgb)
                    save_specular_debug_images(spec_vis_dir, iteration, gt_image, image, spec_prior, base_mask_prior, spec_map.squeeze(0), spec_rgb, final_rgb)

            # Optimizer step
            if iteration < opt.iteration:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if spec_active:
                    spec_optimizer.step()
                    spec_optimizer.zero_grad(set_to_none=True)

            # Full checkpoint (for training resume)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt_data = (gaussians.capture(), iteration)
                torch.save(ckpt_data, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if use_spec:
                    _save_spec_checkpoint(scene, spec_branch, pose_embed, spec_optimizer, iteration)

    # Final log
    with open(log_file_path, "a") as file:
        file.write("\n" + "BEST MODEL" + "\n")
        file.write(json.dumps(best_model) + "\n")
        file.write(json.dumps({"NUM GAUSSIANS": len(gaussians.get_xyz)}))

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                    scene: Scene, renderFunc, renderArgs, train_test_exp,
                    log_file_path="results.txt",
                    spec_branch=None, pose_embed=None, spec_args=None):
    """
    Evaluate and log metrics. When spec_branch is provided, report both
    base-only and final (base+spec) metrics.
    """
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    psnr_avg_base = 0.0
    psnr_avg_final = 0.0

    if iteration in testing_iterations:
        import lpips
        from pytorch_msssim import ms_ssim

        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        use_spec_eval = spec_branch is not None and pose_embed is not None
        train_cams = scene.getTrainCameras()

        if use_spec_eval:
            print(f"\n[Specular] Evaluating with spec branch | pose-based latent")

        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # Base metrics
                l1_list_b, psnr_list_b, lpips_list_b, ssim_list_b, mssim_list_b = [], [], [], [], []
                # Final metrics (base+spec)
                l1_list_f, psnr_list_f, lpips_list_f, ssim_list_f, mssim_list_f = [], [], [], [], []

                for idx, viewpoint in enumerate(config['cameras']):
                    base_img = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if train_test_exp:
                        base_img = base_img[..., base_img.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]

                    # Base metrics
                    l1_list_b.append(l1_loss(base_img, gt_image).mean().double())
                    psnr_list_b.append(psnr(base_img, gt_image).mean().double())
                    ssim_list_b.append(ssim(base_img, gt_image).mean().double())
                    img_b = base_img.unsqueeze(0)
                    gt_b = gt_image.unsqueeze(0)
                    lpips_list_b.append(lpips_fn(img_b, gt_b).mean().double())
                    mssim_list_b.append(ms_ssim(img_b, gt_b, data_range=1.0).mean().double())

                    # Final metrics (with spec branch)
                    if use_spec_eval:
                        z_t = pose_embed(viewpoint.R, viewpoint.T)
                        s_rgb, s_map, _ = spec_branch(base_img, z_t)
                        final_img = (base_img + s_rgb.squeeze(0)).clamp(0, 1)
                    else:
                        final_img = base_img

                    l1_list_f.append(l1_loss(final_img, gt_image).mean().double())
                    psnr_list_f.append(psnr(final_img, gt_image).mean().double())
                    ssim_list_f.append(ssim(final_img, gt_image).mean().double())
                    img_f = final_img.unsqueeze(0)
                    lpips_list_f.append(lpips_fn(img_f, gt_b).mean().double())
                    mssim_list_f.append(ms_ssim(img_f, gt_b, data_range=1.0).mean().double())

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render_base".format(viewpoint.image_name), base_img[None], global_step=iteration)
                        if use_spec_eval:
                            tb_writer.add_images(config['name'] + "_view_{}/render_final".format(viewpoint.image_name), final_img[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                def _avg(lst):
                    return sum(lst) / len(lst)
                def _std(lst):
                    return torch.std(torch.tensor(lst)).item() if len(lst) > 1 else 0.0

                psnr_b = _avg(psnr_list_b)
                psnr_f = _avg(psnr_list_f)

                if config['name'] == 'test':
                    psnr_avg_base = psnr_b.item() if torch.is_tensor(psnr_b) else psnr_b
                    psnr_avg_final = psnr_f.item() if torch.is_tensor(psnr_f) else psnr_f

                    if use_spec_eval:
                        print(f"[ITER {iteration}] test PSNR base={psnr_avg_base:.2f}  final={psnr_avg_final:.2f}  (delta={psnr_avg_final-psnr_avg_base:+.3f})")
                    else:
                        print(f"\n[ITER {iteration}] Evaluating {config['name']}: PSNR {psnr_avg_base:.2f}")

                    if log_file_path:
                        with open(log_file_path, "a") as f:
                            f.write("\n")
                            f.write(f"[ITER {iteration}] {config['name']} BASE  avg: L1 {_avg(l1_list_b):.4f} PSNR {psnr_avg_base:.2f} LPIPS {_avg(lpips_list_b):.4f} SSIM {_avg(ssim_list_b):.4f} MSSIM {_avg(mssim_list_b):.4f}\n")
                            f.write(f"[ITER {iteration}] {config['name']} BASE  std: L1 {_std(l1_list_b):.4f} PSNR {_std(psnr_list_b):.2f} LPIPS {_std(lpips_list_b):.4f} SSIM {_std(ssim_list_b):.4f} MSSIM {_std(mssim_list_b):.4f}\n")
                            if use_spec_eval:
                                f.write(f"[ITER {iteration}] {config['name']} FINAL avg: L1 {_avg(l1_list_f):.4f} PSNR {psnr_avg_final:.2f} LPIPS {_avg(lpips_list_f):.4f} SSIM {_avg(ssim_list_f):.4f} MSSIM {_avg(mssim_list_f):.4f}\n")
                                f.write(f"[ITER {iteration}] {config['name']} FINAL std: L1 {_std(l1_list_f):.4f} PSNR {_std(psnr_list_f):.2f} LPIPS {_std(lpips_list_f):.4f} SSIM {_std(ssim_list_f):.4f} MSSIM {_std(mssim_list_f):.4f}\n")

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/psnr_base', _avg(psnr_list_b), iteration)
                    tb_writer.add_scalar(config['name'] + '/lpips_base', _avg(lpips_list_b), iteration)
                    if use_spec_eval:
                        tb_writer.add_scalar(config['name'] + '/psnr_final', _avg(psnr_list_f), iteration)
                        tb_writer.add_scalar(config['name'] + '/lpips_final', _avg(lpips_list_f), iteration)

        torch.cuda.empty_cache()

    return psnr_avg_base, psnr_avg_final


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sp = SpecularParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 10000, 20000, 30000, 40000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iteration)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, spec_args=sp.extract(args))

    print("\nTraining complete.")
