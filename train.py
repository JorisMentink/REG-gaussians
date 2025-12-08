#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import os
import os.path as osp
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import shutil

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, DeformationParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from r2_gaussian.utils.plot_utils import show_two_slice

from Hexplane.deformation import deform_network
from Hexplane.additive_loss_funcs import grid_smoothness_loss, geometric_regularization_loss, velocity_coherence_loss
import nibabel as nib
import matplotlib.pyplot as plt



def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    deform_params: DeformationParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        deformation_net=deform_net,
        time_value=viewpoint_cam.timepoint,
        state=state
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    print("Initializing deformation network...")
    deform_net = deform_network(deform_params).cuda()
    deform_optimizer = deform_net.setup_deform_optimizer(deform_net,deform_params)
    print("Deformation network initialized.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    #===================================================================================#
    #EXTRA CODE TO SAVE LOSS MAPS OR OTHER VISUALIZATIONS IF NEEDED
    #===================================================================================#
    # Prepare loss CSV and in-memory history for plotting
    loss_csv_path = osp.join(scene.model_path, "losses.csv")
    if not osp.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f:
            f.write("iter,total,render,dssim,tv,Additive,xyz_lr\n")
    loss_history = {"iter": [], "total": [], "render": [], "dssim": [], "tv": [], "Additive": []}
    loss_plot_interval = 500
    loss_plot_path = osp.join(scene.model_path, "loss_curve.png")
    #===================================================================================#
    #EXTRA CODE TO SAVE LOSS MAPS OR OTHER VISUALIZATIONS IF NEEDED
    #===================================================================================#

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    warmup_its = deform_params.warmup_its
    state="rough"
    checkpoint_render_folder = "Checkpoint renders"
    # Clear the folder if it exists

    if os.path.exists(checkpoint_render_folder):
        shutil.rmtree(checkpoint_render_folder) # Clears checkpoint render folder
    os.makedirs(checkpoint_render_folder, exist_ok=True)

    batch_size_fine = 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        if deform_params.use_deform_lr_scheduler:
            deform_net.update_deform_net_learning_rate(iteration)

        if iteration > warmup_its:
            state = "fine"

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        #Select batch of viewpoints
        if state == "fine" and batch_size_fine > 1:
            batch_views = []
            for _ in range(batch_size_fine):
                if len(viewpoint_stack) == 0:
                    viewpoint_stack = scene.getTrainCameras().copy()
                idx = randint(0, len(viewpoint_stack) - 1)
                batch_views.append(viewpoint_stack.pop(idx))
        else:
            # fallback to single view (same as before)
            if len(viewpoint_stack) == 0:
                viewpoint_stack = scene.getTrainCameras().copy()
            batch_views = [viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))]


        #TODO buffer prints
        loss = {
                "total": torch.tensor(0.0, device='cuda'),
                "render": torch.tensor(0.0, device='cuda'),
                "dssim": torch.tensor(0.0, device='cuda'),
                "tv": torch.tensor(0.0, device='cuda'),
                "Additive": torch.tensor(0.0, device='cuda'),
                "geometric_reg": torch.tensor(0.0, device='cuda'),
            }
        for viewpoint_cam in batch_views:
            render_pkg = render(viewpoint_cam, gaussians, pipe,deformation_net=deform_net,state=state)
            image, viewspace_point_tensor, visibility_filter, radii, final_gaussian_params = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["final_gaussian_params"],
            )
          
            gt_image = viewpoint_cam.original_image.cuda()

            render_loss = l1_loss(image, gt_image)
            loss["render"] = render_loss
            loss["total"] = loss["total"] + loss["render"]
            print(f" L1 loss at iter {iteration}: {loss['render'].item()}")
            if opt.lambda_dssim > 0:
                loss_dssim = 1.0 - ssim(image, gt_image)
                loss["dssim"] = loss_dssim
                loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
                print(f" DSSIM loss at iter {iteration}: {loss['dssim'].item()}")
            
            #GEOMETRIC REGULARIZATION LOSS
            # if (iteration > deform_params.geom_loss_from_iter) and (deform_params.lambda_geo > 0) and (iteration % deform_params.geom_loss_interval == 0):
            #     loss["geometric_reg"] = geometric_regularization_loss(final_gaussian_params["scales"], r_m=1.0, r_ani=3.0, reduction='mean')
            #     loss["total"] = loss["total"] + deform_params.lambda_geo * loss["geometric_reg"]
            #     print(f" Geometric regularization loss at iter {iteration}: {loss['geometric_reg'].item()}")


            # if (iteration > deform_params.velocity_coherence_from_iter) and (deform_params.lambda_velocity_coherence > 0) and (iteration % deform_params.velocity_coherence_interval == 0):
            #     loss["velocity_coherence"] = torch.tensor(0.0, device='cuda')
            #     print(f"iteration {iteration}, rest: {iteration % deform_params.velocity_coherence_interval}")
            #     loss["velocity_coherence"] = velocity_coherence_loss(final_gaussian_params["means3D"], final_gaussian_params["velocities"], k=5)
            #     loss["total"] = loss["total"] + deform_params.lambda_velocity_coherence * loss["velocity_coherence"]
            #     print(f" Velocity coherence loss at iter {iteration}: {loss['velocity_coherence'].item()}")

            if (state=="fine") and (deform_params.time_smoothness_weight > 0):
                loss["Additive"] = grid_smoothness_loss(deformation_network = deform_net, 
                                                                time_smoothness_weight = deform_params.time_smoothness_weight, 
                                                                l1_time_planes_weight = deform_params.l1_time_planes, 
                                                                plane_tv_weight = deform_params.plane_tv_weight)
                loss["total"] = loss["total"] + loss["Additive"]
                print(f"Additive loss at iter {iteration}: {loss['Additive'].item()}")

            else:
                # 3D TV loss
                if use_tv:
                    tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                        bbox[1] - tv_vol_sVoxel - bbox[0]
                    ) * torch.rand(3)
                    vol_pred = query(
                        gaussians,
                        tv_vol_center,
                        tv_vol_nVoxel,
                        tv_vol_sVoxel,
                        pipe,
                        deformation_net=deform_net,                # <-- Pass the deformation net
                        time_value=viewpoint_cam.timepoint,
                        state=state)["vol"]
                    if torch.isnan(vol_pred).any() or torch.isinf(vol_pred).any():
                        print("NaN/Inf detected in vol_pred!")
                    loss_tv = tv_3d_loss(vol_pred, reduction="mean")
                    loss["tv"] = loss_tv
                    loss["total"] = loss["total"] + opt.lambda_tv * loss_tv
                    print(f" TV loss at iter {iteration}: {loss['tv'].item()}")


        loss["total"] = loss["total"] / len(batch_views)
        loss["total"].backward()

        if iteration % 100 == 0 or iteration == 5001:

            pred_np = image[0].detach().cpu().numpy()
            gt_np = gt_image[0].detach().cpu().numpy()

            # If single-channel, squeeze; if multi-channel, you can select channel 0
            if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                pred_np = pred_np[0]
                gt_np = gt_np[0]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            im0 = axes[0].imshow(gt_np, cmap='gray')
            axes[0].set_title(f"GT (iter {iteration}, time {viewpoint_cam.timepoint})")
            axes[0].axis('off')
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

            im1 = axes[1].imshow(pred_np, cmap='gray')
            axes[1].set_title(f"Predicted, time {viewpoint_cam.timepoint}")
            axes[1].axis('off')
            fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            fig.tight_layout()
            plt.savefig(os.path.join(checkpoint_render_folder, f"proj_gt_pred_iter_{iteration}.png"))
            plt.close()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter and opt.densify_during_fine:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration <= warmup_its or opt.use_joint_optimization:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration > warmup_its and deform_optimizer is not None:
                deform_optimizer.step()
                deform_optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in checkpoint_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                #tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                # torch.save(
                #     (gaussians.capture(), iteration),
                #     ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                # )
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    {
                        "gaussians": gaussians.capture(),
                        "iteration": iteration,
                        "deformation_net_state_dict": deform_net.state_dict() if deform_net is not None else None,
                        "scanner_cfg": scanner_cfg,  # Add scanner_cfg to the checkpoint
                    },
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]

            #===================================================================================#
            #EXTRA CODE TO SAVE LOSS MAPS OR OTHER VISUALIZATIONS IF NEEDED
            #===================================================================================#
            # Append to CSV and update in-memory history, then plot periodically
            try:
                # find xyz lr
                xyz_lr = None
                for pg in gaussians.optimizer.param_groups:
                    if pg.get('name') == 'xyz':
                        xyz_lr = pg['lr']
                        break
                with open(osp.join(scene.model_path, 'losses.csv'), 'a') as f:
                    f.write(f"{iteration},{metrics.get('loss_total',0):.8e},{metrics.get('loss_render',0):.8e},{metrics.get('loss_dssim',0):.8e},{metrics.get('loss_tv',0):.8e},{metrics.get('loss_Additive',0):.8e},{xyz_lr if xyz_lr is not None else 0}\n")
            except Exception:
                pass

            try:
                loss_history['iter'].append(iteration)
                loss_history['total'].append(metrics.get('loss_total', 0))
                loss_history['render'].append(metrics.get('loss_render', 0))
                loss_history['dssim'].append(metrics.get('loss_dssim', 0))
                loss_history['tv'].append(metrics.get('loss_tv', 0))
                loss_history['Additive'].append(metrics.get('loss_Additive', 0))
            except Exception:
                pass

            # Periodically save a PNG of the loss curves
            try:
                if iteration % loss_plot_interval == 0 and len(loss_history['iter'])>0:
                    plt.figure(figsize=(8, 4))
                    it = loss_history['iter']
                    if any(v > 0 for v in loss_history['total']):
                        plt.semilogy(it, loss_history['total'], label='total')
                    if any(v > 0 for v in loss_history['render']):
                        plt.semilogy(it, loss_history['render'], label='render')
                    if any(v > 0 for v in loss_history['dssim']):
                        plt.semilogy(it, loss_history['dssim'], label='dssim')
                    if any(v > 0 for v in loss_history['tv']):
                        plt.semilogy(it, loss_history['tv'], label='tv')
                    if any(v > 0 for v in loss_history['Additive']):
                        plt.semilogy(it, loss_history['Additive'], label='Additive')
                    plt.xlabel('iteration')
                    plt.ylabel('loss (log)')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(loss_plot_path)
                    plt.close()
            except Exception:
                pass
            #===================================================================================#
            #EXTRA CODE TO SAVE LOSS MAPS OR OTHER VISUALIZATIONS IF NEEDED
            #===================================================================================#

if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DeformationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        dp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
