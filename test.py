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
import os.path as osp
import sys
import torch
from tqdm import tqdm, trange
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser
from random import randint
import SimpleITK as sitk
import nibabel as nib

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
    DeformationParams,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.image_utils import metric_vol, metric_proj
from Hexplane.deformation import deform_network



def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    deform_params: DeformationParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
):
    
    # Set up dataset
    scene = Scene(
        dataset,
        shuffle=False,
    )

    # Set up Gaussians
    gaussians = GaussianModel(None)  # scale_bound will be loaded later
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians

    # Initialize deformation network
    deform_net = deform_network(deform_params).cuda()
    deform_net_checkpoint_path = os.path.join(dataset.model_path, "ckpt", f"chkpnt{iteration}.pth")
    checkpoint = torch.load(deform_net_checkpoint_path, weights_only=False)
    deform_net.load_state_dict(checkpoint["deformation_net_state_dict"])
    deform_net.eval()

    #construct 4D ground truth volume
    gt_vols_4d = scene.info.gt_vols

    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )

    # Evaluate projection train
    if not skip_render_train:
        evaluate_render(
            save_path,
            "render_train",
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            deform_net,
        )
    # Evaluate projection test
    if not skip_render_test:
        evaluate_render(
            save_path,
            "render_test",
            scene.getTestCameras(),
            gaussians,
            pipeline,
            deform_net,
        )
    # Evaluate volume reconstruction
    if not skip_recon:
        evaluate_volume(
            save_path,
            "reconstruction",
            scene.scanner_cfg,
            gaussians,
            pipeline,
            gt_vols_4d,
            deform_net,
        )


def evaluate_volume(
    save_path,
    name,
    scanner_cfg,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    gt_vols,
    deform_net,
):
    """Evaluate volume reconstruction."""
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    #Load predicted volumes and stack into 4D array
    vol_pred_list = []
    for phasebin in range(len(gt_vols)):
        print(f"Querying volume for timepoint {phasebin}...")
        query_pkg = query(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipeline,
            deformation_net=deform_net,
            time_value=phasebin,
            state="fine" 
        )
        vol_pred = query_pkg["vol"]
        vol_pred_list.append(vol_pred)

    eval_dicts_4d = []
    eval_dict_4d = {
            "psnr_4d": 0.0,
            "Average_psnr_3d": 0.0,
            "Average_ssim_3d": 0.0,
            "Average_ssim_3d_x": 0.0,
            "Average_ssim_3d_y": 0.0,
            "Average_ssim_3d_z": 0.0,
        }

    for phasebin in range(len(gt_vols)):
        vol_pred = vol_pred_list[phasebin]
        vol_gt = gt_vols[phasebin]

        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")

        multithread_write(
            [vol_gt[..., i][None] for i in range(vol_gt.shape[2])],
            slice_save_path,
            "_gt",
        )
        multithread_write(
            [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
            slice_save_path,
            "_pred",
        )
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
        }

        eval_dict_4d["Average_psnr_3d"] += psnr_3d / len(gt_vols)
        eval_dict_4d["Average_ssim_3d"] += ssim_3d / len(gt_vols)
        eval_dict_4d["Average_ssim_3d_x"] += ssim_3d_axis[0] / len(gt_vols)
        eval_dict_4d["Average_ssim_3d_y"] += ssim_3d_axis[1] / len(gt_vols)
        eval_dict_4d["Average_ssim_3d_z"] += ssim_3d_axis[2] / len(gt_vols)
        eval_dicts_4d.append(eval_dict)

    vol_pred = np.stack([t.cpu().numpy() for t in vol_pred_list], axis=-1)
    vol_gt = np.stack([t.cpu().numpy() for t in gt_vols], axis=-1)

    psnr_4d, _ = metric_vol(vol_gt, vol_pred, "4dpsnr")

    eval_dict_4d["psnr_4d"] = float(psnr_4d)

    with open(osp.join(save_path, "eval4d.yml"), "w") as f:
        yaml.dump(eval_dict_4d, f, default_flow_style=False, sort_keys=False)

    with open(osp.join(save_path, "eval4d_per_phasebin.yml"), "w") as f:
        for phase, eval_dict in enumerate(eval_dicts_4d):
            f.write(f"Phasebin {phase}\n")
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
            f.write("\n")  # Add a newline for better readability between entries

    print([t.shape for t in vol_pred_list])
    print([t.shape for t in gt_vols])
    print(vol_gt.shape)
    print(vol_pred.shape)


    np.save(osp.join(save_path, "vol_gt.npy"), t2a(vol_gt))
    np.save(osp.join(save_path, "vol_pred.npy"), t2a(vol_pred))

    vol_gt_nifti = nib.Nifti1Image(vol_gt, affine=np.eye(4))  # Replace np.eye(4) with the correct affine if available
    vol_pred_nifti = nib.Nifti1Image(vol_pred, affine=np.eye(4))  # Replace np.eye(4) with the correct affine if available

    # Save the Nifti1Image objects
    nib.save(vol_gt_nifti, os.path.join(save_path, "vol_gt.nii.gz"))
    nib.save(vol_pred_nifti, os.path.join(save_path, "vol_pred.nii.gz"))

    print(f"{name} complete. psnr_4d: {eval_dict_4d['psnr_4d']}, average psnr_3d: {eval_dict_4d['Average_psnr_3d']}, average ssim_3d: {eval_dict_4d['Average_ssim_3d']}")


def evaluate_render(save_path, name, views, gaussians, pipeline, deform_net):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline, deformation_net=deform_net,state="fine")["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)
    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
    ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
    eval_dict = {
        "psnr_2d": psnr_2d,
        "ssim_2d": ssim_2d,
        "psnr_2d_projs": psnr_2d_projs,
        "ssim_2d_projs": ssim_2d_projs,
    }
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
    print(
        f"{name} complete. psnr_2d: {eval_dict['psnr_2d']}, ssim_2d: {eval_dict['ssim_2d']}."
    )


def multithread_write(image_list, path, suffix):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(
                image, osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
            )
            np.save(
                osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
                image.cpu().numpy()[0],
            )
            return count, True
        except:
            return count, False

    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    deform_params = DeformationParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            deform_params.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
        )
