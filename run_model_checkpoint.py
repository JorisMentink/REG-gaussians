import os
import os.path as osp
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from random import randint

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
    DeformationParams,
)

from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from Hexplane.deformation import deform_network
import nibabel as nib



def query_model_checkpoint(
    dataset: ModelParams,
    pipeline: PipelineParams,
    deform_params: DeformationParams,
    iteration: int,
    queried_timepoints=list(range(0, 10, 1))
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
    scanner_cfg = scene.scanner_cfg

    # Initialize deformation network
    deform_net = deform_network(deform_params).cuda()
    deform_net_checkpoint_path = os.path.join(dataset.model_path, "ckpt", f"chkpnt{iteration}.pth")
    checkpoint = torch.load(deform_net_checkpoint_path, weights_only=False)
    deform_net.load_state_dict(checkpoint["deformation_net_state_dict"])
    deform_net.eval()

    save_path = osp.join(
        dataset.model_path,
        "checkpoint_inference"
    )

    os.makedirs(save_path, exist_ok=True)    

    queried_volumes = []

    niftii_savepath = osp.join(save_path, f"it_{loaded_iter}_reconstruction.nii.gz")
    numpy_savepath = osp.join(save_path, f"it_{loaded_iter}_reconstruction.npy")

    for timepoint in queried_timepoints:
        query_pkg = query(
            gaussians,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipeline,
            deformation_net=deform_net,
            time_value=timepoint,
            state="fine"
        )

        queried_volumes.append(query_pkg["vol"].cpu().numpy()) 

    volumes_4d = np.stack(queried_volumes, axis=-1)
    nifti_img = nib.Nifti1Image(volumes_4d, affine=np.eye(4))
    nib.save(nifti_img, niftii_savepath)
    print(f"Saved 4D NIfTI file to {niftii_savepath}")
    np.save(numpy_savepath, volumes_4d)
    print(f"Saved 4D NumPy file to {numpy_savepath}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    deform_params = DeformationParams(parser)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        query_model_checkpoint(
            model.extract(args),
            pipeline.extract(args),
            deform_params.extract(args),
            args.iteration
        )
