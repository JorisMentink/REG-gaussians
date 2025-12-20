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
import sys
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)

sys.path.append("./")
from Static_Gaussian.gaussian.gaussian_model import GaussianModel
from Static_Gaussian.dataset.cameras import Camera
from Static_Gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    deformation_net=None,           # <-- Add this
    time_value=None,                 # <-- Optional: pass time externally
    state=None
):
    """
    Query a volume with voxelization.
    """
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if state == "rough" or state ==None:
        means3D_final = means3D
        scales_final = scales
        rotations_final = rotations
        density_final = density

    # Apply deformation if provided
    if state == "fine" and deformation_net is not None:
        # Use time_value if provided, otherwise get from viewpoint_camera
        time = torch.full((means3D.shape[0], 1), float(time_value/10), device=means3D.device)
        means3D_final, scales_final, rotations_final, density_final = deformation_net(means3D, scales, rotations, density, time)
        scales_final = pc.scaling_activation(scales_final)
        #scales_final = torch.clamp(scales_final, min=1e-3, max=1.0)
        rotations_final = pc.rotation_activation(rotations_final)


    # Check shapes and devices
    assert means3D.is_cuda, "means3D not on CUDA"
    assert density.is_cuda, "density not on CUDA"
    if scales is not None: assert scales.is_cuda, "scales not on CUDA"
    if rotations is not None: assert rotations.is_cuda, "rotations not on CUDA"
    if cov3D_precomp is not None: assert cov3D_precomp.is_cuda, "cov3D_precomp not on CUDA"

    n = means3D.shape[0]
    assert density.shape[0] == n, "density shape mismatch"
    if scales is not None: assert scales.shape[0] == n, "scales shape mismatch"
    if rotations is not None: assert rotations.shape[0] == n, "rotations shape mismatch"

    # Check for NaNs/infs
    assert torch.isfinite(means3D).all(), "means3D contains NaN/Inf"
    assert torch.isfinite(density).all(), "density contains NaN/Inf"
    if scales is not None: assert torch.isfinite(scales).all(), "scales contains NaN/Inf"
    if rotations is not None: assert torch.isfinite(rotations).all(), "rotations contains NaN/Inf"
    if cov3D_precomp is not None: assert torch.isfinite(cov3D_precomp).all(), "cov3D_precomp contains NaN/Inf"


    # print("means3D_final min/max:", means3D_final.min(0).values, means3D_final.max(0).values)
    # if scales_final is not None:
    #     print("scales_final min/max:", scales_final.min().item(), scales_final.max().item())
    # print("density min/max:", density.min().item(), density.max().item())
    # print("voxelizer settings:", voxel_settings)

    # print("====================")
        
    # print("means3D range before:", means3D.min(0).values, means3D.max(0).values)
    # print("means3D range after :", means3D_final.min(0).values, means3D_final.max(0).values)

    # print("=====================")



    vol_pred, radii = voxelizer(
        means3D=means3D_final,
        opacities=density_final,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
    deformation_net=None,      # <-- Add this
    state=None
):
    """
    Render an X-ray projection with rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density
    time_value = viewpoint_camera.timepoint

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if state == "rough" or state == None:
        means3D_final = means3D
        scales_final = scales
        rotations_final = rotations
        density_final = density

    # Apply deformation if provided
    if state == "fine" and deformation_net is not None:
        # Use time_value if provided, otherwise get from viewpoint_camera
        time = torch.full((means3D.shape[0], 1), float(time_value/10), device=means3D.device)
        means3D_final, scales_final, rotations_final, density_final = deformation_net(means3D, scales, rotations, density, time)
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)


    velocities = (means3D_final - means3D) / (time_value + 1e-5) #adding 1e-5 to avoid division by zero. if time = 0 means3D_final - means3D = 0 anyway.


    # Check shapes and devices
    assert means3D.is_cuda, "means3D not on CUDA"
    assert density.is_cuda, "density not on CUDA"
    if scales is not None: assert scales.is_cuda, "scales not on CUDA"
    if rotations is not None: assert rotations.is_cuda, "rotations not on CUDA"
    if cov3D_precomp is not None: assert cov3D_precomp.is_cuda, "cov3D_precomp not on CUDA"

    n = means3D.shape[0]
    assert density.shape[0] == n, "density shape mismatch"
    if scales is not None: assert scales.shape[0] == n, "scales shape mismatch"
    if rotations is not None: assert rotations.shape[0] == n, "rotations shape mismatch"

    # Check for NaNs/infs
    assert torch.isfinite(means3D).all(), "means3D contains NaN/Inf"
    assert torch.isfinite(density).all(), "density contains NaN/Inf"
    if scales is not None: assert torch.isfinite(scales).all(), "scales contains NaN/Inf"
    if rotations is not None: assert torch.isfinite(rotations).all(), "rotations contains NaN/Inf"
    if cov3D_precomp is not None: assert torch.isfinite(cov3D_precomp).all(), "cov3D_precomp contains NaN/Inf"



    rendered_image, radii = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        opacities=density_final,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp,
    )

    final_gaussian_params = {"means3D": means3D_final,
                             "means3D_original": means3D,
                             "scales": scales_final,
                             "rotations": rotations_final,
                             "density": density_final,
                             "velocities": velocities,
                             }

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "final_gaussian_params": final_gaussian_params,
    }


