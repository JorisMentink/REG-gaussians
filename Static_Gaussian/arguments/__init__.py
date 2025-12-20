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
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from Static_Gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = "output/300Projections_Case3_B_2"
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000
        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5e-5 #WATCH OUT I SET THIS TO 1E-5 FOR THE FIRST FULL RUN :( TRY WITH 5E-5#Used it with 1e-5 before, was interesting HAS POTENTIAL. original value 5e-5
        self.densify_scale_threshold = 0.1  # percent of volume size was 0.1 before
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 150_000
        
        self.densify_during_fine = True #True #True #Determines wheter or not gaussian densification/pruning is used in the fine state of training
        self.use_joint_optimization = True #True #Determines whether or not to optimize the initial gaussian state during the fine training stage
        
        super().__init__(parser, "Optimization Parameters")


class DeformationParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 1024 #1024 works well so far. 2048 is too big for the GPU memory #CHANGED FROM 128 for full res training 30K # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4  # useless
        self.defor_depth = 2  # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10  # useless
        self.scale_rotation_pe = 2  # useless
        self.opacity_pe = 2
        self.density_pe = 2  # useless
        self.timenet_width = 64  # useless
        self.timenet_output = 32  # useless
        self.bounds = 1.6
        
        self.warmup_its = 5_000 # Amount of iterations where only the static gaussians will be trained

        self.use_deform_lr_scheduler = True

        self.pos_deform_lr_init = 0.00016#0.00016
        self.pos_deform_lr_final = 0.000016
        self.pos_deform_lr_max_steps = 30_000

        self.scales_deform_lr_init = 0.01#0.005
        self.scales_deform_lr_final = 0.0005
        self.scales_deform_lr_max_steps = 30_000
        
        self.rotations_deform_lr_init = 0.002
        self.rotations_deform_lr_final = 0.0001
        self.rotations_deform_lr_max_steps = 30_000
        
        self.feature_out_lr_init = 0.00016
        self.feature_out_lr_final = 0.000016
        self.feature_out_lr_max_steps = 30_000
        
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016
        self.grid_lr_max_steps = 30_000

        self.lambda_geo = 0.001 #0.001  # Weight for geometric regularization loss
        self.geom_loss_from_iter = 6000 # Iteration to start applying geometric regularization loss
        self.geom_loss_interval = 1 # Interval (in iterations) to apply geometric regularization loss

        self.lambda_velocity_coherence = 0.000001  # Weight for velocity coherence loss
        self.velocity_coherence_from_iter = 10000  # Iteration to start applying velocity
        self.velocity_coherence_interval = 1  # Interval (in iterations) to apply velocity coherence loss

        self.plane_tv_weight = 0.0001  # TV loss of spatial grid
        self.time_smoothness_weight = 0#0.01  # TV loss of temporal grid #CHANGE TO 0 TO REMOVE K-PLANE SMOOTHING LOSS.
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        
        self.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [128, 128, 128, 10] #Original grid resolution: [64,64,64,50] It is advised to use temporal resolution equal to the amount of simulated frames (half didnt work fsr)
        }
        self.multires = [1, 2, 4, 8]  # multi resolution of voxel grid
        self.no_dx = False  # cancel the deformation of Gaussians' position
        self.no_grid = False  # cancel the spatial-temporal hexplane.
        self.no_ds = False  # cancel the deformation of Gaussians' scaling
        self.no_dr = False  # cancel the deformation of Gaussians' rotations
        self.no_do = True  # cancel the deformation of Gaussians' opacity
        self.empty_voxel = False  # useless
        self.grid_pe = 0  # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp = False  # useless
        self.apply_rotation = False  # useless


        super().__init__(parser, "Deformation Parameters")
    
    def extract(self, args):
        g = super().extract(args)
        # Add the kplanes_config that the deform_network needs
        g.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64,64,64,50]
        }
        g.multires = [1, 2, 4, 8]
        return g


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
