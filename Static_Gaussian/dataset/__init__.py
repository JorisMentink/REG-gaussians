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
import random
import numpy as np
import os.path as osp
import torch
import json

sys.path.append("./")
from Static_Gaussian.gaussian import GaussianModel
from Static_Gaussian.arguments import ModelParams
from Static_Gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from Static_Gaussian.utils.camera_utils import cameraList_from_camInfos
from Static_Gaussian.utils.general_utils import t2a


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_is_dynamic = None

        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Peek into meta_data.json to decide loader
            with open(osp.join(args.source_path, "meta_data.json"), "r") as f:
                meta_data = json.load(f)
            if "gt_vols" in meta_data:
                loader_key = "DynBlender"
                self.scene_is_dynamic = True
            else:
                loader_key = "Blender"
                self.scene_is_dynamic = False
            
            scene_info = sceneLoadTypeCallbacks[loader_key](
                args.source_path,
                args.eval,
            )

            self.info = scene_info

        #Useless elif cause we dont use this but we ball.
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        if not self.scene_is_dynamic:
            self.vol_gt = scene_info.vol
        else:
            self.vol_gt = scene_info.gt_vols[0]  # Use t=0 for initialization

        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras