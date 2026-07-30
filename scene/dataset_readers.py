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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # for "OSError: unrecognized data stream contents when reading image file"
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud
from natsort import natsorted
import glob
import random
import torch.nn.functional as F
import torch
import os
from utils.system_utils import searchForMaxIteration
from enum import Enum
from utils.general_utils import build_rotation

MAX_N_POINTS = 700_000

class PinkColor(float, Enum):
    R = 188/255.0
    G = 99/255.0
    B = 76/255.0

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

def readColonSceneInfo(path):
    
    images_folder = os.path.join(path)
    campose_folder = os.path.join(path.replace("C3VD", "C3VD_endogslam_optimized")) 

    params_path = os.path.join(campose_folder, "params.npz") 

    found_iter = searchForMaxIteration(os.path.join(campose_folder, "point_cloud"))
    point_cloud_path = os.path.join(campose_folder, "point_cloud", f"iteration_{found_iter}", "point_cloud.ply")

    # path to save auxiliary ply file (slightly converted point_cloud.ply required by original gaussians)
    ply_path = os.path.join(campose_folder, "point_cloud", f"iteration_{found_iter}", "point_cloud_auxiliary.ply")

    
    # read camera parameters
    params = np.load(params_path) 
    color_paths = natsorted(glob.glob(f"{images_folder}/color/*.png"))
    depth_paths = natsorted(glob.glob(f"{images_folder}/depth/*.tiff"))

    
    train_cameras = []

    for time_idx in range(len(params["gt_w2c_all_frames"])):
        # Get the estimated rotation & translation
        curr_cam_rot = F.normalize(torch.tensor(params['cam_unnorm_rots'][..., time_idx]))
        curr_cam_tran = params['cam_trans'][..., time_idx]
        R = build_rotation(curr_cam_rot).transpose(1, 2).squeeze().cpu().numpy()
        T = curr_cam_tran

        k = params["intrinsics"]
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]

        image = Image.open(color_paths[time_idx])
        orig_w, orig_h = image.size
        depth = np.array(Image.open(depth_paths[time_idx]), dtype=np.float64)
        depth = depth / 2.55 # depth scale for C3VD
        depth =(depth * 255).astype(np.float32)
        depth = Image.fromarray(depth) 
    
        FovY = focal2fov(fy, orig_h)
        FovX = focal2fov(fx, orig_w)

        # train_cameras.append(CameraInfo(
        #     uid=time_idx,
        #     R=R,
        #     T=T,
        #     FovY=FovY,
        #     FovX=FovX,
        #     image=image,
        #     depth=depth,
        #     width=orig_w,
        #     height=orig_h,
        #     image_path=color_paths[time_idx],
        #     image_name=f"{time_idx}",
        #     depth_path=depth_paths[time_idx],
        #     depth_name="",
        #     cx=cx,
        #     cy=cy,
        # ))
        
        train_cameras.append(CameraInfo(
            uid=time_idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=None, # 원래 정의에 있던 필드
            image_path=color_paths[time_idx],
            image_name=f"{time_idx}",
            depth_path=depth_paths[time_idx],
            width=orig_w,
            height=orig_h,
            is_test=False
        ))
    
    num_cameras = len(train_cameras)
    all_idx = set(range(num_cameras))
    eval_idx = set(range(7, num_cameras, 8)) # we don't expect eval in early frames. We dont do eval now, params npz only for train

    train_idx = all_idx - eval_idx
    eval_idx = sorted(list(eval_idx))
    train_idx = sorted(list(train_idx))
    train_cameras_final = [train_cameras[i] for i in train_idx]
    test_cameras_final = [train_cameras[i] for i in eval_idx]

    random.seed(42)
    random.shuffle(train_cameras_final)
    random.shuffle(test_cameras_final)

    # normalization
    nerf_normalization = getNerfppNorm(train_cameras_final)
    
    # point cloud
    init_pt_cld = PlyData.read(point_cloud_path)
    f_dc_0 = np.array(init_pt_cld['vertex'].data['f_dc_0'])
    f_dc_1 = np.array(init_pt_cld['vertex'].data['f_dc_1'])
    f_dc_2 = np.array(init_pt_cld['vertex'].data['f_dc_2'])
    shs = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)
    rgb_colors = SH2RGB(shs)
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    fixed_ratio = 0.5 #we can even set 1 and start from orange values. Should be ok
    rgb_colors[:,0] = rgb_colors[:,0]*(1-fixed_ratio) + PinkColor.R * fixed_ratio
    rgb_colors[:,1] = rgb_colors[:,1]*(1-fixed_ratio) + PinkColor.G * fixed_ratio
    rgb_colors[:,2] = rgb_colors[:,2]*(1-fixed_ratio) + PinkColor.B * fixed_ratio
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    x = np.array(init_pt_cld['vertex'].data['x'])
    y = np.array(init_pt_cld['vertex'].data['y'])
    z = np.array(init_pt_cld['vertex'].data['z'])
    xyz = np.stack([x, y, z], axis=1)

    # Limit to max N init points
    total_init_points = min(xyz.shape[0], MAX_N_POINTS)  
    indices = np.random.choice(xyz.shape[0], total_init_points, replace=False)
    sampled_xyz = xyz[indices]
    sampled_rgb_colors = rgb_colors[indices]

    pcd = BasicPointCloud(points=sampled_xyz, colors=sampled_rgb_colors, normals=np.zeros((len(sampled_xyz), 3)))
    storePly(ply_path, sampled_xyz, sampled_rgb_colors * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cameras_final,
                           test_cameras=test_cameras_final,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "colon": readColonSceneInfo, #Dataset from endoGslam
}