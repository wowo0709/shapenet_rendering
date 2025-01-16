# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import argparse

import blobfile as bf
from blender.render import render_model
from view_data import Front3DBlenderViewData
from point_cloud import PointCloud
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', required=True, type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--rendered_folder', required=True, type=str, default='./tmp',
    help='path for processed rendered images folder')
parser.add_argument(
    '--camera_folder', required=True, type=str, default='./tmp', 
    help='path for cameras used to render images')
parser.add_argument(
    '--num_images', type=int, default=20,
    help='number of rendered images')
parser.add_argument(
    '--num_pts', type=int, default=20480,
    help='number of points to sample from RGBAD images'
)
args = parser.parse_args()

dir_name = args.rendered_folder.split("/")[-1]
for idx, model_id in enumerate(tqdm(os.listdir(args.rendered_folder))):
    out_dir_path = os.path.join(args.save_folder, dir_name, model_id)
    if os.path.exists(out_dir_path) and len(os.listdir(out_dir_path)) > 0:
        if os.listdir(out_dir_path)[0].endswith(".npz") or os.listdir(out_dir_path)[0].endswith(".ply"):
            print(f"{out_dir_path} has already processed!")
            continue
    os.makedirs(out_dir_path, exist_ok=True)

    # rendered_zip_path = os.path.join(args.rendered_folder, cat_id, model_id, f"rendered_images.zip")
    
    render_path = os.path.join(args.rendered_folder, model_id)
    camera_path = os.path.join(args.camera_folder, model_id)

    try:
        vd = Front3DBlenderViewData(render_path=render_path, camera_path=camera_path)
        pc = PointCloud.from_rgbd(vd, args.num_images)
    except Exception as e:
        print("Exception:", e)
        continue
    
    # sampled_pc = pc.farthest_point_sample(args.num_pts) # too long time
    sampled_pc = pc.random_sample(args.num_pts)
    
    # with bf.BlobFile(os.path.join(out_dir_path, f"colored_pc_{args.num_pts}.ply"), "wb") as writer:
    #     sampled_pc.write_ply(writer)
    
    sampled_pc.save(os.path.join(out_dir_path, f"colored_pc_{args.num_pts}.npz"))
        
