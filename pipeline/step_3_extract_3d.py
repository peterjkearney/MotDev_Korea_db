#!/usr/bin/env python3
"""
step_2_extract_3d.py — CUDA-only pass: 
Motionbert pass 2: use average betas from pass 1 to extract fixed-proportion vertices
run MotionBERT-Mesh (whole-clip
crop_scale, per-frame windowed inference) on a camera's already-extracted
2D detections, saving keypoints for H36M skeleton and keypoints/rotations for SMPL-24 skeleton

Runs inside the motor-dev container (needs torch/CUDA, not GL/X11):
    docker run --rm --runtime nvidia \
      -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
      -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
      motor-dev:latest \
      python3 step_3_extract_3d.py --user User28 --action P28_CMJM_01
"""

import os
import sys
import argparse
import json

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
from config import TRIAL_DIR as _TRIAL_DIR
from config import MB_DIR as _MB_DIR      # resolved in config.py; ../MotionBERT no longer holds here

for _d in (_MB_DIR, _SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import crop_scale
from lib.model.model_mesh import MeshRegressor

_MB_MESH_CFG  = os.path.join(_MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml')
_MB_MESH_CKPT = os.path.join(_MB_DIR, 'checkpoint', 'mesh',
                               'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin')

_CLIP_LEN = 81

def run_backbone(model, batch, clip_len, device):
    if device == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feat = model.backbone.get_representation(batch)
    else:
        feat = model.backbone.get_representation(batch)
    feat = feat.reshape(1, clip_len, model.feat_J, -1)
    return feat[:, clip_len // 2 : clip_len // 2 + 1, :, :]


def _reflect_index(i, n_frames):
    """
    Map a possibly out-of-range window index to [0, n_frames-1] by
    reflection (mirroring at the boundaries) rather than clamping/repeating
    the edge frame. For clips shorter than CLIP_LEN (routine for these
    short hop clips — see extract_hop_features.py's companion discussion),
    clamping means the window near either edge is dominated by dozens of
    copies of a single frozen frame, a badly out-of-distribution input for
    a transformer trained on continuously-varying motion. Reflection gives
    it continuously-varying (if mirrored) motion instead.
    """
    if n_frames == 1:
        return 0
    period = 2 * (n_frames - 1)
    i = i % period
    if i >= n_frames:
        i = period - i
    return i

def mesh_height_from_betas(smpl, betas, device):
    """
    Crown-to-sole height (metres) of the SMPL mesh for a given betas vector,
    in T-pose (all 24 joint rotations = identity).

    smpl   : mesh_model.head.smpl
    betas  : (10,) numpy or torch
    """
    betas = torch.as_tensor(betas, dtype=torch.float32).reshape(1, 10).to(device)
    ident = torch.eye(3, dtype=torch.float32, device=device).repeat(1, 24, 1, 1)  # (1,24,3,3)
    with torch.no_grad():
        out = smpl(betas=betas,
                   body_pose=ident[:, 1:],        # (1,23,3,3)
                   global_orient=ident[:, 0:1],   # (1,1,3,3)
                   pose2rot=False)
    y = out.vertices[0, :, 1]                     # SMPL canonical frame is y-up
    return float(y.max() - y.min())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', default='User03')
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    args = ap.parse_args()
    cameras = args.cameras.split(',')

    _input_dir = os.path.join(_TRIAL_DIR,args.user,args.action,'Analysis','keypoints')
    _output_dir = os.path.join(_TRIAL_DIR,args.user,args.action,'Analysis','keypoints','mesh')
    os.makedirs(_output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("[init] MotionBERT-Mesh ...")
    mb_args = get_config(_MB_MESH_CFG)
    mb_args.data_root = os.path.join(_MB_DIR, 'data', 'mesh')
    backbone = load_backbone(mb_args)
    mesh_model = MeshRegressor(mb_args, backbone=backbone,
                                dim_rep=mb_args.dim_rep,
                                hidden_dim=mb_args.hidden_dim,
                                dropout_ratio=mb_args.dropout)
    ckpt = torch.load(_MB_MESH_CKPT, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    mesh_model.load_state_dict(sd, strict=True)
    mesh_model.eval()
    if device == 'cuda':
        mesh_model = mesh_model.cuda()
    print(f"       mesh on {device}")

    # Known height of the subject (crown to sole).  The mesh's own T-pose
    # height is computed per camera below, since betas are per camera now.
    user_meta_path = os.path.join(_TRIAL_DIR,args.user,'user_meta.json')
    if not os.path.exists(user_meta_path):
        raise FileNotFoundError(
        f"{user_meta_path} not found -- step_3 needs the subject's stature "
        f"(create it with {{\"stature_m\": <height in metres>}})")
    with open(user_meta_path) as f:
        user_meta = json.load(f)
    if 'stature_m' not in user_meta:
        raise KeyError(f"{user_meta_path} has no 'stature_m' field")
    actual_height = user_meta['stature_m']

    for camera in cameras:

        # Betas from pass 1, for THIS camera only (step_2b no longer averages
        # across views): shape evidence comes from the same single view the
        # skeleton is placed from.
        fb_path = os.path.join(_output_dir, f'{camera}_final_betas.npz')
        if not os.path.exists(fb_path):
            print(f"=== camera {camera}: no {fb_path}, skipping ===")
            continue
        final_betas = torch.from_numpy(np.load(fb_path)['betas']).reshape(1, 10).to(device)
        mesh_height = mesh_height_from_betas(mesh_model.head.smpl, final_betas, device)
        mb_scale_factor = actual_height / mesh_height
        print(f"  mesh T-pose height {mesh_height:.3f} m, subject {actual_height:.3f} m "
              f"-> scale x{mb_scale_factor:.3f}")

        # Loading rotmats from pass 1
        beta_cam_path = os.path.join(_output_dir,f'{camera}_betas.npz')
        if not os.path.exists(beta_cam_path):
            print(f"=== camera {camera}: no cached betas, skipping ===")
            continue
        rotmats_pass1 = torch.from_numpy(np.load(beta_cam_path)['rotmats'])
       

        path_2d = os.path.join(_input_dir, f'{camera}_2d.npz')
        if not os.path.exists(path_2d):
            print(f"=== camera {camera}: no cached h36m_2d, skipping ===")
            continue
        print(f"=== camera {camera} ===")
        yolo_2d = np.load(path_2d)
        h36m_2d = yolo_2d['h36m_2d']
        n_frames = h36m_2d.shape[0]

        # motion_norm = crop_scale(h36m_2d.astype(np.float32), scale_range=[1, 1])
        # batch = torch.zeros(1, _CLIP_LEN, 17, 3, dtype=torch.float32, device=device)
        # half = _CLIP_LEN // 2
        all_kp_H36M_scaled = np.zeros((n_frames, 17, 3), dtype=np.float32)
        all_kp_SMPL24_scaled = np.zeros((n_frames, 24, 3), dtype=np.float32)

        for frame_n in range(n_frames):

            # Loading rotmats from pass 1
            pred_rotmats = rotmats_pass1[frame_n,1:]
            pred_rotmats = pred_rotmats.unsqueeze(0)
            pred_rotmats= pred_rotmats.to(device)
            global_orient = rotmats_pass1[frame_n, 0:1]
            global_orient = global_orient.unsqueeze(0)
            global_orient = global_orient.to(device)

            with torch.no_grad():
                smpl_out = mesh_model.head.smpl(betas=final_betas,body_pose=pred_rotmats,global_orient=global_orient,pose2rot=False)
            
            kp_H36M_raw = (mesh_model.head.smpl.J_regressor_h36m @ smpl_out.vertices)[0].cpu().numpy() # extracting H36M keypoints from mesh vertices
            all_kp_H36M_scaled[frame_n] = kp_H36M_raw * mb_scale_factor

            kp_SMPL24_raw = (mesh_model.head.smpl.J_regressor @ smpl_out.vertices)[0].cpu().numpy() # extracting SMPL-24 keypoints from mesh vertices
            all_kp_SMPL24_scaled[frame_n] = kp_SMPL24_raw * mb_scale_factor


            if frame_n % 100 == 0:
                print(f"  {frame_n}/{n_frames}")

        out_path = os.path.join(_output_dir, f'{camera}_mesh_pose.npz')
        np.savez(out_path, kps_H36M_scaled=all_kp_H36M_scaled, kps_SMPL24_scaled = all_kp_SMPL24_scaled, rotmats=rotmats_pass1.numpy(), scale_factor = mb_scale_factor,
                 mesh_height_m=mesh_height, stature_m=actual_height, betas=final_betas.cpu().numpy().ravel())
        print(f"  -> {out_path}")


if __name__ == '__main__':
    main()
