#!/usr/bin/env python3
"""
step_2a_extract_betas.py — CUDA-only pass: 
Motionbert pass 1 to extract betas for averaging

run MotionBERT-Mesh (whole-clip
crop_scale, per-frame windowed inference) on a camera's already-extracted
2D detections, saving just the betas and rotations theta

Runs inside the motor-dev container:
    docker run --rm --runtime nvidia \
      -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
      -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
      motor-dev:latest \
      python3 step_2a_extract_betas.py --user User28 --action P28_CMJM_01
"""

import os
import sys
import argparse

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
from utils.missing_joints import fill_missing

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



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', default='User03')
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--missing', choices=['fill', 'zero'], default='fill',
                    help='joints the detector did not return: filled from the nearest seen frame (default), '
                         'or left at (0,0,0) as before -- which crop_scale puts in the corner of the crop '
                         '(see utils/missing_joints.py; zero is kept for tools/ladder.py)')
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

    for camera in cameras:
        path_2d = os.path.join(_input_dir, f'{camera}_2d.npz')
        if not os.path.exists(path_2d):
            print(f"=== camera {camera}: no cached h36m_2d, skipping ===")
            continue
        print(f"=== camera {camera} ===")
        yolo_2d = np.load(path_2d)
        h36m_2d = yolo_2d['h36m_2d']
        n_frames = h36m_2d.shape[0]
        n_missing = int((h36m_2d[:, :, 2] <= 0).sum())
        if args.missing == 'fill' and n_missing:
            h36m_2d = fill_missing(h36m_2d)
            print(f"  {n_missing} missing joints ({100 * n_missing / (n_frames * 17):.1f}%) filled from the nearest seen frame")

        motion_norm = crop_scale(h36m_2d.astype(np.float32), scale_range=[1, 1])
        batch = torch.zeros(1, _CLIP_LEN, 17, 3, dtype=torch.float32, device=device)
        half = _CLIP_LEN // 2
        
        all_rotmats = np.zeros((n_frames, 24, 3, 3), dtype=np.float32)
        all_betas = np.zeros((n_frames,10),dtype=np.float32)

        for frame_n in range(n_frames):
            indices = [_reflect_index(frame_n - half + j, n_frames) for j in range(_CLIP_LEN)] # array of indices for this window (accounting for relfecting at boundaries)
            win_np = motion_norm[indices] # pulling batch of numpy 2D keypoints for frames in window

            with torch.no_grad():
                batch.copy_(torch.from_numpy(win_np[None])) 
                feat_center = run_backbone(mesh_model, batch, _CLIP_LEN, device) # Runs the DSTformer backbone over the full 81-frame window 
                            # (so every frame's representation is informed by 40 frames of context on each side), then slices out just the 
                            # single feature vector for the center frame (clip_len//2 = 40, i.e. frame_n itself). 
                smpl_out = mesh_model.head(feat_center.float()) # Feeds that one frame's feature vector through the SMPL regression head, producing a dict with the mesh model's outputs for this single frame: 3D keypoints and SMPL pose parameters.

            pose_aa = smpl_out[0]['theta'][0, :72].cpu().numpy().reshape(24, 3) # pose in axis-angle format
            all_rotmats[frame_n] = _Rot.from_rotvec(pose_aa).as_matrix().astype(np.float32)

            betas = smpl_out[0]['theta'][0, 72:].cpu().numpy()
            all_betas[frame_n] = betas

            if frame_n % 100 == 0:
                print(f"  {frame_n}/{n_frames}")

        avg_betas = np.mean(all_betas,axis = 0)

        out_path = os.path.join(_output_dir, f'{camera}_betas.npz')
        np.savez(out_path, rotmats=all_rotmats, betas = all_betas, avg_betas = avg_betas)
        print(f"  -> {out_path}")

    

if __name__ == '__main__':
    main()
