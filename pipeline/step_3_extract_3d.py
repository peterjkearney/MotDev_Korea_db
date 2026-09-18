#!/usr/bin/env python3
"""step_3_extract_3d.py -- MotionBERT pass 2: the skeleton with a fixed body shape.

SMPL is run with step_2b's betas (one shape per camera) and step_2a's per-frame
rotations; the H36M-17 and SMPL-24 joints are regressed from the mesh and
scaled so the mesh's T-pose height matches the subject's stature.

    reads   {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz         (rotmats)
            {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_final_betas.npz
            {user}/user_meta.json  {"stature_m": ...}                                   (TRIAL_DIR, else OUT_DIR)
    writes  {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_mesh_pose.npz

With no --user / --action it does every camera with final betas from
--detector; cameras already done are skipped (--force redoes them).  Needs a
GPU in practice.

    python3 step_3_extract_3d.py                          # everything, OpenPose
    python3 step_3_extract_3d.py --detector yolo --user User03
"""

import argparse
import json
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
from config import (DETECTORS, OUT_DIR as _OUT_DIR, betas_path as _betas_path, final_betas_path as _final_path,
                    mesh_pose_path as _mesh_path, stature_path, find_cameras, require_out_dir)
from config import MB_DIR as _MB_DIR      # resolved in config.py; ../MotionBERT no longer holds here

from config import require_motionbert
require_motionbert()
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

def load_model(device):
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
    return mesh_model


def stature_of(user, cache={}):
    """Crown-to-sole height (m) from user_meta.json, read once per user."""
    if user not in cache:
        p = stature_path(user)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found -- step_3 needs the subject's stature "
                                    f"(create it with {{\"stature_m\": <height in metres>}})")
        with open(p) as f:
            cache[user] = float(json.load(f)['stature_m'])
    return cache[user]


def build(user, action, camera, mesh_model, device, args):
    """One camera -> {cam}_mesh_pose.npz.  Returns a summary line."""
    actual_height = stature_of(user)
    # Betas from pass 1, for THIS camera only (step_2b no longer averages across views):
    # shape evidence comes from the same single view the skeleton is placed from.
    final_betas = torch.from_numpy(np.load(_final_path(user, action, args.detector, camera))['betas']).reshape(1, 10).to(device)
    mesh_height = mesh_height_from_betas(mesh_model.head.smpl, final_betas, device)
    mb_scale_factor = actual_height / mesh_height
    rotmats_pass1 = torch.from_numpy(np.load(_betas_path(user, action, args.detector, camera))['rotmats'])
    n_frames = rotmats_pass1.shape[0]

    all_kp_H36M_scaled = np.zeros((n_frames, 17, 3), dtype=np.float32)
    all_kp_SMPL24_scaled = np.zeros((n_frames, 24, 3), dtype=np.float32)
    for frame_n in range(n_frames):
        pred_rotmats = rotmats_pass1[frame_n, 1:].unsqueeze(0).to(device)
        global_orient = rotmats_pass1[frame_n, 0:1].unsqueeze(0).to(device)
        with torch.no_grad():
            smpl_out = mesh_model.head.smpl(betas=final_betas, body_pose=pred_rotmats, global_orient=global_orient, pose2rot=False)
        kp_H36M_raw = (mesh_model.head.smpl.J_regressor_h36m @ smpl_out.vertices)[0].cpu().numpy()   # H36M joints from the mesh
        all_kp_H36M_scaled[frame_n] = kp_H36M_raw * mb_scale_factor
        kp_SMPL24_raw = (mesh_model.head.smpl.J_regressor @ smpl_out.vertices)[0].cpu().numpy()      # SMPL-24 joints from the mesh
        all_kp_SMPL24_scaled[frame_n] = kp_SMPL24_raw * mb_scale_factor

    out_path = _mesh_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, kps_H36M_scaled=all_kp_H36M_scaled, kps_SMPL24_scaled=all_kp_SMPL24_scaled,
             rotmats=rotmats_pass1.numpy(), scale_factor=mb_scale_factor, mesh_height_m=mesh_height,
             stature_m=actual_height, betas=final_betas.cpu().numpy().ravel(), detector=np.array(args.detector))
    return f'{n_frames} frames, mesh T-pose {mesh_height:.3f} m vs subject {actual_height:.2f} m -> scale x{mb_scale_factor:.3f}'


def main():
    ap = argparse.ArgumentParser(description='SMPL skeleton with fixed betas. With no --user/--action, every camera '
                                             'with final betas from --detector; cameras already done are skipped.')
    ap.add_argument('--detector', choices=DETECTORS, default='openpose')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with final betas')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be run, load nothing')
    args = ap.parse_args()

    require_out_dir()
    jobs = find_cameras(_final_path, args.detector, args.user, args.action,
                        set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {args.detector} {{cam}}_final_betas.npz under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_2b_finalise_betas.py first')
    todo = [j for j in jobs if args.force or not os.path.exists(_mesh_path(j[0], j[1], args.detector, j[2]))]
    print(f'{args.detector}: {len(jobs)} camera(s) with final betas under {_OUT_DIR}: {len(todo)} to run, '
          f'{len(jobs) - len(todo)} already done')
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return
    if not todo:
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh_model = load_model(device)
    n_done, failed = 0, []
    for i, (user, action, cam) in enumerate(todo, 1):
        try:
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: {build(user, action, cam, mesh_model, device, args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
