#!/usr/bin/env python3
"""step_2a_extract_betas.py -- MotionBERT pass 1: per-frame SMPL rotations and betas.

Runs MotionBERT-Mesh (whole-clip crop_scale, per-frame 81-frame windowed
inference) on one detector's 2D and keeps just the rotations (theta) and the
per-frame betas; step_2b picks the body shape from them and step_3 builds the
skeleton with it.

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
    writes  {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz

--detector picks whose 2D goes in (openpose, the default, or yolo); every
later step takes the same argument and writes to a matching subfolder.  With
no --user / --action it does every camera that has 2D; cameras already done
are skipped (--force redoes them).  Needs a GPU in practice.

Missing joints (confidence 0) are filled from the nearest seen frame before
MotionBERT sees them -- see utils/missing_joints.py; --missing zero keeps
them at (0,0,0) as the pipeline once did (tools/ladder.py's rung C).

    python3 step_2a_extract_betas.py                                   # everything, OpenPose
    python3 step_2a_extract_betas.py --detector yolo --user User03 --action P03_CMJM_01
    python3 step_2a_extract_betas.py --dry-run
"""

import argparse
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
from config import DETECTORS, twod_path as _twod_path, betas_path as _betas_path, require_out_dir, OUT_DIR as _OUT_DIR, find_cameras
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


def extract(user, action, camera, mesh_model, device, args):
    """One camera's 2D -> {cam}_betas.npz.  Returns a summary line."""
    h36m_2d = np.load(_twod_path(user, action, args.detector, camera))['h36m_2d']
    n_frames = h36m_2d.shape[0]
    n_missing = int((h36m_2d[:, :, 2] <= 0).sum())
    if args.missing == 'fill' and n_missing:
        h36m_2d = fill_missing(h36m_2d)

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
        all_betas[frame_n] = smpl_out[0]['theta'][0, 72:].cpu().numpy()

    out_path = _betas_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, rotmats=all_rotmats, betas=all_betas, avg_betas=np.mean(all_betas, axis=0),
             detector=np.array(args.detector), missing=np.array(args.missing), n_missing_joints=n_missing)
    return (f'{n_frames} frames' + (f', {n_missing} missing joints ({100 * n_missing / (n_frames * 17):.1f}%) '
                                    f'{"filled" if args.missing == "fill" else "left at zero"}' if n_missing else ''))


def main():
    ap = argparse.ArgumentParser(description='MotionBERT pass 1 (rotations + betas). With no --user/--action, '
                                             'every camera with 2D from --detector; cameras already done are skipped.')
    ap.add_argument('--detector', choices=DETECTORS, default='openpose', help="whose 2D goes in (default openpose)")
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with 2D')
    ap.add_argument('--missing', choices=['fill', 'zero'], default='fill',
                    help='joints the detector did not return: filled from the nearest seen frame (default), '
                         'or left at (0,0,0) as before -- which crop_scale puts in the corner of the crop '
                         '(see utils/missing_joints.py; zero is kept for tools/ladder.py)')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be run, load nothing')
    args = ap.parse_args()

    require_out_dir()
    jobs = find_cameras(_twod_path, args.detector, args.user, args.action, set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {args.detector} {{cam}}_2d.npz under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_1_extract_2d.py / step_1_openpose_2d.py first')
    todo = [j for j in jobs if args.force or not os.path.exists(_betas_path(j[0], j[1], args.detector, j[2]))]
    print(f'{args.detector}: {len(jobs)} camera(s) with 2D under {_OUT_DIR}: {len(todo)} to run, '
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
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: {extract(user, action, cam, mesh_model, device, args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
