#!/usr/bin/env python3
"""sample_step2a_window.py -- step_2a (MotionBERT pass 1) with the temporal window as an option.

step_2a feeds MotionBERT an 81-frame window at the video's 60 fps (1.35 s) and keeps the centre
frame.  The mesh head was fine-tuned on 16-frame clips at 30 / 50 fps (0.3-0.5 s).  This variant
builds, for EVERY output frame, a window of --clip-len frames sampled every --stride input frames
and centred on that frame, so the output stays at the full frame rate while the network sees the
tempo it was trained at (e.g. --clip-len 16 --stride 2 on 60 fps video = 16 frames at 30 fps).

--crop trial   normalise the 2D once over the whole trial, as step_2a does (default)
--crop window  normalise each window on its own box, as the training clips were

Writes the same {cam}_betas.npz as step_2a into OUT_DIR (BIOCV_OUT), so step_2b/3/4/8 run on it
unchanged.  Nothing under pipeline/ changes.

    BIOCV_OUT=<variant OUT_DIR> ~/anaconda3/envs/motEnv/bin/python sample_step2a_window.py \\
        --detector yolo --user P03 --action P03_CMJM_01 --cameras 00,05,07 --clip-len 16 --stride 2
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pipeline'))
import step_2a_extract_betas as s2a                      # noqa: E402  (loads MotionBERT paths via config)
from scipy.spatial.transform import Rotation as Rot     # noqa: E402


def extract(user, action, camera, mesh_model, device, args):
    h36m_2d = np.load(s2a._twod_path(user, action, args.detector, camera))['h36m_2d']
    n_frames = h36m_2d.shape[0]
    n_missing = int((h36m_2d[:, :, 2] <= 0).sum())
    if args.missing == 'fill' and n_missing:
        h36m_2d = s2a.fill_missing(h36m_2d)
    h36m_2d = h36m_2d.astype(np.float32)
    trial_norm = s2a.crop_scale(h36m_2d, scale_range=[1, 1]) if args.crop == 'trial' else None

    L, S = args.clip_len, args.stride
    half = L // 2
    batch = torch.zeros(1, L, 17, 3, dtype=torch.float32, device=device)
    all_rotmats = np.zeros((n_frames, 24, 3, 3), dtype=np.float32)
    all_betas = np.zeros((n_frames, 10), dtype=np.float32)
    for f in range(n_frames):
        idx = [s2a._reflect_index(f + (j - half) * S, n_frames) for j in range(L)]
        win = trial_norm[idx] if trial_norm is not None else s2a.crop_scale(h36m_2d[idx], scale_range=[1, 1])
        with torch.no_grad():
            batch.copy_(torch.from_numpy(win[None]))
            feat_c = s2a.run_backbone(mesh_model, batch, L, device)      # centre frame (index L//2 == f)
            out = mesh_model.head(feat_c.float())
        theta = out[0]['theta'][0].cpu().numpy()
        all_rotmats[f] = Rot.from_rotvec(theta[:72].reshape(24, 3)).as_matrix().astype(np.float32)
        all_betas[f] = theta[72:]
    out_path = s2a._betas_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, rotmats=all_rotmats, betas=all_betas, avg_betas=np.mean(all_betas, axis=0),
             detector=np.array(args.detector), missing=np.array(args.missing), n_missing_joints=n_missing,
             window=np.array(f'clip_len={L} stride={S} crop={args.crop}'))
    return f'{n_frames} frames, window {L} x stride {S} ({L * S / 60:.2f} s at 60 fps), crop per {args.crop}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detector', choices=s2a.DETECTORS, default='openpose')
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cameras', required=True, help='comma-separated')
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--stride', type=int, default=2, help='input frames between window samples')
    ap.add_argument('--crop', choices=['trial', 'window'], default='trial')
    ap.add_argument('--missing', choices=['fill', 'zero'], default='fill')
    args = ap.parse_args()
    s2a.require_out_dir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = s2a.load_model(device)
    for cam in args.cameras.split(','):
        print(f'{args.user}/{args.action} {cam}: {extract(args.user, args.action, cam, model, device, args)}', flush=True)


if __name__ == '__main__':
    main()
