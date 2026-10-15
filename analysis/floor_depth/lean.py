#!/usr/bin/env python3
"""lean.py -- whole-body lean of step_4b's placed skeleton relative to a target, in the BODY frame.

Per frame: translate the placed skeleton so its ankle midpoint sits on the target's, then take the shoulder
midpoint's offset along the target's forward axis (forward = (RHip->LHip) x up, up = lab Z) and lateral axis;
+forward = the model's upper body ahead of the subject's (model leans forward).  Also the same offset along
the camera's viewing axis (+ = further from the camera), as variant_summary.py's 'lean'.  Medians per camera.
Frames are matched through source_frame_idx, so 60 fps 2D against 200 fps mocap works.

    python3 ../analysis/floor_depth/lean.py <OUT_DIR> <config> [--target mocap|tri]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from utils.smpl_lite import J  # noqa: E402

SH, AN = ('RShoulder', 'LShoulder'), ('RAnkle', 'LAnkle')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('config')
    ap.add_argument('--target', choices=['mocap', 'tri'], default='tri')
    ap.add_argument('--detector', default='openpose')
    args = ap.parse_args()
    tname = 'mocap_h36m.npz' if args.target == 'mocap' else 'openpose_tri_h36m.npz'
    allf, allc = [], []
    for p in sorted(glob.glob(f'{args.out_dir}/*/*/Analysis/PnP/{args.detector}/{args.config}/*_pnp.npz')):
        u, a = p[len(args.out_dir) + 1:].split(os.sep)[:2]
        cam = os.path.basename(p).split('_')[0]
        base = f'{args.out_dir}/{u}/{a}/Analysis'
        d = np.load(p, allow_pickle=True)
        tz = np.load(f'{base}/H36M/{tname}')
        src = np.load(f'{base}/keypoints/{args.detector}/{cam}_2d.npz')['source_frame_idx']
        R, t = d['cam_L_ext'][:3, :3], d['cam_L_ext'][:3, 3]
        X = d['kps_H36M_placed'].astype(float) * 1000.0
        X = (X - t) @ R                                                     # camera mm -> lab mm
        n = min(len(X), len(src))
        _, ix, it = np.intersect1d(src[:n], tz['source_frame_idx'], return_indices=True)
        X, T = X[ix], tz['kps3d'][it].astype(float)
        sh = lambda Y: Y[:, [J[j] for j in SH]].mean(1)
        an = lambda Y: Y[:, [J[j] for j in AN]].mean(1)
        off = (sh(X) - an(X)) - (sh(T) - an(T))                             # model minus target, ankles aligned
        lat = T[:, J['LHip']] - T[:, J['RHip']]
        lat[:, 2] = 0
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        fwd = np.cross(lat, [0, 0, 1.0])
        view = R[2]                                                         # camera z axis in lab
        ok = np.isfinite(off).all(1) & np.isfinite(fwd).all(1)
        f = np.median((off[ok] * fwd[ok]).sum(1))
        s = np.median((off[ok] * lat[ok]).sum(1))
        v = np.median(off[ok] @ view)
        az = np.degrees(np.arctan2(*(np.median(fwd[ok], 0)[[1, 0]])))
        cam_dir = np.degrees(np.arctan2(-view[1], -view[0]))                # direction from subject to camera
        print(f'{a} cam{cam}: {ok.sum():4d} frames | forward {f:+5.0f} mm, left {s:+5.0f} mm | away from camera {v:+5.0f} mm | '
              f'camera at {((cam_dir - az + 180) % 360) - 180:+5.0f} deg from facing')
        allf.append(f)
        allc.append(v)
    print(f'mean over {len(allf)} cameras: forward {np.mean(allf):+.0f} mm, away from camera {np.mean(allc):+.0f} mm')


if __name__ == '__main__':
    main()
