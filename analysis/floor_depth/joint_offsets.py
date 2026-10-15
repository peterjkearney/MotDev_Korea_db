#!/usr/bin/env python3
"""joint_offsets.py -- systematic per-joint offsets between step_4b's placed joints and the target, with the
pose's placement and orientation removed: per frame a rigid (no scale) Kabsch fit of the placed skeleton
onto the target over the 12 limb joints + Hip/Thorax, then each joint's residual expressed in the TARGET's
body frame (up = Hip->Thorax, lateral = RHip->LHip made orthogonal, forward = up x lateral... sign set so
the nose is forward), median over frames.  A consistent offset is a joint-definition or proportion difference
between the model's joints and the OpenPose-built target; random pose error averages out.

    python3 ../analysis/floor_depth/joint_offsets.py <tree> <config> [<tree> <config> ...]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from utils.smpl_lite import H36M, J, kabsch  # noqa: E402

FIT = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Thorax',
       'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
SHOW = FIT


def offsets(tree, cfg):
    res, ext = [], []
    for p in sorted(glob.glob(f'{tree}/*/*/Analysis/PnP/openpose/{cfg}/*_pnp.npz')):
        u, a = p[len(tree) + 1:].split(os.sep)[:2]
        d = np.load(p, allow_pickle=True)
        R, t = d['cam_L_ext'][:3, :3], d['cam_L_ext'][:3, 3]
        X = d['kps_H36M_placed'].astype(float) * 1000.0                     # camera mm
        T = np.load(f'{tree}/{u}/{a}/Analysis/H36M/openpose_tri_h36m.npz')['kps3d'].astype(float) @ R.T + t
        idx = [J[n] for n in FIT]
        for i in range(len(X)):
            ok = np.isfinite(X[i, idx]).all(1) & np.isfinite(T[i, idx]).all(1)
            if ok.sum() < 12 or not np.isfinite(T[i, [J['Nose'], J['Head']]]).all():
                continue
            P, Q = X[i, idx][ok], T[i, idx][ok]
            Rk, tk = kabsch(P, Q)                                             # Q ~ P @ Rk.T + tk (checked below)
            Y = X[i] @ Rk.T + tk
            up = T[i, J['Thorax']] - T[i, J['Hip']]; up /= np.linalg.norm(up)
            lat = T[i, J['LHip']] - T[i, J['RHip']]; lat -= (lat @ up) * up; lat /= np.linalg.norm(lat)
            fwd = np.cross(up, lat)
            if (T[i, J['Nose']] - T[i, J['Head']]) @ fwd < 0:
                fwd = -fwd
            B = np.stack([lat, up, fwd])                                      # rows: lateral(+L), up, forward
            res.append((Y - T[i]) @ B.T)
            ext.append((np.linalg.norm(Y[J['Thorax']] - Y[J['Hip']]), np.linalg.norm(T[i, J['Thorax']] - T[i, J['Hip']])))
    return np.array(res), np.array(ext)


args = sys.argv[1:]
out = {}
for k in range(0, len(args), 2):
    out[f'{os.path.basename(args[k])}:{args[k + 1]}'] = offsets(args[k], args[k + 1])
labels = list(out)
print('median offset, placed minus target, after a per-frame rigid fit (mm): lateral (+ toward the left side) / up the trunk / forward')
print(f'{"joint":10s} ' + ' | '.join(f'{l:>26s}' for l in labels))
for n in SHOW:
    cells = []
    for l in labels:
        r = out[l][0][:, J[n]]
        m = np.nanmedian(r, 0)
        cells.append(f'{m[0]:+7.0f} {m[1]:+7.0f} {m[2]:+7.0f}   ')
    print(f'{n:10s} ' + ' | '.join(f'{c:>26s}' for c in cells))
for l in labels:
    e = out[l][1]
    print(f'{l}: {len(out[l][0])} frames; Hip->Thorax placed {np.median(e[:, 0]):.0f} vs target {np.median(e[:, 1]):.0f} mm')
