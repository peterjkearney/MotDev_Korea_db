#!/usr/bin/env python3
"""head_error.py -- how well each configuration places the head, scored on the points OpenPose detects.

step_8 leaves Nose and Head out (the H36M regressor's nose and head are not OpenPose's points).  Here every
configuration is scored the same way, from its placed MESH: the nose-tip vertex against the target's Nose (OpenPose's
nose) and the mid-ear vertices against its Head (mid-ears), per camera against that camera's leave-one-out target, as
step_8 scores (mid-ears: the all-camera target -- the leave-one-out ones never have both ears).  Also the head
DIRECTION error: the angle between mid-ears -> nose in the mesh and in the target.

    reads   {tree}/{user}/{action}/Analysis/PnP/{detector}/{config}/{cam}_pnp.npz, keypoints/{detector}/{cam}_2d.npz,
            H36M/openpose_tri_h36m.npz

    python analysis/joint_calib/head_error.py <tree> --configs C14,C19,C21
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline', 'tools'))
from render_mesh import placed_vertices  # noqa: E402
from utils.smpl_lite import J, OPRF_FACE  # noqa: E402

NOSE_V = list(OPRF_FACE['Nose'])[0]
EARS_V = list(OPRF_FACE['Head'])


def score_rep(rep_dir, cfg, detector):
    tri = np.load(os.path.join(rep_dir, 'Analysis', 'H36M', 'openpose_tri_h36m.npz'))
    cams = [str(c) for c in tri['cameras']]
    out = []
    for p in sorted(glob.glob(os.path.join(rep_dir, 'Analysis', 'PnP', detector, cfg, '*_pnp.npz'))):
        cam = os.path.basename(p).split('_')[0]
        d = np.load(p, allow_pickle=True)
        V, _, _, ok, _ = placed_vertices(d, 'placed')
        sfi = np.load(os.path.join(rep_dir, 'Analysis', 'keypoints', detector, f'{cam}_2d.npz'))['source_frame_idx'].astype(int)
        T = min(len(V), len(sfi))
        L = d['cam_L_ext']
        G = tri['kps3d_loo'][cams.index(cam)] if 'kps3d_loo' in tri.files else tri['kps3d']
        def to_cam(X):
            g = np.full((T, 17, 3), np.nan)
            inside = sfi[:T] < len(X)
            g[inside] = (X[sfi[:T][inside]] @ L[:3, :3].T + L[:3, 3]) / 1000.0
            return g
        g, g_all = to_cam(G), to_cam(tri['kps3d'])
        nose_m, head_m = V[:T, NOSE_V], V[:T, EARS_V].mean(1)
        # the leave-one-out targets never hold Head (both ears from two cameras): mid-ears from the all-camera one
        nose_t, head_t = g[:, J['Nose']], g_all[:, J['Head']]
        e_nose = np.linalg.norm(nose_m - nose_t, axis=-1) * 1000.0
        e_head = np.linalg.norm(head_m - head_t, axis=-1) * 1000.0
        dm, dt = nose_m - head_m, nose_t - head_t
        cos = (dm * dt).sum(-1) / (np.linalg.norm(dm, axis=-1) * np.linalg.norm(dt, axis=-1))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        good = ok[:T]
        out.append((np.nanmean(e_nose[good]), np.nanmean(e_head[good]), np.nanmedian(ang[good]),
                    np.isfinite(e_head[good]).mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('tree')
    ap.add_argument('--configs', default='C14,C19,C21')
    ap.add_argument('--detector', default='openpose')
    args = ap.parse_args()
    cfgs = args.configs.split(',')
    reps = sorted({os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))
                   for p in glob.glob(os.path.join(args.tree, '*', '*', 'Analysis', 'PnP', args.detector, cfgs[0]))})
    print(f'{"rep":14s} {"cfg":4s} {"nose mm":>8s} {"mid-ears mm":>11s} {"head dir deg":>12s} {"ears seen":>9s}')
    tot = {c: [] for c in cfgs}
    for r in reps:
        for c in cfgs:
            s = score_rep(r, c, args.detector)
            if not s:
                continue
            m = np.nanmean(s, 0)
            tot[c].append(m)
            print(f'{os.path.basename(r):14s} {c:4s} {m[0]:8.1f} {m[1]:11.1f} {m[2]:12.1f} {100 * m[3]:8.0f}%')
    print('mean over reps:')
    for c in cfgs:
        if tot[c]:
            m = np.nanmean(tot[c], 0)
            print(f'  {c:4s} nose {m[0]:6.1f} mm   mid-ears {m[1]:6.1f} mm   head direction {m[2]:5.1f} deg')


if __name__ == '__main__':
    main()
