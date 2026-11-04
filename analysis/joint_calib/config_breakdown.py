#!/usr/bin/env python3
"""config_breakdown.py -- step_8's scores split by joint group, plus the head's pitch against gravity, per configuration.

Body (from step_8's error_metrics_tri.npz, mean over cameras): placed and PA error over every scored joint, over the
limbs (knees, ankles, elbows, wrists) and over the hips + shoulders -- the joints whose definition differs between
joint sets.  Head: the pitch of the mesh's ears -> nose direction above the floor, minus the target's (all-camera
triangulation; + = model head tipped up), median over frames -- the same reference for every configuration, so joint
definitions cannot skew it.  --face picks the mesh's face points: 'smplx' the single vertices (biased on children:
the nose -> mid-ears line sits 4-7.5 deg off OpenPose's, face_check.py) or 'calibrated' the face regressor
(smpl_lite.opr_rows(face=True)) at the body's kid weight.

    python analysis/joint_calib/config_breakdown.py <tree> --configs C8,C11,C12,C14,C19
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
from utils.smpl_lite import J, OPRF_FACE, opr_rows  # noqa: E402

LIMBS = ['RKnee', 'RAnkle', 'LKnee', 'LAnkle', 'LElbow', 'LWrist', 'RElbow', 'RWrist']
HS = ['RHip', 'LHip', 'LShoulder', 'RShoulder']
NOSE_V, EARS_V = list(OPRF_FACE['Nose'])[0], list(OPRF_FACE['Head'])


def body(rep_dir, cfg, detector):
    p = os.path.join(rep_dir, 'Analysis', 'diagnostics', detector, cfg, 'error_metrics_tri.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    jn = list(z['joint_names'])
    li, hi = [jn.index(j) for j in LIMBS], [jn.index(j) for j in HS]
    pj, pa = z['perjoint_placed_all'], z['perjoint_pa']
    return [np.nanmean(z['err_placed_all']), np.nanmean(z['pa_mpjpe']), np.nanmean(pj[:, li]), np.nanmean(pj[:, hi]),
            np.nanmean(pa[:, li])]


def head_pitch(rep_dir, cfg, detector, face='smplx'):
    tri = np.load(os.path.join(rep_dir, 'Analysis', 'H36M', 'openpose_tri_h36m.npz'))
    errs = []
    for p in sorted(glob.glob(os.path.join(rep_dir, 'Analysis', 'PnP', detector, cfg, '*_pnp.npz'))):
        cam = os.path.basename(p).split('_')[0]
        d = np.load(p, allow_pickle=True)
        V, _, _, ok, _ = placed_vertices(d, 'placed')
        if face == 'calibrated':
            sh = np.asarray(d['shape']).ravel()
            f = opr_rows(float(sh[10]) if sh.size > 10 else 0.0, face=True)
            nose = np.einsum('tvc,v->tc', V, f['Nose'])
            ears = np.einsum('tvc,v->tc', V, 0.5 * (f['LEar'] + f['REar']))
        else:
            nose, ears = V[:, NOSE_V], V[:, EARS_V].mean(1)
        sfi = np.load(os.path.join(rep_dir, 'Analysis', 'keypoints', detector, f'{cam}_2d.npz'))['source_frame_idx'].astype(int)
        T = min(len(V), len(sfi))
        L = d['cam_L_ext']
        up = L[:3, :3][:, 2]                                       # lab +z in camera coordinates
        G = tri['kps3d']
        g = np.full((T, 17, 3), np.nan)
        inside = sfi[:T] < len(G)
        g[inside] = (G[sfi[:T][inside]] @ L[:3, :3].T + L[:3, 3]) / 1000.0

        def pitch(v):
            return np.degrees(np.arcsin((v @ up) / np.linalg.norm(v, axis=-1)))
        e = pitch(nose[:T] - ears[:T]) - pitch(g[:, J['Nose']] - g[:, J['Head']])
        errs.append(e[ok[:T] & np.isfinite(e)])
    return np.concatenate(errs) if errs else np.array([])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('tree')
    ap.add_argument('--configs', default='C8,C11,C12,C14,C19')
    ap.add_argument('--detector', default='openpose')
    ap.add_argument('--face', choices=('smplx', 'calibrated'), default='smplx', help="the mesh's face points for head pitch")
    args = ap.parse_args()
    cfgs = args.configs.split(',')
    reps = sorted({p.split(os.sep + 'Analysis')[0]
                   for p in glob.glob(os.path.join(args.tree, '*', '*', 'Analysis', 'PnP', args.detector, '*'))})
    print(f'{"config":7s} {"placed":>7s} {"PA":>6s} {"limbs":>6s} {"hips+sh":>8s} {"PA limbs":>9s}   head pitch vs gravity')
    for c in cfgs:
        B = [b for b in (body(r, c, args.detector) for r in reps) if b is not None]
        H = [head_pitch(r, c, args.detector, args.face) for r in reps]
        H = np.concatenate([h for h in H if h.size]) if any(h.size for h in H) else np.array([])
        bm = np.mean(B, 0) if B else [np.nan] * 5
        hp = f'{np.median(H):+6.1f} deg ({100 * (H > 0).mean():3.0f}% up)' if H.size else '--'
        print(f'{c:7s} ' + ' '.join(f'{x:6.1f}' for x in bm[:2]) + ' ' + f'{bm[2]:6.1f} {bm[3]:8.1f} {bm[4]:9.1f}'
              + f'   {hp}   ({len(B)} reps)')


if __name__ == '__main__':
    main()
