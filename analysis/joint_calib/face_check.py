#!/usr/bin/env python3
"""face_check.py -- do smplx's single face vertices sit where OpenPose puts the nose, eyes and ears?

smpl_op_fit.py fits each subject's head to the triangulated nose, eyes and ears through smplx's vertex ids
(FACE_VERTICES), so in its fits the head is oriented by the face data.  What is left -- the triangulated keypoint
minus its vertex, in the Head bone's frame (lat, up, fwd; lateral away from the midline), median over frames per
subject -- is the definition mismatch, if it is consistent across subjects.  Also the nose -> mid-ears vector, mesh
vs target: its elevation in the head's sagittal plane and its length.  A monocular 2D fit (step_4b's 'oprf') matches
that vector's image projection, so a mesh vector with the wrong elevation or length is matched with the head
pitched wrong.

    python analysis/joint_calib/face_check.py analysis/joint_calib/out/fit_v3 --splits pipeline/datasets/korea/splits.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline'))
from smpl_op_fit import FACE_VERTICES  # noqa: E402
from utils.smpl_lite import MiniSMPL, SMPL24, rodrigues  # noqa: E402
from utils.openpose import BODY25_NAMES  # noqa: E402

B = {n: i for i, n in enumerate(BODY25_NAMES)}
HEAD = SMPL24.index('Head')
FACE = ('Nose', 'LEye', 'REye', 'LEar', 'REar')


def subject(path, max_frames, rng):
    """Per-frame face residuals (mm, head frame) and nose -> mid-ears vectors (mesh, target; head frame)."""
    z = np.load(path, allow_pickle=True)
    T = len(z['root'])
    keep = np.sort(rng.choice(T, max_frames, replace=False)) if T > max_frames else np.arange(T)
    smpl = MiniSMPL(z['shape'], full=True)
    aa = torch.tensor(np.concatenate([z['root'][keep, None], z['pose'][keep]], 1))
    with torch.no_grad():
        R = rodrigues(aa)
        A = smpl.bone_transforms(R)
        V = smpl.vertices(R, A).numpy() + z['trans'][keep, None]
    G = A[:, HEAD, :3, :3].numpy()                                 # head bone -> lab; rest axes x lat, y up, z fwd
    X = z['X'][keep]
    to_head = lambda d: np.einsum('tji,tj->ti', G, d) * 1000.0  # noqa: E731
    res = {}
    for k in FACE:
        loc = to_head(X[:, B[k]] - V[:, FACE_VERTICES[k]])
        if k[0] == 'R':
            loc[:, 0] *= -1                                         # lateral away from the midline
        elif k[0] == 'L':
            pass
        res[k] = loc
    ears_m = V[:, [FACE_VERTICES['LEar'], FACE_VERTICES['REar']]].mean(1)
    ears_t = X[:, [B['LEar'], B['REar']]].mean(1)
    vec_m = to_head(ears_m - V[:, FACE_VERTICES['Nose']])
    vec_t = to_head(ears_t - X[:, B['Nose']])
    return res, vec_m, vec_t, float(z['size_mm'])


def elev(v):
    """Elevation of the ears above the nose, degrees, in the sagittal (up, fwd) plane."""
    return np.degrees(np.arctan2(v[:, 1], -v[:, 2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('fit_dir')
    ap.add_argument('--splits', default=None, help='splits.json: only its train children (plus every adult)')
    ap.add_argument('--max-frames', type=int, default=120)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    paths = sorted(glob.glob(os.path.join(args.fit_dir, '*.npz')))
    if args.splits:
        with open(args.splits) as f:
            train = set(json.load(f)['train'])
        paths = [p for p in paths if os.path.basename(p)[0] == 'P' or os.path.basename(p)[:-4] in train]
    rows = {}
    for p in paths:
        name = os.path.basename(p)[:-4]
        res, vm, vt, size = subject(p, args.max_frames, rng)
        ok = np.isfinite(vt).all(1) & np.isfinite(vm).all(1)
        r = {k: np.nanmedian(v, 0) for k, v in res.items()}
        r['elev_m'], r['elev_t'] = np.median(elev(vm[ok])), np.median(elev(vt[ok]))
        r['len_m'], r['len_t'] = np.median(np.linalg.norm(vm[ok], axis=1)), np.median(np.linalg.norm(vt[ok], axis=1))
        r['n'] = int(ok.sum())
        rows[name] = r
    groups = sorted({('adult' if n[0] == 'P' else n[0]) for n in rows})
    print(f'{len(rows)} subjects; target minus vertex, mm, head frame (lat away from midline, up, fwd), '
          f'median over frames then over subjects [IQR]\n')
    for g in groups:
        sel = [r for n, r in rows.items() if ('adult' if n[0] == 'P' else n[0]) == g]
        print(f'{g} ({len(sel)} subjects)')
        for k in ('Nose', 'Eye', 'Ear'):
            ks = [k] if k == 'Nose' else ['L' + k, 'R' + k]
            M = np.array([r[kk] for r in sel for kk in ks])
            med, q1, q3 = np.median(M, 0), np.percentile(M, 25, 0), np.percentile(M, 75, 0)
            print(f'  {k:5s} ' + '  '.join(f'{c} {m:+6.1f} [{a:+5.1f},{b:+5.1f}]'
                                         for c, m, a, b in zip(('lat', 'up', 'fwd'), med, q1, q3)))
        em, et = np.array([r['elev_m'] for r in sel]), np.array([r['elev_t'] for r in sel])
        lm, lt = np.array([r['len_m'] for r in sel]), np.array([r['len_t'] for r in sel])
        print(f'  nose -> mid-ears: elevation mesh {np.median(em):+5.1f} vs target {np.median(et):+5.1f} deg '
              f'(target - mesh {np.median(et - em):+5.1f} [{np.percentile(et - em, 25):+.1f},{np.percentile(et - em, 75):+.1f}]);'
              f'  length mesh {np.median(lm):5.1f} vs target {np.median(lt):5.1f} mm')


if __name__ == '__main__':
    main()
