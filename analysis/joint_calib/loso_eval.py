#!/usr/bin/env python3
"""loso_eval.py -- leave-one-subject-out check of the OpenPose offset models (op_offsets.py): calibrate on the
other subjects, predict the held-out subject's triangulated OpenPose joints from its mocap joint centres, and
report the per-joint median residual and the predicted vs actual OpenPose segment lengths.
Models: none (OpenPose = joint centre, i.e. SMPL-24), one-bone (each offset rides one segment frame), two-bone
(parent + child, ridge).

    python3 analysis/joint_calib/loso_eval.py <tree> [--ridge 3]
"""
import argparse
import glob
import os

import numpy as np

from op_offsets import J, BLEND_FRAMES, frames, fit_blend, load_trial, seg_len

SEGS = [('thigh', 'Hip', 'Knee'), ('shank', 'Knee', 'Ankle'), ('upperarm', 'Shoulder', 'Elbow'),
        ('forearm', 'Elbow', 'Wrist'), ('hip w', 'LHip', 'RHip'), ('sh w', 'LShoulder', 'RShoulder')]


def predict(M, model):
    """model {joint: (p, c)} -> M with offsets applied (OpenPose-style), mm."""
    F = frames(M)
    out = M.copy()
    for jn, (p, c) in model.items():
        fp, fc = BLEND_FRAMES[jn]
        for s in 'LR':
            S = np.diag([-1.0 if s == 'R' else 1.0, 1.0, 1.0])
            Rp = F[fp, s if fp not in ('pelvis', 'thorax') else '']
            out[:, J[s + jn]] += np.einsum('tji,j->ti', Rp, S @ p)
            if fc:
                out[:, J[s + jn]] += np.einsum('tji,j->ti', F[fc, s], S @ c)
    return out


def size_of(M):
    return (np.median(seg_len(M, 'Hip', 'Thorax')) + np.median(seg_len(M, 'LHip', 'LKnee'))
            + np.median(seg_len(M, 'LKnee', 'LAnkle')))


def seglens(K):
    out = []
    for name, a, b in SEGS:
        if a.startswith(('L', 'R')) and b.startswith(('L', 'R')):
            out.append(np.median(seg_len(K, a, b)))
        else:
            out.append(np.median(np.r_[seg_len(K, 'L' + a, 'L' + b), seg_len(K, 'R' + a, 'R' + b)]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tree')
    ap.add_argument('--ridge', type=float, default=3.0, help='two-bone ridge, per sqrt(frame) (mm per mm)')
    args = ap.parse_args()
    by = {}
    for d in sorted(glob.glob(os.path.join(args.tree, '*', '*', 'Analysis', 'H36M'))):
        if os.path.exists(os.path.join(d, 'openpose_tri_h36m.npz')):
            by.setdefault(d.split(os.sep)[-4], []).append(load_trial(d))
    MO = {u: (np.concatenate([m for m, _ in v]), np.concatenate([o for _, o in v])) for u, v in by.items()}
    subj = sorted(MO)
    size = {u: size_of(MO[u][0]) for u in subj}
    fits = {}
    for u in subj:
        n = len(MO[u][0])
        one, two = {}, {}
        for jn in BLEND_FRAMES:
            p1, _, _, _ = fit_blend(*MO[u], jn) if BLEND_FRAMES[jn][1] is None else (None,) * 4
            if p1 is None:
                from scipy.optimize import least_squares  # noqa: F401  (one-bone = two-bone with child forced 0)
                p1 = _one_bone(MO[u], jn)
            one[jn] = (p1 / size[u], np.zeros(3))
            p, c, _, _ = fit_blend(*MO[u], jn, ridge_mm=args.ridge * np.sqrt(2 * n))
            two[jn] = (p / size[u], c / size[u])
        fits[u] = dict(one=one, two=two)

    names = ['RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow',
             'RWrist', 'LWrist']
    print('held-out subject: median |OpenPose - predicted| per joint (mm), mean over the 12 limb joints')
    print('and predicted - actual OpenPose segment length (mm): ' + ' '.join(s[0] for s in SEGS))
    for u in subj:
        M, O = MO[u]
        for mname in ('none', 'one', 'two'):
            if mname == 'none':
                P = M
            else:
                others = [v for v in subj if v != u]
                model = {jn: tuple(np.mean([fits[v][mname][jn][k] for v in others], 0) * size[u] for k in (0, 1))
                         for jn in BLEND_FRAMES}
                P = predict(M, model)
            e = np.array([np.median(np.linalg.norm(O[:, J[n]] - P[:, J[n]], axis=-1)) for n in names])
            dl = seglens(P) - seglens(O)
            print(f'  {u} {mname:4s}  joint {e.mean():5.1f}  ' + ' '.join(f'{n[:-0] if False else n}:{x:.0f}' for n, x in zip(names, e))
                  + '   seg ' + ' '.join(f'{x:+4.0f}' for x in dl))
    print('\ntwo-bone (ridge) fit per subject, mm at that subject\'s size: p | c')
    for jn in BLEND_FRAMES:
        print(f'  {jn:8s} ' + '   '.join(f'{u}: ' + ','.join(f'{x:+4.0f}' for x in fits[u]['two'][jn][0] * size[u]) + ' | '
                                        + ','.join(f'{x:+4.0f}' for x in fits[u]['two'][jn][1] * size[u]) for u in subj))


def _one_bone(mo, jn):
    import op_offsets
    fp, fc = BLEND_FRAMES[jn]
    saved = BLEND_FRAMES[jn]
    op_offsets.BLEND_FRAMES[jn] = (fp, None)
    try:
        return fit_blend(*mo, jn)[0]
    finally:
        op_offsets.BLEND_FRAMES[jn] = saved


if __name__ == '__main__':
    main()
