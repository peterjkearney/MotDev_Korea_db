#!/usr/bin/env python3
"""op_offsets.py -- where OpenPose puts each joint relative to the anatomical joint centre, on BioCV.

Per frame, triangulated OpenPose (step_1b, all cameras) minus mocap's joint centre (step_0), both H36M-17,
lab mm, expressed in a body-segment frame built from the MOCAP joints:
    pelvis  x = RHip->LHip, y = Hip->Thorax (made orthogonal), z = x cross y (forward)
    thorax  x = RShoulder->LShoulder, y = Hip->Thorax (orthogonal), z forward
    thigh / shank / upper arm / forearm   y = distal->proximal joint (up the segment), the other axes from
            the pelvis (legs) or thorax (arms) frame made orthogonal to y
Components are reported as (lateral, up, forward): x is mirrored on the right side so +lateral is away from
the midline for both sides.  For limb segments "up" is along the segment towards the proximal joint.

Each joint is reported in a primary frame (the one a SMPL offset will rotate with) and an alternative, with
the median and IQR per subject: a joint whose offset is tight in one frame and loose in the other says which
bone it rides on.  Also per-subject segment lengths (mocap vs OpenPose) to see whether offsets scale with size.

    python3 analysis/joint_calib/op_offsets.py <tree> [--out offsets.npz]
<tree>: {tree}/{user}/{action}/Analysis/H36M/{mocap_h36m,openpose_tri_h36m}.npz
"""
import argparse
import glob
import os

import numpy as np

H36M = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
J = {n: i for i, n in enumerate(H36M)}

# joint: (primary frame, alternative frame); frames are (segment, side) built by frames()
JOINT_FRAMES = {'Hip': ('pelvis', 'thigh'), 'Knee': ('thigh', 'shank'), 'Ankle': ('shank', 'thigh'),
                'Shoulder': ('thorax', 'upperarm'), 'Elbow': ('upperarm', 'forearm'), 'Wrist': ('forearm', 'upperarm')}
SEGS = {'thigh': ('Hip', 'Knee'), 'shank': ('Knee', 'Ankle'), 'upperarm': ('Shoulder', 'Elbow'),
        'forearm': ('Elbow', 'Wrist')}


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _orth(v, axis):
    return _unit(v - (v * axis).sum(-1, keepdims=True) * axis)


def frames(M):
    """M (T,17,3) mocap -> {(segment, side): R (T,3,3)}, rows = x, y, z axes in lab coordinates."""
    out = {}
    xp = _unit(M[:, J['LHip']] - M[:, J['RHip']])
    yp = _orth(M[:, J['Thorax']] - M[:, J['Hip']], xp)
    out['pelvis', ''] = np.stack([xp, yp, np.cross(xp, yp)], 1)
    xt = _unit(M[:, J['LShoulder']] - M[:, J['RShoulder']])
    yt = _orth(M[:, J['Thorax']] - M[:, J['Hip']], xt)
    zt = np.cross(xt, yt)
    out['thorax', ''] = np.stack([xt, yt, zt], 1)
    for s in 'LR':
        for seg, (a, b) in SEGS.items():
            y = _unit(M[:, J[s + a]] - M[:, J[s + b]])
            if seg in ('thigh', 'shank'):
                x = _orth(xp, y)
                out[seg, s] = np.stack([x, y, np.cross(x, y)], 1)
            else:
                z = _orth(zt, y)
                out[seg, s] = np.stack([np.cross(y, z), y, z], 1)
    return out


def local_offsets(M, O):
    """{(joint, frame): (T,2,3) local (lateral, up, forward) offset O - M, L then R}."""
    F = frames(M)
    res = {}
    for jn, fr in JOINT_FRAMES.items():
        for f in fr:
            per_side = []
            for s in 'LR':
                R = F[f, s if f not in ('pelvis', 'thorax') else '']
                d = O[:, J[s + jn]] - M[:, J[s + jn]]
                loc = np.einsum('tij,tj->ti', R, d)
                if s == 'R':
                    loc[:, 0] *= -1                       # +lateral away from the midline on both sides
                per_side.append(loc)
            res[jn, f] = np.stack(per_side, 1)
    return res


# two-bone model: offset = R_parent^T p + R_child^T c (each in its own segment frame, lateral mirrored on the
# right), so a keypoint can ride on both bones of a joint -- e.g. move into the bend as a knee or elbow flexes
BLEND_FRAMES = {'Hip': ('pelvis', 'thigh'), 'Knee': ('thigh', 'shank'), 'Ankle': ('shank', None),
                'Shoulder': ('thorax', 'upperarm'), 'Elbow': ('upperarm', 'forearm'), 'Wrist': ('forearm', None)}


def fit_blend(M, O, jn, ridge_mm=0.0):
    """Robust (soft-L1, 15 mm) least squares for p, c (lateral, up, forward mm) over frames, L/R pooled.
    Returns p, c (c = 0 for one-bone joints), the residual RMS and the residual RMS of the one-bone fit."""
    from scipy.optimize import least_squares
    F = frames(M)
    fp, fc = BLEND_FRAMES[jn]
    rows = []
    for s in 'LR':
        S = np.diag([-1.0 if s == 'R' else 1.0, 1.0, 1.0])
        Rp = F[fp, s if fp not in ('pelvis', 'thorax') else '']
        Rc = F[fc, s] if fc else None
        d = O[:, J[s + jn]] - M[:, J[s + jn]]
        rows.append((np.transpose(Rp, (0, 2, 1)) @ S, None if Rc is None else np.transpose(Rc, (0, 2, 1)) @ S, d))
    Bp = np.concatenate([r[0] for r in rows])
    Bc = None if fc is None else np.concatenate([r[1] for r in rows])
    D = np.concatenate([r[2] for r in rows])

    def res(x, two):
        e = np.einsum('tij,j->ti', Bp, x[:3]) - D
        if two:
            e = e + np.einsum('tij,j->ti', Bc, x[3:])
        return np.r_[e.ravel(), ridge_mm * x] if ridge_mm else e.ravel()

    one = least_squares(res, np.zeros(3), args=(False,), loss='soft_l1', f_scale=15.0)
    rms1 = np.sqrt(np.median((res(one.x, False)[:D.size].reshape(-1, 3) ** 2).sum(1)))
    if fc is None:
        return one.x, np.zeros(3), rms1, rms1
    two = least_squares(res, np.zeros(6), args=(True,), loss='soft_l1', f_scale=15.0)
    rms2 = np.sqrt(np.median((res(two.x, True)[:D.size].reshape(-1, 3) ** 2).sum(1)))
    return two.x[:3], two.x[3:], rms2, rms1


def seg_len(K, a, b):
    return np.linalg.norm(K[:, J[a]] - K[:, J[b]], axis=-1)


def load_trial(d):
    m = np.load(os.path.join(d, 'mocap_h36m.npz'))
    o = np.load(os.path.join(d, 'openpose_tri_h36m.npz'))
    M, O = m['kps3d'].astype(float), o['kps3d'].astype(float)
    n = min(len(M), len(O))
    M, O = M[:n], O[:n]
    core = [J[k] for k in H36M if k not in ('Spine', 'Nose', 'Head')]
    ok = np.isfinite(M[:, core]).all((1, 2)) & np.isfinite(O[:, core]).all((1, 2))
    # gross triangulation failures (wrong person, L/R swap) would dominate a mean; medians are used anyway,
    # but drop frames where any core joint is > 150 mm off
    err = np.linalg.norm(O[:, core] - M[:, core], axis=-1)
    ok &= np.nan_to_num(err, nan=1e9).max(1) < 150
    return M[ok], O[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tree')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    by_subj = {}
    for d in sorted(glob.glob(os.path.join(args.tree, '*', '*', 'Analysis', 'H36M'))):
        if not os.path.exists(os.path.join(d, 'openpose_tri_h36m.npz')):
            continue
        user, action = d.split(os.sep)[-4:-2]
        M, O = load_trial(d)
        print(f'{action}: {len(M)} frames')
        by_subj.setdefault(user, []).append((M, O))
    subj = sorted(by_subj)
    MO = {u: (np.concatenate([m for m, _ in v]), np.concatenate([o for _, o in v])) for u, v in by_subj.items()}

    print('\nsegment lengths, median mm (mocap / OpenPose / diff), L+R pooled')
    for name, (a, b) in [('thigh', ('Hip', 'Knee')), ('shank', ('Knee', 'Ankle')),
                         ('upperarm', ('Shoulder', 'Elbow')), ('forearm', ('Elbow', 'Wrist'))]:
        row = []
        for u in subj:
            M, O = MO[u]
            lm = np.median(np.r_[seg_len(M, 'L' + a, 'L' + b), seg_len(M, 'R' + a, 'R' + b)])
            lo = np.median(np.r_[seg_len(O, 'L' + a, 'L' + b), seg_len(O, 'R' + a, 'R' + b)])
            row.append(f'{u} {lm:4.0f}/{lo:4.0f}/{lo - lm:+4.0f}')
        print(f'  {name:9s} ' + '   '.join(row))
    for name, (a, b) in [('hip w', ('LHip', 'RHip')), ('shoulder w', ('LShoulder', 'RShoulder')),
                         ('trunk', ('Hip', 'Thorax'))]:
        row = []
        for u in subj:
            M, O = MO[u]
            lm, lo = np.median(seg_len(M, a, b)), np.median(seg_len(O, a, b))
            row.append(f'{u} {lm:4.0f}/{lo:4.0f}/{lo - lm:+4.0f}')
        print(f'  {name:9s} ' + '   '.join(row))

    print('\noffset OpenPose - mocap, median (lateral, up, forward) mm [IQR of each], L/R pooled')
    table = {}
    for u in subj:
        table[u] = local_offsets(*MO[u])
    for (jn, f) in table[subj[0]]:
        tag = '*' if JOINT_FRAMES[jn][0] == f else ' '
        cells = []
        for u in subj:
            v = table[u][jn, f].reshape(-1, 3)
            med = np.median(v, 0)
            iqr = np.subtract(*np.percentile(v, [75, 25], 0))
            cells.append('(' + ','.join(f'{x:+4.0f}' for x in med) + ') [' + ','.join(f'{x:2.0f}' for x in iqr) + ']')
        print(f' {tag}{jn:8s} {f:8s} ' + '  '.join(f'{u}: {c}' for u, c in zip(subj, cells)))

    print('\nL vs R, primary frame, all subjects pooled')
    for jn, (f, _) in JOINT_FRAMES.items():
        v = np.concatenate([table[u][jn, f] for u in subj])
        print(f'  {jn:8s} L ' + ','.join(f'{x:+4.0f}' for x in np.median(v[:, 0], 0))
              + '   R ' + ','.join(f'{x:+4.0f}' for x in np.median(v[:, 1], 0)))

    # size: trunk + thigh + shank (mocap), the standing Thorax-to-ankle height; SMPL's rest-pose equivalent
    # (Thorax_y - ankle_y) is what smpl_lite scales the offsets by
    size = {}
    for u in subj:
        M = MO[u][0]
        size[u] = float(np.median(seg_len(M, 'Hip', 'Thorax')) + np.median(np.r_[
            seg_len(M, 'LHip', 'LKnee'), seg_len(M, 'RHip', 'RKnee')]) + np.median(np.r_[
            seg_len(M, 'LKnee', 'LAnkle'), seg_len(M, 'RKnee', 'RAnkle')]))
    s_cal = float(np.mean(list(size.values())))
    print('\nsize (trunk+thigh+shank, mm): ' + '  '.join(f'{u} {size[u]:.0f}' for u in subj) + f'   mean {s_cal:.0f}')
    print(f'calibrated offsets, primary frame, L/R and subjects pooled (mean of per-subject medians / size), '
          f'mm at size {s_cal:.0f}:')
    out = {}
    for jn, (f, _) in JOINT_FRAMES.items():
        per = np.array([np.median(table[u][jn, f].reshape(-1, 3), 0) / size[u] for u in subj])
        out[jn] = per.mean(0) * s_cal
        sd = per.std(0) * s_cal
        print(f'  {jn:8s} {f:8s} ' + ','.join(f'{x:+5.1f}' for x in out[jn])
              + '   (sd over subjects ' + ','.join(f'{x:4.1f}' for x in sd) + ')')
    print(f'\ntwo-bone fit (parent p, child c; lateral, up, forward), mm at size {s_cal:.0f}, mean over subjects '
          f'[sd]; median |residual| one-bone -> two-bone per subject')
    blend = {}
    for jn, (fp, fc) in BLEND_FRAMES.items():
        per = [fit_blend(*MO[u], jn) for u in subj]
        P = np.array([x[0] / size[u] for x, u in zip(per, subj)]) * s_cal
        C = np.array([x[1] / size[u] for x, u in zip(per, subj)]) * s_cal
        blend[jn] = (P.mean(0), C.mean(0))
        print(f'  {jn:8s} p/{fp:8s} ' + ','.join(f'{x:+5.1f}' for x in P.mean(0)) + ' [' + ','.join(f'{x:4.1f}' for x in P.std(0))
              + ']' + (f'  c/{fc:8s} ' + ','.join(f'{x:+5.1f}' for x in C.mean(0)) + ' [' + ','.join(f'{x:4.1f}' for x in C.std(0)) + ']' if fc else ' ' * 47)
              + '  ' + ' '.join(f'{x[3]:.0f}->{x[2]:.0f}' for x in per))
    if args.out:
        np.savez(args.out, subjects=np.array(subj), size_mm=np.array([size[u] for u in subj]), size_cal_mm=s_cal,
                 frames=np.array([JOINT_FRAMES[k][0] for k in out]), joints=np.array(list(out)),
                 offsets_mm=np.array([out[k] for k in out]),
                 blend_joints=np.array(list(blend)), blend_parent_mm=np.array([blend[k][0] for k in blend]),
                 blend_child_mm=np.array([blend[k][1] for k in blend]))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
