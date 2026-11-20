#!/usr/bin/env python3
"""bp_compare.py -- ours (single camera, C19) vs MediaPipe BlazePose against mocap on BioCV adults.

Both systems are scored on the SAME frames: the 30 fps source frames of each chosen camera
(bp_cameras.json) where mocap has all 15 marker joints and BOTH systems produced all of them.  How often
each system produced nothing is reported separately (det_rate).

Joint sets
    limb    knees, ankles, elbows, wrists (8): the headline -- hip/shoulder/trunk centres are defined
            differently by mocap, OpenPose/SMPL and BlazePose, limb joints much less so
    all     the 15 mocap joints (no Nose/Head), as each system outputs them
    off     all 15 after a per-system, per-joint offset (in the skeleton's own pelvis frame) estimated
            on the OTHER test subjects (leave-one-subject-out; P03 uses all 10 test subjects)

Per-frame errors (mm)
    pa      Procrustes (rotation, translation, scale) on the set
    root    lab frame, pelvis on mocap pelvis, nothing else (our rotation via the camera extrinsic; BlazePose's
            world axes are camera-like, rotated the same way)
    rootS   root + the best single scale about the pelvis
    placed  ours only: absolute lab position (BlazePose has no placement)
Joint angles (deg, each skeleton's own joints, lab frame): knee / elbow flexion, hip flexion and abduction
(pelvis frame), shoulder elevation (upper arm vs trunk), trunk lean (forward from vertical).  Per frame
|error|; per trial-camera peak and range-of-motion error (per TRIAL, reps are not segmented).
Per trial-camera: limb-length CV and tails (length / own median: pooled p5-p95, worst limb), jitter (mean
|acceleration| of pelvis-relative joints, mm/frame^2 at 30 fps, consecutive frames only), depth tilt (limb
angle out of the image plane minus mocap's, deg; negative = flatter).

Statistics: subject = median over that subject's frames (or trial-cameras); cohort = median of the 10
subject medians; wins = subjects where ours is lower; Wilcoxon signed-rank on subject medians; bootstrap
95% CI (subjects resampled) of the median paired difference ours - BlazePose.  P03 is reported separately.

Writes {BIOCV_OUT}/bp_compare/{frames.csv.gz, trials.csv, summary.txt}.

    python3 analysis/blazepose/bp_compare.py [--config C19]
    python3 analysis/blazepose/bp_compare.py --shapes-only       # step_2c's fitted shapes, per subject
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(REPO, 'pipeline'))
sys.path.insert(0, os.path.join(REPO, 'pipeline', 'datasets', 'biocv'))
from config import OUT_DIR, calib_path, mocap_path, pnp_path, shapes_path, stature_path, twod_path  # noqa: E402
from utils.calibration import load_calib, reproject  # noqa: E402
from bp_views import trial_mocap, view_angle, view_bin  # noqa: E402
from bp_extract import MP_TO_COCO, coco_to_h36m, mp_path  # noqa: E402

SYSTEMS = ('ours', 'blazepose')
P03_USER = 'User03'
LIMB_J = [2, 3, 5, 6, 12, 13, 15, 16]
LIMBS = {'R thigh': (1, 2), 'R shank': (2, 3), 'L thigh': (4, 5), 'L shank': (5, 6),
         'L upper arm': (11, 12), 'L forearm': (12, 13), 'R upper arm': (14, 15), 'R forearm': (15, 16)}
UP = np.array([0.0, 0.0, 1.0])
ANGLES = ('knee_flex', 'hip_flex', 'hip_abd', 'elbow_flex', 'shoulder_elev', 'trunk_lean')
WRONG_REL = 0.35       # BlazePose 2D this far (median joint, fraction of the subject's 2D height) from mocap = someone else
POS_METRICS = ('pa_limb', 'pa_all', 'pa_off', 'root_limb', 'root_all', 'root_off',
               'rootS_limb', 'rootS_all', 'rootS_off')


# ---------------------------------------------------------------- geometry
def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pa_err(P, G):
    """Per-frame Procrustes error, (N,J,3) each -> (N,)  (utils/metrics._procrustes, batched)."""
    mp, mg = P.mean(1, keepdims=True), G.mean(1, keepdims=True)
    P0, G0 = P - mp, G - mg
    U, S, Vt = np.linalg.svd(np.einsum('nji,njk->nik', P0, G0))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(P), 3))
    D[:, 2] = d
    R = U @ (D[:, :, None] * Vt)
    s = (S * D).sum(1) / (P0 ** 2).sum((1, 2))
    A = s[:, None, None] * P0 @ R + mg
    return np.linalg.norm(A - G, axis=-1).mean(1)


def root_err(P, G, J, scale):
    Pr, Gr = P[:, J] - P[:, :1], G[:, J] - G[:, :1]
    if scale:
        s = (Pr * Gr).sum((1, 2)) / np.maximum((Pr ** 2).sum((1, 2)), 1e-9)
        Pr = Pr * s[:, None, None]
    return np.linalg.norm(Pr - Gr, axis=-1).mean(1)


def body_basis(P):
    """(N,3,3) rows = left, up, forward of the skeleton's own pelvis (H36M joints, lab frame)."""
    lat = _unit(P[:, 4] - P[:, 1])
    up = P[:, 8] - P[:, 0]
    up = _unit(up - (up * lat).sum(-1, keepdims=True) * lat)
    return np.stack([lat, up, np.cross(lat, up)], 1)


def _ang(a, b):
    return np.degrees(np.arccos(np.clip((_unit(a) * _unit(b)).sum(-1), -1, 1)))


def joint_angles(P):
    """{name_side: (N,)} in degrees from H36M-17 lab-frame joints."""
    B = body_basis(P)
    lat, up, fwd = B[:, 0], B[:, 1], B[:, 2]
    out = {}
    for s, (hip, knee, ank, sh, el, wr) in (('R', (1, 2, 3, 14, 15, 16)), ('L', (4, 5, 6, 11, 12, 13))):
        out[f'knee_flex_{s}'] = 180 - _ang(P[:, hip] - P[:, knee], P[:, ank] - P[:, knee])
        out[f'elbow_flex_{s}'] = 180 - _ang(P[:, sh] - P[:, el], P[:, wr] - P[:, el])
        th = _unit(P[:, knee] - P[:, hip])
        out[f'hip_flex_{s}'] = np.degrees(np.arctan2((th * fwd).sum(-1), -(th * up).sum(-1)))
        outward = -lat if s == 'R' else lat
        out[f'hip_abd_{s}'] = np.degrees(np.arcsin(np.clip((th * outward).sum(-1), -1, 1)))
        out[f'shoulder_elev_{s}'] = _ang(P[:, el] - P[:, sh], P[:, 0] - P[:, 8])
    trunk = P[:, 8] - P[:, 0]
    fh = fwd - (fwd @ UP)[:, None] * UP
    fh = _unit(fh)
    out['trunk_lean'] = np.degrees(np.arctan2((trunk * fh).sum(-1), trunk @ UP))
    return out


def limb_tails(P):
    pooled, lo, hi, cv = [], [], [], []
    for a, b in LIMBS.values():
        L = np.linalg.norm(P[:, b] - P[:, a], axis=-1)
        if len(L) < 10:
            continue
        r = L / np.median(L)
        pooled.append(r)
        lo.append(np.percentile(r, 5))
        hi.append(np.percentile(r, 95))
        cv.append(100 * L.std() / L.mean())
    if not pooled:
        return dict(limb_cv=np.nan, len_p5=np.nan, len_p95=np.nan, len_worst_p5=np.nan, len_worst_p95=np.nan)
    r = np.concatenate(pooled)
    return dict(limb_cv=float(np.median(cv)), len_p5=float(np.percentile(r, 5)), len_p95=float(np.percentile(r, 95)),
                len_worst_p5=float(min(lo)), len_worst_p95=float(max(hi)))


def jitter(P, idx):
    """Mean |second difference| of pelvis-relative joints over runs of consecutive frames (mm/frame^2)."""
    X = P - P[:, :1]
    ok = (idx[2:] - idx[1:-1] == 1) & (idx[1:-1] - idx[:-2] == 1)
    if not ok.any():
        return np.nan
    a = X[2:] - 2 * X[1:-1] + X[:-2]
    return float(np.linalg.norm(a[ok][:, 1:], axis=-1).mean())


def tilt(Pc):
    """(N,8) limb angle out of the image plane (deg) from camera-frame joints."""
    v = np.stack([Pc[:, b] - Pc[:, a] for a, b in LIMBS.values()], 1)
    return np.degrees(np.arcsin(np.clip(np.abs(v[..., 2]) / np.linalg.norm(v, axis=-1), 0, 1)))


# ---------------------------------------------------------------- loading
def bp_2d_rel(image33, G, mask, calib):
    """(T,) median distance of BlazePose's 2D joints from the projected mocap, as a fraction of the subject's projected
    height (NaN where either is missing): ~0.05-0.15 for a sloppy pose of the right person, >> WRONG_REL for another."""
    W, H, K, L, dist = calib
    cam = G @ L[:3, :3].T + L[:3, 3]
    T = len(G)
    uv = reproject(cam.reshape(-1, 3), K, dist.reshape(1, 5)).reshape(T, 17, 2)
    uv[cam[..., 2] <= 0] = np.nan
    bp = coco_to_h36m(image33[:, MP_TO_COCO, :2].astype(float)) * np.array([W, H])
    j = np.where(mask)[0]
    d = np.nanmedian(np.linalg.norm(bp[:, j] - uv[:, j], axis=-1), axis=1)
    h = np.nanmax(uv[:, j, 1], axis=1) - np.nanmin(uv[:, j, 1], axis=1)
    return d / h


def load_view(u, a, cam, config, bp_suffix=''):
    d = np.load(twod_path(u, a, 'openpose', cam))
    sfi = d['source_frame_idx'].astype(int)
    G = trial_mocap(u, a, sfi)
    mask = np.asarray(np.load(mocap_path(u, a))['valid_joint_mask'], bool)
    calib = load_calib(calib_path(u, cam, a))
    L = calib[3]
    R, t = L[:3, :3], L[:3, 3]
    ours_c = np.load(pnp_path(u, a, 'openpose', cam, config))['kps_H36M_placed'].astype(float) * 1000.0
    bp = np.load(mp_path(u, a, cam).replace('_mp.npz', f'_mp{bp_suffix}.npz'))
    bp_c = bp['world'].astype(float)
    T = min(len(sfi), len(ours_c), len(bp_c))
    if not len(sfi) == len(ours_c) == len(bp_c):
        print(f'  WARNING {u}/{a} cam {cam}: rows 2d {len(sfi)} ours {len(ours_c)} blazepose {len(bp_c)} -> {T}')
    sfi, G, ours_c, bp_c = sfi[:T], G[:T], ours_c[:T], bp_c[:T]
    import warnings
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)              # all-NaN rows: frames with no detection
        rel = bp_2d_rel(bp['image33'][:T], G, mask, calib)
    return dict(sfi=sfi, G=G, Gc=G @ R.T + t, mask=mask, L=L, bp_rel=rel,
                sys={'ours': ((ours_c - t) @ R, ours_c), 'blazepose': (bp_c @ R, bp_c)})


# ---------------------------------------------------------------- statistics
def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    if len(d) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.median(d[rng.integers(0, len(d), (n, len(d)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(sub):
    """sub: DataFrame index subject, columns ours/blazepose -> one summary string."""
    sub = sub.dropna()
    n = len(sub)
    if n == 0:
        return 'n/a'
    o, b = sub['ours'].values, sub['blazepose'].values
    s = f'{np.median(o):7.1f} {np.median(b):7.1f}   wins {int((o < b).sum())}/{n}'
    if n >= 5:
        from scipy.stats import wilcoxon
        p = wilcoxon(o, b).pvalue
        lo, hi = boot_ci(o - b)
        s += f'   d {np.median(o - b):+6.1f} [{lo:+.1f}, {hi:+.1f}]  p {p:.3f}'
    return s


def summarise(F, Tr, title):
    """Text summary for one subject group: F frame table, Tr trial-camera table."""
    out = [f'==== {title}: {F.subject.nunique()} subjects, {F.groupby(["action", "cam"]).ngroups} trial-cameras, '
           f'{len(F) // 2} paired frames', '']
    cols = 'ours  blazepose'
    b = F[F.system == 'blazepose']
    out += ['-- BlazePose identity: % of mocap frames with no detection | % of paired frames on someone else '
            f'(2D median joint > {WRONG_REL:.2f} x subject height from projected mocap); worst trial-cameras']
    wr = b.groupby(['action', 'cam']).wrong_person.mean().sort_values(ascending=False)
    nd = 1 - Tr[Tr.system == 'blazepose'].det_rate
    out.append(f'   no detection {100 * nd.mean():.1f}%   wrong person {100 * b.wrong_person.mean():.2f}%   ' +
               '  '.join(f'{a} {c} {100 * x:.0f}%' for (a, c), x in wr.head(5).items() if x > 0))
    ok_f = F.set_index(['action', 'cam', 'frame']).index.isin(
        b[~b.wrong_person].set_index(['action', 'cam', 'frame']).index)
    Fr = F[ok_f]
    out.append(f'   without wrong-person frames ({cols}): ' + '   '.join(
        f'{m} ' + paired(Fr.groupby(['subject', 'system'])[m].median().unstack()) for m in ('pa_limb', 'root_limb')))
    out.append('')
    out += ['-- position error (mm), pooled frames: mean / median / p5 / p95 / % > 100 mm',
            f'{"metric":>11} {"system":>10} {"mean":>7} {"med":>7} {"p5":>7} {"p95":>7} {">100":>6}']
    for m in POS_METRICS + ('placed_all',):
        for s in SYSTEMS:
            x = F.loc[F.system == s, m].dropna().values
            if len(x):
                out.append(f'{m:>11} {s:>10} {x.mean():7.1f} {np.median(x):7.1f} {np.percentile(x, 5):7.1f} '
                           f'{np.percentile(x, 95):7.1f} {100 * (x > 100).mean():5.1f}%')
    out += ['', f'-- subject medians: cohort median ({cols}), wins, median diff [95% CI], Wilcoxon p']
    for m in POS_METRICS:
        out.append(f'{m:>11}  ' + paired(F.groupby(['subject', 'system'])[m].median().unstack()))
    for by, label in (('movement', 'movement'), ('role', 'chosen camera'), ('view', 'per-frame view')):
        out += ['', f'-- by {label}: cohort median of subject medians ({cols}), wins ...']
        for m in ('pa_limb', 'root_limb', 'pa_off'):
            for k, g in F.groupby(by):
                out.append(f'{m:>11} {k:>8}  ' + paired(g.groupby(['subject', 'system'])[m].median().unstack()))
    out += ['', '-- joint angles (deg): pooled MAE | signed error p5 / p95 | median over trial-cams of '
                'peak error, ROM error | subject-median MAE: cohort, wins, ...']
    for an in ANGLES:
        for s in SYSTEMS:
            f = F[F.system == s]
            e = f[f'{an}_err'].dropna().values
            tr = Tr[Tr.system == s]
            out.append(f'{an:>14} {s:>10} {np.abs(e).mean():6.1f} | {np.percentile(e, 5):+6.1f} '
                       f'{np.percentile(e, 95):+6.1f} | {tr[f"{an}_peak_err"].median():5.1f} '
                       f'{tr[f"{an}_rom_err"].median():5.1f}')
        F2 = F.assign(ae=F[f'{an}_err'].abs())
        out.append(f'{"":>14} {"subjects":>10}  ' + paired(F2.groupby(['subject', 'system']).ae.median().unstack()))
    out += ['', '-- per trial-camera (median over trial-cams): limb CV %, length/median p5 p95, worst limb p5 p95, '
                'jitter mm/fr^2, depth tilt signed / abs deg, detection rate']
    for s in SYSTEMS + ('mocap',):
        tr = Tr[Tr.system == s]
        out.append(f'{s:>10} {tr.limb_cv.median():6.2f} {tr.len_p5.median():6.3f} {tr.len_p95.median():6.3f} '
                   f'{tr.len_worst_p5.median():6.3f} {tr.len_worst_p95.median():6.3f} {tr.jitter.median():6.2f} '
                   f'{tr.tilt_signed.median():+6.1f} {tr.tilt_abs.median():5.1f} {tr.det_rate.median():6.1%}')
    for m in ('limb_cv', 'jitter', 'tilt_abs'):
        sub = Tr[Tr.system.isin(SYSTEMS)].groupby(['subject', 'system'])[m].median().unstack()
        out.append(f'{m:>11}  subjects ' + paired(sub))
    return out


# ---------------------------------------------------------------- main
def shapes_report():
    users = sorted(d for d in os.listdir(OUT_DIR) if d.startswith('User'))
    print(f'{"user":>7} {"stature":>8} {"kid_oprg h":>10} {"kid w":>6} {"rms mm":>6}')
    for u in users:
        p = shapes_path(u)
        if not os.path.exists(p):
            print(f'{u:>7}  no shapes.npz')
            continue
        s = np.load(p)
        st = json.load(open(stature_path(u))).get('stature_m', np.nan)
        print(f'{u:>7} {st:8.3f} {float(s["kid_oprg_height_m"]):10.3f} {float(s["kid_oprg"][-1]):6.2f} '
              f'{float(s["kid_oprg_rms_mm"]):6.1f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='C19')
    ap.add_argument('--shapes-only', action='store_true')
    ap.add_argument('--bp-suffix', default='', help="BlazePose cache: '' as is, '_crop' subject given (bp_extract --crop)")
    ap.add_argument('--out', default=None, help='default {BIOCV_OUT}/bp_compare{bp_suffix}')
    args = ap.parse_args()
    args.out = args.out or os.path.join(OUT_DIR, f'bp_compare{args.bp_suffix}')
    if args.shapes_only:
        shapes_report()
        return
    sel = json.load(open(os.path.join(OUT_DIR, 'bp_cameras.json')))

    views = []
    for u, acts in sorted(sel.items()):
        for a, r in sorted(acts.items()):
            for role in ('front', 'side', 'oblique'):
                if role not in r:
                    continue
                cam = r[role]
                need = [pnp_path(u, a, 'openpose', cam, args.config),
                        mp_path(u, a, cam).replace('_mp.npz', f'_mp{args.bp_suffix}.npz')]
                if not all(os.path.exists(p) for p in need):
                    print(f'{u}/{a} cam {cam}: missing ' + ', '.join(os.path.relpath(p, OUT_DIR) for p in need
                                                                     if not os.path.exists(p)))
                    continue
                v = load_view(u, a, cam, args.config, args.bp_suffix)
                v.update(user=u, action=a, cam=cam, role=role)
                views.append(v)
    if not views:
        raise SystemExit('nothing to compare')

    # pass 1: paired frames, pelvis-frame residuals for the per-joint offsets
    resid = {}                                  # (system, user) -> [sum (17,3), count]
    for v in views:
        m = v['mask']
        ok = np.isfinite(v['G'][:, m]).all((1, 2))
        v['gt_ok'] = ok.copy()
        for s in SYSTEMS:
            v[f'{s}_ok'] = np.isfinite(v['sys'][s][0][:, m]).all((1, 2))
            ok &= v[f'{s}_ok']
        v['ok'] = ok
        for s in SYSTEMS:
            P, G = v['sys'][s][0][ok], v['G'][ok]
            r = G - (P - P[:, :1] + G[:, :1])
            loc = np.einsum('nij,nkj->nki', body_basis(P), r)       # residual in the skeleton's pelvis frame
            acc = resid.setdefault((s, v['user']), [np.zeros((17, 3)), 0])
            acc[0] += np.nan_to_num(loc).sum(0)
            acc[1] += len(P)
    test_users = sorted({v['user'] for v in views} - {P03_USER})

    def offsets(s, u):
        pool = [w for w in test_users if w != u]
        tot, n = np.zeros((17, 3)), 0          # no other subjects (e.g. P03 alone): no offset
        for w in pool:
            if (s, w) in resid:
                tot, n = tot + resid[s, w][0], n + resid[s, w][1]
        return tot / max(n, 1)

    # pass 2: metrics
    frames, trials = [], []
    for v in views:
        ok, m = v['ok'], v['mask']
        all_j = np.where(m)[0]
        idx = np.where(ok)[0]
        G, Gc = v['G'][ok], v['Gc'][ok]
        ang = view_angle(G, v['L'])
        base = dict(subject=v['user'], action=v['action'], movement=v['action'].split('_')[1], cam=v['cam'],
                    role=v['role'])
        g_ang = joint_angles(G)
        g_tilt = tilt(Gc)
        for s in SYSTEMS:
            P, Pc = v['sys'][s][0][ok], v['sys'][s][1][ok]
            Po = P + np.einsum('nij,ki->nkj', body_basis(P), offsets(s, v['user']))
            rec = dict(pa_limb=pa_err(P[:, LIMB_J], G[:, LIMB_J]), pa_all=pa_err(P[:, all_j], G[:, all_j]),
                       pa_off=pa_err(Po[:, all_j], G[:, all_j]),
                       root_limb=root_err(P, G, LIMB_J, False), root_all=root_err(P, G, all_j, False),
                       root_off=root_err(Po, G, all_j, False),
                       rootS_limb=root_err(P, G, LIMB_J, True), rootS_all=root_err(P, G, all_j, True),
                       rootS_off=root_err(Po, G, all_j, True),
                       placed_all=(np.linalg.norm(P[:, all_j] - G[:, all_j], axis=-1).mean(1) if s == 'ours'
                                   else np.full(len(P), np.nan)))
            p_ang = joint_angles(P)
            tr = dict(base, system=s, n_paired=int(ok.sum()), n_gt=int(v['gt_ok'].sum()),
                      det_rate=float(v[f'{s}_ok'][v['gt_ok']].mean()) if v['gt_ok'].any() else np.nan,
                      **limb_tails(P), jitter=jitter(P[:, all_j], idx))
            pt = tilt(Pc) - g_tilt
            tr.update(tilt_signed=float(np.nanmean(pt)), tilt_abs=float(np.nanmean(np.abs(pt))))
            for an in ANGLES:
                keys = [an] if an == 'trunk_lean' else [f'{an}_R', f'{an}_L']
                rec[f'{an}_err'] = np.nanmean([p_ang[k] - g_ang[k] for k in keys], 0)
                tr[f'{an}_peak_err'] = float(np.mean([abs(np.nanmax(p_ang[k]) - np.nanmax(g_ang[k])) for k in keys]))
                rom = lambda x: np.nanmax(x) - np.nanmin(x)                 # noqa: E731
                tr[f'{an}_rom_err'] = float(np.mean([abs(rom(p_ang[k]) - rom(g_ang[k])) for k in keys]))
            rel = v['bp_rel'][ok] if s == 'blazepose' else np.full(len(idx), np.nan)
            frames.append(pd.DataFrame(dict(base, system=s, frame=idx, source_frame=v['sfi'][ok], view_angle=ang,
                                            view=view_bin(ang), bp_2d_rel=rel, wrong_person=rel > WRONG_REL, **rec)))
            trials.append(tr)
        trials.append(dict(base, system='mocap', n_paired=int(ok.sum()), **limb_tails(G), jitter=jitter(G[:, all_j], idx),
                           det_rate=np.nan, tilt_signed=0.0, tilt_abs=0.0))

    F = pd.concat(frames, ignore_index=True)
    Tr = pd.DataFrame(trials)
    os.makedirs(args.out, exist_ok=True)
    F.to_csv(os.path.join(args.out, 'frames.csv.gz'), index=False, float_format='%.2f')
    Tr.to_csv(os.path.join(args.out, 'trials.csv'), index=False, float_format='%.4f')
    lines = [f'ours = OpenPose 2D 30 fps -> fine-tuned MotionBERT-Mesh -> {args.config}; '
             f'BlazePose = heavy, video mode, ' + ('cropped to our OpenPose subject (subject given)' if args.bp_suffix == '_crop'
             else 'whole frame (as is)') + '; single camera each; vs mocap', '']
    for title, users in (('TEST (10 adults)', test_users), ('P03 (pipeline tuned on it)', [P03_USER])):
        f, t = F[F.subject.isin(users)], Tr[Tr.subject.isin(users)]
        if len(f):
            lines += summarise(f, t, title) + ['']
    text = '\n'.join(lines)
    with open(os.path.join(args.out, 'summary.txt'), 'w') as fh:
        fh.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
