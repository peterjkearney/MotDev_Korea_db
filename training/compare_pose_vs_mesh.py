#!/usr/bin/env python3
"""compare_pose_vs_mesh.py -- base MotionBERT (3D pose) against the MotionBERT-Mesh runs, on the Korea dev views.

Every lifter gets the same input -- each camera's OpenPose 2D as H36M-17, missing joints filled from the nearest
frame (utils/missing_joints, as step_2a) -- and is scored against the same target on the same frames and joints:

    target   the triangulated H36M joints (openpose_tri_h36m.npz) in that camera's frame: --tri loo (default) the
             target built WITHOUT the scored camera, so its own 2D cannot shape what it is scored against; --tri all
             the all-camera triangulation step_8 uses by default
    mask     joints the target has (valid_joint_mask: Nose / Head out) AND every lifter has, frame by frame -- so
             all rows of the table are averaged over exactly the same joint-frames
    metrics  PA-MPJPE (rotation, translation and scale removed: pose shape) and N-MPJPE (translation and scale only:
             also orientation in the camera).  Base MotionBERT predicts normalised, root-relative units with no
             placement, so placed error has no counterpart and is not compared.

Lifters (rows):
    pose        MotionBERT 3D pose, MB_ft_h36m (H36M fine-tuned, 243-frame windows, flip-averaged as its own eval)
    pose_lite   MotionBERT-Lite 3D pose, FT_MB_lite_MB_ft_h36m_global_lite (243 frames, flip-averaged)
    mesh_*      a pipeline run's raw MotionBERT-Mesh skeleton (step4: {cam}_pnp.npz kps_H36M_placed) or a refined
                configuration (--configs; e.g. C19's {config}/{cam}_pnp.npz) -- named --runs label=tree
    Refined configurations output the 'opr' joint set (OpenPose-style hips / shoulders), which matches the
    OpenPose-triangulated target by construction; the raw skeletons (pose, step4) use H36M's mocap definitions.

Results: one row per (view, lifter) in {out}/pose_vs_mesh_{tri}.csv, and a summary (cohort median of subject
medians, per-subject wins and paired differences against --reference) in {out}/pose_vs_mesh_{tri}.txt.  The base
models' outputs are cached per view under {out}/cache/, so re-scoring (e.g. --tri all after loo) is quick.

    python3 training/compare_pose_vs_mesh.py \\
        --runs rel16=.../pipeline_runs/korea_dev_clip16,ft16=.../pipeline_runs/korea_dev_ft1 \\
        --configs ft16:C19 --out .../pipeline_runs/pose_vs_mesh
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
from config import MB_DIR, require_motionbert  # noqa: E402
require_motionbert()
sys.path.insert(0, MB_DIR)
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.utils_data import crop_scale, flip_data  # noqa: E402
from utils import metrics as M  # noqa: E402
from utils.missing_joints import fill_missing  # noqa: E402

POSE_MODELS = {
    'pose': ('MB_ft_h36m.yaml', 'MB_ft_h36m'),
    'pose_lite': ('MB_ft_h36m_global_lite.yaml', 'FT_MB_lite_MB_ft_h36m_global_lite'),
}


def load_pose(name):
    cfg_file, ck_dir = POSE_MODELS[name]
    cfg = get_config(os.path.join(MB_DIR, 'configs', 'pose3d', cfg_file))
    model = load_backbone(cfg)
    ck = torch.load(os.path.join(MB_DIR, 'checkpoint', 'pose3d', ck_dir, 'best_epoch.bin'), map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ck['model_pos'].items()}, strict=True)
    return model.eval(), int(cfg.maxlen)


@torch.no_grad()
def run_pose(model, maxlen, h36m_2d):
    """(T,17,3) px -> (T,17,3) root-relative, normalised units, in the camera's axes (x right, y down, z depth).
    Sequences longer than maxlen go in maxlen windows, the last one ending on the last frame."""
    x = crop_scale(fill_missing(h36m_2d).astype(np.float32), scale_range=[1, 1]).astype(np.float32)
    T = len(x)
    out = np.zeros((T, 17, 3), np.float32)
    starts = list(range(0, max(T - maxlen, 0) + 1, maxlen))
    if T > maxlen and starts[-1] != T - maxlen:
        starts.append(T - maxlen)
    for s in starts:
        seg = torch.from_numpy(x[s:s + maxlen])[None]
        p = model(seg)
        p = (p + flip_data(model(flip_data(seg)))) / 2               # MotionBERT's test-time flip
        p = p - p[:, :, :1]                                          # rootrel
        out[s:s + seg.shape[1]] = p[0].numpy()
    return out


LIMBS = {'R thigh': (1, 2), 'R shank': (2, 3), 'L thigh': (4, 5), 'L shank': (5, 6),
         'L upper arm': (11, 12), 'L forearm': (12, 13), 'R upper arm': (14, 15), 'R forearm': (15, 16)}


def limb_tails(P, mask):
    """The tails of limb length over the rep: each limb's length / its median over the rep, pooled over the 8 limbs
    -> (5th, 95th percentile) and the worst limb's (5th, 95th).  1.0 = a limb at its usual length."""
    pooled, lo, hi = [], [], []
    for a, b in LIMBS.values():
        ok = mask[:, a] & mask[:, b]
        if ok.sum() < 10:
            continue
        L = np.linalg.norm(P[ok, b] - P[ok, a], axis=-1)
        r = L / np.median(L)
        pooled.append(r)
        lo.append(np.percentile(r, 5))
        hi.append(np.percentile(r, 95))
    if not pooled:
        return dict(p5=np.nan, p95=np.nan, worst_p5=np.nan, worst_p95=np.nan)
    r = np.concatenate(pooled)
    return dict(p5=float(np.percentile(r, 5)), p95=float(np.percentile(r, 95)), worst_p5=float(min(lo)),
                worst_p95=float(max(hi)))


def limb_stats(P, mask):
    """Per limb over the rep: (coefficient of variation of its length %, per-frame out-of-image-plane angle deg).
    CV is scale-free (std / mean), so a lifter in normalised units compares with one in mm."""
    cv, ang = {}, {}
    for name, (a, b) in LIMBS.items():
        ok = mask[:, a] & mask[:, b]
        v = P[:, b] - P[:, a]
        L = np.linalg.norm(v, axis=-1)
        ang[name] = np.where(ok, np.degrees(np.arcsin(np.clip(np.abs(v[:, 2]) / np.maximum(L, 1e-9), 0, 1))), np.nan)
        cv[name] = 100.0 * np.nanstd(L[ok]) / np.nanmean(L[ok]) if ok.sum() >= 10 else np.nan
    return cv, ang


def seq_pa(P, G, mask):
    """Mean error after ONE similarity transform for the whole rep (each frame pelvis-centred first): frame-to-frame
    changes of scale or orientation, which per-frame PA aligns away, count as error here."""
    ok = mask & mask[:, :1]
    if ok.sum() < 20:
        return np.nan
    Pr, Gr = P - P[:, :1], G - G[:, :1]
    A = M._procrustes(Pr[ok], Gr[ok])
    return float(np.linalg.norm(A - Gr[ok], axis=-1).mean())


def views(tree):
    """(user, action, camera) for every 2D file in the tree."""
    out = []
    for f in sorted(glob.glob(os.path.join(tree, '*', '*', 'Analysis', 'keypoints', 'openpose', '*_2d.npz'))):
        parts = f.split(os.sep)
        out.append((parts[-6], parts[-5], os.path.basename(f).split('_')[0]))
    return out


def target_cam(tree, user, action, cam, tri, L_ext, sfi):
    g = np.load(os.path.join(tree, user, action, 'Analysis', 'H36M', 'openpose_tri_h36m.npz'))
    X = g['kps3d'] if tri == 'all' else g['kps3d_loo'][[str(c) for c in g['cameras']].index(cam)]
    rows = np.full((len(sfi), 17, 3), np.nan)
    inside = sfi < len(X)
    rows[inside] = X[sfi[inside]]
    return rows @ L_ext[:3, :3].T + L_ext[:3, 3], g['valid_joint_mask'].astype(bool)      # mm, camera frame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', required=True, help='label=tree,... : pipeline run folders (the first supplies the 2D, '
                                                  'target and calibration; all hold the same trials)')
    ap.add_argument('--configs', default='', help='label:config,... refined configurations to add, e.g. ft16:C19')
    ap.add_argument('--pose', default='pose,pose_lite', help='base MotionBERT models to run (pose, pose_lite)')
    ap.add_argument('--tri', choices=('loo', 'all'), default='loo')
    ap.add_argument('--reference', default=None, help='lifter the others are paired against (default: the first run)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='first N views only (a quick check)')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    runs = dict(r.split('=', 1) for r in a.runs.split(','))
    cfgs = [tuple(c.split(':')) for c in a.configs.split(',') if c]
    pose_names = [p for p in a.pose.split(',') if p]
    ref_tree = next(iter(runs.values()))
    os.makedirs(os.path.join(a.out, 'cache'), exist_ok=True)

    models = {}
    rows = []
    vs = views(ref_tree)[:a.limit or None]
    t0 = time.time()
    for i, (u, act, cam) in enumerate(vs, 1):
        d2 = np.load(os.path.join(ref_tree, u, act, 'Analysis', 'keypoints', 'openpose', f'{cam}_2d.npz'))
        h2d, sfi = d2['h36m_2d'], np.asarray(d2['source_frame_idx'], int)
        preds, L_ext = {}, None
        for label, tree in runs.items():                                   # mesh skeletons, metres -> mm
            p = os.path.join(tree, u, act, 'Analysis', 'PnP', 'openpose', f'{cam}_pnp.npz')
            if os.path.exists(p):
                z = np.load(p)
                preds[f'mesh_{label}'] = z['kps_H36M_placed'] * 1000.0
                L_ext = z['cam_L_ext'] if L_ext is None else L_ext
        for label, cfg in cfgs:
            p = os.path.join(runs[label], u, act, 'Analysis', 'PnP', 'openpose', cfg, f'{cam}_pnp.npz')
            if os.path.exists(p):
                z = np.load(p)
                preds[f'mesh_{label}_{cfg}'] = z['kps_H36M_placed'] * 1000.0
                L_ext = z['cam_L_ext'] if L_ext is None else L_ext
        if L_ext is None:
            print(f'{u}/{act} cam {cam}: no PnP output in any run, skipped', flush=True)
            continue
        for name in pose_names:                                             # base models, cached
            c = os.path.join(a.out, 'cache', f'{act}_cam{cam}_{name}.npy')
            if os.path.exists(c):
                preds[name] = np.load(c)
                continue
            if name not in models:
                models[name] = load_pose(name)
            preds[name] = run_pose(*models[name], h2d)
            np.save(c, preds[name])
        T = min(len(h2d), len(sfi), *(len(v) for v in preds.values()))
        gt, vmask = target_cam(ref_tree, u, act, cam, a.tri, L_ext, sfi[:T])
        mask = vmask[None] & np.isfinite(gt).all(-1)
        for v in preds.values():
            mask &= np.isfinite(v[:T]).all(-1)
        gcv, gang = limb_stats(gt, mask)
        gtail = limb_tails(gt, mask)
        for name, v in preds.items():
            tail = limb_tails(v[:T], mask)
            pa, _ = M.pa_mpjpe(v[:T], gt, mask)
            nm, _ = M.n_mpjpe(v[:T], gt, mask)
            cv, ang = limb_stats(v[:T], mask)
            dang = np.concatenate([(ang[k] - gang[k])[np.isfinite(ang[k] - gang[k])] for k in LIMBS])
            rows.append(dict(user=u, action=act, camera=cam, lifter=name, n_frames=int(np.isfinite(pa).sum()),
                             n_joint_frames=int(mask.sum()), pa_mm=float(np.nanmean(pa)) if np.isfinite(pa).any()
                             else np.nan, n_mm=float(np.nanmean(nm)) if np.isfinite(nm).any() else np.nan,
                             seq_pa_mm=seq_pa(v[:T], gt, mask),
                             limb_cv_pct=float(np.nanmedian(list(cv.values()))),
                             target_limb_cv_pct=float(np.nanmedian(list(gcv.values()))),
                             depth_bias_deg=float(dang.mean()) if len(dang) else np.nan,
                             depth_abs_deg=float(np.abs(dang).mean()) if len(dang) else np.nan,
                             **{f'limb_{k}': x for k, x in tail.items()},
                             **{f'target_limb_{k}': x for k, x in gtail.items()}))
        if i % 25 == 0 or i == len(vs):
            print(f'  {i}/{len(vs)} views, {(time.time() - t0) / 60:.1f} min', flush=True)

    df = pd.DataFrame(rows)
    csv = os.path.join(a.out, f'pose_vs_mesh_{a.tri}.csv')
    df.to_csv(csv, index=False)
    # only views every lifter scored
    full = df.groupby(['user', 'action', 'camera'])['lifter'].transform('count') == df['lifter'].nunique()
    df = df[full & df['pa_mm'].notna()]
    lifters = list(dict.fromkeys(df['lifter']))
    ref = a.reference or f'mesh_{next(iter(runs))}'
    lines = [f'target: {"leave-one-out" if a.tri == "loo" else "all-camera"} triangulation; '
             f'{df.groupby(["user", "action", "camera"]).ngroups} views, {df["user"].nunique()} subjects; '
             f'same joint-frames for every lifter',
             'cohort median of subject medians (mm); vs reference: subjects where the lifter is lower, '
             'median of subject differences, median trial-level difference', '',
             f'{"lifter":22s} {"PA":>6s} {"N-MPJPE":>8s}   vs {ref}: PA wins / med d / trial d   N wins / med d']
    subj = df.groupby(['lifter', 'user'])[['pa_mm', 'n_mm']].median()
    wide = df.pivot_table(index=['user', 'action', 'camera'], columns='lifter', values=['pa_mm', 'n_mm'])
    for lf in lifters:
        s = subj.loc[lf]
        line = f'{lf:22s} {s["pa_mm"].median():6.1f} {s["n_mm"].median():8.1f}'
        if lf != ref and ref in lifters:
            d = s - subj.loc[ref]
            td = (wide['pa_mm'][lf] - wide['pa_mm'][ref]).median()
            line += (f'   {(d["pa_mm"] < 0).sum():2d}/{len(d)} {d["pa_mm"].median():+6.1f} {td:+6.1f}'
                     f'   {(d["n_mm"] < 0).sum():2d}/{len(d)} {d["n_mm"].median():+6.1f}')
        lines.append(line)
    extra = ['seq_pa_mm', 'limb_cv_pct', 'depth_bias_deg', 'depth_abs_deg']
    if all(c in df for c in extra):
        sx = df.groupby(['lifter', 'user'])[extra + ['target_limb_cv_pct']].median()
        lines += ['', 'temporal / depth (cohort median of subject medians):',
                  '  seq PA     one alignment for the whole rep (mm): size or orientation swings count',
                  '  limb CV    median over the 8 limb bones of length std / mean over the rep (%)',
                  '  depth      limb tilt out of the image plane minus the target\'s (deg): negative = flatter',
                  '             than the target (foreshortened); |depth| its mean absolute size', '',
                  f'{"lifter":22s} {"seq PA":>7s} {"limb CV":>8s} {"depth":>7s} {"|depth|":>8s}']
        for lf in lifters:
            s = sx.loc[lf].median()
            lines.append(f'{lf:22s} {s["seq_pa_mm"]:7.1f} {s["limb_cv_pct"]:8.1f} {s["depth_bias_deg"]:+7.1f} '
                         f'{s["depth_abs_deg"]:8.1f}')
        lines.append(f'{"(target itself)":22s} {"":7s} {sx["target_limb_cv_pct"].groupby("user").first().median():8.1f}')
    tails = ['limb_p5', 'limb_p95', 'limb_worst_p5', 'limb_worst_p95']
    if all(c in df for c in tails):
        st = df.groupby(['lifter', 'user'])[tails + ['target_' + c for c in tails]].median()
        lines += ['', 'limb-length tails (each limb\'s length / its median over the rep; cohort median of subject '
                  'medians):', '  p5 / p95      5th and 95th percentile, all 8 limbs pooled',
                  '  worst limb    the lowest 5th and highest 95th percentile of any single limb', '',
                  f'{"lifter":22s} {"p5":>6s} {"p95":>6s}   {"worst p5":>8s} {"worst p95":>9s}']
        for lf in lifters:
            s = st.loc[lf].median()
            lines.append(f'{lf:22s} {s["limb_p5"]:6.2f} {s["limb_p95"]:6.2f}   {s["limb_worst_p5"]:8.2f} '
                         f'{s["limb_worst_p95"]:9.2f}')
        s = st[['target_' + c for c in tails]].groupby('user').first().median()
        lines.append(f'{"(target itself)":22s} {s["target_limb_p5"]:6.2f} {s["target_limb_p95"]:6.2f}   '
                     f'{s["target_limb_worst_p5"]:8.2f} {s["target_limb_worst_p95"]:9.2f}')
    txt = os.path.join(a.out, f'pose_vs_mesh_{a.tri}.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nwrote {csv}\n      {txt}')


if __name__ == '__main__':
    main()
