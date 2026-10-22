#!/usr/bin/env python3
"""step_5_report.py -- the numbers: how well each anonymiser covers the face,
and what each blur does to 2D keypoints, lifted 3D and triangulated 3D.

Every blur is compared against `reenc` (the same video re-encoded without
blur), on frames where the subject was detected in both, so neither
compression nor a detector's dropped frames is counted as blur error -- the
dropped frames are reported on their own (detect_rate).

report/anonymiser.csv  per method x camera: recall (face visible to the camera
                       -- head turned < 90 deg towards it -- and nose + eyes
                       inside a blur region), by facing angle; false-positive
                       regions (no face in them) and how many blurred part of
                       the subject's body; seconds per frame
report/twod.csv        per condition x detector x camera: detection rate, median
                       and 95th-percentile keypoint shift (px and % of the
                       subject's height in the image) for face and body joints,
                       mean confidence change, body-joint error to the
                       projected mocap
report/twod_joints.csv the same shift per COCO joint, pooled over cameras
report/lift.csv        per condition x detector x camera x lifter: PA-MPJPE to
                       mocap (mocap's 15 marker-backed joints), its change from
                       baseline, and how far blur moved the body joints and the
                       Nose/Head joints (mm, each aligned to mocap)
report/tri.csv         the same for the all-camera triangulation, no alignment
report/summary.md      the headline tables
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from common import (Trial, det_path, faces_path, head_path, lift_path, tri_path, out_dir, procrustes_fit,
                    CAMS, BASELINE, FACE, COCO_NAMES, H36M_NAMES)
from blur import inside
from utils.metrics import pa_mpjpe

BODY = [j for j in range(17) if j not in FACE]
HEAD_H36M = [9, 10]
CONF = 0.3


def anonymiser_table(action, trial, head):
    rows = []
    face3d = head['face'][:, :3]                                      # nose, l_eye, r_eye
    for method in ('haar', 'deface', 'oracle'):
        for ci, cam in enumerate(CAMS):
            p = faces_path(action, method, cam)
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            regs, shape = z['regions'], str(z['shape'])
            base = np.load(det_path(action, BASELINE, 'yolo', cam))['coco']
            pts = trial.cams[cam].project(face3d)                     # (T,3,2)
            ctr = trial.cams[cam].project(head['centre'])             # (T,2)
            facing = head['facing'][:, list(head['cams']).index(cam)]
            covered = np.full(trial.T, np.nan)
            fp = np.zeros(trial.T, int)
            fp_body = np.zeros(trial.T, int)
            for f in range(trial.T):
                r = np.asarray(regs[f]).reshape(-1, 5)
                ok = np.isfinite(pts[f]).all(1)
                q = pts[f][ok] if ok.any() else ctr[f][None]
                if np.isfinite(q).all():
                    covered[f] = inside(r, shape, q).all()
                for reg in r:
                    if np.isfinite(ctr[f]).all() and inside(reg[None], shape, np.vstack([q, ctr[f]])).any():
                        continue
                    fp[f] += 1
                    body = base[f, BODY]
                    fp_body[f] += bool(inside(reg[None], shape, body[body[:, 2] >= CONF, :2]).any())
            vis = (facing < 90) & np.isfinite(covered)
            row = dict(method=method, cam=cam, frames_face_visible=int(vis.sum()),
                       recall=np.mean(covered[vis]) if vis.any() else np.nan)
            for lo, hi in ((0, 30), (30, 60), (60, 90)):
                m = vis & (facing >= lo) & (facing < hi)
                row[f'recall_{lo}_{hi}deg'] = np.mean(covered[m]) if m.sum() >= 5 else np.nan
                row[f'n_{lo}_{hi}deg'] = int(m.sum())
            row.update(fp_per_frame=fp.mean(), frames_with_fp=np.mean(fp > 0),
                       frames_fp_on_body=np.mean(fp_body > 0),
                       sec_per_frame=float(np.mean(z['seconds'])) if len(z['seconds']) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _height_px(coco):
    """Per-frame vertical extent of the confident keypoints -- the subject's size in the image."""
    k = np.where(coco[..., 2:3] >= CONF, coco[..., :2], np.nan)
    return np.nanmax(k[..., 1], 1) - np.nanmin(k[..., 1], 1)


def twod_tables(action, trial, conds, dets):
    rows, joint_rows = [], []
    for d in dets:
        for cam in CAMS:
            bp = det_path(action, BASELINE, d, cam)
            if not os.path.exists(bp):
                continue
            b = np.load(bp)
            m2 = trial.mocap2d(cam)
            hpx = np.nanmedian(_height_px(b['coco']))
            for cond in conds:
                cp = det_path(action, cond, d, cam)
                if not os.path.exists(cp):
                    continue
                c = np.load(cp)
                both = np.isfinite(b['match_px']) & np.isfinite(c['match_px'])
                ok = both[:, None] & (b['coco'][..., 2] >= CONF) & (c['coco'][..., 2] >= CONF)
                shift = np.linalg.norm(c['coco'][..., :2] - b['coco'][..., :2], axis=-1)
                row = dict(cond=cond, det=d, cam=cam, detect_rate=np.isfinite(c['match_px']).mean(),
                           detect_rate_baseline=np.isfinite(b['match_px']).mean(), height_px=hpx)
                for part, js in (('face', FACE), ('body', BODY)):
                    s = shift[:, js][ok[:, js]]
                    row[f'{part}_shift_med_px'] = np.median(s) if len(s) else np.nan
                    row[f'{part}_shift_p95_px'] = np.percentile(s, 95) if len(s) else np.nan
                    row[f'{part}_shift_med_pct'] = 100 * row[f'{part}_shift_med_px'] / hpx
                    dc = (c['coco'][..., 2] - b['coco'][..., 2])[:, js][both]
                    row[f'{part}_conf_change'] = dc.mean() if dc.size else np.nan
                    row[f'{part}_frames_lost'] = np.mean((b['coco'][both][:, js, 2] >= CONF)
                                                         & (c['coco'][both][:, js, 2] < CONF)) if both.any() else np.nan
                err = []
                for h, cj in [(1, 12), (2, 14), (3, 16), (4, 11), (5, 13), (6, 15),
                              (11, 5), (12, 7), (13, 9), (14, 6), (15, 8), (16, 10)]:
                    m = both & (c['coco'][:, cj, 2] >= CONF) & np.isfinite(m2[:, h]).all(1)
                    err += list(np.linalg.norm(c['coco'][m, cj, :2] - m2[m, h], axis=1))
                row['body_err_to_mocap_px'] = np.median(err) if err else np.nan
                rows.append(row)
                for j in range(17):
                    s = shift[:, j][ok[:, j]]
                    joint_rows += [dict(cond=cond, det=d, cam=cam, joint=COCO_NAMES[j], shift_px=v,
                                        shift_pct=100 * v / hpx) for v in s]
    twod = pd.DataFrame(rows)
    jr = pd.DataFrame(joint_rows)
    joints = (jr.groupby(['cond', 'det', 'joint'], sort=False)
              .agg(median_px=('shift_px', 'median'), p95_px=('shift_px', lambda v: np.percentile(v, 95)),
                   median_pct=('shift_pct', 'median')).reset_index()) if len(jr) else jr
    return twod, joints


def _aligned(pred, gt, mask):
    """Each frame of pred similarity-aligned to gt on the masked joints, all 17 joints moved."""
    out = np.full(pred.shape, np.nan)
    for f in range(len(pred)):
        m = mask & np.isfinite(gt[f]).all(1) & np.isfinite(pred[f]).all(1)
        if m.sum() >= 4:
            s, R, t = procrustes_fit(pred[f, m], gt[f, m])
            out[f] = s * pred[f] @ R + t
    return out


def lift_table(action, trial, conds, dets):
    rows = []
    mask = trial.mocap_mask
    body = [j for j in range(17) if mask[j]]
    for d in dets:
        for cam in CAMS:
            bp = lift_path(action, BASELINE, d, cam)
            if not os.path.exists(bp):
                continue
            b = np.load(bp)
            for cond in conds:
                cp = lift_path(action, cond, d, cam)
                if not os.path.exists(cp):
                    continue
                c = np.load(cp)
                both = b['detected'] & c['detected']
                for lf in [k for k in c.files if k not in ('detected',) and k in b.files]:
                    P_b, P_c = b[lf].astype(float), c[lf].astype(float)
                    m = np.broadcast_to(mask, P_c.shape[:2]) & both[:, None]
                    e_b, _ = pa_mpjpe(P_b, trial.mocap, m)
                    e_c, _ = pa_mpjpe(P_c, trial.mocap, m)
                    A_b, A_c = _aligned(P_b, trial.mocap, mask), _aligned(P_c, trial.mocap, mask)
                    sh = np.linalg.norm(A_c - A_b, axis=-1)[both]
                    rows.append(dict(cond=cond, det=d, cam=cam, lifter=lf, frames=int(both.sum()),
                                     pa_mpjpe_baseline=np.nanmean(e_b), pa_mpjpe=np.nanmean(e_c),
                                     pa_mpjpe_change=np.nanmean(e_c) - np.nanmean(e_b),
                                     body_shift_mm=np.nanmean(sh[:, body]),
                                     head_shift_mm=np.nanmean(sh[:, HEAD_H36M])))
    return pd.DataFrame(rows)


def tri_table(action, trial, conds, dets):
    rows = []
    mask = trial.mocap_mask
    for d in dets:
        bp = tri_path(action, BASELINE, d)
        if not os.path.exists(bp):
            continue
        Xb = np.load(bp)['X']
        for cond in conds:
            cp = tri_path(action, cond, d)
            if not os.path.exists(cp):
                continue
            Xc = np.load(cp)['X']
            def err(X):
                e = np.linalg.norm(X - trial.mocap, axis=-1)[:, mask]
                return np.nanmean(e)
            sh = np.linalg.norm(Xc - Xb, axis=-1)
            row = dict(cond=cond, det=d, mpjpe_baseline=err(Xb), mpjpe=err(Xc), mpjpe_change=err(Xc) - err(Xb),
                       solved=np.isfinite(Xc[..., 0]).mean(), solved_baseline=np.isfinite(Xb[..., 0]).mean(),
                       body_shift_mm=np.nanmean(sh[:, mask]), head_shift_mm=np.nanmean(sh[:, HEAD_H36M]))
            for j in range(17):
                row[f'shift_{H36M_NAMES[j]}'] = np.nanmean(sh[:, j])
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    args = ap.parse_args()
    trial = Trial(args.action)
    head = np.load(head_path(args.action))
    conds = [c for c in ('raw', 'haar', 'deface', 'oracle')
             if glob.glob(out_dir(args.action, '2d', c, '*', '*.npz'))]
    dets = [d for d in ('yolo', 'rtmpose', 'mediapipe', 'openpose')
            if glob.glob(out_dir(args.action, '2d', BASELINE, d, '*.npz'))]
    rep = out_dir(args.action, 'report')
    os.makedirs(rep, exist_ok=True)

    anon = anonymiser_table(args.action, trial, head)
    twod, joints = twod_tables(args.action, trial, conds, dets)
    lift = lift_table(args.action, trial, conds, dets)
    tri = tri_table(args.action, trial, conds, dets)
    for name, df in (('anonymiser', anon), ('twod', twod), ('twod_joints', joints), ('lift', lift), ('tri', tri)):
        df.to_csv(os.path.join(rep, f'{name}.csv'), index=False, float_format='%.4g')

    fmt = lambda df: df.to_markdown(index=False, floatfmt='.3g') if len(df) else '(none)'   # noqa: E731
    md = [f'# Face blur study -- {args.action}, cameras {" ".join(CAMS)}\n',
          f'Baseline: `{BASELINE}`.  Conditions: {", ".join(conds)}.  Detectors: {", ".join(dets)}.\n',
          '## Anonymiser coverage (mean over cameras)\n',
          fmt(anon.groupby('method').mean(numeric_only=True).reset_index()) if len(anon) else '(none)',
          '\n## 2D keypoint shift vs baseline (median over cameras)\n',
          fmt(twod.groupby(['cond', 'det']).median(numeric_only=True).reset_index()[
              ['cond', 'det', 'detect_rate', 'detect_rate_baseline', 'face_shift_med_px', 'face_shift_p95_px',
               'body_shift_med_px', 'body_shift_p95_px', 'body_shift_med_pct', 'face_frames_lost',
               'body_err_to_mocap_px']]) if len(twod) else '(none)',
          '\n## Lifted 3D (mean over cameras)\n',
          fmt(lift.groupby(['cond', 'det', 'lifter']).mean(numeric_only=True).reset_index()[
              ['cond', 'det', 'lifter', 'pa_mpjpe_baseline', 'pa_mpjpe', 'pa_mpjpe_change',
               'body_shift_mm', 'head_shift_mm']]) if len(lift) else '(none)',
          '\n## Triangulated 3D\n',
          fmt(tri[['cond', 'det', 'mpjpe_baseline', 'mpjpe', 'mpjpe_change', 'solved', 'body_shift_mm',
                   'head_shift_mm']]) if len(tri) else '(none)']
    with open(os.path.join(rep, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\n-> {rep}')


if __name__ == '__main__':
    main()
