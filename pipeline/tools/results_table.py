#!/usr/bin/env python3
"""results_table.py -- every step_8 score under OUT_DIR as one long table.

One row per (user, action, camera, configuration): the step_8 metrics (placed / smooth MPJPE in
mm and % of stature, PA-MPJPE, N-MPJPE, per-joint error, bone ratios, signed bias), the
configuration's settings, the shape it used (betas, kid weight) and the trial's frame counts.
Configuration 'step4' is step_4's own placement; the others are step_4b's (configs.py).

This is the store.  Wide views (one column per configuration) are one pivot away and hold one
metric each: tools/compare_configs.py makes them, with the paired differences per subject.

    python3 tools/results_table.py --gt triangulated --out /content/drive/.../results_table_tri_KOREA.csv
    python3 tools/results_table.py --gt mocap --detector yolo --out .../results_table_BioCV.csv
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import OUT_DIR, DETECTORS, metrics_path, pnp_path, final_betas_path, shapes_path, require_out_dir  # noqa: E402
import configs  # noqa: E402

SERIES = ('placed_all', 'placed_conf', 'smooth_all', 'smooth_conf')


def trials_with_metrics(detector, gt):
    """[(user, action, config)] for every error_metrics file, step_4's as config 'step4'."""
    out = []
    for p in sorted(glob.glob(metrics_path('*', '*', detector, gt))):
        u, a = os.path.relpath(p, OUT_DIR).split(os.sep)[:2]
        out.append((u, a, 'step4'))
    for cid in configs.ORDER:
        for p in sorted(glob.glob(metrics_path('*', '*', detector, gt, cid))):
            u, a = os.path.relpath(p, OUT_DIR).split(os.sep)[:2]
            out.append((u, a, cid))
    return out


def rows_for(user, action, cid, detector, gt):
    m = np.load(metrics_path(user, action, detector, gt, None if cid == 'step4' else cid), allow_pickle=True)
    names = [str(x) for x in m['joint_names']]
    cams = [str(c) for c in m['cameras']]
    stature = float(m['stature_mm']) if 'stature_mm' in m.files and np.isfinite(float(m['stature_mm'])) else np.nan
    bones = [str(b) for b in m['bone_names']] if 'bone_names' in m.files else []
    cfg = configs.CONFIGS.get(cid, {})
    rows = []
    for i, cam in enumerate(cams):
        row = dict(user=user, action=action, camera=cam, config=cid, gt=gt, detector=detector,
                   n_cameras=len(cams), angle_deg=round(float(m['angles_deg'][i]), 1), n_frames=int(m['n_frames'][i]),
                   stature_mm='' if np.isnan(stature) else round(stature, 1))
        for key in ('shape', 'placement', 'body', 'exclude_2d', 'w_foot', 'w_floor', 'w_floor_contact', 'note'):
            row[f'cfg_{key}'] = cfg.get(key, 'motionbert' if key == 'shape' else 'pnp' if key == 'placement'
                                     else False if key == 'body' else '' if key in ('exclude_2d', 'note') else 0.0)
        # the shape actually used
        pp = pnp_path(user, action, detector, cam, None if cid == 'step4' else cid)
        shape = None
        if cid != 'step4' and os.path.exists(pp):
            z = np.load(pp, allow_pickle=True)
            shape = z['shape'] if 'shape' in z.files else None
            row['scale_factor'] = round(float(z['scale_factor']), 4) if 'scale_factor' in z.files else ''
            row['n_frames_placed'] = int(z['pnp_ok'].sum()) if 'pnp_ok' in z.files else ''
        if shape is None:
            bp = final_betas_path(user, action, detector, cam)
            shape = np.load(bp, allow_pickle=True)['betas'].ravel()[:10] if os.path.exists(bp) else np.full(10, np.nan)
        for k in range(10):
            row[f'beta_{k:02d}'] = round(float(shape[k]), 4)
        row['kid_weight'] = round(float(shape[10]), 4) if len(shape) > 10 else ''
        for s in SERIES:
            e = float(m[f'err_{s}'][i])
            row[f'err_{s}_mm'] = round(e, 2)
            row[f'err_{s}_pct_stature'] = '' if np.isnan(stature) else round(100 * e / stature, 2)
        row['pa_mpjpe_mm'] = round(float(m['pa_mpjpe'][i]), 2)
        row['n_mpjpe_mm'] = round(float(m['n_mpjpe'][i]), 2)
        for j, nm in enumerate(names):
            v = float(m['perjoint_placed_all'][i, j])
            row[f'errj_{nm}_mm'] = '' if np.isnan(v) else round(v, 2)
        for j, b in enumerate(bones):
            v = float(m['bone_ratio'][i, j])
            row[f'bone_{b}_ratio'] = '' if np.isnan(v) else round(v, 3)
        if 'bias_xyz' in m.files:
            bz = np.nanmean(m['bias_xyz'][i], axis=0)
            row['bias_x_mm'], row['bias_y_mm'], row['bias_z_mm'] = [round(float(v), 1) for v in bz]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--detector', choices=DETECTORS, default='openpose')
    ap.add_argument('--gt', choices=['mocap', 'triangulated'], default='triangulated')
    ap.add_argument('--out', default=None, help='CSV path (default {OUT_DIR}/results_table[_tri].csv)')
    args = ap.parse_args()
    require_out_dir()
    out = args.out or os.path.join(OUT_DIR, 'results_table.csv' if args.gt == 'mocap' else 'results_table_tri.csv')
    rows = []
    for u, a, cid in trials_with_metrics(args.detector, args.gt):
        try:
            rows += rows_for(u, a, cid, args.detector, args.gt)
        except Exception as e:
            print(f'{u}/{a} {cid}: skipped -- {type(e).__name__}: {e}')
    if not rows:
        raise SystemExit(f'no error_metrics under {OUT_DIR} for {args.detector} vs {args.gt}')
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    cfgs = sorted({r['config'] for r in rows}, key=lambda c: (c != 'step4', configs.ORDER.index(c) if c in configs.ORDER else 0))
    print(f'{len(rows)} rows ({len({(r["user"], r["action"]) for r in rows})} trials, {len({r["user"] for r in rows})} subjects, '
          f'configurations {", ".join(cfgs)}) -> {out}')


if __name__ == '__main__':
    main()
