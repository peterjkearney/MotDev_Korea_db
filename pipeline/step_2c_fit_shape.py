#!/usr/bin/env python3
"""step_2c_fit_shape.py -- one body shape per SUBJECT from the bone lengths of the triangulated target.

MotionBERT's betas are regressed from bounding-box-normalised 2D and come out near the adult
mean shape for every subject (step_2b), so a child gets adult proportions.  A triangulated
skeleton carries the subject's own bone lengths; this fits two shapes to them:

    adult   SMPL's 10 betas                                (cannot reach a child's leg-to-torso ratio)
    kid     the 10 betas + AGORA's kid template as an 11th direction, weight in [0,1]

The segment lengths (utils/smpl_lite.SEGMENTS) are medians over every frame of every laid-out rep
of the subject, in the target's own units (mm), so the fitted body is metric and step_4b does not
rescale it.  Nothing here depends on the detector: the target is the triangulated OpenPose skeleton.

    reads   {OUT_DIR}/{user}/{action}/Analysis/H36M/openpose_tri_h36m.npz   (every rep of the subject)
    writes  {OUT_DIR}/{user}/shapes.npz                                     adult (10,), kid (11,), fit residuals

--cohort, after every subject is fitted: the median kid shape over subjects, for the
'cohort_kid' configuration (the version a single-camera product could use, having no
triangulation of its own):

    writes  {OUT_DIR}/cohort_shapes.npz

    python3 step_2c_fit_shape.py                 # every subject with a target; done ones skipped
    python3 step_2c_fit_shape.py --user B010 --force
    python3 step_2c_fit_shape.py --cohort
"""
import argparse
import glob
import os

import numpy as np

from config import OUT_DIR, tri_target_path, shapes_path, cohort_shapes_path, require_out_dir
from utils.smpl_lite import (SEGMENTS, load_linear, fit_betas, seg_lengths_mm, data_seg_lengths, mesh_height_m)


def subjects_with_targets(user=None):
    out = {}
    for p in sorted(glob.glob(tri_target_path(user or '*', '*'))):
        a = os.path.dirname(os.path.dirname(os.path.dirname(p)))          # .../{user}/{action}
        u = os.path.basename(os.path.dirname(a))
        out.setdefault(u, []).append(p)
    return out


def fit_subject(user, targets, lin_adult, lin_kid, args):
    kps = []
    for p in targets:
        z = np.load(p, allow_pickle=True)
        kps.append(z['kps3d'].astype(np.float64))
    kps = np.concatenate(kps)
    tgt, iqr, n = data_seg_lengths(kps, SEGMENTS)
    if np.isfinite(tgt).sum() < 6:
        raise ValueError(f'only {np.isfinite(tgt).sum()} segments with data over {len(targets)} reps')
    adult, rms_a = fit_betas(lin_adult, tgt, SEGMENTS, lam=args.lam, sigma_mm=args.sigma_mm)
    kid, rms_k = fit_betas(lin_kid, tgt, SEGMENTS, lam=args.lam, sigma_mm=args.sigma_mm)
    out = shapes_path(user)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, adult=adult.astype(np.float32), kid=kid.astype(np.float32),
             adult_rms_mm=rms_a, kid_rms_mm=rms_k,
             adult_height_m=mesh_height_m(lin_adult, adult), kid_height_m=mesh_height_m(lin_kid, kid),
             segments=np.array([f'{a}-{b}' for a, b in SEGMENTS]), target_mm=tgt, target_iqr_mm=iqr, target_n=n,
             adult_fit_mm=seg_lengths_mm(lin_adult, adult), kid_fit_mm=seg_lengths_mm(lin_kid, kid),
             n_reps=len(targets), n_frames=int(np.isfinite(kps).all(-1).any(-1).sum()),
             lam=args.lam, sigma_mm=args.sigma_mm, source=np.array('openpose_tri_h36m.npz, all-camera, every rep'))
    return (f'{len(targets)} reps, {int(n.max())} frames: adult betas max |{np.abs(adult).max():.1f}| RMS {rms_a:.1f} mm; '
            f'kid weight {kid[-1]:.2f}, betas max |{np.abs(kid[:-1]).max():.1f}|, RMS {rms_k:.1f} mm')


def cohort(args):
    files = sorted(glob.glob(shapes_path('*')))
    if not files:
        raise SystemExit(f'no {{user}}/shapes.npz under {OUT_DIR} -- fit the subjects first')
    subs, kids, adults = [], [], []
    for p in files:
        z = np.load(p, allow_pickle=True)
        subs.append(os.path.basename(os.path.dirname(p)))
        kids.append(z['kid'])
        adults.append(z['adult'])
    kids, adults = np.array(kids), np.array(adults)
    med_kid, med_adult = np.median(kids, 0), np.median(adults, 0)
    np.savez(cohort_shapes_path(), kid=med_kid.astype(np.float32), adult=med_adult.astype(np.float32),
             subjects=np.array(subs), kid_all=kids.astype(np.float32), adult_all=adults.astype(np.float32),
             note=np.array('median over subjects of step_2c shapes'))
    w = kids[:, -1]
    print(f'{len(subs)} subjects: kid weight median {np.median(w):.2f}, range {w.min():.2f}-{w.max():.2f}, '
          f'IQR {np.percentile(w, 25):.2f}-{np.percentile(w, 75):.2f}')
    print(f'cohort kid shape -> {cohort_shapes_path()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', default=None, help='default: every subject with a triangulated target')
    ap.add_argument('--lam', type=float, default=1.0, help='ridge weight toward the mean shape')
    ap.add_argument('--sigma-mm', type=float, default=10.0, help='segment-length noise scale')
    ap.add_argument('--cohort', action='store_true', help='write the cohort-median shape from the fitted subjects')
    ap.add_argument('--force', action='store_true', help='redo subjects whose shapes.npz exists')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    require_out_dir()
    if args.cohort:
        cohort(args)
        return
    subs = subjects_with_targets(args.user)
    if not subs:
        raise SystemExit(f'no openpose_tri_h36m.npz under {OUT_DIR} for user={args.user or "*"}')
    todo = [u for u in subs if args.force or not os.path.exists(shapes_path(u))]
    print(f'{len(subs)} subject(s) with targets under {OUT_DIR}: {len(todo)} to fit, {len(subs) - len(todo)} already done')
    if args.dry_run:
        for u in todo:
            print(f'  {u}: {len(subs[u])} reps')
        return
    lin_adult, lin_kid = load_linear(10), load_linear(10, kid=True)
    n_ok, failed = 0, []
    for u in todo:
        try:
            print(f'{u}: {fit_subject(u, subs[u], lin_adult, lin_kid, args)}', flush=True)
            n_ok += 1
        except Exception as e:
            failed.append(u)
            print(f'{u}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\nfitted {n_ok}, already done {len(subs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
