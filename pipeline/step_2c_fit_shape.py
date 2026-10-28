#!/usr/bin/env python3
"""step_2c_fit_shape.py -- one body shape per SUBJECT from the bone lengths of the triangulated target.

MotionBERT's betas are regressed from bounding-box-normalised 2D and come out near the adult
mean shape for every subject (step_2b), so a child gets adult proportions.  A triangulated
skeleton carries the subject's own bone lengths; this fits two shapes to them:

    adult   SMPL's 10 betas                                (cannot reach a child's leg-to-torso ratio)
    kid     the 10 betas + AGORA's kid template as an 11th direction, weight in [0,1]

and each again WITHOUT the hip and shoulder widths (adult_nw, kid_nw): those segments measure the
joint definition (OpenPose vs SMPL's regressor) as much as the body and are the only terms that set
girth, so the _nw shapes take length and proportion from the bones, girth from the prior (--lam-rest
on betas 1-9), and compare Hip->Thorax with the model's hip/shoulder midpoints, as the target builds it.
adult_s24 / kid_s24 are the _nw fits with the model's joints from SMPL's own 24-joint regressor
(smpl_lite.joint_regressor('smpl24')): hips at the femoral head, not up on the pelvis as H36M's are,
so the model's thigh is the target's thigh.  step_4b runs these with joints='smpl24'.
adult_s24g / kid_s24g replace the _s24 ridge (which pins proportions with girth) with a girth prior:
hip, waist and chest girth held to the mean body of the same size and kid weight (--sigma-girth-mm),
betas 1-9 under a light ridge (--lam-rest-girth) so they can fit the torso and limb proportions.
adult_opg / kid_opg are the _s24g fits with OpenPose-style model joints (smpl_lite joints='op': SMPL-24 joint +
per-joint offset calibrated on BioCV, lengths measured standing with the arms down), so the fitted body's
anatomical bones are what OpenPose's lengths imply; _hsg the same with only the hip and shoulder offsets across
the body (joints='op_hs').  _opgw / _hsgw add the hip and shoulder widths back (fair evidence once the model's
joints follow OpenPose's definition).  step_4b runs these with joints='op' / 'op_hs'.
adult_oprg / kid_oprg are the _s24g fits with the OpenPose vertex-regressor hips and shoulders (smpl_lite joints='opr',
models/openpose_regressor.npz; the kid fit iterates on the kid weight the regressor depends on); _oprgw with the
hip and shoulder widths; _oprw the widths WITHOUT the girth prior (girth from the widths and the light ridge only).
step_4b runs these with joints='opr'.

The segment lengths (utils/smpl_lite.SEGMENTS) are medians over every frame of every laid-out rep
of the subject, in the target's own units (mm), so the fitted body is metric and step_4b does not
rescale it.  Nothing here depends on the detector: the target is the triangulated OpenPose skeleton.

    reads   {OUT_DIR}/{user}/{action}/Analysis/H36M/openpose_tri_h36m.npz   (every rep of the subject)
    writes  {OUT_DIR}/{user}/shapes.npz                                     adult (10,), kid (11,), adult_nw, kid_nw, adult_s24, kid_s24, adult_s24g, kid_s24g,
                                                                            adult/kid_opg, _opgw, _hsg, _hsgw, fit residuals

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
from utils.smpl_lite import (SEGMENTS, SEGMENTS_NO_WIDTH, load_linear, midpoint_centres, fit_betas, seg_lengths_mm,
                             data_seg_lengths, mesh_height_m, girth_rings, girths_mm)


def subjects_with_targets(user=None):
    out = {}
    for p in sorted(glob.glob(tri_target_path(user or '*', '*'))):
        a = os.path.dirname(os.path.dirname(os.path.dirname(p)))          # .../{user}/{action}
        u = os.path.basename(os.path.dirname(a))
        out.setdefault(u, []).append(p)
    return out


def fit_subject(user, targets, lin_adult, lin_kid, lin_adult24, lin_kid24, lin_op, rings_adult, rings_kid, args):
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
    tgt_nw = np.array([tgt[SEGMENTS.index(s)] for s in SEGMENTS_NO_WIDTH])
    mid_adult, mid_kid = midpoint_centres(lin_adult), midpoint_centres(lin_kid)
    adult_s24, rms_a24 = fit_betas(lin_adult24, tgt_nw, SEGMENTS_NO_WIDTH, lam=args.lam, sigma_mm=args.sigma_mm,
                                   lam_rest=args.lam_rest)
    kid_s24, rms_k24 = fit_betas(lin_kid24, tgt_nw, SEGMENTS_NO_WIDTH, lam=args.lam, sigma_mm=args.sigma_mm,
                                 lam_rest=args.lam_rest)
    gfit = dict(lam=args.lam, sigma_mm=args.sigma_mm, lam_rest=args.lam_rest_girth, sigma_girth_mm=args.sigma_girth_mm)
    adult_s24g, rms_a24g = fit_betas(lin_adult24, tgt_nw, SEGMENTS_NO_WIDTH, girth=rings_adult, **gfit)
    kid_s24g, rms_k24g = fit_betas(lin_kid24, tgt_nw, SEGMENTS_NO_WIDTH, girth=rings_kid, **gfit)
    kid_g = girths_mm(lin_kid24, kid_s24g, rings_kid)
    op = {}
    for tag, joints, segs, t in (('opg', 'op', SEGMENTS_NO_WIDTH, tgt_nw), ('opgw', 'op', SEGMENTS, tgt),
                                 ('hsg', 'op_hs', SEGMENTS_NO_WIDTH, tgt_nw), ('hsgw', 'op_hs', SEGMENTS, tgt),
                                 ('hwg', 'op_hw', SEGMENTS_NO_WIDTH, tgt_nw)):
        for body, lin, rings in (('adult', lin_op[joints, False], rings_adult), ('kid', lin_op[joints, True], rings_kid)):
            b, rms = fit_betas(lin, t, segs, girth=rings, **gfit)
            op[f'{body}_{tag}'] = b.astype(np.float32)
            op[f'{body}_{tag}_rms_mm'] = rms
            op[f'{body}_{tag}_height_m'] = mesh_height_m(lin, b)
            op[f'{body}_{tag}_fit_mm'] = seg_lengths_mm(lin, b, segs)
    # 'opr' (OpenPose vertex regressor): its hips/shoulders depend on the kid weight, which the kid fit sets --
    # fit with the regressor at the C14 shape's kid weight, move it to the fitted one, refit until it settles
    # _oprw: the widths with NO girth prior (betas 1-9 only under the light lam_rest_girth ridge) -- does girth stay
    # sensible once the widths, now fair evidence, are in?
    for tag, segs, t, prior in (('oprg', SEGMENTS_NO_WIDTH, tgt_nw, True), ('oprgw', SEGMENTS, tgt, True),
                                ('oprw', SEGMENTS, tgt, False)):
        ra, rk = (rings_adult, rings_kid) if prior else (None, None)
        fits = {'adult': (lin_op['opr', False],) + fit_betas(lin_op['opr', False], t, segs, girth=ra, **gfit)}
        kw = float(kid_s24g[-1])
        for _ in range(4):
            lin = load_linear(10, kid=True, joints='opr', opr_kw=kw)
            b, rms = fit_betas(lin, t, segs, girth=rk, **gfit)
            settled, kw = abs(b[-1] - kw) < 0.005, float(b[-1])
            if settled:
                break
        fits['kid'] = (lin, b, rms)
        for body, (lin, b, rms) in fits.items():
            op[f'{body}_{tag}'] = b.astype(np.float32)
            op[f'{body}_{tag}_rms_mm'] = rms
            op[f'{body}_{tag}_height_m'] = mesh_height_m(lin, b)
            op[f'{body}_{tag}_fit_mm'] = seg_lengths_mm(lin, b, segs)
            g = girths_mm(lin, b, rings_kid if body == 'kid' else rings_adult)
            op[f'{body}_{tag}_girth_mm'] = np.array([g[k] for k in g])
    adult_nw, rms_anw = fit_betas(mid_adult, tgt_nw, SEGMENTS_NO_WIDTH, lam=args.lam, sigma_mm=args.sigma_mm,
                                  lam_rest=args.lam_rest)
    kid_nw, rms_knw = fit_betas(mid_kid, tgt_nw, SEGMENTS_NO_WIDTH, lam=args.lam, sigma_mm=args.sigma_mm,
                                lam_rest=args.lam_rest)
    out = shapes_path(user)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, adult=adult.astype(np.float32), kid=kid.astype(np.float32),
             adult_rms_mm=rms_a, kid_rms_mm=rms_k,
             adult_height_m=mesh_height_m(lin_adult, adult), kid_height_m=mesh_height_m(lin_kid, kid),
             segments=np.array([f'{a}-{b}' for a, b in SEGMENTS]), target_mm=tgt, target_iqr_mm=iqr, target_n=n,
             adult_fit_mm=seg_lengths_mm(lin_adult, adult), kid_fit_mm=seg_lengths_mm(lin_kid, kid),
             adult_nw=adult_nw.astype(np.float32), kid_nw=kid_nw.astype(np.float32),
             adult_nw_rms_mm=rms_anw, kid_nw_rms_mm=rms_knw,
             adult_nw_height_m=mesh_height_m(lin_adult, adult_nw), kid_nw_height_m=mesh_height_m(lin_kid, kid_nw),
             segments_nw=np.array([f'{a}-{b}' for a, b in SEGMENTS_NO_WIDTH]),
             adult_nw_fit_mm=seg_lengths_mm(mid_adult, adult_nw, SEGMENTS_NO_WIDTH),
             kid_nw_fit_mm=seg_lengths_mm(mid_kid, kid_nw, SEGMENTS_NO_WIDTH), lam_rest=args.lam_rest,
             adult_s24=adult_s24.astype(np.float32), kid_s24=kid_s24.astype(np.float32),
             adult_s24_rms_mm=rms_a24, kid_s24_rms_mm=rms_k24,
             adult_s24_height_m=mesh_height_m(lin_adult, adult_s24), kid_s24_height_m=mesh_height_m(lin_kid, kid_s24),
             adult_s24_fit_mm=seg_lengths_mm(lin_adult24, adult_s24, SEGMENTS_NO_WIDTH),
             kid_s24_fit_mm=seg_lengths_mm(lin_kid24, kid_s24, SEGMENTS_NO_WIDTH),
             adult_s24g=adult_s24g.astype(np.float32), kid_s24g=kid_s24g.astype(np.float32),
             adult_s24g_rms_mm=rms_a24g, kid_s24g_rms_mm=rms_k24g,
             adult_s24g_height_m=mesh_height_m(lin_adult, adult_s24g), kid_s24g_height_m=mesh_height_m(lin_kid, kid_s24g),
             adult_s24g_fit_mm=seg_lengths_mm(lin_adult24, adult_s24g, SEGMENTS_NO_WIDTH),
             kid_s24g_fit_mm=seg_lengths_mm(lin_kid24, kid_s24g, SEGMENTS_NO_WIDTH),
             kid_s24g_girth_mm=np.array([kid_g[k] for k in kid_g]), girth_names=np.array(list(kid_g)),
             lam_rest_girth=args.lam_rest_girth, sigma_girth_mm=args.sigma_girth_mm, **op,
             n_reps=len(targets), n_frames=int(np.isfinite(kps).all(-1).any(-1).sum()),
             lam=args.lam, sigma_mm=args.sigma_mm, source=np.array('openpose_tri_h36m.npz, all-camera, every rep'))
    return (f'{len(targets)} reps, {int(n.max())} frames: adult betas max |{np.abs(adult).max():.1f}| RMS {rms_a:.1f} mm; '
            f'kid weight {kid[-1]:.2f}, betas max |{np.abs(kid[:-1]).max():.1f}|, RMS {rms_k:.1f} mm; '
            f'no widths: kid weight {kid_nw[-1]:.2f}, RMS {rms_knw:.1f} mm; '
            f'SMPL-24 joints: kid weight {kid_s24[-1]:.2f}, RMS {rms_k24:.1f} mm; '
            f'+ girth prior: kid weight {kid_s24g[-1]:.2f}, RMS {rms_k24g:.1f} mm, waist {kid_g["waist"]:.0f} mm; '
            + '; '.join(f'{k}: kid weight {op["kid_" + k][-1]:.2f}, RMS {op["kid_" + k + "_rms_mm"]:.1f} mm'
                        for k in ('opg', 'opgw', 'hsg', 'hsgw', 'hwg', 'oprg', 'oprgw')))


def cohort(args):
    files = sorted(glob.glob(shapes_path('*')))
    if not files:
        raise SystemExit(f'no {{user}}/shapes.npz under {OUT_DIR} -- fit the subjects first')
    subs, kids, adults, more = [], [], [], {'kid_nw': [], 'kid_s24': [], 'kid_s24g': [], 'kid_opg': [], 'kid_opgw': [], 'kid_hsg': [],
                                            'kid_hsgw': [], 'kid_hwg': [], 'kid_oprg': [], 'kid_oprgw': [], 'kid_oprw': []}
    for p in files:
        z = np.load(p, allow_pickle=True)
        subs.append(os.path.basename(os.path.dirname(p)))
        kids.append(z['kid'])
        adults.append(z['adult'])
        for k in more:
            if k in z.files:
                more[k].append(z[k])
    kids, adults = np.array(kids), np.array(adults)
    med_kid, med_adult = np.median(kids, 0), np.median(adults, 0)
    extra = {}
    for k, v in more.items():
        if len(v) == len(subs):
            extra[k], extra[k + '_all'] = np.median(v, 0).astype(np.float32), np.array(v, dtype=np.float32)
        else:
            print(f'note: {len(subs) - len(v)} subject(s) have no {k} (refit with --force); no cohort {k} written')
    np.savez(cohort_shapes_path(), kid=med_kid.astype(np.float32), adult=med_adult.astype(np.float32),
             subjects=np.array(subs), kid_all=kids.astype(np.float32), adult_all=adults.astype(np.float32),
             note=np.array('median over subjects of step_2c shapes'), **extra)
    w = kids[:, -1]
    print(f'{len(subs)} subjects: kid weight median {np.median(w):.2f}, range {w.min():.2f}-{w.max():.2f}, '
          f'IQR {np.percentile(w, 25):.2f}-{np.percentile(w, 75):.2f}')
    print(f'cohort kid shape -> {cohort_shapes_path()}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', default=None, help='default: every subject with a triangulated target')
    ap.add_argument('--lam', type=float, default=1.0, help='ridge weight toward the mean shape')
    ap.add_argument('--sigma-mm', type=float, default=10.0, help='segment-length noise scale')
    ap.add_argument('--lam-rest', type=float, default=10.0,
                    help='_nw shapes: ridge weight on betas 1-9 (girth and proportions; beta 0 keeps --lam)')
    ap.add_argument('--lam-rest-girth', type=float, default=0.1, help='_s24g shapes: ridge weight on betas 1-9')
    ap.add_argument('--sigma-girth-mm', type=float, default=10.0,
                    help='_s24g shapes: girth prior scale (hip/waist/chest vs the mean body of the same size)')
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
    lin_adult24, lin_kid24 = load_linear(10, joints='smpl24'), load_linear(10, kid=True, joints='smpl24')
    lin_op = {(j, k): load_linear(10, kid=k, joints=j) for j in ('op', 'op_hs', 'op_hw') for k in (False, True)}
    lin_op['opr', False] = load_linear(10, joints='opr', opr_kw=0.0)           # the kid 'opr' fit builds its own
    rings_adult, rings_kid = girth_rings(lin_adult24, np.zeros(10)), girth_rings(lin_kid24, np.r_[np.zeros(10), 0.5])
    n_ok, failed = 0, []
    for u in todo:
        try:
            msg = fit_subject(u, subs[u], lin_adult, lin_kid, lin_adult24, lin_kid24, lin_op, rings_adult,
                              rings_kid, args)
            print(f'{u}: {msg}', flush=True)
            n_ok += 1
        except Exception as e:
            failed.append(u)
            print(f'{u}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\nfitted {n_ok}, already done {len(subs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
