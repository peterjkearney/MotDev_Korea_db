#!/usr/bin/env python3
"""build_child_gt.py -- child 3D ground truth for the MotionBERT test.

For every session (subject) in the Korea B set:

  1. calibrate the three cameras from the pose data (camcal),
  2. triangulate every frame of every rep with those cameras, dropping a view
     when it disagrees with the other two (catches left/right label swaps),
  3. measure the child's crown-to-sole stature from upright standing frames,
  4. assess the session, each rep and each frame against quality gates,
  5. save each usable rep with, per camera c:
       - the pipeline INPUT: camera c's OpenPose 2D converted to the H36M-17
         (x, y, conf) array step_1 would have produced, plus intrinsics and
         stature -- enough to run step_2a onwards;
       - the TARGET: the skeleton triangulated from the OTHER two cameras,
         expressed in camera c's own coordinate frame.

Why there is no YOLO and why the target is leave-one-out.  The B_video files
are rendered OpenPose stick figures on black, not camera footage (the dataset
is anonymised), so step_1 cannot run and the only 2D is OpenPose's.  Input and
target would then share a detector: a target triangulated from all three
cameras contains camera c's own detection errors, and scoring camera c's
prediction against it would flatter the model.  Triangulating the target from
the other two cameras removes that.  The all-camera skeleton is saved too, for
proportions, stature and QA, but not for scoring.

Scale.  Images fix everything but overall scale, and for this test scale
cancels: the pipeline is fed the stature measured from the SAME reconstruction
the target comes from, and PnP on a stature-scaled skeleton is
scale-equivariant, so errors expressed as a % of stature are unaffected.  The
default (--scale stature) sets each session's units so the measured stature
equals the cohort median (1.02 m).  That is only a choice of unit -- the model
is fed the same number -- but it keeps every session close to true metres,
which matters for the RTS smoother's fixed metre-valued noise settings.
--scale aimpoint instead anchors on camera 1 being 3 m from the rig's aim
point (known to be 25% off in some sessions).

Stature is measured, not assumed: ear height above the fitted floor plus a
crown offset proportional to the measured ear-to-ear width, in upright frames.
A proportional offset keeps the measurement in the reconstruction's own units;
a fixed "+10 cm" would quietly break the scale argument above.

Joint conventions.  H36M-17 is built from BODY_25 exactly as PnP_depth_clean's
step_1 coco2h36m builds it from YOLO: pelvis = mid-hip, thorax = mid-shoulder,
spine = mid(pelvis, thorax), head = mid-ear.  `eval_joint_mask_h36m` excludes
Nose and Head, matching step_0's mocap `valid_joint_mask`.

Usage:
    python build_child_gt.py --out /Volumes/Expansion/MotorDevelopment/Korea/B/GT3D
    python build_child_gt.py --out ... --subjects B010,B011
Resumable: sessions with an existing session_summary.json are skipped unless
--overwrite.  Per-session and per-rep QA tables are rebuilt at the end.
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camcal as cc

DATA_DIR = '/Volumes/Expansion/MotorDevelopment/Korea/B/Aligned'
JSON_DIR = '/Volumes/Expansion/MotorDevelopment/Korea/B'
VIDEO_DIR = '/Volumes/Expansion/MotorDevelopment/Korea/B_video'

IMAGE_SIZE = (1920, 1080)          # the OpenPose keypoints' pixel frame
VIDEO_SIZE = (1280, 720)           # the re-encoded videos YOLO will run on
ANCHOR_DISTANCE = 3.0              # m, camera 1 -> aim point (--scale aimpoint)
COHORT_STATURE = 1.02              # m, median stature at ~3.75 y (WHO)
CROWN_K = 0.9                      # crown above ear level, x ear-to-ear width

GATES = dict(
    # session -- calibration data (enforced by camcal.select_spread)
    calib_conf_thresh=0.4,         # a joint seen above this in >= 2 views is a correspondence
    calib_min_corr_per_frame=4,    # frames with fewer carry almost nothing
    min_calib_frames_strict=80,    # strict tier: complete skeletons in all 3 views
    min_calib_frames=40,           # fallback tier
    min_calib_corr=1500,
    max_reproj_over_epipolar=1.6,  # calibration reprojection vs the model-free noise floor
    max_abs_roll_deg=5.0,          # tripods: roll must come out near zero
    max_long_bone_cv_pct=8.0,      # depth distortion makes bones breathe
    min_floor_inlier_frac=0.6,
    min_upright_frames=15,         # to measure stature at all
    # triangulation
    conf_thresh=0.5,
    reproj_thresh_px=15.0,         # per view, 1080p pixels
    # frame
    min_core_joints=10,            # of the 12 H36M limb joints
    max_frame_bone_dev=0.15,       # any bone vs its session median
    # rep
    min_usable_frames=30,          # one second at the dataset's 30 fps
    min_usable_frac=0.25,
    max_rep_reproj_over_session=1.5,   # catches a camera bumped mid-session
    max_rep_bone_dev=0.12,             # catches a different person / bad tracking
)

H36M_NAMES = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
              'Spine', 'Thorax', 'Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
H36M_CORE = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]   # limb joints, no midpoints
LONG_BONES = ('torso', 'upperarm', 'forearm', 'thigh', 'shank')


# --------------------------------------------------------------------------
# Joint conversion and anthropometry
# --------------------------------------------------------------------------

def body25_to_h36m(P, ok):
    """BODY_25 (N,25,3) -> H36M-17 (N,17,3), mirroring step_1's coco2h36m.

    Midpoints commute with rigid transforms, so this can be applied in any
    frame.  A composite joint is valid only if all its parts are.
    """
    N = P.shape[0]
    out = np.full((N, 17, 3), np.nan)
    valid = np.zeros((N, 17), bool)
    src = {0: [9, 12], 1: [9], 2: [10], 3: [11], 4: [12], 5: [13], 6: [14],
           8: [2, 5], 9: [0], 10: [17, 18],
           11: [5], 12: [6], 13: [7], 14: [2], 15: [3], 16: [4]}
    for k, idx in src.items():
        out[:, k] = P[:, idx].mean(axis=1)
        valid[:, k] = ok[:, idx].all(axis=1)
    out[:, 7] = 0.5 * (out[:, 0] + out[:, 8])
    valid[:, 7] = valid[:, 0] & valid[:, 8]
    out[~valid] = np.nan
    return out, valid


def openpose_to_h36m_2d(xy, conf):
    """OpenPose BODY_25 2D -> the (T,17,3) x, y, conf array step_1 would save.

    Same midpoint conventions as step_1's coco2h36m, with one difference that
    matters: OpenPose writes a missing joint as (0, 0) with confidence 0,
    which YOLO never does.  coco2h36m averages positions and confidences
    unconditionally, so a midpoint with one missing part would land halfway
    to the image corner with about half the present part's confidence --
    enough to pass step_4's PnP gate of 0.4.  Here a composite joint with ANY
    missing part is itself written as missing (0, 0, 0), which MotionBERT's
    crop_scale also ignores (it keys on conf != 0).

    xy (T,25,2), conf (T,25) -> (T,17,3)
    """
    T = xy.shape[0]
    present = (conf > 0) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    src = {0: [9, 12], 1: [9], 2: [10], 3: [11], 4: [12], 5: [13], 6: [14],
           8: [2, 5], 9: [0], 10: [17, 18],
           11: [5], 12: [6], 13: [7], 14: [2], 15: [3], 16: [4]}
    out = np.zeros((T, 17, 3), np.float32)
    have = np.zeros((T, 17), bool)
    for k, idx in src.items():
        out[:, k, :2] = xy[:, idx].mean(axis=1)
        out[:, k, 2] = conf[:, idx].mean(axis=1)
        have[:, k] = present[:, idx].all(axis=1)
    out[:, 7] = 0.5 * (out[:, 0] + out[:, 8])
    have[:, 7] = have[:, 0] & have[:, 8]
    out[~have] = 0.0
    return out


def bone_lengths(P, ok):
    """{bone key: (N, n_sides) lengths}, NaN where an end is missing."""
    out = {}
    for a, b, key in cc.BONES:
        ln = np.linalg.norm(P[:, a] - P[:, b], axis=1)
        ln[~(ok[:, a] & ok[:, b])] = np.nan
        out.setdefault(key, []).append(ln)
    return {k: np.stack(v, axis=1) for k, v in out.items()}


def _angle_deg(u, v):
    cu = np.einsum('ij,ij->i', u, v) / (
        np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cu, -1, 1)))


def measure_stature(P, ok):
    """Crown-to-sole height from upright standing frames.

    P (N,25,3) in the floor frame (z up, floor at z=0).  Upright = trunk near
    vertical, both knees straight, both ankles down at the floor.  Crown =
    ear-midpoint height + CROWN_K x median ear-to-ear width.  Strict criteria
    first, relaxed if too few frames qualify.
    """
    need = [1, 8, 9, 10, 11, 12, 13, 14, 17, 18]
    base = ok[:, need].all(axis=1)
    up = np.broadcast_to([0.0, 0.0, 1.0], (len(P), 3))
    with np.errstate(invalid='ignore'):
        tilt = _angle_deg(P[:, 1] - P[:, 8], up)
        knee_r = _angle_deg(P[:, 9] - P[:, 10], P[:, 11] - P[:, 10])
        knee_l = _angle_deg(P[:, 12] - P[:, 13], P[:, 14] - P[:, 13])
        neck_z = P[:, 1, 2]
        low = (P[:, 11, 2] < 0.15 * neck_z) & (P[:, 14, 2] < 0.15 * neck_z)
        ear_z = 0.5 * (P[:, 17, 2] + P[:, 18, 2])
        ear_w = np.linalg.norm(P[:, 17] - P[:, 18], axis=1)

    sel = np.zeros(len(P), bool)
    for tilt_max, knee_min, label in ((12, 160, 'strict'), (20, 150, 'relaxed')):
        sel = (base & (tilt < tilt_max) & (knee_r > knee_min) & (knee_l > knee_min)
               & low & (neck_z > 0))
        if sel.sum() >= GATES['min_upright_frames']:
            w = float(np.median(ear_w[sel]))
            crown = ear_z[sel] + CROWN_K * w
            q25, q75 = np.percentile(crown, [25, 75])
            return dict(stature=float(np.median(crown)), n_upright=int(sel.sum()),
                        iqr=float(q75 - q25), criteria=label, ear_width=w)
    return dict(stature=float('nan'), n_upright=int(sel.sum()), iqr=float('nan'),
                criteria='none', ear_width=float('nan'))


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------

def build_video_index(video_dir):
    idx = {}
    for p in glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True):
        name = os.path.basename(p)
        if name.startswith('._'):
            continue
        idx.setdefault(os.path.splitext(name)[0].replace('[SHANA]', ''), p)
    return idx


def frame_mapping(json_dir, stub, cam, aligned_xy):
    """Source-video frame index of every aligned row, for one camera.

    merge_cameras trimmed each camera's start by its sync lag but did not
    save the lags, so recover the offset by matching the aligned keypoints
    against the camera's own JSON (identical floats).  Returns (video_idx,
    json_frame_index), or (None, None) if no unique match.
    """
    path = os.path.join(json_dir, f'{stub}_{cam}.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    xy = np.array([fr['skeleton'][0]['pose'] for fr in d['data']]).reshape(-1, 25, 2)
    fi = np.array([fr['frame_index'] for fr in d['data']])
    T = len(aligned_xy)
    hits = [o for o in range(len(xy) - T + 1) if np.array_equal(xy[o:o + T], aligned_xy)]
    if len(hits) != 1:
        return None, None
    rows = hits[0] + np.arange(T)
    return rows, fi[rows]


def _sampson(F, p1, p2):
    p1h = np.hstack([p1, np.ones((len(p1), 1))])
    p2h = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1, Ftx2 = p1h @ F.T, p2h @ F
    num = np.einsum('ij,ij->i', p2h, Fx1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def epipolar_floor(obs):
    """Model-free 2D noise level: median Sampson distance under a RANSAC F."""
    meds = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        m = obs.vis[:, a, :] & obs.vis[:, b, :]
        p1, p2 = obs.xy[:, a][m], obs.xy[:, b][m]
        if len(p1) < 50:
            continue
        F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 3.0, 0.999, 5000)
        if F is None or F.shape != (3, 3):
            continue
        meds.append(float(np.median(_sampson(F, p1, p2))))
    return float(np.median(meds)) if meds else float('nan')


def scaled(cams, s):
    return cc.CameraSet(f=cams.f.copy(), k1=cams.k1.copy(), k2=cams.k2.copy(),
                        rvec=cams.rvec.copy(), t=cams.t * s,
                        cx=cams.cx.copy(), cy=cams.cy.copy(),
                        image_size=cams.image_size)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------

def calibrate(subject, args):
    # Two tiers.  Measured on common clean frames, calibrating only on frames
    # where the whole body is well detected in every camera is slightly more
    # accurate (~5% lower held-out-camera error on B010, a tie on B011): a
    # complete, confident skeleton is a decent proxy for "OpenPose is having a
    # good frame".  But that rule starves sessions where some joints are rarely
    # seen by all three cameras, so fall back to treating every joint seen by
    # two or more cameras as a correspondence.  Never worse than the strict
    # rule alone; the fallback is recorded in the summary.
    try:
        cand = cc.load_candidates(args.data, subjects=[subject], conf_thresh=0.6,
                                  min_joints=20, stride=6, verbose=False)
        obs = cc.select_spread(cand, n_frames=250, min_frames=GATES['min_calib_frames_strict'])
        rule = 'strict'
    except cc.InsufficientData:
        cand = cc.load_candidates(args.data, subjects=[subject],
                                  conf_thresh=GATES['calib_conf_thresh'],
                                  min_views=2, min_corr=GATES['calib_min_corr_per_frame'],
                                  stride=args.calib_stride, verbose=False)
        obs = cc.select_spread(cand, n_frames=args.calib_max_frames,
                               target_corr=args.calib_corr,
                               min_frames=GATES['min_calib_frames'],
                               min_corr_total=GATES['min_calib_corr'])
        rule = 'fallback'
    fb = cc.focal_bounds_for_camera('sony-rx100', IMAGE_SIZE[0])
    f0 = cc.init_focal_from_spec(fb, 3, 'wide')
    base12 = float(np.linalg.norm(cc.DEFAULT_LAYOUT[1] - cc.DEFAULT_LAYOUT[0]))
    cams0 = cc.init_from_essential(obs, f0, base12, image_size=IMAGE_SIZE,
                                   verbose=False)
    opts = cc.BAOptions(fit_k1=False, f_bounds=fb, use_bone_prior=False,
                        verbose=0, max_nfev=200)
    cams, L, X, _, prob = cc.solve(cams0, obs, opts=opts, sigma_px=4.0,
                                   verbose=False)
    normal, offset, inl, _ = cc.fit_floor(X, prob, obs, thresh=0.04)
    # Put the reconstruction in roughly-metric units before anything else;
    # the final unit is chosen after stature is measured.
    cams, X, L, _, s, info = cc.rescale_to_convergence(
        cams, X, L, prob, normal, offset, ANCHOR_DISTANCE, 0, 'ground',
        verbose=False)
    return dict(obs=obs, cams=cams, X=X, L=L, prob=prob, normal=normal, rule=rule,
                offset=offset * s, floor_inlier_frac=float(inl.mean()),
                axis_miss_m=info['miss'])


def process_session(subject, args, video_index):
    t0 = time.time()
    sess_dir = os.path.join(args.out, subject)
    os.makedirs(sess_dir, exist_ok=True)
    summary = dict(subject=subject, status='failed', usable=False, reasons=[],
                   gates=GATES, scale_mode=args.scale)
    reps_out = []

    def finish():
        summary['seconds'] = round(time.time() - t0, 1)
        summary['reps'] = reps_out
        with open(os.path.join(sess_dir, 'session_summary.json'), 'w') as f:
            json.dump(jsonable(summary), f, indent=1)
        return summary

    # ---- 1. calibrate ----------------------------------------------------
    try:
        cal = calibrate(subject, args)
    except cc.InsufficientData as e:
        summary.update(status='skipped', reasons=[str(e).split('.')[0]])
        return finish()
    except Exception as e:
        summary.update(status='failed', reasons=[f'calibration: {type(e).__name__}: {e}'],
                       traceback=traceback.format_exc())
        return finish()

    cams, prob, obs = cal['cams'], cal['prob'], cal['obs']
    reproj = cc.reprojection_errors(cams, prob, cal['X'])
    calib_med = np.array([reproj[c]['median'] for c in range(3)])
    epi = epipolar_floor(obs)
    bstats = cc.bone_stats(prob, cal['X'], cal['L'])
    long_cv = float(np.median([bstats[k]['cv_pct'] for k in LONG_BONES if k in bstats]))

    # ---- 2. triangulate every frame of every rep -------------------------
    files = sorted(f for f in os.listdir(args.data)
                   if f.startswith(subject + '_') and f.endswith('.npz')
                   and not f.startswith('.'))
    reps = []
    for fn in files:
        d = np.load(os.path.join(args.data, fn))
        xy = np.stack([d['xy1'], d['xy2'], d['xy3']], axis=1).astype(float)
        conf = np.stack([d['score1'], d['score2'], d['score3']], axis=1).astype(float)
        vis = (conf > GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        X, ok, used, err = cc.triangulate_robust(cams, xy, vis, GATES['reproj_thresh_px'])
        # Leave-one-camera-out targets.  The model's input is camera c's
        # OpenPose 2D, so camera c's detections must not also shape the
        # target it is scored against: triangulate from the other two only.
        X_loo, ok_loo = [], []
        for c in range(3):
            v = vis.copy()
            v[:, c, :] = False
            Xl, okl, _, _ = cc.triangulate_robust(cams, xy, v, GATES['reproj_thresh_px'])
            X_loo.append(Xl)
            ok_loo.append(okl)
        reps.append(dict(stub=fn[:-4], xy=xy, conf=conf, X=X, ok=ok,
                         used=used, err=err,
                         X_loo=np.stack(X_loo), ok_loo=np.stack(ok_loo)))

    # ---- 3. stature, then the final unit ---------------------------------
    floor0 = cc.gravity_frame(cams, cal['normal'], cal['offset'])
    Rw, org = floor0['R_world_to_floor'], floor0['origin']
    allX = np.concatenate([r['X'] for r in reps])
    allok = np.concatenate([r['ok'] for r in reps])
    st = measure_stature((allX - org) @ Rw.T, allok)

    s = 1.0
    if args.scale == 'stature' and np.isfinite(st['stature']):
        s = COHORT_STATURE / st['stature']
    cams = scaled(cams, s)
    offset = cal['offset'] * s
    for r in reps:
        r['X'] = r['X'] * s
        r['X_loo'] = r['X_loo'] * s
    stature = st['stature'] * s
    floor = cc.gravity_frame(cams, cal['normal'], offset)
    Rw, org = floor['R_world_to_floor'], floor['origin']

    # session-median bone lengths, for the frame and rep gates
    allX = np.concatenate([r['X'] for r in reps])
    sess_bones = {k: float(np.nanmedian(v))
                  for k, v in bone_lengths(allX, allok).items()}

    # ---- 4. session gates ------------------------------------------------
    ratio = float(calib_med.max() / epi) if np.isfinite(epi) and epi > 0 else float('nan')
    roll = float(np.abs(floor['roll_deg']).max())
    checks = [
        (ratio <= GATES['max_reproj_over_epipolar'],
         f'reprojection {calib_med.max():.2f}px is {ratio:.2f}x the epipolar floor '
         f'{epi:.2f}px (max {GATES["max_reproj_over_epipolar"]})'),
        (roll <= GATES['max_abs_roll_deg'],
         f'camera roll {roll:.1f} deg (max {GATES["max_abs_roll_deg"]})'),
        (long_cv <= GATES['max_long_bone_cv_pct'],
         f'long-bone CV {long_cv:.1f}% (max {GATES["max_long_bone_cv_pct"]})'),
        (cal['floor_inlier_frac'] >= GATES['min_floor_inlier_frac'],
         f'floor inliers {cal["floor_inlier_frac"]:.2f} (min {GATES["min_floor_inlier_frac"]})'),
        (np.isfinite(st['stature']),
         f'stature unmeasurable: {st["n_upright"]} upright frames '
         f'(need {GATES["min_upright_frames"]})'),
    ]
    session_ok = all(p for p, _ in checks)
    summary.update(
        status='ok', usable=session_ok,
        reasons=[msg for p, msg in checks if not p],
        n_calib_frames=int(obs.xy.shape[0]), calib_rule=cal['rule'],
        calib_median_reproj_px=calib_med, epipolar_floor_px=epi,
        reproj_over_epipolar=ratio, long_bone_cv_pct=long_cv,
        floor_inlier_frac=cal['floor_inlier_frac'],
        focal_px_1080=cams.f, focal_35mm=cc.focal_35mm_equiv(cams.f),
        camera_height=floor['height'], camera_tilt_deg=floor['tilt_deg'],
        camera_roll_deg=floor['roll_deg'], camera_plan=floor['plan'],
        axis_miss=cal['axis_miss_m'] * s,
        stature=stature, stature_measure=st, unit_scale=s,
        session_bone_median=sess_bones)

    if np.isfinite(stature):
        with open(os.path.join(sess_dir, 'user_meta.json'), 'w') as f:
            json.dump({'stature_m': round(float(stature), 4),
                       'note': 'crown-to-sole, measured from the triangulated '
                               'reconstruction, in the same units as the saved '
                               'skeletons (see session_summary.json unit_scale)'},
                      f, indent=1)

    # ---- 5. per rep: gates, camera frames, save ---------------------------
    R_wc = cams.R                                           # (3,3,3)
    K1080 = np.stack([[[cams.f[c], 0, cams.cx[c]], [0, cams.f[c], cams.cy[c]], [0, 0, 1.0]]
                      for c in range(3)])
    sv = VIDEO_SIZE[0] / IMAGE_SIZE[0]
    K720 = K1080.copy()
    K720[:, :2, :] *= sv

    for r in reps:
        X, ok, T = r['X'], r['ok'], r['X'].shape[0]
        Xh, okh = body25_to_h36m(X, ok)

        # frame gates: every bone (both sides) against its session median
        bl = bone_lengths(X, ok)
        dev = np.concatenate([np.abs(bl[k] / sess_bones[k] - 1) for k in bl], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN rows are expected
            frame_dev = np.nanmax(dev, axis=1)
            frame_reproj = np.nanmedian(r['err'].reshape(T, -1), axis=1)
            rep_med = np.array([np.nanmedian(r['err'][:, c]) for c in range(3)])
            rep_bone = {k: np.nanmedian(v) for k, v in bl.items()}
        n_core = okh[:, H36M_CORE].sum(axis=1)
        frame_ok = ((n_core >= GATES['min_core_joints'])
                    & (frame_dev <= GATES['max_frame_bone_dev']))   # NaN compares False

        # the same frame gates, applied to each camera's leave-one-out target
        frame_ok_loo = np.zeros((3, T), bool)
        Xh_loo, okh_loo = [], []
        for c in range(3):
            Xl, okl = r['X_loo'][c], r['ok_loo'][c]
            blc = bone_lengths(Xl, okl)
            devc = np.concatenate([np.abs(blc[k] / sess_bones[k] - 1) for k in blc], axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                fdev = np.nanmax(devc, axis=1)
            hl, hok = body25_to_h36m(Xl, okl)
            Xh_loo.append(hl)
            okh_loo.append(hok)
            frame_ok_loo[c] = ((hok[:, H36M_CORE].sum(axis=1) >= GATES['min_core_joints'])
                               & (fdev <= GATES['max_frame_bone_dev']))

        # rep gates
        rep_ratio = float(np.nanmax(rep_med / calib_med)) if np.isfinite(rep_med).any() else float('inf')
        devs = [abs(rep_bone[k] / sess_bones[k] - 1) for k in LONG_BONES
                if np.isfinite(rep_bone.get(k, np.nan))]
        rep_dev = float(max(devs)) if devs else float('inf')
        # Provenance only: the videos are rendered stick figures, not footage,
        # so nothing downstream needs them -- but the original JSON frame of
        # every row is worth keeping.
        maps = [frame_mapping(args.json_dir, r['stub'], c + 1, r['xy'][:, c]) for c in range(3)]
        videos = [video_index.get(f"{r['stub']}_{c + 1}", '') for c in range(3)]
        n_ok = int(frame_ok.sum())

        rchecks = [
            (session_ok, 'session not usable'),
            (n_ok >= GATES['min_usable_frames'],
             f'{n_ok} usable frames (min {GATES["min_usable_frames"]})'),
            (n_ok / max(T, 1) >= GATES['min_usable_frac'],
             f'usable fraction {n_ok / max(T, 1):.2f} (min {GATES["min_usable_frac"]})'),
            (rep_ratio <= GATES['max_rep_reproj_over_session'],
             f'reprojection {rep_ratio:.2f}x the session calibration '
             f'(max {GATES["max_rep_reproj_over_session"]})'),
            (rep_dev <= GATES['max_rep_bone_dev'],
             f'bone lengths {100 * rep_dev:.0f}% off the session median '
             f'(max {100 * GATES["max_rep_bone_dev"]:.0f}%)'),
        ]
        rep_ok = all(p for p, _ in rchecks)
        rec = dict(stub=r['stub'], usable=rep_ok,
                   reasons=[m for p, m in rchecks if not p],
                   n_frames=T, n_usable_frames=n_ok,
                   usable_frac=n_ok / max(T, 1),
                   median_reproj_px=rep_med, reproj_over_session=rep_ratio,
                   bone_dev=rep_dev,
                   view_dropped_frac=float(((r['used'].sum(axis=1) == 2)
                                            & (r['conf'] > GATES['conf_thresh']).all(axis=1)
                                            & ok).sum() / max(ok.sum(), 1)),
                   n_usable_frames_loo=frame_ok_loo.sum(axis=1),
                   frame_mapping_found=[m[0] is not None for m in maps],
                   saved=False)

        if rep_ok or args.save_all:
            Xc = np.stack([X @ R_wc[c].T + cams.t[c] for c in range(3)])        # (3,T,25,3)
            Xhc = np.stack([body25_to_h36m(Xc[c], ok)[0] for c in range(3)])  # (3,T,17,3)
            Xh_floor, _ = body25_to_h36m((X - org) @ Rw.T, ok)
            # leave-one-out target for camera c, in camera c's own frame
            Xh_loo_c = np.stack([Xh_loo[c] @ R_wc[c].T + cams.t[c] for c in range(3)])
            h36m_2d = np.stack([openpose_to_h36m_2d(r['xy'][:, c], r['conf'][:, c])
                                for c in range(3)])
            vid_idx = np.stack([m[0] if m[0] is not None else np.full(T, -1) for m in maps])
            json_idx = np.stack([m[1] if m[1] is not None else np.full(T, -1) for m in maps])

            np.savez_compressed(
                os.path.join(sess_dir, r['stub'] + '.npz'),
                subject=subject, stub=r['stub'], usable=rep_ok,
                cameras=np.array([1, 2, 3]),
                fps=np.float64(args.fps if args.fps else np.nan),   # capture rate, if supplied
                # Layout: every per-camera array leads with the camera axis (C,T,...).
                #
                # PIPELINE INPUT for camera c: h36m_2d[c] replaces step_1's output
                # (there is no RGB video to run YOLO on), with K_1080[c].
                h36m_2d=h36m_2d,
                stature_m=np.float64(stature),
                K_1080=K1080, dist=np.zeros((3, 5)), image_size_1080=np.array(IMAGE_SIZE),
                #
                # TARGET for camera c: X_h36m_cam_loo[c], scored on frame_usable_loo[c].
                # Triangulated from the OTHER two cameras, so the input's own
                # detections do not shape the target.
                X_h36m_cam_loo=Xh_loo_c.astype(np.float32),
                valid_h36m_loo=np.stack(okh_loo),
                frame_usable_loo=frame_ok_loo,
                eval_joint_mask_h36m=np.array([n not in ('Nose', 'Head') for n in H36M_NAMES]),
                h36m_joint_names=np.array(H36M_NAMES),
                #
                # All-three-camera skeleton: best 3D estimate, for proportions,
                # stature and QA -- NOT for scoring camera c, since it includes
                # camera c's own detections.
                X_h36m_cam=Xhc.astype(np.float32),
                X_body25_cam=Xc.astype(np.float32),
                valid_h36m=okh, valid_body25=ok,
                X_h36m_floor=Xh_floor.astype(np.float32),
                frame_usable=frame_ok,
                #
                # provenance: the videos are rendered stick figures (no footage)
                K_720=K720, image_size_720=np.array(VIDEO_SIZE),
                video_path=np.array(videos), video_frame_idx=vid_idx,
                json_frame_index=json_idx,
                # geometry
                R_world_to_cam=R_wc, t_world_to_cam=cams.t,
                R_world_to_floor=Rw, floor_origin=org,
                # quality
                frame_reproj_px=frame_reproj.astype(np.float32),
                frame_bone_dev=frame_dev.astype(np.float32),
                n_views_body25=r['used'].sum(axis=1).astype(np.int8),
                reproj_px_body25=r['err'].transpose(1, 0, 2).astype(np.float32),
                # the source 2D, for reference
                xy_openpose_1080=r['xy'].transpose(1, 0, 2, 3).astype(np.float32),
                conf_openpose=r['conf'].transpose(1, 0, 2).astype(np.float32),
                units=np.array(f'scaled so measured stature = {stature:.3f} '
                               f'(scale mode "{args.scale}"); world = camera 1 frame'),
            )
            rec['saved'] = True
        reps_out.append(rec)

    summary['n_reps'] = len(reps_out)
    summary['n_reps_usable'] = sum(r['usable'] for r in reps_out)
    summary['n_frames_usable'] = sum(r['n_usable_frames'] for r in reps_out if r['usable'])
    return finish()


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------

def write_tables(out_root):
    sess_rows, rep_rows = [], []
    for p in sorted(glob.glob(os.path.join(out_root, '*', 'session_summary.json'))):
        with open(p) as f:
            s = json.load(f)
        row = {k: s.get(k) for k in ('subject', 'status', 'usable', 'n_calib_frames', 'calib_rule',
                                     'reproj_over_epipolar', 'epipolar_floor_px',
                                     'long_bone_cv_pct', 'floor_inlier_frac',
                                     'stature', 'unit_scale', 'n_reps',
                                     'n_reps_usable', 'n_frames_usable', 'seconds')}
        for key in ('calib_median_reproj_px', 'camera_height', 'camera_tilt_deg',
                    'camera_roll_deg', 'focal_px_1080'):
            v = s.get(key) or [None] * 3
            for c in range(3):
                row[f'{key}_{c + 1}'] = v[c]
        row['n_upright'] = (s.get('stature_measure') or {}).get('n_upright')
        row['reasons'] = '; '.join(s.get('reasons', []))
        sess_rows.append(row)
        for r in s.get('reps', []):
            rr = {'subject': s['subject']}
            rr.update({k: r.get(k) for k in ('stub', 'usable', 'n_frames', 'n_usable_frames',
                                             'usable_frac', 'reproj_over_session',
                                             'bone_dev', 'view_dropped_frac', 'saved')})
            loo = r.get('n_usable_frames_loo') or [None] * 3
            for c in range(3):
                rr[f'n_usable_frames_loo_cam{c + 1}'] = loo[c]
            rr['reasons'] = '; '.join(r.get('reasons', []))
            rep_rows.append(rr)
    pd.DataFrame(sess_rows).to_csv(os.path.join(out_root, 'sessions.csv'), index=False)
    pd.DataFrame(rep_rows).to_csv(os.path.join(out_root, 'reps.csv'), index=False)
    return pd.DataFrame(sess_rows), pd.DataFrame(rep_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True, help='output root')
    ap.add_argument('--data', default=DATA_DIR, help='aligned .npz directory')
    ap.add_argument('--json-dir', default=JSON_DIR, help='per-camera OpenPose JSONs')
    ap.add_argument('--video-dir', default=VIDEO_DIR)
    ap.add_argument('--subjects', default=None, help='comma-separated; default all')
    ap.add_argument('--calib-corr', type=int, default=6000,
                    help='fallback-tier calibration budget, in correspondences. Measured on '
                         'B010: 3x more (20000) changed nothing, at 4x the runtime.')
    ap.add_argument('--calib-max-frames', type=int, default=800,
                    help='cap on calibration frames')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='capture frame rate, saved for the pipeline (step_4 smoothing uses it). '
                         'The Korea data is 30 fps; the B_video renders\' 60 fps container '
                         'rate is an artefact of re-encoding.')
    ap.add_argument('--calib-stride', type=int, default=6,
                    help='take every Nth qualifying frame before selection')
    ap.add_argument('--scale', choices=['stature', 'aimpoint'], default='stature')
    ap.add_argument('--save-all', action='store_true',
                    help='also save reps that fail the gates (flagged unusable)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    subjects = sorted({f.split('_')[0] for f in os.listdir(args.data)
                       if f.endswith('.npz') and not f.startswith('.')})
    if args.subjects:
        want = set(args.subjects.split(','))
        subjects = [s for s in subjects if s in want]
    video_index = build_video_index(args.video_dir)
    print(f'{len(subjects)} sessions, {len(video_index)} videos indexed -> {args.out}')

    for i, sub in enumerate(subjects, 1):
        done = os.path.join(args.out, sub, 'session_summary.json')
        if os.path.exists(done) and not args.overwrite:
            print(f'[{i}/{len(subjects)}] {sub}: already done, skipping')
            continue
        s = process_session(sub, args, video_index)
        tag = ('USABLE' if s['usable'] else s['status'].upper() if s['status'] != 'ok'
               else 'NOT USABLE')
        extra = ''
        if s['status'] == 'ok':
            extra = (f"  reps {s['n_reps_usable']}/{s['n_reps']}"
                     f"  frames {s['n_frames_usable']}"
                     f"  stature {s['stature']:.3f}")
        print(f"[{i}/{len(subjects)}] {sub}: {tag}{extra}  ({s['seconds']}s)")
        for why in s['reasons']:
            print(f'      - {why}')

    sess, reps = write_tables(args.out)
    ok = sess[sess['usable'] == True]
    print(f"\n{len(ok)}/{len(sess)} sessions usable, "
          f"{int(reps['usable'].sum()) if len(reps) else 0}/{len(reps)} reps, "
          f"{int(ok['n_frames_usable'].sum()) if len(ok) else 0} usable frames")
    print(f"tables: {os.path.join(args.out, 'sessions.csv')}, "
          f"{os.path.join(args.out, 'reps.csv')}")


if __name__ == '__main__':
    main()
