#!/usr/bin/env python3
"""build_child_gt.py -- child 3D ground truth for the MotionBERT test.

For every session (subject) in the Korea B set:

  1. calibrate the three cameras from the pose data (camcal),
  2. refine every rep's camera sync: align_cameras.py's offsets are whole frames
     from one motion signal, so try +-sync_window frames per camera and keep the
     offsets whose three views triangulate best (needs align_cameras.py's
     untrimmed tracks; older Aligned/ files are used as they are),
  3. find reps where one camera disagrees with the other two on nearly every
     joint -- the camera was moved for those reps -- and drop them,
  4. recalibrate on the remaining, re-synced reps and refine the sync once more
     (a calibration fitted to mistimed frames is slightly bent: B010's zooms moved
     1-2% on recalibration, and the second refinement changed 1 rep of 20),
  5. triangulate every frame of every rep with those cameras, dropping a view
     when it disagrees with the other two (catches left/right label swaps),
  6. measure the child's crown-to-sole stature from upright standing frames,
  7. assess the session, each rep and each joint against quality gates.  Joints
     are gated per bone (a bone > joint_bone_dev from its session median loses
     its two joints) and a rep is judged on the frames that keep >= 10 of the 12
     limb joints -- the same rule step_1_korea_2d.py scores with,
  8. save the session's cameras (always, so failed sessions can be inspected)
     and each usable rep with, per camera c:
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
    max_abs_roll_deg=10.0,          # tripods: roll must come out near zero
    max_long_bone_cv_pct=8.0,      # depth distortion makes bones breathe
    min_floor_inlier_frac=0.6,
    min_upright_frames=15,         # to measure stature at all
    # sync refinement
    sync_window=4,                 # frames tried either side, per camera, per pass
    sync_max_passes=3,             # re-centre and search again while the best sits at the edge
    sync_stride=2,                 # frames used when scoring an offset
    sync_min_joints=200,           # 3-view joints needed to judge an offset at all
    # moved camera: kept on few joints while the other two agree
    moved_max_kept_pct=50.0,       # % of its 3-view joints the robust triangulation keeps
    moved_min_others_kept_pct=80.0,
    moved_min_joints=50,
    # triangulation
    conf_thresh=0.5,
    reproj_thresh_px=15.0,         # per view, 1080p pixels
    # joint / frame
    joint_bone_dev=0.25,           # a bone this far from its session median loses its two joints
    min_core_joints=10,            # of the 12 H36M limb joints, after the joint gate
    max_frame_bone_dev=0.15,       # the old whole-frame rule, still saved as frame_usable
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


# The bones of cc.BONES on H36M-17 joints (BODY_25 neck = H36M Thorax, mid-hip = Hip), so the
# session medians computed on BODY_25 apply unchanged.
H36M_BONES = {'torso': [('Hip', 'Thorax')],
              'clavicle': [('Thorax', 'RShoulder'), ('Thorax', 'LShoulder')],
              'upperarm': [('RShoulder', 'RElbow'), ('LShoulder', 'LElbow')],
              'forearm': [('RElbow', 'RWrist'), ('LElbow', 'LWrist')],
              'hip_offset': [('Hip', 'RHip'), ('Hip', 'LHip')],
              'thigh': [('RHip', 'RKnee'), ('LHip', 'LKnee')],
              'shank': [('RKnee', 'RAnkle'), ('LKnee', 'LAnkle')]}


def joint_gate(Xh, okh, sess_bones, thresh=GATES['joint_bone_dev']):
    """Per-bone check on H36M joints: a bone further than `thresh` (fraction) from its session
    median clears both its joints.  Xh (T,17,3) in any rigid frame, same units as sess_bones.
    Returns the gated (T,17) mask.  step_1_korea_2d.py reads it rather than recomputing."""
    ok = okh.copy()
    drop = np.zeros_like(ok)
    J = {n: i for i, n in enumerate(H36M_NAMES)}
    for name, pairs in H36M_BONES.items():
        ref = sess_bones.get(name)
        if not ref:
            continue
        for a, b in pairs:
            ia, ib = J[a], J[b]
            both = ok[:, ia] & ok[:, ib]
            L = np.linalg.norm(Xh[:, ia] - Xh[:, ib], axis=-1)
            bad = both & (np.abs(L / ref - 1) > thresh)
            drop[bad, ia] = True
            drop[bad, ib] = True
    return ok & ~drop


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


def load_reps(data_dir, subject):
    """Every synced rep of one session, as dicts the sync refinement can re-trim.

    align_cameras.py files carry each camera's untrimmed track and the offsets it chose
    (`lag` = (lag12, lag13): camera 1 row i+lag1c shows the same instant as camera c row i).
    Older Aligned/ files (merge_cameras.ipynb) hold only the trimmed span: they are taken as
    their own 'raw' track at offset (0, 0), so the refinement can still shift them, losing a
    frame or two at the ends.
    """
    reps = []
    for fn in sorted(f for f in os.listdir(data_dir)
                     if f.startswith(subject + '_') and f.endswith('.npz') and not f.startswith('.')):
        d = np.load(os.path.join(data_dir, fn))
        full = 'raw_xy1' in d
        key = 'raw_' if full else ''
        reps.append(dict(
            stub=fn[:-4],
            raw_xy=[d[f'{key}xy{c}'].astype(float) for c in (1, 2, 3)],
            raw_conf=[d[f'{key}score{c}'].astype(float) for c in (1, 2, 3)],
            frame_index=[d[f'frame_index{c}'] for c in (1, 2, 3)] if full else None,
            lag0=(int(d['lag12']), int(d['lag13'])) if full else (0, 0),
            people=[],
        ))
        reps[-1]['lag'] = reps[-1]['lag0']
        # align_cameras.py found more than one combination of people that moves alike in all
        # three cameras: keep them all, choose_person settles it by body size.
        if 'candidates' in d and len(d['candidates']) > 1:
            for row in d['candidates']:
                combo = [int(i) for i in row[1:4]]
                reps[-1]['people'].append(dict(
                    combo=combo, lag=(int(row[4]), int(row[5])),
                    raw_xy=[d[f'tracks_xy{c + 1}'][combo[c]].astype(float) for c in range(3)],
                    raw_conf=[d[f'tracks_score{c + 1}'][combo[c]].astype(float) for c in range(3)]))
    return reps


def choose_person(cams, rep):
    """For a rep with several candidate people: the smallest body among those that triangulate well.

    An adult moving with the child (demonstrating, holding hands) can correlate across the
    cameras as well as the child does, so movement alone cannot always tell them apart; size
    can.  Each candidate is triangulated with the session's cameras; those within 1.5x the best
    reprojection are compared by the median of torso + thigh + shank, and the smallest is kept.
    Returns a record of the choice (empty if there was nothing to choose).
    """
    if len(rep['people']) < 2:
        return {}
    scored = []
    for k, p in enumerate(rep['people']):
        trial = dict(rep, raw_xy=p['raw_xy'], raw_conf=p['raw_conf'], lag=p['lag'])
        err = sync_error(cams, trial, trial['lag'])
        xy, conf, _ = rep_span(trial)
        vis = (conf > GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        X, ok, _, _ = cc.triangulate_robust(cams, xy, vis, GATES['reproj_thresh_px'])
        bl = bone_lengths(X, ok)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            size = float(np.nansum([np.nanmedian(bl[k2]) for k2 in ('torso', 'thigh', 'shank') if k2 in bl]))
        scored.append((k, err, size if size > 0 else np.nan))
    errs = np.array([e for _, e, _ in scored])
    if not np.isfinite(errs).any():
        return {}
    good = [s for s in scored if np.isfinite(s[1]) and s[1] <= 1.5 * np.nanmin(errs) and np.isfinite(s[2])]
    k = min(good, key=lambda s: s[2])[0] if good else int(np.nanargmin(errs))
    p = rep['people'][k]
    rep.update(raw_xy=p['raw_xy'], raw_conf=p['raw_conf'], lag0=p['lag'], lag=p['lag'])
    return dict(person_candidates=len(scored), person_chosen=k, person_combo=p['combo'],
                person_err_px=[round(e, 2) for _, e, _ in scored], person_size=[round(s, 3) for _, _, s in scored])


def rep_span(rep, lag=None):
    """(xy (T,3,25,2), conf (T,3,25), start (3,)) for a rep at offsets `lag` (default: its current)."""
    l12, l13 = rep['lag'] if lag is None else lag
    s1 = max(0, l12, l13)
    start = np.array([s1, s1 - l12, s1 - l13])
    T = int(min(len(x) - s for x, s in zip(rep['raw_xy'], start)))
    if T <= 0:
        return np.zeros((0, 3, 25, 2)), np.zeros((0, 3, 25)), start
    xy = np.stack([x[s:s + T] for x, s in zip(rep['raw_xy'], start)], axis=1)
    conf = np.stack([c[s:s + T] for c, s in zip(rep['raw_conf'], start)], axis=1)
    return xy, conf, start


def stored_frame_mapping(rep, start, T, cam):
    """(video_idx, json_frame_index) of every synced row, from align_cameras.py's saved tracks."""
    rows = start[cam] + np.arange(T)
    return rows, rep['frame_index'][cam][rows]


def frame_mapping(json_dir, stub, cam, aligned_xy):
    """Source-video frame index of every aligned row, for one camera.

    merge_cameras trimmed each camera's start by its sync lag but did not
    save the lags, so recover the offset by matching the aligned keypoints
    against the camera's own JSON (identical floats).  Returns (video_idx,
    json_frame_index), or (None, None) if no unique match.  Only needed for
    the old Aligned/ files; align_cameras.py's carry the mapping.
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


JSON_DECIMALS = 2                     # the summary is for reading; the rep .npz files keep full precision
JSON_PRECISE = {'session_bone_median', 'unit_scale'}   # read back by code (step_1_korea_2d's bone check): 4 decimals


def jsonable(x, decimals=JSON_DECIMALS):
    if isinstance(x, dict):
        return {k: jsonable(v, 4 if k in JSON_PRECISE else decimals) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v, decimals) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist(), decimals)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), decimals)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# --------------------------------------------------------------------------
# One session
# --------------------------------------------------------------------------

def calibrate(subject, reps, args):
    """One calibration from the given reps (each at its current sync offsets)."""
    # Two tiers.  Measured on common clean frames, calibrating only on frames
    # where the whole body is well detected in every camera is slightly more
    # accurate (~5% lower held-out-camera error on B010, a tie on B011): a
    # complete, confident skeleton is a decent proxy for "OpenPose is having a
    # good frame".  But that rule starves sessions where some joints are rarely
    # seen by all three cameras, so fall back to treating every joint seen by
    # two or more cameras as a correspondence.  Never worse than the strict
    # rule alone; the fallback is recorded in the summary.
    def arrays():
        for r in reps:
            xy, conf, _ = rep_span(r)
            yield r['stub'], xy, conf

    try:
        cand = cc.candidates_from_arrays(arrays(), conf_thresh=0.6, min_joints=20, stride=6)
        obs = cc.select_spread(cand, n_frames=250, min_frames=GATES['min_calib_frames_strict'])
        rule = 'strict'
    except (cc.InsufficientData, RuntimeError):
        cand = cc.candidates_from_arrays(arrays(), conf_thresh=GATES['calib_conf_thresh'],
                                         min_views=2, min_corr=GATES['calib_min_corr_per_frame'],
                                         stride=args.calib_stride)
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


def sync_error(cams, rep, lag, stride=GATES['sync_stride']):
    """Median reprojection (px) of the joints all three cameras see, with the rep at offsets `lag`.

    With the right offsets the three views show one instant and their rays meet; a frame off,
    a moving joint is in three different places and the error rises.  NaN if too few joints.
    """
    xy, conf, _ = rep_span(rep, lag)
    xy, conf = xy[::stride], conf[::stride]
    if len(xy) < 10:
        return float('nan')
    vis = (conf > GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    vis &= vis.all(axis=1, keepdims=True)
    if vis[:, 0].sum() < GATES['sync_min_joints']:
        return float('nan')
    X, _ = cc.triangulate(cams, xy, vis)
    return float(np.nanmedian(cc.reprojection_per_view(cams, X, xy, vis)))


def refine_sync(cams, rep):
    """Move the rep's offsets to the best-triangulating pair within +-sync_window per camera.

    Re-centres and searches again while the best sits on the window's edge (up to
    sync_max_passes).  Returns (error at the starting offsets, error at the chosen ones).
    """
    W = GATES['sync_window']
    lag = tuple(rep['lag'])
    err0 = sync_error(cams, rep, lag)
    if not np.isfinite(err0):
        return err0, err0
    best_err = err0
    for _ in range(GATES['sync_max_passes']):
        errs = {(lag[0] + a, lag[1] + b): sync_error(cams, rep, (lag[0] + a, lag[1] + b))
                for a in range(-W, W + 1) for b in range(-W, W + 1)}
        errs = {k: e for k, e in errs.items() if np.isfinite(e)}
        best = min(errs, key=errs.get)
        best_err = errs[best]
        at_edge = max(abs(best[0] - lag[0]), abs(best[1] - lag[1])) == W
        lag = best
        if not at_edge:
            break
    rep['lag'] = lag
    return err0, best_err


def camera_check(cams, rep):
    """Per camera: % of its 3-view joints the robust triangulation keeps.

    A camera in the right place is kept on ~95%; one that was moved for this rep disagrees
    with the other two on every joint and is kept on ~0%.  NaN with too few joints.
    """
    xy, conf, _ = rep_span(rep)
    vis = (conf > GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    _, _, used, _ = cc.triangulate_robust(cams, xy, vis, GATES['reproj_thresh_px'])
    three = vis.all(axis=1)
    return np.array([100 * used[:, c][three].mean() if three.sum() >= GATES['moved_min_joints']
                     else np.nan for c in range(3)])


def moved_cameras(kept):
    """Cameras kept on few joints while both others are kept on most."""
    out = []
    for c in range(3):
        others = np.delete(kept, c)
        if (np.isfinite(kept[c]) and kept[c] < GATES['moved_max_kept_pct']
                and np.all(others >= GATES['moved_min_others_kept_pct'])):
            out.append(c)
    return out


def calibrate_and_sync(subject, reps, args):
    """Calibrate, refine every rep's sync, drop moved-camera reps, recalibrate, refine again.

    Returns (cal, sync_info, moved) where sync_info[stub] records the offsets and errors and
    moved[stub] lists the camera indices judged moved under the final calibration.
    """
    info = {r['stub']: dict(lag_align=r['lag0']) for r in reps}
    cal = calibrate(subject, reps, args)
    n_cal = 1
    for r in reps:
        choice = choose_person(cal['cams'], r)
        if choice:
            info[r['stub']].update(choice, lag_align=r['lag0'])
    for r in reps:
        e0, e1 = refine_sync(cal['cams'], r)
        info[r['stub']].update(err_align=e0, lag_pass1=r['lag'], err_pass1=e1)
    kept = {r['stub']: camera_check(cal['cams'], r) for r in reps}
    moved = {s: moved_cameras(k) for s, k in kept.items()}
    good = [r for r in reps if not moved[r['stub']]]
    changed = (sum(info[r['stub']]['lag_pass1'] != r['lag0'] for r in reps)
               + sum(info[r['stub']].get('person_chosen', 0) != 0 for r in reps))
    if changed or len(good) < len(reps):
        try:
            cal = calibrate(subject, good, args)
            n_cal = 2
        except cc.InsufficientData:
            pass                              # keep the first; recorded below
        for r in good:
            e0, e1 = refine_sync(cal['cams'], r)
            info[r['stub']].update(lag_pass2=r['lag'], err_pass2=e1)
        kept = {r['stub']: camera_check(cal['cams'], r) for r in reps}
        moved = {s: moved_cameras(k) for s, k in kept.items()}
    for s in info:
        info[s]['kept_pct'] = kept[s]
        info[s]['lag_final'] = next(tuple(r['lag']) for r in reps if r['stub'] == s)
    cal['n_calibrations'] = n_cal
    cal['n_sync_changed'] = sum(info[s]['lag_final'] != info[s]['lag_align'] for s in info)
    return cal, info, moved


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

    # ---- 1-4. calibrate, refine sync, drop moved-camera reps, recalibrate --
    loaded = load_reps(args.data, subject)
    if not loaded:
        summary.update(status='skipped', reasons=['no synced reps'])
        return finish()
    try:
        cal, sync_info, moved = calibrate_and_sync(subject, loaded, args)
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

    # ---- 5. triangulate every frame of every rep -------------------------
    reps = []
    for src in loaded:
        xy, conf, start = rep_span(src)
        if len(xy) == 0:
            continue
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
        reps.append(dict(stub=src['stub'], src=src, start=start, xy=xy, conf=conf, X=X, ok=ok,
                         used=used, err=err, moved=moved[src['stub']],
                         X_loo=np.stack(X_loo), ok_loo=np.stack(ok_loo)))

    # ---- 6. stature, then the final unit ---------------------------------
    # Moved-camera reps are triangulated (their two good cameras still agree) but kept out of
    # the session's stature and bone medians.
    calib_reps = [r for r in reps if not r['moved']] or reps
    floor0 = cc.gravity_frame(cams, cal['normal'], cal['offset'])
    Rw, org = floor0['R_world_to_floor'], floor0['origin']
    allX = np.concatenate([r['X'] for r in calib_reps])
    allok = np.concatenate([r['ok'] for r in calib_reps])
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

    # session-median bone lengths, for the joint, frame and rep gates
    allX = np.concatenate([r['X'] for r in calib_reps])
    sess_bones = {k: float(np.nanmedian(v))
                  for k, v in bone_lengths(allX, allok).items()}

    # ---- 7. session gates ------------------------------------------------
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
        session_bone_median=sess_bones,
        n_calibrations=cal['n_calibrations'], n_reps_sync_changed=cal['n_sync_changed'],
        moved_camera_reps={k: [c + 1 for c in v] for k, v in moved.items() if v})

    # The session's cameras, whatever the verdict: failed sessions can then be inspected
    # without recalibrating.  Offsets are per rep (camera 1 row i+lag1c == camera c row i).
    np.savez(os.path.join(sess_dir, 'session_cameras.npz'),
             R_world_to_cam=cams.R, t_world_to_cam=cams.t, f=cams.f, cx=cams.cx, cy=cams.cy,
             R_world_to_floor=Rw, floor_origin=org, unit_scale=s, usable=session_ok,
             stubs=np.array([r['stub'] for r in reps]),
             lag_align=np.array([sync_info[r['stub']]['lag_align'] for r in reps]),
             lag_final=np.array([sync_info[r['stub']]['lag_final'] for r in reps]),
             kept_pct=np.array([sync_info[r['stub']]['kept_pct'] for r in reps]))

    if np.isfinite(stature):
        with open(os.path.join(sess_dir, 'user_meta.json'), 'w') as f:
            json.dump({'stature_m': round(float(stature), 4),
                       'note': 'crown-to-sole, measured from the triangulated '
                               'reconstruction, in the same units as the saved '
                               'skeletons (see session_summary.json unit_scale)'},
                      f, indent=1)

    # ---- 8. per rep: gates, camera frames, save ---------------------------
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
        # The joint gate: what is scored, and what the rep is judged on.
        okg = joint_gate(Xh, okh, sess_bones)
        frame_ok_joint = okg[:, H36M_CORE].sum(axis=1) >= GATES['min_core_joints']

        # the same gates, applied to each camera's leave-one-out target
        frame_ok_loo = np.zeros((3, T), bool)
        frame_ok_loo_joint = np.zeros((3, T), bool)
        Xh_loo, okh_loo, okg_loo = [], [], []
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
            okg_loo.append(joint_gate(hl, hok, sess_bones))
            frame_ok_loo[c] = ((hok[:, H36M_CORE].sum(axis=1) >= GATES['min_core_joints'])
                               & (fdev <= GATES['max_frame_bone_dev']))
            frame_ok_loo_joint[c] = okg_loo[c][:, H36M_CORE].sum(axis=1) >= GATES['min_core_joints']

        # rep gates
        rep_ratio = float(np.nanmax(rep_med / calib_med)) if np.isfinite(rep_med).any() else float('inf')
        devs = [abs(rep_bone[k] / sess_bones[k] - 1) for k in LONG_BONES
                if np.isfinite(rep_bone.get(k, np.nan))]
        rep_dev = float(max(devs)) if devs else float('inf')
        # Provenance only: the videos are rendered stick figures, not footage,
        # so nothing downstream needs them -- but the original JSON frame of
        # every row is worth keeping.
        if r['src']['frame_index'] is not None:
            maps = [stored_frame_mapping(r['src'], r['start'], T, c) for c in range(3)]
        else:
            maps = [frame_mapping(args.json_dir, r['stub'], c + 1, r['xy'][:, c]) for c in range(3)]
        videos = [video_index.get(f"{r['stub']}_{c + 1}", '') for c in range(3)]
        n_ok = int(frame_ok_joint.sum())
        si = sync_info[r['stub']]
        kept = si['kept_pct']

        rchecks = [
            (session_ok, 'session not usable'),
            (not r['moved'],
             '; '.join(f'camera {c + 1} moved (kept on {kept[c]:.0f}% of its joints, the others '
                       f'{np.nanmin(np.delete(kept, c)):.0f}%+)' for c in r['moved'])),
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
                   n_usable_frames_frame_rule=int(frame_ok.sum()),
                   sync_lag_align=si['lag_align'], sync_lag=si['lag_final'],
                   sync_err_align_px=si.get('err_align'),
                   sync_err_px=si.get('err_pass2', si.get('err_pass1')),
                   camera_kept_pct=kept, moved_cameras=[c + 1 for c in r['moved']],
                   person={k: si[k] for k in ('person_candidates', 'person_chosen', 'person_combo',
                                              'person_err_px', 'person_size') if k in si} or None,
                   median_reproj_px=rep_med, reproj_over_session=rep_ratio,
                   bone_dev=rep_dev,
                   view_dropped_frac=float(((r['used'].sum(axis=1) == 2)
                                            & (r['conf'] > GATES['conf_thresh']).all(axis=1)
                                            & ok).sum() / max(ok.sum(), 1)),
                   n_usable_frames_loo=frame_ok_loo_joint.sum(axis=1),
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
                # TARGET for camera c: X_h36m_cam_loo[c], scored where valid_h36m_loo_gated[c].
                # Triangulated from the OTHER two cameras, so the input's own
                # detections do not shape the target.  The gated masks are the joint
                # gate (joint_gate); valid_* are triangulation only; frame_usable* are
                # the old whole-frame rule, kept for comparison.
                X_h36m_cam_loo=Xh_loo_c.astype(np.float32),
                valid_h36m_loo=np.stack(okh_loo),
                valid_h36m_loo_gated=np.stack(okg_loo),
                frame_usable_loo=frame_ok_loo,
                frame_usable_loo_joint=frame_ok_loo_joint,
                eval_joint_mask_h36m=np.array([n not in ('Nose', 'Head') for n in H36M_NAMES]),
                h36m_joint_names=np.array(H36M_NAMES),
                #
                # All-three-camera skeleton: best 3D estimate, for proportions,
                # stature and QA -- NOT for scoring camera c, since it includes
                # camera c's own detections.
                X_h36m_cam=Xhc.astype(np.float32),
                X_body25_cam=Xc.astype(np.float32),
                valid_h36m=okh, valid_body25=ok,
                valid_h36m_gated=okg,
                joint_bone_dev=np.float64(GATES['joint_bone_dev']),
                X_h36m_floor=Xh_floor.astype(np.float32),
                frame_usable=frame_ok,
                frame_usable_joint=frame_ok_joint,
                sync_lag=np.array(si['lag_final']), sync_lag_align=np.array(si['lag_align']),
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
                                     'n_reps_usable', 'n_frames_usable', 'n_calibrations',
                                     'n_reps_sync_changed', 'seconds')}
        row['n_reps_moved_camera'] = len(s.get('moved_camera_reps') or {})
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
                                             'n_usable_frames_frame_rule',
                                             'usable_frac', 'reproj_over_session',
                                             'bone_dev', 'view_dropped_frac',
                                             'sync_err_align_px', 'sync_err_px', 'saved')})
            for key in ('sync_lag_align', 'sync_lag'):
                v = r.get(key)
                rr[key] = f'{v[0]},{v[1]}' if v else None
            rr['moved_cameras'] = ','.join(str(c) for c in r.get('moved_cameras') or [])
            loo = r.get('n_usable_frames_loo') or [None] * 3
            for c in range(3):
                rr[f'n_usable_frames_loo_cam{c + 1}'] = loo[c]
            rr['reasons'] = '; '.join(r.get('reasons', []))
            rep_rows.append(rr)
    pd.DataFrame(sess_rows).to_csv(os.path.join(out_root, 'sessions.csv'), index=False, float_format='%.2f')
    pd.DataFrame(rep_rows).to_csv(os.path.join(out_root, 'reps.csv'), index=False, float_format='%.2f')
    return pd.DataFrame(sess_rows), pd.DataFrame(rep_rows)


def _process_one(task):
    """process_session for the worker pool: one session's crash must not end the batch."""
    subject, args, _ = task
    try:
        return process_session(*task)
    except Exception as e:
        summary = dict(subject=subject, status='failed', usable=False,
                       reasons=[f'{type(e).__name__}: {e}'], traceback=traceback.format_exc(), seconds=0)
        sess_dir = os.path.join(args.out, subject)
        os.makedirs(sess_dir, exist_ok=True)
        with open(os.path.join(sess_dir, 'session_summary.json'), 'w') as f:
            json.dump(jsonable(summary), f, indent=1)
        return summary


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
    ap.add_argument('--workers', type=int, default=1, help='sessions processed in parallel')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    subjects = sorted({f.split('_')[0] for f in os.listdir(args.data)
                       if f.endswith('.npz') and not f.startswith('.')})
    if args.subjects:
        want = set(args.subjects.split(','))
        subjects = [s for s in subjects if s in want]
    video_index = build_video_index(args.video_dir)
    todo = [s for s in subjects
            if args.overwrite or not os.path.exists(os.path.join(args.out, s, 'session_summary.json'))]
    print(f'{len(subjects)} sessions ({len(subjects) - len(todo)} already done), '
          f'{len(video_index)} videos indexed -> {args.out}', flush=True)

    def report(i, s):
        tag = ('USABLE' if s['usable'] else s['status'].upper() if s['status'] != 'ok'
               else 'NOT USABLE')
        extra = ''
        if s['status'] == 'ok':
            extra = (f"  reps {s['n_reps_usable']}/{s['n_reps']}"
                     f"  frames {s['n_frames_usable']}"
                     f"  stature {s['stature']:.3f}"
                     f"  sync changed {s['n_reps_sync_changed']}"
                     f"  moved-camera reps {len(s['moved_camera_reps'])}")
        print(f"[{i}/{len(todo)}] {s['subject']}: {tag}{extra}  ({s['seconds']}s)", flush=True)
        for why in s['reasons']:
            print(f'      - {why}', flush=True)

    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, s in enumerate(pool.imap_unordered(_process_one, [(sub, args, video_index) for sub in todo]), 1):
                report(i, s)
    else:
        for i, sub in enumerate(todo, 1):
            report(i, process_session(sub, args, video_index))

    sess, reps = write_tables(args.out)
    ok = sess[sess['usable'] == True]
    print(f"\n{len(ok)}/{len(sess)} sessions usable, "
          f"{int(reps['usable'].sum()) if len(reps) else 0}/{len(reps)} reps, "
          f"{int(ok['n_frames_usable'].sum()) if len(ok) else 0} usable frames")
    print(f"tables: {os.path.join(args.out, 'sessions.csv')}, "
          f"{os.path.join(args.out, 'reps.csv')}")


if __name__ == '__main__':
    main()
