#!/usr/bin/env python3
"""
step_1_extract_2d.py — YOLO -> H36M 2D pixel-space keypoints for all 9
synchronised BioCV camera views of one trial (e.g. P03_WALK_01), saved separately
from any 3D lifting so the same 2D detections can be reused.

Source video reduced to --target-fps (default 60) via frame_decimation.nearest_frame_
indices().

Run inside the motor-dev container:
    docker run --rm --runtime nvidia \
      -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
      -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
      motor-dev:latest \
      python3 step_1_extract_2d.py --user User28 --action P28_CMJM_01
"""

import os
import glob
import shutil
import tempfile
import argparse

import numpy as np
import cv2
from ultralytics import YOLO

from utils.frame_decimation import nearest_frame_indices
from utils.calibration import load_calib, reproject

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
from config import TRIAL_DIR as _TRIAL_DIR
from config import mocap_path as _mocap_path, twod_path as _twod_path, require_out_dir
from config import MODELS_DIR as _MODELS_DIR     # the original's ../models no longer holds here
_YOLO_MODEL     = os.path.join(_MODELS_DIR, 'yolo26m-pose.pt')


def coco2h36m(kp_coco, conf):
    """
    Input:  kp_coco (17, 2) in COCO17 order
            conf (17,) confidence per joint
    Output: kp_h36m (17, 3) in H36M order

    Copies confidence values or averages between two points when taking mid-point
    """
    kp_with_conf = np.concatenate((kp_coco,conf[:,None]),axis=1)
    V, C = kp_with_conf.shape
    kp_h36m = np.zeros((V, C),dtype=np.float32)

    kp_h36m[0, :] = (kp_with_conf[11, :] + kp_with_conf[12, :]) * 0.5      # pelvis = mid-hip
    kp_h36m[1, :] = kp_with_conf[12, :]                                 # r hip
    kp_h36m[2, :] = kp_with_conf[14, :]                                 # r knee
    kp_h36m[3, :] = kp_with_conf[16, :]                                 # r ankle
    kp_h36m[4, :] = kp_with_conf[11, :]                                 # l hip
    kp_h36m[5, :] = kp_with_conf[13, :]                                 # l knee
    kp_h36m[6, :] = kp_with_conf[15, :]                                 # l ankle
    kp_h36m[8, :] = (kp_with_conf[5, :] + kp_with_conf[6, :]) * 0.5              # neck/thorax = mid-shoulder
    kp_h36m[7, :] = (kp_h36m[0, :] + kp_h36m[8, :]) * 0.5              # spine = mid(pelvis, neck)
    kp_h36m[9, :] = kp_with_conf[0, :]                                   # nose (~head-lower)
    kp_h36m[10, :] = (kp_with_conf[3, :] + kp_with_conf[4, :]) * 0.5             # head-top ≈ mid-ear
    kp_h36m[11, :] = kp_with_conf[5, :]                                  # l shoulder
    kp_h36m[12, :] = kp_with_conf[7, :]                                  # l elbow
    kp_h36m[13, :] = kp_with_conf[9, :]                                  # l wrist
    kp_h36m[14, :] = kp_with_conf[6, :]                                  # r shoulder
    kp_h36m[15, :] = kp_with_conf[8, :]                                  # r elbow
    kp_h36m[16, :] = kp_with_conf[10, :]                                 # r wrist

    return kp_h36m


def yolo_track_all_individuals(video_path, model, source_frame_idx):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  src: {src_w}x{src_h}  {n_frames} frames {src_fps:.1f} fps")

    frame_i = 0
    kept_frames = 0
    id_seen = []
    history = {}

    n_frames = min(n_frames,max(source_frame_idx))
    source_frame_idx = set(source_frame_idx.tolist()) # speeds up the 'frame_i in source_frame_idx' lookup

    print(f'Num frames: {n_frames}')

    while frame_i <= n_frames:
        keep = frame_i in source_frame_idx # only analyse frame if index of frame is in source_frame_idx (the set of frames that align with 60fps)
        if keep:
            ret, frame = cap.read() # ret is boolean indicating if frame was succesfully loaded, 'frame' is the image itself
        else:
            ret = cap.grab() # get frame but don't process/decode image, essentially skipping frame so no next cap call gets next frame
            frame = None
        if not ret:
            break
        if keep:
            r = model.track(frame, persist=True,verbose=False,tracker='botsort_custom.yaml')[0]

            if r.boxes.id is not None:
                ids  = r.boxes.id.int().cpu().numpy()      # (n,)
                kpts = r.keypoints.data.cpu().numpy()      # (n, 17, 3), COCO-17 order

                # Looping through all persons seen in this frame and populating history with extracted keypoints
                for tid, person_coco in zip(ids, kpts):

                    person = coco2h36m(person_coco[:, :2], person_coco[:, 2]).reshape(1,17,3)

                    if tid not in id_seen:
                        if kept_frames > 0:
                            history[tid] = np.zeros((kept_frames,17,3),dtype=np.float32)
                            history[tid] = np.concatenate((history[tid],person),axis=0)
                        else:
                            history[tid] = person
                        id_seen.append(tid)
                    else:  
                        history[tid] = np.concatenate((history[tid],person),axis=0)

                    
                not_seen_this_frame = list(set(id_seen) - set(ids))

                for tid in not_seen_this_frame:
                    
                    history[tid] = np.concatenate((history[tid],np.zeros((1,17,3),dtype=np.float32)),axis=0)

            else:

                # no people detected, set all previously seen people to zeros for this frame
                for tid in id_seen:
                    history[tid] = np.concatenate((history[tid],np.zeros((1,17,3),dtype=np.float32)),axis=0)

            kept_frames +=1

        frame_i += 1
    cap.release()

    return history, src_w, src_h



def project_mocap_to_pixel(calib_path, mocap_kps3d):
    # Projecting mocap into camera coordinates
    w, h, K, L_ext, dist = load_calib(calib_path)
    dist_cv = dist.reshape(1, 5)

    #converting mocap coordinates to camera coordinate system
    R = L_ext[:3, :3]
    t = L_ext[:3, 3]

    mocap_kps3d_cam = mocap_kps3d @ R.T + t          # (T, 17, 3), still mm

    T, J = mocap_kps3d.shape[:2]
    mocap2d = reproject(mocap_kps3d_cam.reshape(-1,3), K, dist_cv).reshape(T,J,2)

    valid2d = mocap_kps3d_cam[..., 2] > 0      # (T, 17) bool
    mocap2d[~valid2d] = np.nan

    return mocap2d


def find_closest_user_to_mocap(history, mocap2d, src_w):
    # history is now H36M-ordered (coco2h36m applied in yolo_track_all_individuals), same as
    # mocap -- so both sides use the same joint indices. Limbs only (skip 0/7/8/9/10: pelvis/
    # spine/neck/head are synthetic midpoints, noisier and less useful for identity matching).
    joints = [1,2,3,4,5,6,11,12,13,14,15,16]
    comparison_pairs = list(zip(joints, joints))

    PENALTY    = src_w / 2      # 960 px — cost of a missing/unusable YOLO joint
    CONF_THR   = 0.3
    MIN_JOINTS = 4               # frames with fewer valid mocap joints are too sparse to trust

    CLOSE_THRESH_PX  = 40        # a frame counts as "close" if its mean matched-joint error is under this
    MIN_PRESENT_FRAC = 0.5       # ...and at least this fraction of the valid mocap joints were detected
    SPLIT_ID_COVERAGE_THRESH = 0.2   # flag as a possible split-identity if 2+ ids each cover this much

    mocap_idx = [m for m,_ in comparison_pairs]
    yolo_idx = [y for _,y in comparison_pairs]
    n = min(len(mocap2d), *(len(h) for h in history.values()))

    # mocap side: which (frame, joint) cells are measurable at all
    mo = mocap2d[:n, mocap_idx, :]                            # (n, J, 2)
    mocap_ok = np.isfinite(mo).all(axis=-1)               # (n, J)
    mocap_ok &= (mocap_ok.sum(axis=1) >= MIN_JOINTS)[:, None]

    valid_frames = mocap_ok.any(axis=1)                   # frames where mocap itself is usable at all
    n_valid_frames = int(valid_frames.sum())
    mocap_joints_per_frame = mocap_ok.sum(axis=1).astype(float)   # (n,) valid mocap joints per frame

    score_by_id = {}
    matched_mean_by_id = {}
    coverage_by_id = {}
    for tid,h in history.items():
        yo = h[:n, yolo_idx, :2].astype(float)
        absent = np.all(h[:n, yolo_idx] == 0, axis=-1) | (h[:n, yolo_idx, 2] < CONF_THR) # marker is 'absent' if coords are all
        yo[absent] = np.nan

        raw_d = np.linalg.norm(mo - yo, axis=-1)                              # NaN where YOLO missing
        penalised_d = np.where(np.isfinite(raw_d), np.minimum(raw_d, PENALTY), PENALTY) # cap real, penalise missing
        score_by_id[tid] = penalised_d[mocap_ok].mean()

        matched_mask = mocap_ok & np.isfinite(raw_d)
        matched_mean_by_id[tid] = raw_d[matched_mask].mean() if matched_mask.any() else float('nan')

        # per-frame "close" test: enough of this frame's valid mocap joints were detected, and
        # the average error among those detected joints is small — distinguishes "this id IS the
        # user, just occluded/noisy sometimes" from "this id is a genuinely different person".
        present_per_frame = np.where(mocap_ok & ~absent, 1.0, 0.0).sum(axis=1)   # (n,)
        with np.errstate(invalid='ignore', divide='ignore'):
            present_frac = np.where(mocap_joints_per_frame > 0, present_per_frame / mocap_joints_per_frame, 0.0)
            mean_dist_present = np.nansum(np.where(matched_mask, raw_d, 0.0), axis=1) / np.maximum(present_per_frame, 1)

        close_frame = valid_frames & (present_frac >= MIN_PRESENT_FRAC) & (mean_dist_present < CLOSE_THRESH_PX)
        coverage_by_id[tid] = close_frame.sum() / n_valid_frames if n_valid_frames else 0.0

    best_id = min(score_by_id, key=score_by_id.get)

    split_candidates = [tid for tid, cov in coverage_by_id.items() if cov >= SPLIT_ID_COVERAGE_THRESH]
    if len(split_candidates) > 1:
        print(f"  WARNING: {len(split_candidates)} ids each cover >= {SPLIT_ID_COVERAGE_THRESH:.0%} of mocap "
              f"frames with a close match — possible identity split (real user tracked under multiple ids): "
              + ", ".join(f"id {tid} ({coverage_by_id[tid]:.1%})" for tid in split_candidates))

    return best_id, score_by_id, matched_mean_by_id, coverage_by_id


_TAB10_BGR = [
    (180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
    (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23),
]  # matplotlib tab10, reordered RGB->BGR for cv2


def animateYoloTrack(mocap2d, history, w, h, out_path, fps=60, out_width=800):
    """Draw mocap (blue) vs every tracked YOLO id (one color each) directly onto video
    frames with OpenCV and write out_path. Avoids matplotlib's per-frame Agg redraw, which
    is far slower than cv2.circle + cv2.VideoWriter for a few hundred/thousand frames."""

    # frames common to both streams (histories can differ in length if the loop broke early)
    n = min(len(mocap2d), *(len(v) for v in history.values())) if history else len(mocap2d)

    def clean(arr, conf_thr=0.3):
        """(n,17,3) YOLO -> (n,17,2), absent/low-confidence joints as NaN so they aren't drawn."""
        xy = arr[:n, :, :2].astype(float).copy()
        missing = np.all(arr[:n] == 0, axis=2) | (arr[:n, :, 2] < conf_thr)
        xy[missing] = np.nan
        return xy

    tracks = {tid: clean(v) for tid, v in history.items()}
    ids = list(tracks)

    scale = out_width / w
    out_h = int(round(h * scale))
    mocap_scaled = mocap2d[:n, :, :2] * scale
    tracks_scaled = {tid: xy * scale for tid, xy in tracks.items()}

    legend = [('mocap', (255, 0, 0))] + [(f'id {tid}', _TAB10_BGR[i % 10]) for i, tid in enumerate(ids)]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_width, out_h))
    try:
        for i in range(n):
            frame = np.full((out_h, out_width, 3), 255, dtype=np.uint8)

            for x, y in mocap_scaled[i]:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(round(x)), int(round(y))), 4, (255, 0, 0), -1)

            for j, tid in enumerate(ids):
                color = _TAB10_BGR[j % 10]
                for x, y in tracks_scaled[tid][i]:
                    if np.isfinite(x) and np.isfinite(y):
                        cv2.circle(frame, (int(round(x)), int(round(y))), 4, color, -1)

            cv2.putText(frame, f'frame {i}/{n - 1}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            for k, (label, color) in enumerate(legend):
                ly = 20 + (k + 1) * 18
                cv2.circle(frame, (out_width - 100, ly - 4), 4, color, -1)
                cv2.putText(frame, label, (out_width - 90, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            writer.write(frame)
    finally:
        writer.release()


def reset_tracker(model):
    """Forget every track before a new video. model.track(persist=True) keeps its tracker
    between calls, which is what links frames within a video -- but across videos it would
    carry one camera's tracks (and id counter) into the next."""
    for t in (getattr(getattr(model, 'predictor', None), 'trackers', None) or []):
        t.reset()


def find_trials(pattern, user=None, action=None):
    """(user, action) for every folder under TRIAL_DIR holding a markers.c3d and videos
    matching `pattern`, narrowed by --user and/or --action when given. The c3d is required
    because the subject is picked by proximity to the projected mocap: calibration folders
    and markerless (ML) trials have videos but nothing to pick by."""
    users = [user] if user else sorted(d for d in os.listdir(_TRIAL_DIR)
                                       if os.path.isdir(os.path.join(_TRIAL_DIR, d)))
    trials = []
    for u in users:
        udir = os.path.join(_TRIAL_DIR, u)
        if not os.path.isdir(udir):
            continue
        for a in ([action] if action else sorted(os.listdir(udir))):
            if (os.path.exists(os.path.join(udir, a, 'markers.c3d'))
                    and glob.glob(os.path.join(udir, a, pattern))):
                trials.append((u, a))
    return trials


def process_trial(user, action, model, args):
    """YOLO 2D for one trial's cameras -> {OUT_DIR}/{user}/{action}/Analysis/keypoints/yolo/.
    Returns (done, skipped, failed) camera counts."""
    video_files = sorted(glob.glob(os.path.join(_TRIAL_DIR, user, action, args.pattern)))
    todo = [v for v in video_files if args.force or not os.path.exists(
        _twod_path(user, action, 'yolo', os.path.splitext(os.path.basename(v))[0]))]
    n_skip = len(video_files) - len(todo)
    if not todo:
        return 0, n_skip, 0

    mocap_path = _mocap_path(user, action)
    if not os.path.exists(mocap_path):
        print(f"{user}/{action}: no {mocap_path} -- run step_0_load_mocap.py first")
        return 0, n_skip, len(todo)
    mocap_data = np.load(mocap_path)
    mocap_world_mm_native = mocap_data['kps3d']           # (T,17,3) mm, world-space, NaN where invalid
    native_fps = float(mocap_data['fps'])

    # mocap_h36m.npz is kept at native (200Hz) resolution (step_0_load_mocap.py), one row per
    # native frame -- decimate down to --target-fps here, on the video/tracking side, with the
    # same nearest_frame_indices() the module docstring describes. The result is still a set of
    # valid NATIVE frame indices, so mocap_world_mm can just be subset by it directly -- no
    # separate mocap-side decimation, and no risk of two independent decimations disagreeing.
    source_frame_idx = nearest_frame_indices(len(mocap_world_mm_native), native_fps, args.target_fps)
    mocap_world_mm = mocap_world_mm_native[source_frame_idx]
    out_fps = min(args.target_fps, native_fps)

    n_done = n_fail = 0
    for video_path in todo:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{user}/{action} {stem}] {video_path}", flush=True)
        try:
            calib_path_cam = os.path.join(_TRIAL_DIR, user, f'{stem}.mp4-mocAligned.calib')
            if not os.path.exists(calib_path_cam):
                raise FileNotFoundError(calib_path_cam)
            reset_tracker(model)
            keypoints_all_users, src_w, src_h = yolo_track_all_individuals(video_path, model, source_frame_idx) # get keypoints for all deteced individuals
            mocap2d = project_mocap_to_pixel(calib_path_cam, mocap_world_mm) # project mocab to pixel space

            # score each detected individual by proximity to mocap points
            best_id, score_by_id, matched_mean_by_id, coverage_by_id = find_closest_user_to_mocap(keypoints_all_users, mocap2d, src_w)
            if best_id is None or best_id not in keypoints_all_users:
                raise RuntimeError('no tracked person matched the mocap')

            print(f'Identified {len(keypoints_all_users)} persons. Best Id: {best_id}')
            print(f'{"id":>4}  {"score":>8}  {"matched-only":>12}  {"coverage":>8}')
            for tid in sorted(score_by_id, key=score_by_id.get):
                print(f'{int(tid):>4}  {float(score_by_id[tid]):>8.2f}  {float(matched_mean_by_id[tid]):>12.2f}  {coverage_by_id[tid]:>8.1%}')

            out_path = _twod_path(user, action, 'yolo', stem)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if not args.no_tracks_video:
                # rendered on local disk, then copied next to the npz (cv2 writing through the
                # Drive mount is slow and can leave a truncated mp4)
                tmp = os.path.join(tempfile.gettempdir(), f'{action}_{stem}_tracks.mp4')
                animateYoloTrack(mocap2d, keypoints_all_users, src_w, src_h, tmp)
                shutil.copy2(tmp, os.path.join(os.path.dirname(out_path), f'{stem}_tracks.mp4'))
                os.remove(tmp)
            # the npz is written LAST: its presence is what marks this camera as done
            np.savez(out_path,
                     h36m_2d=keypoints_all_users[best_id],                # (T, 17, 3) x_px, y_px, conf
                     source_frame_idx=source_frame_idx,          # indices into the original 200Hz stream
                     video=os.path.basename(video_path),
                     fps = out_fps,
                     detector=np.array('yolo'))
            print(f"  -> {out_path}", flush=True)
            n_done += 1
        except Exception as e:                      # one bad video must not stop the batch
            n_fail += 1
            print(f"  FAILED -- {type(e).__name__}: {e}", flush=True)
    return n_done, n_skip, n_fail


def main():
    ap = argparse.ArgumentParser(description='YOLO 2D keypoints. With no --user/--action, every trial under '
                                             'BIOCV_ROOT; cameras already done are skipped, so re-running '
                                             'after a disconnect resumes where it stopped.')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--pattern', default='0*.mp4')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--target-fps', type=float, default=60)
    ap.add_argument('--no-tracks-video', action='store_true',
                    help='skip the {cam}_tracks.mp4 diagnostic (every tracked person vs projected mocap)')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be extracted, run nothing')
    args = ap.parse_args()

    # Reads the videos and calibs from TRIAL_DIR (the local copy); writes only to
    # {OUT_DIR}/{user}/{action}/Analysis/keypoints/yolo/ (config.twod_path).
    require_out_dir()
    trials = find_trials(args.pattern, args.user, args.action)
    if not trials:
        raise SystemExit(f'no videos matching {args.pattern} under {_TRIAL_DIR} for '
                         f'user={args.user or "*"} action={args.action or "*"}')
    print(f'{len(trials)} trial(s) under {_TRIAL_DIR}')
    if args.dry_run:
        n_todo = n_have = 0
        for user, action in trials:
            stems = [os.path.splitext(os.path.basename(v))[0]
                     for v in sorted(glob.glob(os.path.join(_TRIAL_DIR, user, action, args.pattern)))]
            todo = [c for c in stems if args.force or not os.path.exists(_twod_path(user, action, 'yolo', c))]
            n_todo += len(todo); n_have += len(stems) - len(todo)
            if todo:
                print(f'  {user}/{action}: {",".join(todo)}'
                      + ('' if os.path.exists(_mocap_path(user, action)) else '   (no mocap yet -- run step_0)'))
        print(f'{n_todo} camera video(s) to extract, {n_have} already done')
        return
    print(f"Loading YOLO model: {_YOLO_MODEL}")
    model = YOLO(_YOLO_MODEL)

    tot = np.zeros(3, int)
    for i, (user, action) in enumerate(trials, 1):
        counts = process_trial(user, action, model, args)
        tot += counts
        if counts[0] or counts[2]:
            print(f'[{i}/{len(trials)}] {user}/{action}: {counts[0]} done, {counts[1]} already done, '
                  f'{counts[2]} failed', flush=True)
    print(f'\n[done] cameras: {tot[0]} extracted, {tot[1]} already done, {tot[2]} failed')


if __name__ == '__main__':
    main()
