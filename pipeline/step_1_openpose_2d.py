#!/usr/bin/env python3
"""step_1_openpose_2d.py -- OpenPose BODY_25 -> H36M-17 2D, the OpenPose counterpart of step_1.

Stream 1 (adults, BioCV).  Reads the one-JSON-per-video OpenPose output that
tools/run_openpose.py writes and produces the same kind of {cam}_2d.npz
step_1_extract_2d.py does for YOLO (h36m_2d, source_frame_idx, video, fps),
plus body25_2d, which step_1b_triangulate_2d.py needs:

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_openpose.json
            {OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz        (step_0)
            {user}/{cam}.mp4-mocAligned.calib                             (TRIAL_DIR, else OUT_DIR)
    writes  {OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_2d.npz

With no --user / --action it does every trial that has OpenPose JSONs; cameras
already done are skipped (--force redoes them).  It needs no video and no GPU.

Frames.  The JSON says which native video frame each entry is
(`native_frame_idx`), so the 2D, the video and the mocap are on one clock by
construction.  If OpenPose was run on the native 200 Hz video the entries are
thinned to the --target-fps ticks; a decimated run is used as it is.  The
mocap is sampled at those same native frame numbers (video and mocap are
frame-aligned at the native rate, per the mocAligned calibration); rows past
the end of the mocap are missing, and a duration mismatch is reported.

Person selection.  OpenPose has no track ids, so the subject is chosen PER
FRAME: the person whose detected H36M limb joints lie closest to the
projected mocap, provided that person has at least half of the limb joints
the mocap has and sits within --max-match-px of it.  Joints a person lacks
are not counted against them (utils/openpose.pick_person says why).  Frames
with no acceptable match are written as missing (all zeros), and `detected`
records which frames matched.

    python3 step_1_openpose_2d.py                                  # everything
    python3 step_1_openpose_2d.py --user User03 --action P03_CMJM_01 [--cameras 06,07]
    python3 step_1_openpose_2d.py --dry-run
"""
import argparse
import glob
import os

import numpy as np

from config import (TRIAL_DIR as _TRIAL_DIR, OUT_DIR as _OUT_DIR, mocap_path as _mocap_path,
                    twod_path as _twod_path, openpose_json_path as _json_path, require_out_dir)
from utils.calibration import load_calib, reproject
from utils.frame_decimation import nearest_frame_indices
from utils.openpose import H36M_NAMES, body25_to_h36m_2d, load_openpose_video, pick_person


def project_mocap(calib_path, kps3d_mm):
    """Mocap lab-mm -> this camera's pixels; NaN where behind the camera."""
    w, h, K, L, dist = load_calib(calib_path)
    R, t = L[:3, :3], L[:3, 3]
    cam = kps3d_mm @ R.T + t
    T, J = cam.shape[:2]
    uv = reproject(cam.reshape(-1, 3), K, dist.reshape(1, 5)).reshape(T, J, 2)
    uv[cam[..., 2] <= 0] = np.nan
    return uv


def find_cameras(user=None, action=None, cameras=None):
    """[(user, action, cam)] for every {cam}_openpose.json under OUT_DIR, narrowed by the filters."""
    pat = _json_path(user or '*', action or '*', '*')
    out = []
    for p in sorted(glob.glob(pat)):
        cam = os.path.basename(p)[:-len('_openpose.json')]
        a = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))   # .../{user}/{action}
        if cameras is None or cam in cameras:
            out.append((os.path.basename(os.path.dirname(a)), os.path.basename(a), cam))
    return out


def convert(user, action, cam, mocap, args):
    """One camera's OpenPose JSON -> {cam}_2d.npz.  Returns a one-line summary."""
    doc, people = load_openpose_video(_json_path(user, action, cam))
    native = np.asarray(doc['native_frame_idx'], int)
    vfps, vn = float(doc['video_fps']), int(doc['video_n_frames'])
    # a run on the native-rate video is thinned to the target-fps ticks; a decimated run
    # (no more entries than ticks) is used whole
    ticks = nearest_frame_indices(vn, vfps, args.target_fps)
    keep = np.isin(native, ticks) if len(native) > len(ticks) + 1 else np.ones(len(native), bool)
    source_frame_idx = native[keep]
    people = [p for p, k in zip(people, keep) if k]
    T = len(source_frame_idx)
    out_fps = min(args.target_fps, vfps)

    # mocap sampled at the same native frames; beyond its end -> missing
    kps_native, native_fps, valid_joint = mocap['kps3d'], float(mocap['fps']), mocap['valid_joint_mask']
    in_range = source_frame_idx < len(kps_native)
    kps = np.full((T, 17, 3), np.nan)
    kps[in_range] = kps_native[source_frame_idx[in_range]]
    warn = ''
    if abs(vn / vfps - len(kps_native) / native_fps) > 0.25:
        warn = (f'  WARNING video {vn / vfps:.2f} s vs mocap {len(kps_native) / native_fps:.2f} s, '
                f'{int((~in_range).sum())} rows have no mocap')

    calib = next((p for p in (os.path.join(r, user, f'{cam}.mp4-mocAligned.calib') for r in (_TRIAL_DIR, _OUT_DIR))
                  if os.path.exists(p)), None)
    if calib is None:
        raise FileNotFoundError(f'{cam}.mp4-mocAligned.calib under {_TRIAL_DIR}/{user} or {_OUT_DIR}/{user}')
    mocap2d = project_mocap(calib, kps)                     # (T,17,2)

    h36m = np.zeros((T, 17, 3), np.float32)
    body25 = np.zeros((T, 25, 3), np.float32)
    detected = np.zeros(T, bool)
    score = np.full(T, np.nan)
    n_matched = np.zeros(T, int)
    n_people = np.zeros(T, int)
    for i, (xy_p, conf_p) in enumerate(people):
        n_people[i] = len(xy_p)
        ref_ok = valid_joint & np.isfinite(mocap2d[i]).all(axis=1)
        k, s, nm = pick_person(xy_p, conf_p, mocap2d[i], ref_ok, conf_thresh=args.conf)
        if k is None or s > args.max_match_px:
            continue
        n_matched[i] = nm
        body25[i, :, :2] = xy_p[k]
        body25[i, :, 2] = conf_p[k]
        h36m[i] = body25_to_h36m_2d(xy_p[k:k + 1], conf_p[k:k + 1])[0]
        detected[i] = True
        score[i] = s

    out = _twod_path(user, action, 'openpose', cam)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, h36m_2d=h36m, body25_2d=body25, source_frame_idx=source_frame_idx,
             video=f'{cam}.mp4', fps=out_fps, detected=detected, match_px=score,
             n_matched_joints=n_matched, max_match_px=args.max_match_px, n_people=n_people, joint_names=np.array(H36M_NAMES),
             detector=np.array('openpose_body25'),
             video_fps=vfps, video_n_frames=vn, mocap_fps=native_fps, mocap_n_frames=len(kps_native))
    med = np.nanmedian(score) if detected.any() else np.nan
    return (f'{T} frames, subject matched on {detected.sum()} (median {med:.1f} px to mocap; '
            f'{(n_people > 1).mean():.0%} of frames had >1 person){warn}')


def main():
    ap = argparse.ArgumentParser(description='OpenPose JSON -> H36M 2D. With no --user/--action, every trial '
                                             'with OpenPose JSONs; cameras already done are skipped.')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with a JSON')
    ap.add_argument('--target-fps', type=float, default=60)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--max-match-px', type=float, default=80.0,
                    help='reject the closest person if the mean distance to the projected mocap, over the limb '
                         'joints that person has, exceeds this (the subject is typically 5-25 px away)')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be converted, run nothing')
    args = ap.parse_args()

    require_out_dir()
    jobs = find_cameras(args.user, args.action, set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {{cam}}_openpose.json under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run tools/run_openpose.py first')
    todo = [j for j in jobs if args.force or not os.path.exists(_twod_path(j[0], j[1], 'openpose', j[2]))]
    print(f'{len(jobs)} camera(s) with OpenPose JSONs under {_OUT_DIR}: {len(todo)} to convert, '
          f'{len(jobs) - len(todo)} already done')
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return

    n_done, failed, mocap_cache = 0, [], (None, None)
    for user, action, cam in todo:
        try:
            if mocap_cache[0] != (user, action):
                mp = _mocap_path(user, action)
                if not os.path.exists(mp):
                    raise FileNotFoundError(f'{mp} -- run step_0_load_mocap.py first')
                mocap_cache = ((user, action), np.load(mp))
            print(f'{user}/{action} {cam}: {convert(user, action, cam, mocap_cache[1], args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'{user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\nconverted {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
