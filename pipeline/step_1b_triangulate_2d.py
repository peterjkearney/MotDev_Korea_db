#!/usr/bin/env python3
"""step_1b_triangulate_2d.py -- triangulated-OpenPose ground truth for BioCV.

Stream 1's second target.  Korea has no mocap; its ground truth is OpenPose
triangulated across three cameras.  To compare children with adults on the
SAME yardstick, build that kind of target on BioCV too, from the OpenPose 2D
step_1_openpose_2d.py wrote and the known calibration, and record how far it
sits from mocap -- the cost of the yardstick itself.

Targets are leave-one-out: the target for camera c is triangulated from every
camera EXCEPT c, so the input a model is given never shapes the target it is
scored against.  (On Korea the same rule leaves 2 of 3 cameras; here 8 of 9.)

Reads  {OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_2d.npz   (step_1_openpose_2d)
       {OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz                 (step_0)
       {user}/{cam}.mp4-mocAligned.calib                                      (TRIAL_DIR, else OUT_DIR)
Writes {OUT_DIR}/{user}/{action}/Analysis/H36M/openpose_tri_h36m.npz, in mocap_h36m.npz's
layout (lab frame, mm, native frame resolution with NaN rows where no 2D exists):
    kps3d          (T_native,17,3)   all-camera triangulation
    kps3d_loo      (C,T_native,17,3) target per held-out camera
    cameras        (C,) labels matching kps3d_loo's first axis
plus tri_vs_mocap.npz next to it: per-joint errors against mocap, also printed.

With no --user / --action it does every trial with OpenPose 2D for >= 3 cameras;
trials already done are skipped (--force redoes them).  Needs no video and no GPU.

    python3 step_1b_triangulate_2d.py                                 # everything
    python3 step_1b_triangulate_2d.py --user User03 --action P03_CMJM_01
    python3 step_1b_triangulate_2d.py --dry-run
"""
import argparse
import glob
import os

import numpy as np

from config import (TRIAL_DIR as _TRIAL_DIR, OUT_DIR as _OUT_DIR, mocap_path as _mocap_path,
                    twod_path as _twod_path, tri_target_path, tri_vs_mocap_path, require_out_dir)
from utils.calibration import load_calib
from utils.openpose import H36M_NAMES, body25_to_h36m_3d, present_mask
from utils.triangulate import triangulate_robust

# Joints a triangulated-OpenPose target cannot score fairly: Nose and Head
# (step_0 marks them invalid in mocap too) and Spine.  Spine is a synthesised
# pelvis-thorax midpoint here, exactly as in step_1's YOLO conversion, while
# mocap's is anatomical (T10-based): on P08 the two differ by ~90 mm with
# every detected joint within 7 mm.  That is a convention, not an error, so
# it is kept out of the mask rather than left to inflate every mean.
EVAL_EXCLUDE = ('Nose', 'Head', 'Spine')


def openpose_cameras(user, action):
    d = os.path.dirname(_twod_path(user, action, 'openpose', 'x'))
    return sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(d, '*_2d.npz')))


def find_trials(user=None, action=None):
    """(user, action) for every trial under OUT_DIR with OpenPose 2D for >= 3 cameras."""
    pat = os.path.dirname(_twod_path(user or '*', action or '*', 'openpose', 'x'))
    out = set()
    for p in glob.glob(os.path.join(pat, '*_2d.npz')):
        a = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))     # .../{user}/{action}
        out.add((os.path.basename(os.path.dirname(a)), os.path.basename(a)))
    return sorted(t for t in out if len(openpose_cameras(*t)) >= 3)


def triangulate_trial(user, action, args):
    """One trial -> openpose_tri_h36m.npz (+ tri_vs_mocap.npz).  Returns a summary line."""
    cams = args.cameras.split(',') if args.cameras else openpose_cameras(user, action)
    xy, conf, calibs, used_cams, sfi = [], [], [], [], None
    for cam in cams:
        p = _twod_path(user, action, 'openpose', cam)
        if not os.path.exists(p):
            print(f'  camera {cam}: no {p}, skipping')
            continue
        d = np.load(p)
        if 'body25_2d' not in d.files:
            raise ValueError(f'{p} has no body25_2d -- not written by step_1_openpose_2d.py')
        # Frame lists come from each camera's own video, so a camera whose video is a frame
        # shorter has a shorter list; use the common prefix, refuse if they genuinely disagree.
        s_i = d['source_frame_idx']
        if sfi is None:
            sfi = s_i
        else:
            n = min(len(sfi), len(s_i))
            if not np.array_equal(sfi[:n], s_i[:n]):
                raise ValueError(f'camera {cam}: source_frame_idx disagrees with the first camera')
            sfi = sfi[:n]
        b = d['body25_2d']
        xy.append(b[:, :, :2])
        conf.append(b[:, :, 2])
        calib = next((c for c in (os.path.join(r, user, f'{cam}.mp4-mocAligned.calib') for r in (_TRIAL_DIR, _OUT_DIR))
                      if os.path.exists(c)), None)
        if calib is None:
            raise FileNotFoundError(f'{cam}.mp4-mocAligned.calib under {_TRIAL_DIR}/{user} or {_OUT_DIR}/{user}')
        w, h, K, L, dist = load_calib(calib)
        calibs.append((K, dist, L[:3, :3], L[:3, 3]))
        used_cams.append(cam)
    if len(used_cams) < 3:
        raise ValueError(f'need >= 3 cameras for leave-one-out targets, have {len(used_cams)}')

    n = len(sfi)
    xy = np.stack([x[:n] for x in xy], axis=1).astype(float)          # (T,C,25,2)
    conf = np.stack([c[:n] for c in conf], axis=1).astype(float)
    vis = (conf > args.conf) & present_mask(xy, conf)
    T, C = xy.shape[:2]

    X_all, used, err = triangulate_robust(xy, vis, calibs, args.reproj_thresh)
    ok_all = np.isfinite(X_all).all(-1)
    H_all, okh_all = body25_to_h36m_3d(X_all, ok_all)
    H_loo = np.full((C, T, 17, 3), np.nan)
    for c in range(C):
        v = vis.copy()
        v[:, c, :] = False
        Xl, _, _ = triangulate_robust(xy, v, calibs, args.reproj_thresh)
        H_loo[c], _ = body25_to_h36m_3d(Xl, np.isfinite(Xl).all(-1))

    # native-resolution layout, like mocap_h36m.npz
    mocap_path = _mocap_path(user, action)
    if os.path.exists(mocap_path):
        mocap = np.load(mocap_path)
        n_native, fps_native = len(mocap['kps3d']), float(mocap['fps'])
    else:
        mocap, n_native = None, 0
        fps_native = float(np.load(_twod_path(user, action, 'openpose', used_cams[0]))['video_fps'])
    n_native = max(n_native, int(sfi.max()) + 1)     # a video longer than the mocap
    kps3d = np.full((n_native, 17, 3), np.nan, np.float32)
    kps3d[sfi] = H_all
    kps3d_loo = np.full((C, n_native, 17, 3), np.nan, np.float32)
    kps3d_loo[:, sfi] = H_loo
    valid_joint_mask = np.array([nm not in EVAL_EXCLUDE for nm in H36M_NAMES])

    core_ok = okh_all[:, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]]
    summary = (f'{T} frames from {C} cameras: {core_ok.mean():.0%} of core joints solved, '
               f'median {np.nanmedian(used.sum(axis=1)[ok_all]):.0f} views per joint')

    # the cost of this yardstick: triangulated vs mocap, same joints step_8 scores
    if mocap is not None:
        M = np.full((len(sfi), 17, 3), np.nan)
        ok = sfi < len(mocap['kps3d'])
        M[ok] = mocap['kps3d'][sfi[ok]]
        vj = mocap['valid_joint_mask'] & valid_joint_mask
        d_all = np.linalg.norm(H_all - M, axis=-1)
        d_loo = np.linalg.norm(H_loo - M[None], axis=-1)
        pj_all = np.array([np.nanmean(d_all[:, j]) if vj[j] else np.nan for j in range(17)])
        pj_loo = np.array([[np.nanmean(d_loo[c, :, j]) if vj[j] else np.nan for j in range(17)]
                           for c in range(C)])
        summary += (f'; vs mocap {np.nanmean(pj_all):.1f} mm all-camera, leave-one-out '
                    + ' '.join(f'{c}:{np.nanmean(pj_loo[i]):.0f}' for i, c in enumerate(used_cams)))
        if args.verbose:
            summary += '\n  per joint (all-camera): ' + ' '.join(
                f'{H36M_NAMES[j]} {pj_all[j]:.0f}' for j in range(17) if vj[j])
        vp = tri_vs_mocap_path(user, action)
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        np.savez(vp, cameras=np.array(used_cams), perjoint_all=pj_all, perjoint_loo=pj_loo,
                 joint_names=np.array(H36M_NAMES), mean_all=np.nanmean(pj_all), mean_loo=np.nanmean(pj_loo, axis=1))

    out = tri_target_path(user, action)          # written last: its presence marks the trial done
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, kps3d=kps3d, kps3d_loo=kps3d_loo, cameras=np.array(used_cams),
             source_frame_idx=np.arange(n_native), fps=fps_native, units=np.array('mm'),
             valid_joint_mask=valid_joint_mask, joint_names=np.array(H36M_NAMES),
             n_views_body25=used.sum(axis=1).astype(np.int8),
             reproj_thresh_px=args.reproj_thresh, conf_thresh=args.conf,
             frame=np.array('lab (same as mocap); X_cam = R @ X_lab + t per calib'))
    return summary


def main():
    ap = argparse.ArgumentParser(description='Triangulated-OpenPose target. With no --user/--action, every '
                                             'trial with OpenPose 2D for >= 3 cameras; trials already done are skipped.')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='default: every camera with OpenPose 2D')
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--reproj-thresh', type=float, default=15.0, help='px, per view')
    ap.add_argument('--verbose', action='store_true', help='also print the per-joint error vs mocap')
    ap.add_argument('--force', action='store_true', help='redo trials whose target already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be triangulated, run nothing')
    args = ap.parse_args()

    require_out_dir()
    trials = find_trials(args.user, args.action)
    if not trials:
        raise SystemExit(f'no trial with OpenPose 2D for >= 3 cameras under {_OUT_DIR} for '
                         f'user={args.user or "*"} action={args.action or "*"} -- run step_1_openpose_2d.py first')
    todo = [t for t in trials if args.force or not os.path.exists(tri_target_path(*t))]
    print(f'{len(trials)} trial(s) with OpenPose 2D under {_OUT_DIR}: {len(todo)} to triangulate, '
          f'{len(trials) - len(todo)} already done')
    if args.dry_run:
        for user, action in todo:
            print(f'  {user}/{action}: cameras {",".join(openpose_cameras(user, action))}')
        return
    n_done, failed = 0, []
    for user, action in todo:
        try:
            print(f'{user}/{action}: {triangulate_trial(user, action, args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad trial must not stop the batch
            failed.append((user, action))
            print(f'{user}/{action}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ntriangulated {n_done}, already done {len(trials) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
