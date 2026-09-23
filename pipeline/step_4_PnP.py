#!/usr/bin/env python3
"""step_4_PnP.py -- place the skeleton in the camera by PnP, one frame at a time.

Per frame, PnP aligns step_3's H36M-17 joints with the detector's 2D H36M
joints (joints with 2D confidence > 0.4, at least 6 of them).  The
rotation/translation found is applied to the SMPL-24 joints too, which are
what drives the avatar.  The root is then smoothed along time (RTS) to take
out per-frame depth jitter.

PnP assumes each 2D/3D pair is the same body point.  That holds for the limb
joints, but not for the hips: the 3D hips are H36M joint centres regressed
from the SMPL mesh (wide, at femoral-head height) while the detector's hips
are COCO/OpenPose surface points (narrower, lower).  --exclude-joints leaves
named joints out of the correspondence set, e.g. --exclude-joints Hip,RHip,LHip;
the joints used are recorded in the output (pnp_joints_used).

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
            {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_mesh_pose.npz
            {user}/{cam}.mp4-mocAligned.calib                        (TRIAL_DIR, else OUT_DIR)
    writes  {OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{cam}_pnp.npz

With no --user / --action it does every camera with a mesh pose from
--detector; cameras already done are skipped (--force redoes them).  numpy +
cv2 only, no GPU.

    python3 step_4_PnP.py                                  # everything, OpenPose
    python3 step_4_PnP.py --detector yolo --user User03
    python3 step_4_PnP.py --exclude-joints Hip,RHip,LHip   # limb joints only
"""

import argparse
import os

import cv2
import numpy as np

from config import (DETECTORS, OUT_DIR as _OUT_DIR, twod_path as _twod_path, mesh_pose_path as _mesh_path,
                    pnp_path as _pnp_path, calib_path, find_cameras, require_out_dir)
from utils.calibration import load_calib
from utils.openpose import H36M_NAMES
from utils.rts_smoother import rts_smooth_3d

_PNP_CONF_THRESH = 0.4
_PNP_MIN_POINTS = 6


def parse_joints(spec):
    """'Hip,RHip,4' -> (17,) bool mask of the joints to USE (all True for an empty spec)."""
    use = np.ones(17, bool)
    for tok in (t.strip() for t in (spec or '').split(',') if t.strip()):
        if tok.isdigit():
            j = int(tok)
        else:
            names = {n.lower(): i for i, n in enumerate(H36M_NAMES)}
            if tok.lower() not in names:
                raise SystemExit(f'--exclude-joints: unknown joint {tok!r}; H36M names are {", ".join(H36M_NAMES)}')
            j = names[tok.lower()]
        if not 0 <= j < 17:
            raise SystemExit(f'--exclude-joints: joint index {j} out of range 0-16')
        use[j] = False
    return use


def solve_root_pose(kps3d_m, kps2d_px, conf, K, dist_cv, use_joint=None):
    """One frame.  use_joint: (17,) bool, joints allowed as correspondences (default all)."""
    mask = conf > _PNP_CONF_THRESH
    if use_joint is not None:
        mask &= use_joint
    if mask.sum() < _PNP_MIN_POINTS:
        return None
    obj_pts = np.ascontiguousarray(kps3d_m[mask], dtype=np.float64)
    img_pts = np.ascontiguousarray(kps2d_px[mask], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_cv, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        # SQPnP asserts on a degenerate point set (e.g. a collapsed skeleton);
        # that is a failed frame, not a reason to abort the whole camera.
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float32), tvec.reshape(3).astype(np.float32), rvec, tvec


def place(user, action, camera, args):
    """One camera -> {cam}_pnp.npz.  Returns a summary line."""
    yolo_data = np.load(_twod_path(user, action, args.detector, camera))
    all_kp_2d = yolo_data['h36m_2d']
    fps = yolo_data['fps']
    mesh_pose = np.load(_mesh_path(user, action, args.detector, camera))
    all_kp_H36M_raw = mesh_pose['kps_H36M_scaled']    # (T,17,3) metres, mesh's own camera-space
    all_kp_SMPL24_raw = mesh_pose['kps_SMPL24_scaled']
    n_frames = min(all_kp_2d.shape[0], all_kp_H36M_raw.shape[0])
    if all_kp_2d.shape[0] != all_kp_H36M_raw.shape[0]:
        print(f'  WARNING 2D has {all_kp_2d.shape[0]} rows, mesh pose {all_kp_H36M_raw.shape[0]}; using {n_frames}')
    w, h, K, L_ext, dist = load_calib(calib_path(user, camera))
    dist_cv = dist.reshape(1, 5)
    use_joint = parse_joints(args.exclude_joints)

    n_pnp_ok = 0
    all_kp_H36M_placed = np.full((n_frames, 17, 3), np.nan, np.float32)
    all_kp_SMPL24_placed = np.full((n_frames, 24, 3), np.nan, np.float32)
    pnp_ok_mask = np.zeros(n_frames, dtype=bool)
    for tt in range(n_frames):
        kp_H36M_raw = all_kp_H36M_raw[tt]
        kp_SMPL24_raw = all_kp_SMPL24_raw[tt]
        # PnP aligns the H36M joints with the 2D keypoints (see the module docstring on the hips)
        solved = solve_root_pose(kp_H36M_raw, all_kp_2d[tt, :, :2], all_kp_2d[tt, :, 2], K, dist_cv, use_joint)
        if solved is not None:
            R, t, _, _ = solved
            all_kp_H36M_placed[tt]   = (R @ kp_H36M_raw.T).T   + t
            all_kp_SMPL24_placed[tt] = (R @ kp_SMPL24_raw.T).T + t
            pnp_ok_mask[tt] = True
            n_pnp_ok += 1

    if n_pnp_ok > 0:
        # RTS smoothing of the H36M root to reduce depth jitter, applied to every joint
        root_H36M_smooth, _ = rts_smooth_3d(all_kp_H36M_placed[:, 0, :], pnp_ok_mask, all_kp_H36M_placed[:, 0, :], 1 / fps,
                                            sigma_along=0.15, sigma_perp=0.02, process_accel_std=args.process_accel_std)
        adjustment_H36M = root_H36M_smooth - all_kp_H36M_placed[:, 0, :]
        all_kp_H36M_smooth = all_kp_H36M_placed + adjustment_H36M[:, None, :]
        all_kp_SMPL24_smooth = all_kp_SMPL24_placed + adjustment_H36M[:, None, :]
    else:
        all_kp_H36M_smooth = all_kp_H36M_placed
        all_kp_SMPL24_smooth = all_kp_SMPL24_placed

    out_path = _pnp_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             kps_H36M_placed=all_kp_H36M_placed,
             kps_SMPL24_placed=all_kp_SMPL24_placed,
             kps_H36M_smooth=all_kp_H36M_smooth,
             kps_SMPL24_smooth=all_kp_SMPL24_smooth,
             pnp_ok=pnp_ok_mask, detector=np.array(args.detector),
             pnp_joints_used=use_joint, pnp_conf_thresh=_PNP_CONF_THRESH,
             cam_w=w, cam_h=h, cam_K=K, cam_L_ext=L_ext, cam_dist_cv=dist_cv)
    left_out = [H36M_NAMES[j] for j in np.where(~use_joint)[0]]
    return (f'solvePnP ok on {n_pnp_ok}/{n_frames} frames' + ('' if n_pnp_ok else ' -- no smoothing')
            + (f' (without {", ".join(left_out)})' if left_out else ''))


def main():
    ap = argparse.ArgumentParser(description='PnP placement + root smoothing. With no --user/--action, every '
                                             'camera with a mesh pose from --detector; cameras already done are skipped.')
    ap.add_argument('--detector', choices=DETECTORS, default='openpose')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with a mesh pose')
    ap.add_argument('--process-accel-std', default=10.0, type=float)
    ap.add_argument('--exclude-joints', default='',
                    help='H36M joints left out of the PnP correspondences, by name or index, e.g. Hip,RHip,LHip '
                         '(the detector\'s hips are not the points the mesh\'s H36M hips are); default none')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be run')
    args = ap.parse_args()

    require_out_dir()
    jobs = find_cameras(_mesh_path, args.detector, args.user, args.action,
                        set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {args.detector} {{cam}}_mesh_pose.npz under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_3_extract_3d.py first')
    todo = [j for j in jobs if args.force or not os.path.exists(_pnp_path(j[0], j[1], args.detector, j[2]))]
    print(f'{args.detector}: {len(jobs)} camera(s) with a mesh pose under {_OUT_DIR}: {len(todo)} to run, '
          f'{len(jobs) - len(todo)} already done')
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return
    n_done, failed = 0, []
    for user, action, cam in todo:
        try:
            print(f'{user}/{action} {cam}: {place(user, action, cam, args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'{user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
