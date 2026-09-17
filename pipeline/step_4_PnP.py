#!/usr/bin/env python3
"""
step_3_PnP.py

Applies PnP to each individual frame to align motionbert's H36M 3D keypoints with 
Yolo's 2D keypoints. The results rotation/translation is applied to motionbert's SMPL-24
keypoints as these are the keypoints used to drive the avatar.

All placed keypoints are saved for separate visualisation.

Runs inside the motor-dev container (needs torch/CUDA, not GL/X11):
    docker run --rm --runtime nvidia \
    -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
    -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
    motor-dev:latest \
    python3 step_4_PnP.py --user User28 --action P28_CMJM_01

"""

import os
import sys
import argparse
from utils.rts_smoother import rts_smooth_3d
from utils.calibration import load_calib

import numpy as np
import cv2

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from config import TRIAL_DIR as _TRIAL_DIR

_PNP_CONF_THRESH = 0.4
_PNP_MIN_POINTS = 6


def solve_root_pose(kps3d_m, kps2d_px, conf, K, dist_cv):
    
    mask = conf > _PNP_CONF_THRESH
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--user', default='User03')
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--process-accel-std', default=10.0, type=float)
    
    args = ap.parse_args()
    cameras = args.cameras.split(',')
    yolo_2d_dir = os.path.join(_TRIAL_DIR, args.user, args.action,'Analysis','keypoints')
    mesh_pose_dir = os.path.join(_TRIAL_DIR, args.user, args.action,'Analysis','keypoints', 'mesh')
    calib_dir = os.path.join(_TRIAL_DIR, args.user)
    output_dir = os.path.join(_TRIAL_DIR, args.user, args.action,'Analysis','keypoints','PnP')
    os.makedirs(output_dir, exist_ok=True)


    for camera in cameras:
        print(f"=== camera {camera} ===")
        # Loading 2D keypoints
        yolo_2d_path = os.path.join(yolo_2d_dir, f'{camera}_2d.npz')
        if not os.path.exists(yolo_2d_path):
            print(f"  no {yolo_2d_path} -- run step_1_extract_2d.py --cameras {camera} first, skipping")
            continue
        yolo_data = np.load(yolo_2d_path)
        all_kp_2d = yolo_data['h36m_2d']
        fps = yolo_data['fps']

        # Loading 3D keypoints and rotations
        mesh_pose_path = os.path.join(mesh_pose_dir, f'{camera}_mesh_pose.npz')
        if not os.path.exists(mesh_pose_path):
            print(f"  no {mesh_pose_path} -- run step_2_extract_3d.py --cameras {camera} first, skipping")
            continue
        mesh_pose = np.load(mesh_pose_path)
        
        all_kp_H36M_raw = mesh_pose['kps_H36M_scaled']    # (T,17,3) metres, mesh's own camera-space
        all_kp_SMPL24_raw = mesh_pose['kps_SMPL24_scaled']
        all_rotmats = mesh_pose['rotmats']        # (T,24,3,3)

        n_frames = all_kp_2d.shape[0]
        assert all_kp_H36M_raw.shape[0] == n_frames, "mesh_pose/variants frame count mismatch"
        
        # extracting camera calibration parameters (needed for PnP)
        calib_path = os.path.join(calib_dir, f'{camera}.mp4-mocAligned.calib')
        w, h, K, L_ext, dist = load_calib(calib_path)
        dist_cv = dist.reshape(1, 5)

        
        n_pnp_ok = 0
        all_kp_H36M_placed = np.full_like(all_kp_H36M_raw, np.nan)
        all_kp_SMPL24_placed =  np.full_like(all_kp_SMPL24_raw, np.nan)
        pnp_ok_mask = np.zeros(n_frames, dtype=bool)
        for tt in range(n_frames):
            kp_H36M_raw = all_kp_H36M_raw[tt]
            kp_SMPL24_raw = all_kp_SMPL24_raw[tt]

            # Using PnP to align the H36M markers with the Yolo 2D keypoints (these markers refer to identical anatomical markers)
            solved = solve_root_pose(kp_H36M_raw, all_kp_2d[tt, :, :2], all_kp_2d[tt, :, 2], K, dist_cv)
            if solved is not None:
                R, t, _, _ = solved

                # Applying rotation to both H36M keypoints and SMPL24 keypoints
                all_kp_H36M_placed[tt]   = (R @ kp_H36M_raw.T).T   + t
                all_kp_SMPL24_placed[tt] = (R @ kp_SMPL24_raw.T).T + t

                pnp_ok_mask[tt] = True
                n_pnp_ok += 1
        print(f"  solvePnP ok: {n_pnp_ok}/{n_frames}  (avatar hidden on the other {n_frames - n_pnp_ok})")

        if n_pnp_ok >0:
            # Applying RTS Smoothing to root position of H36M skeleton to reduce depth jitter
            root_H36M_smooth,_ = rts_smooth_3d( all_kp_H36M_placed[:,0,:], pnp_ok_mask, all_kp_H36M_placed[:,0,:], 1/fps,
                    sigma_along=0.15, sigma_perp=0.02, process_accel_std=args.process_accel_std)

            # Adding smoothing adjustment to each point in skeleton
            adjustment_H36M = root_H36M_smooth - all_kp_H36M_placed[:,0,:]
            all_kp_H36M_smooth = all_kp_H36M_placed + adjustment_H36M[:,None,:]
            all_kp_SMPL24_smooth = all_kp_SMPL24_placed + adjustment_H36M[:,None,:]
        else:
            print(f'No valid PnP frames for camera {camera}, smoothing not performed')
            all_kp_H36M_smooth = all_kp_H36M_placed
            all_kp_SMPL24_smooth = all_kp_SMPL24_placed
            


        # Saving placed keypoints and camera parameters
        out_path = os.path.join(output_dir, f'{camera}_pnp.npz')
        np.savez(out_path, 
                 kps_H36M_placed=all_kp_H36M_placed, 
                 kps_SMPL24_placed = all_kp_SMPL24_placed, 
                 kps_H36M_smooth=all_kp_H36M_smooth,
                 kps_SMPL24_smooth=all_kp_SMPL24_smooth,
                 cam_w=w, cam_h=h, cam_K = K, cam_L_ext = L_ext, cam_dist_cv = dist_cv)
                

if __name__ == '__main__':
    main()
