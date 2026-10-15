"""Grounded-ankle depth: ray of the camera's own 2D ankle intersected with the floor plane at the contact height,
vs step_4b's placed ankle, both against the (all-camera) target ankle, in the camera frame (mm).
Also the camera's 2D ankle vs the target ankle reprojected into that camera (px)."""
import glob, os, sys
import cv2
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.smpl_lite import J
from floorlib import grounded_ankles

S, tree, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
REP = {'B010': 'B010_GMS_1_1', 'B011': 'B011_GMS_2_3', 'B014': 'B014_GMS_1_2', 'B023': 'B023_GMS_2_2'}
print(f'{tree} {cfg}: grounded ankle frames; depth error = camera z minus target (median, mm); 2D offset = '
      f'camera OpenPose ankle minus reprojected target ankle (median px, u right / v down)')
print(f'{"rep":5s}{"cam":>4s}{"n":>5s} | {"ray-plane":>9s} {"placed":>7s} | {"2D du":>6s} {"dv":>6s} | {"placed ankle height":>19s}')
for u, rep in REP.items():
    base = f'{S}/{tree}/{u}/{rep}/Analysis'
    tgt = np.load(f'{base}/H36M/openpose_tri_h36m.npz')['kps3d'].astype(float)
    allX = np.concatenate([np.load(p)['kps3d'].astype(float) for p in glob.glob(f'{S}/{tree}/{u}/*/Analysis/H36M/openpose_tri_h36m.npz')])
    h = np.median(grounded_ankles(allX)[:, 2])
    for cam in ('1', '2', '3'):
        p = np.load(f'{base}/PnP/openpose/{cfg}/{cam}_pnp.npz', allow_pickle=True)
        kp = np.load(f'{base}/keypoints/openpose/{cam}_2d.npz')['h36m_2d'].astype(float)
        K, L, dist = p['cam_K'], p['cam_L_ext'], p['cam_dist_cv']
        R, t = L[:3, :3], L[:3, 3]
        C = -R.T @ t
        placed = p['kps_H36M_placed'].astype(float) * 1000.0                      # camera frame, mm
        rp, pl, d2, hz = [], [], [], []
        for col, a in enumerate(('RAnkle', 'LAnkle')):
            g = p['foot_grounded'][:, col] & (kp[:, J[a], 2] > 0.3) & np.isfinite(tgt[:, J[a]]).all(1) & np.isfinite(placed[:, J[a]]).all(1)
            if not g.any():
                continue
            uv = cv2.undistortPoints(kp[g, J[a], :2].reshape(-1, 1, 2), K, dist).reshape(-1, 2)
            d = np.c_[uv, np.ones(len(uv))] @ R                                    # lab directions (R^T d per row)
            s = (h - C[2]) / d[:, 2]
            P = C + s[:, None] * d                                                 # lab mm
            Pc = P @ R.T + t
            Tc = tgt[g, J[a]] @ R.T + t
            rp.append(Pc[:, 2] - Tc[:, 2])
            pl.append(placed[g, J[a], 2] - Tc[:, 2])
            proj = cv2.projectPoints(Tc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
            d2.append(kp[g, J[a], :2] - proj)
            hz.append(((placed[g, J[a]] - t) @ R)[:, 2])                           # placed ankle lab height
        rp, pl, d2, hz = map(np.concatenate, (rp, pl, d2, hz))
        print(f'{u:5s}{cam:>4s}{len(rp):5d} | {np.median(rp):+9.0f} {np.median(pl):+7.0f} | {np.median(d2[:, 0]):+6.1f} {np.median(d2[:, 1]):+6.1f} | '
              f'{np.median(hz):8.0f} (target {h:.0f})')
