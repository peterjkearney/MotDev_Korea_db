"""Re-level a Korea OUT_DIR copy's lab frame to the plane of the planted ankles (flat-floor assumption).
Rotates about the planted ankles' centroid (their mean height is kept); targets (kps3d, kps3d_loo) and every
calib (L_new = L_old @ inv(T)) get the same rigid transform, so camera-frame quantities are unchanged.
Run from pipeline/:  python relevel.py <OUT_DIR copy> B023 B014 ..."""
import glob, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.calibration import load_calib, save_calib
from floorlib import grounded_ankles, plane


def rot_a_to_b(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


root, users = sys.argv[1], sys.argv[2:]
for u in users:
    targets = sorted(glob.glob(f'{root}/{u}/*/Analysis/H36M/openpose_tri_h36m.npz'))
    P = np.concatenate([grounded_ankles(np.load(p)['kps3d'].astype(float)) for p in targets])
    a, b, _ = plane(P)
    Rc = rot_a_to_b(np.array([-a, -b, 1.0]), np.array([0, 0, 1.0]))
    p0 = P.mean(0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rc, p0 - Rc @ p0
    Tinv = np.linalg.inv(T)
    for p in targets:
        z = dict(np.load(p, allow_pickle=True))
        for k in ('kps3d', 'kps3d_loo'):
            if k in z:
                z[k] = (z[k].astype(np.float64) @ Rc.T + T[:3, 3]).astype(z[k].dtype)
        np.savez(p, **z)
    calibs = glob.glob(f'{root}/{u}/*.mp4-mocAligned.calib') + glob.glob(f'{root}/{u}/*/*.mp4-mocAligned.calib')
    for cp in calibs:
        w, h, K, L, dist = load_calib(cp)
        save_calib(cp, w, h, K, L @ Tinv, dist)
    P2 = np.concatenate([grounded_ankles(np.load(p)['kps3d'].astype(float)) for p in targets])
    a2, b2, _ = plane(P2)
    print(f'{u}: tilt {np.degrees(np.arctan(np.hypot(a, b))):.2f} -> {np.degrees(np.arctan(np.hypot(a2, b2))):.2f} deg; '
          f'planted ankle median {np.median(P[:, 2]):.0f} -> {np.median(P2[:, 2]):.0f} mm; {len(targets)} targets, {len(calibs)} calibs')
