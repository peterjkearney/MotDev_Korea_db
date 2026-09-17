"""triangulate.py -- N-view triangulation of 2D keypoints with known calibration.

Used to build a triangulated-OpenPose ground truth on BioCV, so that adults
and children are scored against the SAME kind of target (Korea has no mocap).

Cameras are given as (K, dist, R, t) with X_cam = R @ X_lab + t, in the units
of t (BioCV: mm).  Pixels are undistorted with cv2 before the DLT, which is
run in normalised coordinates.

`triangulate_robust` drops disagreeing views: with N >= 3 views, if the
worst reprojection error of a joint exceeds `thresh_px`, the worst view is
removed and the joint re-solved, down to 2 views.  A left/right label swap
in one camera shows up as exactly such a disagreement.
"""
import cv2
import numpy as np


def undistort(xy, K, dist):
    """(N,2) pixels -> (N,2) normalised, distortion removed."""
    if len(xy) == 0:
        return np.zeros((0, 2))
    pts = cv2.undistortPoints(np.ascontiguousarray(xy, np.float64).reshape(-1, 1, 2),
                              K, dist.reshape(1, -1))
    return pts.reshape(-1, 2)


def project(X, K, dist, R, t):
    """(N,3) lab -> (N,2) pixels, NaN rows preserved."""
    out = np.full((len(X), 2), np.nan)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
        p, _ = cv2.projectPoints(np.ascontiguousarray(X[ok], np.float64), rvec,
                                 np.asarray(t, np.float64).reshape(3, 1),
                                 np.asarray(K, np.float64), dist.reshape(1, -1))
        out[ok] = p.reshape(-1, 2)
    return out


def dlt(norm_pts, Rs, ts):
    """One point from >= 2 views: normalised (u, v) per view, R, t per view."""
    A = []
    for (u, v), R, t in zip(norm_pts, Rs, ts):
        P = np.hstack([R, np.asarray(t).reshape(3, 1)])
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    h = Vt[-1]
    if abs(h[3]) < 1e-12:
        return None
    return h[:3] / h[3]


def triangulate_robust(xy, vis, cams, thresh_px=15.0, min_views=2):
    """xy (T,C,J,2) pixels, vis (T,C,J) bool, cams: list of (K, dist, R, t).

    Returns X (T,J,3) NaN where unsolved, used (T,C,J) views in the final
    solution, err (T,C,J) reprojection error in used views (NaN elsewhere).
    """
    T, C, J, _ = xy.shape
    X = np.full((T, J, 3), np.nan)
    used = np.zeros((T, C, J), bool)
    err = np.full((T, C, J), np.nan)

    # normalise every observation once
    norm = np.full((T, C, J, 2), np.nan)
    for c, (K, dist, R, t) in enumerate(cams):
        m = vis[:, c, :]
        if m.any():
            norm[:, c, :][m] = undistort(xy[:, c, :][m], K, dist)

    for f in range(T):
        for j in range(J):
            views = [c for c in range(C) if vis[f, c, j]]
            while len(views) >= min_views:
                pts = [norm[f, c, j] for c in views]
                P = dlt(pts, [cams[c][2] for c in views], [cams[c][3] for c in views])
                if P is None:
                    break
                e = []
                for c in views:
                    K, dist, R, t = cams[c]
                    z = (R @ P + np.asarray(t).reshape(3))[2]
                    if z <= 0:
                        e.append(np.inf)
                        continue
                    uv = project(P[None], K, dist, R, t)[0]
                    e.append(float(np.linalg.norm(uv - xy[f, c, j])))
                e = np.array(e)
                if e.max() <= thresh_px:
                    X[f, j] = P
                    for c, ec in zip(views, e):
                        used[f, c, j] = True
                        err[f, c, j] = ec
                    break
                if len(views) == min_views:
                    break                       # a pair cannot outvote itself
                views.pop(int(np.argmax(e)))    # drop the worst view, retry
    return X, used, err
