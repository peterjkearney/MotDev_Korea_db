"""Shared helpers for the floor tests: planted-ankle samples and the plane through them."""
import numpy as np
from utils.smpl_lite import J


def grounded_ankles(X):
    """Planted ankle positions (mm): ankle speed < 3 mm/frame and within 30 mm of the rep's low ankle height."""
    out = []
    for a in ('RAnkle', 'LAnkle'):
        A = X[:, J[a]]
        sp = np.r_[np.nan, np.linalg.norm(np.diff(A, axis=0), axis=1)]
        lo = np.nanpercentile(A[:, 2], 5)
        out.append(A[(sp < 3) & (A[:, 2] < lo + 30) & np.isfinite(A).all(1)])
    return np.concatenate(out)


def plane(P):
    """z = a x + b y + c through the samples -> (a, b, c)."""
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    return np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
