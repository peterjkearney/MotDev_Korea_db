#!/usr/bin/env python3
"""
rts_smoother.py — constant-velocity Kalman filter + RTS (Rauch-Tung-Striebel)
backward smoother for a 3D position trajectory, with per-frame ANISOTROPIC
measurement noise aligned to each frame's own camera ray (large uncertainty
along the ray -- depth -- small perpendicular to it, per the ΔX=ΔZ*(u-cx)/fx
relationship derived in conversation), and native support for missing
observations (frames with no measurement are pure-predicted, both passes).

Why RTS and not a per-frame causal gate: a forward-only gate can't tell
"correct value that only looks wrong because it disagrees with an already-
wrong running reference" apart from "genuinely wrong value" -- if an early
observation is bad, a causal filter can get stuck defending it, rejecting
every subsequently-correct observation for disagreeing with a state it
already (wrongly) trusts. RTS's backward pass means a bad early stretch
gets pulled toward what the REST of the sequence supports, from both
directions, instead of being permanently anchored to a bad start.
"""

import numpy as np


def ray_aligned_R(ray_unit, sigma_along, sigma_perp):
    """3x3 measurement covariance: large variance along ray_unit (the
    depth-ambiguity direction for THIS frame's specific ray), small
    variance in the plane perpendicular to it -- not axis-aligned, since
    the ray itself tilts away from the camera's Z-axis whenever the
    subject is off-centre (see conversation history)."""
    d = ray_unit / np.linalg.norm(ray_unit)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    Rot = np.stack([e1, e2, d], axis=1)   # columns: perp, perp, along-ray
    D = np.diag([sigma_perp ** 2, sigma_perp ** 2, sigma_along ** 2])
    return Rot @ D @ Rot.T


def rts_smooth_3d(obs, obs_valid, ray_units, dt,
                   sigma_along=0.15, sigma_perp=0.02, process_accel_std=2.0):
    """
    obs        : (T,3) raw position observations (metres); ignored where
                 obs_valid is False.
    obs_valid  : (T,) bool -- whether obs[t] is a real measurement.
    ray_units  : (T,3) unit camera-space ray direction per frame (only
                 needs to be valid where obs_valid is True).
    dt         : seconds between frames (constant -- fixed decimated fps).
    sigma_along, sigma_perp : measurement noise std (m), along vs
                 perpendicular to each frame's own ray.
    process_accel_std : std of unmodelled acceleration (m/s^2); ISOTROPIC
                 (a real body has no direction-dependent speed limit,
                 unlike the measurement noise, which does).

    Returns (smoothed (T,3) positions, smoothed (T,3,3) position covariances).
    """
    T = obs.shape[0]
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    q = process_accel_std ** 2
    Qb = q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    Q = np.zeros((6, 6))
    for i in range(3):
        idx = [i, i + 3]
        Q[np.ix_(idx, idx)] = Qb

    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0

    first_valid = np.where(obs_valid)[0]
    if len(first_valid) == 0:
        raise ValueError("no valid observations at all")

    s = np.zeros(6)
    s[:3] = obs[first_valid[0]]
    P = np.eye(6) * 1e4   # position AND velocity effectively unknown at start

    s_filt = np.zeros((T, 6))
    P_filt = np.zeros((T, 6, 6))

    for t in range(T):
        if t > 0:
            s = F @ s
            P = F @ P @ F.T + Q
        if obs_valid[t]:
            R = ray_aligned_R(ray_units[t], sigma_along, sigma_perp)
            y = obs[t] - H @ s
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            s = s + K @ y
            P = (np.eye(6) - K @ H) @ P
        s_filt[t] = s
        P_filt[t] = P

    s_smooth = s_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T - 2, -1, -1):
        P_pred_next = F @ P_filt[t] @ F.T + Q
        C = P_filt[t] @ F.T @ np.linalg.inv(P_pred_next)
        s_smooth[t] = s_filt[t] + C @ (s_smooth[t + 1] - F @ s_filt[t])
        P_smooth[t] = P_filt[t] + C @ (P_smooth[t + 1] - P_pred_next) @ C.T

    return s_smooth[:, :3], P_smooth[:, :3, :3]
