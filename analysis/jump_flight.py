#!/usr/bin/env python3
"""jump_flight.py -- flight-phase diagnostics for the Korea jump reps (B3 broad jump, C3 rope jump: *_GMS_3_*).

Reads step_4b outputs (PnP/{detector}/{config}/{cam}_pnp.npz: kps_H36M_placed, com_placed, foot_grounded) and the
leave-one-out triangulated target (H36M/openpose_tri_h36m.npz), scored as step_8 scores placed error.
Nothing here changes the pipeline; it asks whether a ballistic (parabola) COM term has room to help.

Flight timing (--timing, every subcommand):
    ankle  step_4b's foot_grounded (2D ankle speed, 5-frame box, >= 3-frame contacts): ~4-5 frames early at takeoff
           and ~3-4 late at landing -- it counts push-off and landing absorption as air
    ankle2d  ankle-only, YOLO-compatible (see ankle2d_air): one-sided ankle speed candidates, edges where the trunk's
           2D vertical acceleration points down, veto below 0.7 g in pixels
    toe    this camera's own 2D: a foot is grounded while any of BigToe / SmallToe / Heel moves < --toe-px px/frame
           (one-sided: the smaller of the two neighbouring frame steps); single-camera, deployable
    ref    triangulated BODY_25 toes/heels (DLT over the rep's 3 cameras): airborne while both feet are visible and the
           lowest foot point is > 20 mm above the floor level just before or just after (the higher of the two; the
           landing area can sit ~40 mm lower, e.g. all B023 reps); a side with no floor in view is ignored

    gravity  fitted vertical acceleration of the monocular COM / pelvis vs triangulated pelvis, per flight;
             plus residual about the best parabola (vertical) / straight line (horizontal)
    error    flight vs stance placed error; pelvis-error decomposition over each flight
             (offset / linear drift / curvature / wobble, orthogonal)
    split    flight vs stance: placed error, root-aligned pose error, pelvis offset (and its depth part)
    bridge   replace each flight's COM path by an arc joining the takeoff and landing stance frames
             (--horiz: straight line horizontally, vertical kept) and re-score flight frames
    sync     flight error with the target shifted by -2..+2 frames; placed error vs target pelvis speed
    geom     fps, subject size in pixels, camera distance, horizontal travel and depth change per flight
    timing   score the ankle and toe detectors against the triangulated reference: takeoff / landing frame offsets,
             airtime error, flights invented where the reference is in view (all jump reps, no step_4b output needed
             for toe; ankle needs --configs' outputs)
    scale    can gravity set metric scale?  g fitted (1 frame trimmed at each end) on the triangulated pelvis, a
             triangulated segment-weighted COM (child masses) and the monocular mesh COM, per reference flight

    python3 analysis/jump_flight.py gravity --root data/korea_dev_colab --configs C8,C25
    python3 analysis/jump_flight.py bridge --root data/korea_dev_colab --configs C25 --horiz

Results 2026-11-05 (dev subset, C25): flight 71 vs stance 49 mm placed; pelvis flight error is mostly a smooth depth
slide (offset 36 / drift 44 / curvature 15 / wobble 16 mm); per-flight mean 82 mm -> horizontal bridge 70 mm (14/21
improved), full ballistic arc 156 mm (the detected flight is too long, so g over it gives a bump ~70-130 mm above the
real rise); error rises with speed 46 -> 80 mm; a 1-frame target shift costs ~17 mm.  (All with --timing ankle.)
Timing 2026-11-10 (48 camera-flights, 16 reference flights of 4-14 frames, median 9): ankle -4 / +3 frames, toe
(5 px) 0 / 0 with 60% / 83% within 1 frame, 2 invented flights.  With --timing toe: flight 64 vs stance 51 mm, bridge
69 -> 61 mm (full arc ~= horizontal).  Scale: g per flight has an 11-15% robust CV (tri COM 10.3, tri pelvis 9.0, mono
COM 9.5 m/s^2 median) -- noise-limited by ~6 usable frames at 30 fps, so gravity cannot set metric scale here.
ankle2d (no toes, for YOLO): takeoff -1 / landing 0 frames, 58% / 85% within 1, airtime 50%, 8 invented flights over
all jump reps (vs 38 without the acceleration veto); bridge with it 79 -> 73 (full arc) / 71 (horizontal), 13-15/20.
"""
import argparse
import glob
import os
import warnings

import numpy as np

G = 9.81


# ---------------------------------------------------------------- loading

def reps(root):
    """(rep name, Analysis dir, target npz) for every jump rep with a triangulated target."""
    for rep in sorted(glob.glob(os.path.join(root, '*', '*_GMS_3_*'))):
        an = os.path.join(rep, 'Analysis')
        tri = os.path.join(an, 'H36M', 'openpose_tri_h36m.npz')
        if os.path.exists(tri):
            yield os.path.basename(rep), an, np.load(tri, allow_pickle=True)


def cameras(an, cfg, detector):
    """(cam, pnp npz) for each camera's output of this config."""
    for f in sorted(glob.glob(os.path.join(an, 'PnP', detector, cfg, '*_pnp.npz'))):
        yield os.path.basename(f).split('_')[0], np.load(f, allow_pickle=True)


def calib(d):
    """R, t (mm) with X_cam = R @ X_lab + t; lab is z up, floor at 0."""
    L = d['cam_L_ext']
    t = L[:3, 3] if np.abs(L[:3, 3]).max() > 50 else L[:3, 3] * 1000.0
    return L[:3, :3], t


def target_cam(an, Gt, d, cam, detector, shift=0):
    """Placed H36M (camera frame, mm), LOO target aligned row-for-row (camera frame, mm, NaN where absent), T."""
    sfi = np.load(os.path.join(an, 'keypoints', detector, f'{cam}_2d.npz'))['source_frame_idx']
    X = d['kps_H36M_placed'].astype(float) * 1000.0
    T = min(len(X), len(sfi))
    loo = {str(c): Gt['kps3d_loo'][i] for i, c in enumerate(Gt['cameras'])}[cam]
    s = sfi[:T] + shift
    g = np.full((T, 17, 3), np.nan)
    ok = (s >= 0) & (s < len(loo))
    g[ok] = loo[s[ok]]
    R, t = calib(d)
    return X[:T], g @ R.T + t, T


def runs(mask, lo=1, hi=None):
    """[start, end) runs of True with lo <= length (<= hi)."""
    out, i, T = [], 0, len(mask)
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if j - i >= lo and (hi is None or j - i <= hi):
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def frame_err(X, gc, vm):
    """Per-frame mean joint error (mm) over the target's valid joints."""
    e = np.linalg.norm(X - gc, axis=-1)[:, vm]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                 # frames with no target -> NaN
        return np.nanmean(e, 1)


def rms(Z):
    return float(np.sqrt((Z ** 2).sum(1).mean()))


# ---------------------------------------------------------------- flight timing

FOOT = {'R': [22, 23, 24], 'L': [19, 20, 21]}                  # BODY_25 BigToe, SmallToe, Heel
_TRI = {}


def tri_body25(rep_dir, Gt, conf=0.3):
    """BODY_25 triangulated (DLT on undistorted points, >= 2 views with conf > conf) on the target's rows, lab mm.
    Matches the target's ankles to 0.0 mm.  Cached per rep."""
    if rep_dir in _TRI:
        return _TRI[rep_dir]
    import sys
    import cv2
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pipeline'))
    from utils.calibration import load_calib
    an, N = os.path.join(rep_dir, 'Analysis'), len(Gt['kps3d'])
    rows, Ps = [], []
    for cam in [str(c) for c in Gt['cameras']]:
        _, _, K, L, dist = load_calib(os.path.join(rep_dir, f'{cam}.mp4-mocAligned.calib'))
        k = np.load(os.path.join(an, 'keypoints', 'openpose', f'{cam}_2d.npz'))
        b, sfi = k['body25_2d'], k['source_frame_idx']
        uv = cv2.undistortPoints(b[..., :2].reshape(-1, 1, 2).astype(np.float64), np.asarray(K, float),
                                 np.asarray(dist, float).ravel()).reshape(b.shape[:-1] + (2,))
        r = np.full((N, 25, 3), np.nan)
        ok = (sfi >= 0) & (sfi < N)
        r[sfi[ok], :, :2] = uv[ok]
        r[sfi[ok], :, 2] = b[ok, :, 2]
        rows.append(r)
        Ps.append(np.asarray(L, float)[:3])
    X = np.full((N, 25, 3), np.nan)
    for i in range(N):
        for j in range(25):
            A = [row for r, P in zip(rows, Ps) if r[i, j, 2] > conf
                 for row in (r[i, j, 0] * P[2] - P[0], r[i, j, 1] * P[2] - P[1])]
            if len(A) >= 4:
                v = np.linalg.svd(np.array(A))[2][-1]
                X[i, j] = v[:3] / v[3]
    _TRI[rep_dir] = X
    return X


def ref_flights(X, thr=20.0, W=10, min_peak=40.0):
    """Reference flights on target rows from triangulated feet: [(start, end, clean)], clean = both ends bracketed by
    frames with both feet visible.  Airborne = lowest foot point > thr above the floor level within W frames before or
    after (the higher; a side whose window never comes back near the floor is ignored); peak must clear min_peak."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)                 # feet out of view -> all-NaN rows
        zR = np.nanmin(X[:, FOOT['R'], 2], 1)
        zL = np.nanmin(X[:, FOOT['L'], 2], 1)
        known = np.isfinite(zR) & np.isfinite(zL)
        z = np.where(known, np.fmin(zR, zL), np.nan)
        N = len(z)
        air = np.zeros(N, bool)
        out = []
        for t in np.flatnonzero(known):
            lw, rw = z[max(0, t - W):t + 1], z[t:t + W + 1]
            l, r = np.nanmin(lw), np.nanmin(rw)
            if not r <= l + thr:
                r = -np.inf
            if not l <= r + thr:
                l = -np.inf
            air[t] = z[t] > max(l, r) + thr
        for s, e in runs(air):
            if np.nanmax(z[s:e]) - np.nanmin(z[max(0, s - W):s + 1]) >= min_peak:
                out.append((s, e, bool(s > 0 and e < N and known[s - 1] and known[e])))
    return out, z


def toe_air(b25, thr_px=5.0, conf=0.3):
    """Single-camera flight: neither foot has a BigToe/SmallToe/Heel moving < thr_px px/frame (the smaller of the
    two neighbouring steps), and both feet are visible."""
    T = len(b25)
    g = np.zeros((T, 2), bool)
    vis = np.zeros((T, 2), bool)
    for k, f in enumerate('RL'):
        for j in FOOT[f]:
            xy = b25[:, j, :2].astype(float)
            ok = b25[:, j, 2] > conf
            xy[~ok] = np.nan
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            v = np.full(T, np.nan)
            v[1:-1] = np.fmin(step[1:], step[:-1])
            g[:, k] |= v < thr_px
            vis[:, k] |= ok
    return ~g.any(1) & vis.all(1)


def rows_to_cam(rows, sfi):
    """A mask on target rows -> on this camera's frames."""
    out = np.zeros(len(sfi), bool)
    ok = (sfi >= 0) & (sfi < len(rows))
    out[ok] = rows[sfi[ok]]
    return out


def cam_to_rows(m, sfi, n):
    out = np.zeros(n, bool)
    ok = (sfi >= 0) & (sfi < n)
    out[sfi[ok][m[ok]]] = True
    return out


H_RA, H_LA, H_NOSE = 3, 6, 9
H_TRUNK = [1, 4, 8, 11, 14]                                     # RHip, LHip, Thorax, LShoulder, RShoulder


def ankle2d_air(h, stature_m, fps, thr_px=8.0, min_c=2, pad=4, g_frac=0.7, conf=0.3):
    """Ankle-only single-camera flight (YOLO-compatible: H36M ankles, hips, thorax, shoulders, nose).
    1. candidates: neither ankle has a one-sided 2D speed < thr_px px/frame for >= min_c frames, both ankles visible;
    2. edges: within each candidate +-pad frames, the longest run where the trunk's 2D vertical acceleration points
       down (image y'' > 0, [.25 .5 .25] smoothed) -- push-off and landing accelerate the trunk upward;
    3. veto: keep it only if the trunk's mean downward acceleration over it is >= g_frac x g in pixels
       (g_px = 9.81 x px-per-metre / fps^2, px-per-metre from the nose-to-ankle span ~ 0.87 stature).
    Real flights sit at ~1 g (all kept at 0.9 g, a third lost at 1.1 g)."""
    T = len(h)
    g = np.zeros((T, 2), bool)
    vis = np.zeros((T, 2), bool)
    for k, j in enumerate((H_RA, H_LA)):
        ok = h[:, j, 2] > conf
        vis[:, k] = ok
        xy = np.where(ok[:, None], h[:, j, :2].astype(float), np.nan)
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        v = np.full(T, np.nan)
        v[1:-1] = np.fmin(step[1:], step[:-1])
        for s, e in runs(v < thr_px, min_c):
            g[s:e, k] = True
    cand = ~g.any(1) & vis.all(1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        y = np.nanmean(np.where(h[:, H_TRUNK, 2] > conf, h[:, H_TRUNK, 1], np.nan), 1)
        acc = np.full(T, np.nan)
        acc[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
        acc_s = np.convolve(np.nan_to_num(acc), [.25, .5, .25], 'same')
        okh = (h[:, H_NOSE, 2] > conf) & vis.all(1)
        span = np.nanmedian(np.maximum(h[okh, H_RA, 1], h[okh, H_LA, 1]) - h[okh, H_NOSE, 1]) if okh.any() else np.nan
    g_px = 9.81 * span / (0.87 * stature_m) / fps ** 2
    out = np.zeros(T, bool)
    for s, e in runs(cand):
        p, q = max(1, s - pad), min(T - 1, e + pad)
        rr = runs(acc_s[p:q] > 0)
        if not rr:
            continue
        b0, b1 = max(rr, key=lambda r: r[1] - r[0])
        if b1 - b0 >= 2 and np.nanmean(acc[p + b0:p + b1]) >= g_frac * g_px:
            out[p + b0:p + b1] = True
    return out


def stature(an):
    import json
    with open(os.path.join(os.path.dirname(os.path.dirname(an)), 'user_meta.json')) as f:
        return float(json.load(f)['stature_m'])


def flight_air(a, an, Gt, d, cam, T):
    """Airborne mask on this camera's first T output frames, by --timing."""
    if a.timing == 'ankle':
        return ~d['foot_grounded'][:T].any(1)
    k = np.load(os.path.join(an, 'keypoints', 'openpose', f'{cam}_2d.npz'))
    if a.timing == 'toe':
        return toe_air(k['body25_2d'][:T], a.toe_px)
    if a.timing == 'ankle2d':
        return ankle2d_air(k['h36m_2d'][:T], stature(an), float(k['fps']))
    flights, _ = ref_flights(tri_body25(os.path.dirname(an), Gt))
    rows = np.zeros(len(Gt['kps3d']), bool)
    for s, e, _ in flights:
        rows[s:e] = True
    return rows_to_cam(rows, k['source_frame_idx'][:T])


# ---------------------------------------------------------------- subcommands

def cmd_gravity(a):
    trim = 1

    def fit(z, t):
        m = np.isfinite(z)
        z, t = z[m], t[m]
        if len(z) < 4:
            return np.nan, np.nan
        A = np.stack([t ** 2, t, np.ones_like(t)], 1)
        c, *_ = np.linalg.lstsq(A, z, rcond=None)
        return -2 * c[0], float(np.sqrt(np.mean((A @ c - z) ** 2)))

    def hres(X, t):
        m = np.isfinite(X[:, :2]).all(1)
        if m.sum() < 5:
            return np.nan
        A = np.stack([t[m], np.ones(m.sum())], 1)
        c, *_ = np.linalg.lstsq(A, X[m, :2], rcond=None)
        return 1000 * rms(A @ c - X[m, :2])

    rows = []
    for name, an, Gt in reps(a.root):
        fps = float(Gt['fps'])
        P = Gt['kps3d'][:, 0] / (1000.0 if str(Gt['units']) == 'mm' else 1.0)
        for cfg in a.configs:
            for cam, d in cameras(an, cfg, a.detector):
                R, t = calib(d)
                lab = lambda X: (X - t / 1000.0) @ R
                C, Pm = lab(d['com_placed']), lab(d['kps_H36M_placed'][:, 0])
                air = flight_air(a, an, Gt, d, cam, len(d['pnp_ok'])) & d['pnp_ok']
                for s, e in runs(air, a.min_flight):
                    s, e = s + trim, e - trim
                    tt = np.arange(s, e) / fps
                    gc, rc = fit(C[s:e, 2], tt)
                    gp, _ = fit(Pm[s:e, 2], tt)
                    gt, rt = fit(P[s:e, 2], tt) if e <= len(P) else (np.nan, np.nan)
                    ht = hres(P[s:e], tt) if e <= len(P) else np.nan
                    rows.append((name, cfg, cam, s, e, gc, gp, gt, 1000 * rc, 1000 * rt, hres(C[s:e], tt), ht))
    print(f'{"rep":16s}{"cfg":>4s}{"cam":>4s}{"frames":>9s}{"g_COM":>8s}{"g_pelv":>8s}{"g_triP":>8s}{"rmsCOMmm":>9s}')
    for r in rows:
        print(f'{r[0]:16s}{r[1]:>4s}{r[2]:>4s}{f"{r[3]}-{r[4]}":>9s}{r[5]:8.1f}{r[6]:8.1f}{r[7]:8.1f}{r[8]:9.1f}')
    print('\ntrue flights only (tri pelvis g > 5): residual RMS mm about the best ballistic fit')
    for cfg in a.configs:
        Rr = [r for r in rows if r[1] == cfg and r[7] > 5]
        if not Rr:
            continue
        A = np.array([[r[8], r[9], r[10], r[11]] for r in Rr])
        g = np.array([r[5] for r in Rr])
        gt = np.array([r[7] for r in Rr])
        print(f'{cfg}: n={len(Rr)}  g mono COM {np.median(g):.1f}, tri pelvis {np.median(gt):.1f} m/s^2 | vertical: mono COM '
              f'{np.median(A[:, 0]):.1f} (max {A[:, 0].max():.1f}), tri pelvis {np.nanmedian(A[:, 1]):.1f} | horizontal: '
              f'mono COM {np.median(A[:, 2]):.1f} (max {A[:, 2].max():.1f}), tri pelvis {np.nanmedian(A[:, 3]):.1f}')


def cmd_error(a):
    for cfg in a.configs:
        fl_all, st_all, rows = [], [], []
        for name, an, Gt in reps(a.root):
            vm, fps = Gt['valid_joint_mask'], float(Gt['fps'])
            for cam, d in cameras(an, cfg, a.detector):
                X, gc, T = target_cam(an, Gt, d, cam, a.detector)
                R, _ = calib(d)
                fe = frame_err(X, gc, vm)
                air = flight_air(a, an, Gt, d, cam, T)
                segs = runs(air, a.min_flight)
                fl = np.zeros(T, bool)
                for s, e in segs:
                    fl[s:e] = True
                fl_all += list(fe[fl & np.isfinite(fe)])
                st_all += list(fe[~air & np.isfinite(fe)])
                dv_cam = X[:, 0] - gc[:, 0]
                dv_lab = dv_cam @ R
                for s, e in segs:
                    m = np.isfinite(dv_lab[s:e]).all(1)
                    if m.sum() < 6:
                        continue
                    tt, Y = (np.arange(s, e) - s)[m] / fps, dv_lab[s:e][m]
                    Q, _ = np.linalg.qr(np.stack([np.ones_like(tt), tt, tt ** 2], 1))   # offset, drift, curvature
                    Cq = Q.T @ Y
                    parts = [rms(np.outer(Q[:, k], Cq[k])) for k in range(3)]
                    with np.errstate(all='ignore'):
                        rows.append((name, cam, s, e, np.nanmean(fe[s:e]), np.nanmean(fe[~air]), rms(Y),
                                     np.sqrt(np.nanmean(dv_cam[s:e, 2] ** 2)), *parts, rms(Y - Q @ Cq),
                                     *np.nanmean(dv_lab[s:e], 0)))
        print(f'\n== {cfg}: mean per-joint placed error  flight {np.median(fl_all):.0f} mm (n={len(fl_all)} frames)  '
              f'stance {np.median(st_all):.0f} mm (n={len(st_all)})  [frame medians]')
        print(f'{"rep":14s}{"cam":>4s}{"frames":>8s}{"errFl":>7s}{"errSt":>7s} | pelvis err RMS: {"total":>6s}{"depth":>6s}'
              f'{"offset":>7s}{"drift":>6s}{"curv":>6s}{"wobble":>7s} | mean lab dx,dy,dz')
        for r in rows:
            print(f'{r[0]:14s}{r[1]:>4s}{f"{r[2]}-{r[3]}":>8s}{r[4]:7.0f}{r[5]:7.0f} | {"":15s}{r[6]:6.0f}{r[7]:6.0f}'
                  f'{r[8]:7.0f}{r[9]:6.0f}{r[10]:6.0f}{r[11]:7.0f} | {r[12]:5.0f}{r[13]:5.0f}{r[14]:5.0f}')
        if rows:
            A = np.array([r[4:12] for r in rows])
            names = ['errFl', 'errSt', 'total', 'depth', 'offset', 'drift', 'curv', 'wobble']
            print('median:', ' '.join(f'{n}={v:.0f}' for n, v in zip(names, np.median(A, 0))))


def cmd_split(a):
    for cfg in a.configs:
        acc = {k: [] for k in ['fl_pl', 'st_pl', 'fl_ra', 'st_ra', 'fl_root', 'st_root', 'fl_dep', 'st_dep']}
        for name, an, Gt in reps(a.root):
            vm = Gt['valid_joint_mask']
            for cam, d in cameras(an, cfg, a.detector):
                X, gc, T = target_cam(an, Gt, d, cam, a.detector)
                air = flight_air(a, an, Gt, d, cam, T)
                fl = np.zeros(T, bool)
                for s, e in runs(air, a.min_flight):
                    fl[s:e] = True
                st = ~air
                vals = dict(pl=frame_err(X, gc, vm),
                            ra=frame_err(X - X[:, :1], gc - gc[:, :1], vm),
                            root=np.linalg.norm(X[:, 0] - gc[:, 0], axis=-1),
                            dep=np.abs(X[:, 0, 2] - gc[:, 0, 2]))
                for k, v in vals.items():
                    acc['fl_' + k] += list(v[fl])
                    acc['st_' + k] += list(v[st])
        md = {k: np.nanmedian(v) for k, v in acc.items()}
        print(f'{cfg}: placed  flight {md["fl_pl"]:.0f} / stance {md["st_pl"]:.0f} mm | root-aligned pose  flight '
              f'{md["fl_ra"]:.0f} / stance {md["st_ra"]:.0f} | pelvis offset  flight {md["fl_root"]:.0f} / stance '
              f'{md["st_root"]:.0f} (depth part {md["fl_dep"]:.0f} / {md["st_dep"]:.0f})  '
              f'[n flight {np.isfinite(acc["fl_pl"]).sum()}, stance {np.isfinite(acc["st_pl"]).sum()}]')


def cmd_bridge(a):
    for cfg in a.configs:
        res = []
        for name, an, Gt in reps(a.root):
            vm = Gt['valid_joint_mask']
            for cam, d in cameras(an, cfg, a.detector):
                fps = float(d['fps'])
                X, gc, T = target_cam(an, Gt, d, cam, a.detector)
                R, t = calib(d)
                Xl, gl = (X - t) @ R, (gc - t) @ R
                Cl = (d['com_placed'][:T].astype(float) * 1000.0 - t) @ R
                air = flight_air(a, an, Gt, d, cam, T)
                for s, e in runs(air, a.min_flight):
                    p, q = s - 1, e                                    # stance frames either side
                    if p < 0 or q >= T:
                        continue
                    tt = (np.arange(p, q + 1) - p) / fps
                    tb = tt[-1]
                    arc = Cl[p] + np.outer(tt / tb, Cl[q] - Cl[p])
                    arc[:, 2] = Cl[p:q + 1, 2] if a.horiz else arc[:, 2] + 0.5 * G * 1000.0 * tt * (tb - tt)
                    sh = (arc - Cl[p:q + 1])[1:-1]
                    fr = slice(s, e)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)       # flights with no target rows
                        e0 = np.nanmean(frame_err(Xl[fr], gl[fr], vm))
                        e1 = np.nanmean(frame_err(Xl[fr] + sh[:, None], gl[fr], vm))
                        eab = np.nanmean(frame_err(Xl[[p, q]], gl[[p, q]], vm))
                    res.append((name, cam, s, e, e0, e1, eab, np.abs(sh).max()))
        print(f'== {cfg}  ({"horizontal straight line" if a.horiz else "full ballistic arc"}, flights >= {a.min_flight} frames)')
        print(f'{"rep":14s}{"cam":>4s}{"frames":>8s}{"before":>8s}{"bridged":>8s}{"endpts":>8s}{"maxShift":>9s}')
        for r in res:
            print(f'{r[0]:14s}{r[1]:>4s}{f"{r[2]}-{r[3]}":>8s}{r[4]:8.0f}{r[5]:8.0f}{r[6]:8.0f}{r[7]:9.0f}')
        if res:
            A = np.array([r[4:7] for r in res])
            print(f'median before {np.nanmedian(A[:, 0]):.0f}  bridged {np.nanmedian(A[:, 1]):.0f}  endpoint frames '
                  f'{np.nanmedian(A[:, 2]):.0f};  improved {np.sum(A[:, 1] < A[:, 0])}/{len(A)}\n')


def cmd_sync(a):
    shifts = (-2, -1, 0, 1, 2)
    for cfg in a.configs:
        tab = {k: [] for k in shifts}
        sp, er = [], []
        print(f'== {cfg}')
        for name, an, Gt in reps(a.root):
            vm, fps = Gt['valid_joint_mask'], float(Gt['fps'])
            for cam, d in cameras(an, cfg, a.detector):
                row = {}
                for k in shifts:
                    X, gc, T = target_cam(an, Gt, d, cam, a.detector, shift=k)
                    if k == shifts[0]:
                        air = flight_air(a, an, Gt, d, cam, T)
                        fl = np.zeros(T, bool)
                        for s, e in runs(air, a.min_flight, 20):
                            fl[s:e] = True
                    e = frame_err(X, gc, vm)
                    with np.errstate(all='ignore'):
                        row[k] = np.nanmean(e[fl])
                    tab[k].append(row[k])
                    if k == 0:
                        R, t = calib(d)
                        g_lab = (gc[:, 0] - t) @ R
                        sp += list(np.linalg.norm(np.gradient(g_lab, axis=0), axis=-1) * fps / 1000.0)
                        er += list(e)
                kb = min(row, key=lambda k: np.inf if np.isnan(row[k]) else row[k])
                print(f'{name:14s} cam{cam}  ' + '  '.join(f'{k:+d}:{row[k]:5.0f}' for k in shifts) + f'   best {kb:+d}')
        print('median flight error by target shift:', '  '.join(f'{k:+d}: {np.nanmedian(v):.0f}' for k, v in tab.items()))
        sp, er = np.array(sp), np.array(er)
        m = np.isfinite(sp) & np.isfinite(er)
        for lo, hi in [(0, .3), (.3, .8), (.8, 1.5), (1.5, np.inf)]:
            s = m & (sp >= lo) & (sp < hi)
            print(f'target pelvis speed {lo}-{hi} m/s: placed error median {np.median(er[s]):.0f} mm (n={s.sum()})')


def cmd_geom(a):
    for cfg in a.configs:
        for name, an, Gt in reps(a.root):
            for cam, d in cameras(an, cfg, a.detector):
                fps = float(d['fps'])
                R, t = calib(d)
                X = d['kps_H36M_placed'].astype(float)
                lab = (X[:, 0] - t / 1000.0) @ R
                depth = np.nanmedian(X[:, 0, 2])
                px = float(d['mesh_height_m']) * d['cam_K'][1, 1] / depth
                segs = runs(flight_air(a, an, Gt, d, cam, len(X)), int(round(0.2 * fps)))
                head = (f'{name:14s} {cfg:>4s} cam{cam}  fps {fps:3.0f}  img {int(d["cam_h"])} px  subject ~{px:4.0f} px  '
                        f'camera distance {depth:3.1f} m')
                if not segs:
                    print(head, ' no flight')
                    continue
                s, e = segs[0]
                travel = 1000 * np.linalg.norm(lab[e - 1, :2] - lab[s, :2])
                ddepth = 1000 * (X[e - 1, 0, 2] - X[s, 0, 2])
                print(f'{head}  flight {(e - s) / fps:.2f} s ({e - s} fr)  horizontal travel {travel:4.0f} mm  '
                      f'depth change {ddepth:5.0f} mm')


def cmd_timing(a):
    """Detector takeoff / landing offsets (frames, detected - reference) on clean reference flights."""
    stats, extra = {'ankle': [], 'ankle2d': [], 'toe': []}, {'ankle': 0, 'ankle2d': 0, 'toe': 0}
    n_ref, dur = 0, []
    for name, an, Gt in reps(a.root):
        X = tri_body25(os.path.dirname(an), Gt)
        allf, z = ref_flights(X)
        clean = [(s, e) for s, e, c in allf if c]
        n_ref += len(clean)
        dur += [e - s for s, e in clean]
        known = np.isfinite(z)
        for cam in [str(c) for c in Gt['cameras']]:
            k = np.load(os.path.join(an, 'keypoints', 'openpose', f'{cam}_2d.npz'))
            sfi = k['source_frame_idx']
            det = {'toe': toe_air(k['body25_2d'], a.toe_px),
                   'ankle2d': ankle2d_air(k['h36m_2d'], stature(an), float(k['fps']))}
            pf = os.path.join(an, 'PnP', a.detector, a.configs[0], f'{cam}_pnp.npz')
            if os.path.exists(pf):
                fg = np.load(pf)['foot_grounded']
                det['ankle'] = np.zeros(len(sfi), bool)
                det['ankle'][:len(fg)] = ~fg[:len(sfi)].any(1)
            for dn, air in det.items():
                segs = [(s, e) for s, e in runs(cam_to_rows(air, sfi, len(z)), 2)]
                for s, e in clean:
                    ov = [(p, q) for p, q in segs if p < e and q > s]
                    stats[dn].append((ov[0][0] - s, ov[-1][1] - e) if ov else (np.nan, np.nan))
                extra[dn] += sum(1 for p, q in segs if q - p >= 3 and known[p:q].mean() > 0.8
                                 and not any(p < e and q > s for s, e, _ in allf))
    print(f'reference flights (clean, triangulated toes): {n_ref}, {np.median(dur):.0f} frames median '
          f'(range {min(dur)}-{max(dur)})')
    for dn, st in stats.items():
        if not st:
            print(f'{dn:6s}: no outputs (ankle needs {a.configs[0]} step_4b outputs)')
            continue
        A = np.array(st, float)
        m = np.isfinite(A[:, 0])
        w1 = lambda v: f'{np.mean(np.abs(v) <= 1):.0%}'
        print(f'{dn:7s}: {m.sum()}/{len(A)} camera-flights found; takeoff {np.median(A[m, 0]):+.0f} (within 1 frame '
              f'{w1(A[m, 0])}), landing {np.median(A[m, 1]):+.0f} ({w1(A[m, 1])}), airtime {w1(A[m, 1] - A[m, 0])}; '
              f'invented flights >= 3 frames: {extra[dn]}')


# child-ish segment masses (fraction) and centroid along proximal -> distal, on H36M joints
_SEG = [((8, 9), .14, .5), ((0, 8), .47, .5),
        ((11, 12), .025, .45), ((14, 15), .025, .45), ((12, 13), .02, .6), ((15, 16), .02, .6),
        ((1, 2), .09, .43), ((4, 5), .09, .43), ((2, 3), .055, .6), ((5, 6), .055, .6)]


def seg_com(K):
    return sum(m * (K[:, p] + f * (K[:, q] - K[:, p])) for (p, q), m, f in _SEG) / sum(m for _, m, _ in _SEG)


def cmd_scale(a):
    def gfit(z, fps, s, e, trim=1):
        s, e = s + trim, e - trim
        t, zz = np.arange(s, e) / fps, z[s:e]
        m = np.isfinite(zz)
        return -2 * np.polyfit(t[m], zz[m], 2)[0] if m.sum() >= 4 else np.nan

    rows = []
    for name, an, Gt in reps(a.root):
        fps, K = float(Gt['fps']), Gt['kps3d'] / 1000.0
        flights, z = ref_flights(tri_body25(os.path.dirname(an), Gt))
        C = seg_com(K)
        for s, e, clean in flights:
            if not clean:
                continue
            r = dict(rep=name, s=s, e=e, g_pelv=gfit(K[:, 0, 2], fps, s, e), g_com=gfit(C[:, 2], fps, s, e))
            for cam, d in cameras(an, a.configs[0], a.detector):
                R, t = calib(d)
                k = np.load(os.path.join(an, 'keypoints', 'openpose', f'{cam}_2d.npz'))
                sfi = k['source_frame_idx']
                T = min(len(sfi), len(d['com_placed']))
                Cm = np.full((len(z), 3), np.nan)
                ok = (sfi[:T] >= 0) & (sfi[:T] < len(z))
                Cm[sfi[:T][ok]] = ((d['com_placed'][:T].astype(float) - t / 1000.0) @ R)[ok]
                r[f'ref{cam}'] = gfit(Cm[:, 2], fps, s, e)
                ov = [(p, q) for p, q in runs(cam_to_rows(toe_air(k['body25_2d'], a.toe_px), sfi, len(z)))
                      if p < e and q > s]
                r[f'toe{cam}'] = gfit(Cm[:, 2], fps, ov[0][0], ov[-1][1]) if ov else np.nan
            rows.append(r)
    cams = sorted({k[3:] for r in rows for k in r if k.startswith('ref')})
    print(f'{"rep":14s}{"frames":>8s}{"triPelv":>8s}{"triCOM":>7s}   mono COM ' +
          ' '.join(f'cam{c} ref|toe' for c in cams) + f'   (g, m/s^2; mono = {a.configs[0]})')
    for r in rows:
        fr = f'{r["s"]}-{r["e"]}'
        mono = '  '.join(f'{r.get("ref" + c, np.nan):5.1f}|{r.get("toe" + c, np.nan):5.1f}' for c in cams)
        print(f'{r["rep"]:14s}{fr:>8s}{r["g_pelv"]:8.1f}{r["g_com"]:7.1f}   {mono}')

    def summ(v, label):
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v):
            med = np.median(v)
            print(f'{label:34s} n={len(v):2d}  median {med:5.2f} (/9.81 = {med / G:.2f})  IQR {np.percentile(v, 25):5.2f}-'
                  f'{np.percentile(v, 75):5.2f}  robust CV {1.4826 * np.median(np.abs(v - med)) / med:.0%}')
    summ([r['g_pelv'] for r in rows], 'triangulated pelvis')
    summ([r['g_com'] for r in rows], 'triangulated segment COM')
    summ([r.get('ref' + c, np.nan) for r in rows for c in cams], 'mono mesh COM, reference timing')
    summ([r.get('toe' + c, np.nan) for r in rows for c in cams], 'mono mesh COM, own toe timing')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['gravity', 'error', 'split', 'bridge', 'sync', 'geom', 'timing', 'scale'])
    ap.add_argument('--root', default='data/korea_dev_colab', help='folder of {subject}/{rep} trees')
    ap.add_argument('--configs', default='C8,C25', help='comma-separated step_4b configs')
    ap.add_argument('--detector', default='openpose')
    ap.add_argument('--timing', choices=['ankle', 'ankle2d', 'toe', 'ref'], default='ankle', help='flight timing (see above)')
    ap.add_argument('--toe-px', type=float, default=5.0, help='toe timing: grounded below this 2D speed, px/frame')
    ap.add_argument('--min-flight', type=int, default=None,
                    help='frames for a run to count as a flight (default 8 for ankle timing, 4 otherwise)')
    ap.add_argument('--horiz', action='store_true', help='bridge: horizontal straight line only, keep vertical')
    a = ap.parse_args()
    a.configs = a.configs.split(',')
    if a.min_flight is None:
        a.min_flight = 8 if a.timing == 'ankle' else 4
    globals()['cmd_' + a.cmd](a)


if __name__ == '__main__':
    main()
