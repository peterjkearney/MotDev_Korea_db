#!/usr/bin/env python3
"""sync_two_cams.py -- check the sync of two BioCV cameras with the Korea sync method.

TEMPORARY experiment (not part of the pipeline).  BioCV cameras are hardware-synced, so
the true offset is 0 frames; the question is whether the subject's motion alone recovers it
for a camera pair at ~90 deg: one looking along the running line (runner coming straight at
or away from it), one side-on (runner crossing the image).

Uses korea_gt/align_cameras.py unchanged:
  * candidate_tracks  -- link each camera's OpenPose people into per-person tracks
  * SIGNALS / normalise -- 'vertical' (signed vertical velocity, the Korea default) or 'speed'
  * masked_xcorr      -- best offset within +-max_lag frames
With two cameras there is no lag12/lag13/lag23 consistency triangle, so only the min_corr
gate applies.  Every pair of candidate tracks is tried; the best-correlating pair wins (as
align_cameras.choose_people does for three cameras).

Independent referee (as build_child_gt.refine_sync): triangulate the two views with the BioCV
calibrations at each candidate offset and take the median reprojection error.  Caveat for this
layout: both cameras sit at about the same height, so the epipolar lines are near-horizontal
and the referee only "sees" vertical image motion -- the same information the vertical signal
uses.  It is a consistency check, not fully independent evidence.

--shift S drops the first S frames of camera B (negative: of camera A), so the expected
offset becomes S.  --sweep runs a range of shifts.

Lag convention (align_cameras): lag > 0 means A[i + lag] matches B[i].

    python sync_two_cams.py                       # cams 7 (line of sight) and 1 (side), true lag 0
    python sync_two_cams.py --cams 7 5 --shift 12
    python sync_two_cams.py --sweep -30 30 6
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'korea_gt'))
import align_cameras as ac            # noqa: E402
import camcal as cc                   # noqa: E402

DEFAULT_TRIAL = os.path.join(REPO, 'data', 'sampleBiocv', 'P03_RUN_01')
DEFAULT_CALIB = os.path.join(REPO, 'data', 'sampleBiocv')
CONF = ac.SYNC['conf_thresh']


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_openpose(trial, cam):
    """BioCV openpose_video_v1 JSON -> Korea-style frames[t] = [(xy (25,2), score (25,)), ...]."""
    path = os.path.join(trial, 'Analysis', 'keypoints', 'openpose', f'{cam:02d}_openpose.json')
    with open(path) as f:
        j = json.load(f)
    frames = []
    for fr in j['frames']:
        dets = []
        for p in fr['people']:
            k = np.asarray(p['pose_keypoints_2d'], float).reshape(25, 3)
            dets.append((k[:, :2], k[:, 2]))
        frames.append(dets)
    return frames, np.asarray(j['native_frame_idx']), float(j['target_fps'])


def load_calib(calib_dir, cams):
    """BioCV .calib files (mm) -> camcal.CameraSet (m), cameras in the order given."""
    from scipy.spatial.transform import Rotation
    f, k1, k2, rv, t, cx, cy = [], [], [], [], [], [], []
    for c in cams:
        v = np.array(open(os.path.join(calib_dir, f'{c:02d}.mp4-mocAligned.calib')).read().split(), float)
        K, M, d = v[2:11].reshape(3, 3), v[11:27].reshape(4, 4), v[27:32]
        f.append(K[0, 0]); cx.append(K[0, 2]); cy.append(K[1, 2])
        k1.append(d[0]); k2.append(d[1])
        rv.append(Rotation.from_matrix(M[:3, :3]).as_rotvec()); t.append(M[:3, 3] / 1000.0)
    a = lambda x: np.asarray(x, float)
    return cc.CameraSet(f=a(f), k1=a(k1), k2=a(k2), rvec=a(rv), t=a(t), cx=a(cx), cy=a(cy))


def apply_shift(frames_a, frames_b, shift):
    """Simulate unsynced starts: drop the first `shift` frames of B (or -shift of A)."""
    if shift > 0:
        return frames_a, frames_b[shift:]
    if shift < 0:
        return frames_a[-shift:], frames_b
    return frames_a, frames_b


# --------------------------------------------------------------------------
# Motion-signal sync (the Korea method, two cameras)
# --------------------------------------------------------------------------

def motion_sync(frames_a, frames_b, signal, max_lag):
    """Best (track pair, lag, corr) over every pair of candidate tracks."""
    ta, _ = ac.candidate_tracks(frames_a)
    tb, _ = ac.candidate_tracks(frames_b)
    sa = [ac.normalise(*ac.SIGNALS[signal](x, s)) for x, s in ta]
    sb = [ac.normalise(*ac.SIGNALS[signal](x, s)) for x, s in tb]
    results = []
    for i, j in itertools.product(range(len(ta)), range(len(tb))):
        curve = ac.masked_xcorr(sa[i][0], sb[j][0], sa[i][1], sb[j][1], max_lag)
        if curve:
            lag, r = max(curve, key=lambda x: x[1])
            results.append(dict(track=(i, j), lag=lag, corr=r, curve=curve))
    if not results:
        return None, ta, tb
    results.sort(key=lambda d: -d['corr'])
    best = results[0]
    # how distinct is the peak: best correlation more than 2 frames away from it
    far = [r for l, r in best['curve'] if abs(l - best['lag']) > 2]
    best['runner_up'] = max(far) if far else np.nan
    best['corr_at_0'] = dict(best['curve']).get(0, np.nan)
    best['n_pairs'] = len(results)
    best['status'] = 'OK' if best['corr'] >= ac.SYNC['min_corr'] else 'weak correlation'
    best['signals'] = (sa[best['track'][0]], sb[best['track'][1]])
    return best, ta, tb


# --------------------------------------------------------------------------
# Geometric referee
# --------------------------------------------------------------------------

def pair_at_lag(xa, sa, xb, sb, lag):
    """Overlapping frames of A and B at offset `lag` (A[i+lag] ~ B[i]) -> xy (F,2,25,2), conf (F,2,25)."""
    lo, hi = max(0, -lag), min(len(xb), len(xa) - lag)
    if hi <= lo:
        return None, None
    xy = np.stack([xa[lo + lag:hi + lag], xb[lo:hi]], axis=1)
    cf = np.stack([sa[lo + lag:hi + lag], sb[lo:hi]], axis=1)
    return xy, cf


def geometric_error(cams, xa, sa, xb, sb, lag, min_joints=200):
    """Median two-view reprojection error (px) of joints both cameras see, at offset `lag`."""
    xy, cf = pair_at_lag(xa, sa, xb, sb, lag)
    if xy is None:
        return np.nan, 0
    vis = (cf > CONF) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    vis &= vis.all(axis=1, keepdims=True)
    n = int(vis[:, 0].sum())
    if n < min_joints:
        return np.nan, n
    X, _ = cc.triangulate(cams, xy, vis)
    return float(np.nanmedian(cc.reprojection_per_view(cams, X, xy, vis))), n


def geometric_sync(cams, xa, sa, xb, sb, lags):
    errs = {l: geometric_error(cams, xa, sa, xb, sb, l) for l in lags}
    good = {l: e for l, (e, _) in errs.items() if np.isfinite(e)}
    best = min(good, key=good.get) if good else None
    return best, errs


# --------------------------------------------------------------------------
# Two-stage sync, no calibration: coarse stage picks the step, vertical signal picks the frame
# --------------------------------------------------------------------------

def step_period(sig, valid, lo=8, hi=60):
    """Step period (frames) from the first autocorrelation peak of a vertical signal."""
    ac_ = ac.masked_xcorr(sig, sig, valid, valid, hi, min_overlap=30)
    ac_ = [(l, r) for l, r in ac_ if lo <= l <= hi]
    for (l0, r0), (l1, r1), (l2, r2) in zip(ac_, ac_[1:], ac_[2:]):
        if r1 > r0 and r1 >= r2 and r1 > 0.2:
            return l1
    return np.nan


def progression(xy, sc):
    """(1/torso length, mid-hip x) per frame: distance from a camera the runner runs at, and
    position along the track in a side-on camera -- both ~affine in the runner's position."""
    ok = sc > CONF
    torso = np.where(ok[:, 1] & ok[:, 8], np.linalg.norm(xy[:, 1] - xy[:, 8], axis=1), np.nan)
    return 1.0 / torso, np.where(ok[:, 8], xy[:, 8, 0], np.nan)


def progression_cost(xa, sa, xb, sb, lags):
    """Coarse stage A (FAILS on steady running, kept for the record): relative residual of an
    affine fit between camera A's and camera B's progression signals at each lag.  Each camera
    uses whichever of (1/size, x) changes more over the run.  At constant speed a time shift is
    just an affine offset, so the fit is nearly as good at every lag."""
    def pick(xy, sc):
        inv, x = progression(xy, sc)
        rel = lambda p: np.nanstd(p) / np.nanmean(np.abs(p)) if np.isfinite(p).sum() > 10 else 0
        return inv if rel(inv) > rel(x) else x
    pa, pb = pick(xa, sa), pick(xb, sb)
    out = {}
    for L in lags:
        lo, hi = max(0, -L), min(len(pb), len(pa) - L)
        if hi - lo < 30:
            continue
        a, b = pa[lo + L:hi + L], pb[lo:hi]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30:
            continue
        X = np.c_[np.ones(m.sum()), b[m]]
        coef, *_ = np.linalg.lstsq(X, a[m], rcond=None)
        out[L] = float((a[m] - X @ coef).std() / a[m].std())
    return out


def _sampson(F, p1, p2):
    x1, x2 = np.c_[p1, np.ones(len(p1))], np.c_[p2, np.ones(len(p2))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    return np.sqrt(np.sum(x2 * Fx1, 1) ** 2 /
                   (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2))


def fmatrix_cost(xa, sa, xb, sb, lags, conf=0.5, min_pts=200):
    """Coarse stage B (no calibration): at each lag, fit a fundamental matrix to every joint both
    cameras see (RANSAC, 3 px) and take the median Sampson distance over ALL correspondences.
    Swapping left and right legs (one step off) is not a camera geometry, so step aliasing goes;
    a full stride (two steps) off is still nearly as good -- keep max_lag below a stride."""
    import cv2
    out = {}
    for L in lags:
        xy, cf = pair_at_lag(xa, sa, xb, sb, L)
        if xy is None:
            continue
        vis = (cf > conf).all(1) & ~(xy == 0).all(-1).any(1)
        p1, p2 = xy[:, 0][vis], xy[:, 1][vis]
        if len(p1) < min_pts:
            continue
        F, _ = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 3.0, 0.999, 5000)
        if F is not None:
            out[L] = float(np.median(_sampson(F[:3], p1, p2)))
    return out


def two_stage(best_vertical, xa, sa, xb, sb, max_lag, coarse='fmatrix'):
    """Coarse lag from `coarse`, then the vertical-signal peak within +-half a step of it."""
    (ga, va), (gb, vb) = best_vertical['signals']
    period = step_period(ga, va)
    if not np.isfinite(period):
        period = step_period(gb, vb)
    lags = range(-max_lag, max_lag + 1)
    cost = (fmatrix_cost if coarse == 'fmatrix' else progression_cost)(xa, sa, xb, sb, lags)
    if not cost:
        return None
    c_lag = min(cost, key=cost.get)
    half = int(period // 2) if np.isfinite(period) else 6
    near = [(l, r) for l, r in best_vertical['curve'] if abs(l - c_lag) <= half]
    f_lag, f_r = max(near, key=lambda x: x[1]) if near else (c_lag, np.nan)
    far = [v for l, v in cost.items() if abs(l - c_lag) > half]
    return dict(coarse=coarse, period=period, coarse_lag=c_lag, coarse_cost=cost[c_lag],
                coarse_margin=(min(far) / cost[c_lag]) if far and cost[c_lag] > 0 else np.nan,
                lag=f_lag, corr=f_r, cost=cost)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_one(args, frames_a, frames_b, cams, shift, verbose=True, plot=None):
    fa, fb = apply_shift(frames_a, frames_b, shift)
    out = dict(shift=shift)
    best_tracks = None
    for sig in args.signals:
        best, ta, tb = motion_sync(fa, fb, sig, args.max_lag)
        if best is None:
            out[sig] = None
            continue
        out[sig] = best
        if sig == args.signals[0]:
            best_tracks = (ta[best['track'][0]], tb[best['track'][1]], ta, tb)
        if verbose:
            print(f'  {sig:8s} lag {best["lag"]:+3d} (expected {shift:+d})  corr {best["corr"]:.3f}  '
                  f'corr@expected {dict(best["curve"]).get(shift, np.nan):.3f}  '
                  f'best >2 frames off {best["runner_up"]:.3f}  tracks {best["track"]} '
                  f'of {len(ta)}x{len(tb)}  {best["status"]}')
    if best_tracks is not None:
        (xa, sa), (xb, sb), ta, tb = best_tracks
        if 'vertical' in args.signals and out.get('vertical'):
            for coarse in args.coarse:
                ts = two_stage(out['vertical'], xa, sa, xb, sb, args.max_lag, coarse)
                out[f'two_stage_{coarse}'] = ts
                if verbose and ts:
                    unit = 'Sampson px' if coarse == 'fmatrix' else 'rel. resid'
                    print(f'  2-stage  {coarse:11s} coarse {ts["coarse_lag"]:+3d} ({unit} {ts["coarse_cost"]:.3f}, '
                          f'next best >half step away x{ts["coarse_margin"]:.2f}) -> vertical within '
                          f'+-{int(ts["period"] // 2)} (step {ts["period"]:.0f} fr): lag {ts["lag"]:+3d} '
                          f'(expected {shift:+d})  corr {ts["corr"]:.3f}')
        lags = range(-args.max_lag, args.max_lag + 1, 1)
        gbest, gerr = geometric_sync(cams, xa, sa, xb, sb, lags)
        out['geom'] = (gbest, gerr)
        if verbose and gbest is not None:
            e0 = gerr.get(shift, (np.nan, 0))[0]
            print(f'  geometry lag {gbest:+3d} (expected {shift:+d})  median reproj {gerr[gbest][0]:.1f} px '
                  f'(at expected {e0:.1f} px, {gerr[gbest][1]} joint-frames)')
        if plot:
            make_plot(plot, args, out, xa, sa, xb, sb, shift)
    return out


def make_plot(path, args, out, xa, sa, xb, sb, shift):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a, b = args.cams
    fig, ax = plt.subplots(4, 1, figsize=(10, 13))
    s0 = args.signals[0]
    best = out[s0]
    (ga, va), (gb, vb) = best['signals']
    lag = best['lag']
    t = np.arange(len(gb))
    ia = t + lag
    ok = (ia >= 0) & (ia < len(ga))
    ax[0].plot(t[vb], gb[vb], '-', lw=1, label=f'cam {b:02d}')
    sel = ok.copy(); sel[ok] &= va[ia[ok]]
    ax[0].plot(t[sel], ga[ia[sel]], '-', lw=1, label=f'cam {a:02d} shifted by {lag:+d}')
    ax[0].set_title(f'{s0} signal, aligned at the found lag ({lag:+d}; expected {shift:+d})')
    ax[0].set_xlabel('frame of cam B (60 fps)'); ax[0].legend()
    for sig in args.signals:
        if out.get(sig):
            l, r = zip(*out[sig]['curve'])
            ax[1].plot(l, r, label=f'{sig} (best {out[sig]["lag"]:+d}, r={out[sig]["corr"]:.2f})')
    ax[1].axvline(shift, color='k', ls=':', label='expected')
    ax[1].set_xlabel('lag (frames)'); ax[1].set_ylabel('Pearson r'); ax[1].legend()
    ax[1].set_title('cross-correlation vs lag')
    gbest, gerr = out['geom']
    l = sorted(gerr)
    ax[2].plot(l, [gerr[k][0] for k in l])
    ax[2].axvline(shift, color='k', ls=':', label='expected')
    if gbest is not None:
        ax[2].axvline(gbest, color='r', ls='--', label=f'min {gbest:+d}')
    ax[2].set_xlabel('lag (frames)'); ax[2].set_ylabel('median reprojection (px)')
    ax[2].set_title('geometric referee: two-view triangulation error vs lag'); ax[2].legend()
    for c in args.coarse:
        ts = out.get(f'two_stage_{c}')
        if ts:
            l = sorted(ts['cost'])
            v = np.array([ts['cost'][k] for k in l])
            ax[3].plot(l, v / v.min(), label=f'{c} (min {ts["coarse_lag"]:+d} -> 2-stage {ts["lag"]:+d})')
    ax[3].axvline(shift, color='k', ls=':', label='expected')
    ax[3].set_xlabel('lag (frames)'); ax[3].set_ylabel('cost / its minimum')
    ax[3].set_title('two-stage, coarse stage cost (no calibration)'); ax[3].legend()
    fig.suptitle(f'{os.path.basename(args.trial)}  cams {a:02d} (A) vs {b:02d} (B)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'  plot -> {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--trial', default=DEFAULT_TRIAL)
    ap.add_argument('--calib-dir', default=DEFAULT_CALIB)
    ap.add_argument('--cams', type=int, nargs=2, default=[7, 1], metavar=('A', 'B'),
                    help='camera A and B (default 7 = line of sight, 1 = side-on)')
    ap.add_argument('--signals', nargs='+', default=['vertical', 'speed'], choices=sorted(ac.SIGNALS))
    ap.add_argument('--max-lag', type=int, default=45)
    ap.add_argument('--coarse', nargs='*', default=['fmatrix', 'progression'], choices=['fmatrix', 'progression'],
                    help='coarse stage(s) for the two-stage sync (needs the vertical signal)')
    ap.add_argument('--shift', type=int, default=0, help='artificial offset: drop the first S frames of B (A if <0)')
    ap.add_argument('--sweep', type=int, nargs=3, metavar=('FROM', 'TO', 'STEP'),
                    help='run every shift in range(FROM, TO+1, STEP) and tabulate')
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    a, b = args.cams
    frames_a, idx_a, fps = load_openpose(args.trial, a)
    frames_b, idx_b, _ = load_openpose(args.trial, b)
    assert np.array_equal(idx_a, idx_b), 'cameras sampled at different native frames'
    cams = load_calib(args.calib_dir, args.cams)
    ang = np.degrees(np.arccos(abs(np.dot(cams.axes[0], cams.axes[1]))))
    print(f'{os.path.basename(args.trial)}: cam {a:02d} vs cam {b:02d}, {len(frames_a)} frames at {fps:g} fps, '
          f'optical axes {ang:.0f} deg apart; people per frame max {max(map(len, frames_a))}/{max(map(len, frames_b))}')

    if not args.sweep:
        print(f'shift {args.shift:+d}:')
        run_one(args, frames_a, frames_b, cams, args.shift,
                plot=os.path.join(args.out, f'sync_cam{a:02d}_cam{b:02d}_shift{args.shift:+d}.png'))
        return

    rows = []
    for s in range(args.sweep[0], args.sweep[1] + 1, args.sweep[2]):
        print(f'shift {s:+d}:')
        o = run_one(args, frames_a, frames_b, cams, s)
        row = dict(shift=s)
        for sig in args.signals:
            row[f'{sig}_lag'] = o[sig]['lag'] if o.get(sig) else np.nan
            row[f'{sig}_corr'] = o[sig]['corr'] if o.get(sig) else np.nan
        for c in args.coarse:
            ts = o.get(f'two_stage_{c}')
            row[f'{c}_coarse_lag'] = ts['coarse_lag'] if ts else np.nan
            row[f'2stage_{c}_lag'] = ts['lag'] if ts else np.nan
        row['geom_lag'] = o['geom'][0] if o.get('geom') else np.nan
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    for k in [c for c in df if c.endswith('_lag')]:
        df[k.replace('_lag', '_err')] = df[k] - df['shift']
    path = os.path.join(args.out, f'sweep_cam{a:02d}_cam{b:02d}.csv')
    df.to_csv(path, index=False)
    print(df.to_string(index=False))
    print(f'table -> {path}')


if __name__ == '__main__':
    main()
