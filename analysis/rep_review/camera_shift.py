#!/usr/bin/env python3
"""camera_shift.py -- did a camera move, or is the rep just noisier?  (a better test than the 1.5x reprojection gate)

The rep gate compares the SIZE of a rep's reprojection error with the calibration's.  A moved camera and a noisy rep
both raise it.  What separates them is the PATTERN: after a camera is knocked, every point it sees sits off to the same
side of where the other two cameras put the joint; noise scatters in every direction.

For every rep in a calibrated camera setup, and each camera c:
  - triangulate each body joint from the OTHER two cameras (conf > 0.5, both residuals <= 15 px) -- so camera c can't
    pull the 3D point towards itself and hide its own offset -- and project into camera c;
  - d = camera c's OpenPose point minus that projection (px, 1080p);
  - bias = median d (x, y), over all joint-frames; scatter = median |d - bias|.
Even a fixed camera has some bias (unmodelled lens distortion, OpenPose's own habits, where the child stands), so the
reference is the setup's OTHER usable reps:  shift = |bias - median bias of the setup's other usable reps|.  The usable
reps' own shifts (each against the rest) give the normal range.  Reads data only.

    python analysis/rep_review/camera_shift.py --set B [--group 03] [--out shifts.csv]
"""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeleton as K
import camcal as cc

BODY = list(range(15))
AGREE_PX = 15.0
MIN_N = 100          # joint-frames a camera needs in a rep for a bias


def loo_residuals(cs, xy, conf, c):
    """(T, 15, 2) camera c's point minus the 2-view (other cameras) projection; NaN where not measurable."""
    T = len(xy)
    present = (conf > 0.5) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    o = [k for k in range(3) if k != c]
    v = np.zeros_like(present)
    for k in o:
        v[:, k] = present[:, k]
    Xp, okp = cc.triangulate(cs, xy, v)
    okp &= v[:, o[0]] & v[:, o[1]]
    e = cc.reprojection_per_view(cs, Xp, xy, v)
    okp &= (np.nan_to_num(e[:, o[0]], nan=1e9) <= AGREE_PX) & (np.nan_to_num(e[:, o[1]], nan=1e9) <= AGREE_PX)
    uv = cs.project(np.nan_to_num(Xp.reshape(-1, 3)), c).reshape(T, 25, 2)
    d = xy[:, c] - uv
    m = okp & present[:, c]
    d[~m] = np.nan
    return d[:, BODY]


def stats(d):
    d = d.reshape(-1, 2)
    d = d[np.isfinite(d).all(1)]
    if len(d) < MIN_N:
        return dict(n=len(d), bx=np.nan, by=np.nan, scatter=np.nan)
    b = np.median(d, axis=0)
    return dict(n=len(d), bx=b[0], by=b[1], scatter=float(np.median(np.linalg.norm(d - b, axis=1))))


def pattern(d, b, xy_c):
    """body_frac: share of joints (>= 30 points) whose own bias carries at least half the rep's bias in its direction
    -- a moved camera shifts the whole body, an OpenPose habit one or two joints.
    sync_k: least-squares fit of (d - bias) = k * image velocity (px/frame) -- in frames; a sync error makes the
    residual follow each joint's motion, a moved camera or OpenPose noise doesn't (k ~ 0)."""
    out = dict(body_frac=np.nan, sync_k=np.nan)
    if not np.isfinite(b).all():
        return out
    nb = np.linalg.norm(b)
    if nb > 0:
        u = b / nb
        proj = []
        for j in range(d.shape[1]):
            dj = d[:, j][np.isfinite(d[:, j]).all(1)]
            if len(dj) >= 30:
                proj.append(np.median(dj, axis=0) @ u)
        if proj:
            out['body_frac'] = float(np.mean(np.array(proj) >= 0.5 * nb))
    v = np.full_like(xy_c, np.nan)
    v[1:-1] = (xy_c[2:] - xy_c[:-2]) / 2
    r = d - b
    m = np.isfinite(r).all(-1) & np.isfinite(v).all(-1)
    vv = (v[m] ** 2).sum()
    if m.sum() >= MIN_N and vv > 0:
        out['sync_k'] = float((r[m] * v[m]).sum() / vv)
    return out


def subject(args):
    S, subj = args
    summ, cams = K._session(S, subj)
    if summ is None or cams is None or summ.get('status') != 'ok':
        return []
    rows = []
    for rec in summ['reps']:
        loc, _ = K._locate(S, rec['stub'])
        if loc is None:
            continue
        cs, X, ok, start, lag, used, err, vis, conf, xy = K._triangulate(S, rec['stub'], loc)
        T = len(xy)
        ratio = np.array(rec.get('median_reproj_px') or [np.nan] * 3, float) / \
            np.array(loc[1].get('calib_median_reproj_px') or [np.nan] * 3, float)
        for c in range(3):
            d = loo_residuals(cs, xy, conf, c)
            s_all, s1, s2 = stats(d), stats(d[:T // 2]), stats(d[T // 2:])
            xyc = np.where(((xy[:, c, BODY] == 0).all(-1) | (conf[:, c, BODY] <= 0.5))[..., None], np.nan, xy[:, c, BODY])
            s_all.update(pattern(d, np.array([s_all['bx'], s_all['by']]), xyc))
            rows.append(dict(subject=subj, stub=rec['stub'], setup=loc[4], usable=bool(rec['usable']),
                             setup_usable=bool(loc[1].get('usable')), cam=c + 1, ratio=ratio[c], **s_all,
                             bx1=s1['bx'], by1=s1['by'], bx2=s2['bx'], by2=s2['by']))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='B')
    ap.add_argument('--group', default='03', help='review-sheet group to report (prefix)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rv = pd.read_csv(f'{K.ROOT}/rep_review.csv', dtype=str, keep_default_na=False)
    rv = rv[rv.set == a.set].set_index('stub')
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{K.ROOT}/{a.set}/GT3D/*/session_summary.json'))
    with Pool(a.workers) as pool:
        D = pd.DataFrame([r for rr in pool.map(subject, [(a.set, s) for s in subs]) for r in rr])
    D['group'] = D.stub.map(rv.group).fillna('')
    D['bad_cams'] = D.stub.map(rv.bad_cams).fillna('')

    # shift against the setup's OTHER usable reps (leave the rep itself out)
    D['shift'] = np.nan
    D['n_ref'] = 0
    for (subj, s, c), g in D.groupby(['subject', 'setup', 'cam']):
        u = g[g.usable & g.bx.notna()]
        for i, r in g.iterrows():
            ref = u[u.stub != r.stub]
            if len(ref) < 3 or not np.isfinite(r.bx):
                continue
            b0 = ref[['bx', 'by']].median().values
            D.loc[i, 'shift'] = float(np.hypot(r.bx - b0[0], r.by - b0[1]))
            D.loc[i, 'n_ref'] = len(ref)
            D.loc[i, 'half_shift'] = float(np.hypot(r.bx2 - r.bx1, r.by2 - r.by1))
    if a.out:
        D.to_csv(a.out, index=False)

    U = D[D.usable & D['shift'].notna()]
    q = lambda x, p: np.nanpercentile(x, p)
    p95, p99 = q(U['shift'], 95), q(U['shift'], 99)
    print(f'set {a.set}: shift of a camera\'s bias from its setup\'s other usable reps (px, 1080p)')
    print(f'  usable reps (normal range): median {q(U["shift"], 50):.1f}, p90 {q(U["shift"], 90):.1f}, '
          f'p95 {p95:.1f}, p99 {p99:.1f}   (n = {len(U)} rep-cameras)')
    print(f'  usable reps, scatter: median {q(U.scatter, 50):.1f}, p95 {q(U.scatter, 95):.1f}')
    print(f'  usable reps, first-vs-second-half bias change: median {q(U.half_shift, 50):.1f}, p95 {q(U.half_shift, 95):.1f}')
    print(f'  usable reps, body_frac: median {q(U.body_frac, 50):.2f};  sync_k (frames): median {q(U.sync_k, 50):.2f}, '
          f'p5 {q(U.sync_k, 5):.2f}, p95 {q(U.sync_k, 95):.2f}')
    Us = D[D.usable & D.scatter.notna()]
    sc95 = q(Us.scatter, 95)

    G = D[D.group.str.startswith(a.group)].copy()
    flagged = G[G.apply(lambda r: str(r.cam) in r.bad_cams.split(','), axis=1)]
    print(f'\ngroup {a.group}: {G.stub.nunique()} reps, {len(flagged)} flagged rep-cameras')
    cls = np.select([flagged['shift'] > p95, flagged.scatter > sc95],
                    ['shifted (bias beyond usable p95)', 'noisier (scatter beyond usable p95)'], 'neither beyond p95')
    cls = np.where(flagged['shift'].isna(), 'no reference (<3 usable reps in setup)', cls)
    print('  ' + pd.Series(cls).value_counts().to_string().replace('\n', '\n  '))
    print(f'\n  flagged cameras (shift / scatter / half-to-half change, px; usable p95: {p95:.1f} / {sc95:.1f} / '
          f'{q(U.half_shift, 95):.1f})')
    cols = ['stub', 'cam', 'ratio', 'n', 'bx', 'by', 'shift', 'body_frac', 'scatter', 'sync_k', 'half_shift', 'n_ref']
    print(flagged[cols].round(2).to_string(index=False))

    # persistence: per child/setup/camera with a flagged rep, shift of every rep in order
    print('\n  rep-by-rep shift for each child/camera with a flagged rep (* = flagged by the gate, u = usable):')
    for (subj, s, c), g in D[D.set_index(['subject', 'setup', 'cam']).index.isin(
            flagged.set_index(['subject', 'setup', 'cam']).index)].groupby(['subject', 'setup', 'cam']):
        g = g.assign(k=g.stub.str.split('_').map(lambda p: (int(p[2]), int(p[3])))).sort_values('k')
        seq = ' '.join(f"{p[2]}_{p[3]}:{v:.0f}{'*' if str(c) in b.split(',') else ('u' if u else '')}"
                       for p, v, b, u in zip(g.stub.str.split('_'), g['shift'].fillna(-1), g.bad_cams, g.usable))
        print(f'  {subj} s{s} cam{c}: {seq}')


if __name__ == '__main__':
    main()
