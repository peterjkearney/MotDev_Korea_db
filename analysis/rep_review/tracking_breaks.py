#!/usr/bin/env python3
"""tracking_breaks.py -- count synced reps where align_cameras' person tracking lost the child in one camera.

align_cameras.py tracks every person per camera (Hungarian on joint distance, gaps up to max_gap bridged, fragments
merged) and picks one track per camera.  When the child's track splits into fragments that overlap in time, they are
not merged, and the chosen fragment ends while the child carries on in another one (A069_GMS_4_1 cam 3: chosen track
frames 0-77, child continues in a track covering 61-284 -> no camera-3 points for most of the rep).

A BREAK, per rep and camera, inside the synced span:
  - the chosen track has no body (< MIN_JOINTS of the 15 body joints) for a run of frames, while
  - another track of that camera has a body there, which CONTINUES the child: its body centre within
    NEAR_PX + PX_PER_FRAME x (frames since the chosen track last saw the child) of the chosen track's nearest body,
    and its body height 0.67-1.5x the chosen track's median -- so an adult walking in while the child is simply
    undetected doesn't count;
  - for at least MIN_RUN frames.
Reads Aligned/ only (no change to anything).  Reps that failed sync have no Aligned file and are not covered.

    python analysis/rep_review/tracking_breaks.py [--sets A,B,C] [--out breaks.csv]
"""
import argparse, glob, os
import numpy as np, pandas as pd

ROOT = os.environ.get('KOREA_ROOT', '/Volumes/Sandisk_SSD/MotorDevelopment/Korea')
MIN_JOINTS, MIN_RUN, NEAR_PX, PX_PER_FRAME = 5, 10, 150.0, 10.0
BODY = list(range(15))


def body_stats(xy):
    """per frame: present (bool), centre (2,), height -- from the 15 body joints with a point."""
    pts = xy[:, BODY]
    have = ~((pts[..., 0] == 0) & (pts[..., 1] == 0))
    present = have.sum(1) >= MIN_JOINTS
    with np.errstate(invalid='ignore'):
        p = np.where(have[..., None], pts, np.nan)
        centre = np.nanmean(p, axis=1)
        height = np.nanmax(p[..., 1], axis=1) - np.nanmin(p[..., 1], axis=1)
    return present, centre, height


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def rep_breaks(path):
    a = np.load(path)
    T = len(a['xy1'])
    out = []
    for c in range(3):
        tx = a[f'tracks_xy{c + 1}']
        k0 = int(a['track'][c])
        s0 = int(a['start'][c])
        span = slice(s0, s0 + T)
        pres, cen, hgt = body_stats(tx[k0][span])
        if not pres.any():
            out.append(dict(cam=c + 1, lost=T, broken=0, note='chosen track empty'))
            continue
        h_ref = np.nanmedian(hgt[pres])
        idx = np.where(pres)[0]
        others = [body_stats(tx[k][span]) for k in range(len(tx)) if k != k0]
        lost = broken = 0
        for g0, g1 in runs(~pres):
            lost += g1 - g0
            cover = np.zeros(g1 - g0, bool)
            for po, co, ho in others:
                for f in range(g0, g1):
                    if not po[f]:
                        continue
                    near = idx[np.argmin(np.abs(idx - f))]          # chosen track's nearest frame with the child
                    ok = (np.linalg.norm(co[f] - cen[near]) <= NEAR_PX + PX_PER_FRAME * abs(f - near)
                          and 0.67 <= ho[f] / h_ref <= 1.5)
                    cover[f - g0] |= ok
            for r0, r1 in runs(cover):
                if r1 - r0 >= MIN_RUN:
                    broken += r1 - r0
        out.append(dict(cam=c + 1, lost=lost, broken=broken, note=''))
    return T, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='A,B,C')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rv = pd.read_csv(f'{ROOT}/rep_review.csv', usecols=['set', 'stub', 'group', 'stage_failed'])
    stage = dict(zip(rv.stub, rv.stage_failed))
    rows = []
    for S in args.sets.split(','):
        for p in sorted(glob.glob(f'{ROOT}/{S}/Aligned/{S}*_GMS_*.npz')):
            if os.path.basename(p).startswith('._'):
                continue
            stub = os.path.basename(p)[:-4]
            T, cams = rep_breaks(p)
            for r in cams:
                rows.append(dict(set=S, stub=stub, T=T, stage=stage.get(stub, '?'), **r))
    D = pd.DataFrame(rows)
    if args.out:
        D.to_csv(args.out, index=False)
    R = D.groupby(['set', 'stub', 'stage', 'T']).agg(broken=('broken', 'sum'), worst=('broken', 'max'),
                                                    cams_broken=('broken', lambda x: int((x > 0).sum()))).reset_index()
    R['frac'] = R.worst / R['T']
    print(f'tracking breaks (>= {MIN_RUN} frames of the child in another track while the chosen track is empty)\n')
    for S, g in R.groupby('set'):
        b = g[g.broken > 0]
        print(f'set {S}: {len(b)} of {len(g)} synced reps have a break '
              f'({100 * len(b) / len(g):.1f}%); camera-frames lost to breaks {int(g.broken.sum())} '
              f'of {int(3 * g["T"].sum())} ({100 * g.broken.sum() / (3 * g["T"].sum()):.1f}%)')
        st = b.groupby('stage').agg(reps=('stub', 'size'), median_frac=('frac', 'median'))
        tot = g.groupby('stage').size()
        st['of'] = tot.reindex(st.index)
        st['median_frac'] = (100 * st.median_frac).round(0)
        print('   by stage (median share of the rep lost, worst camera %):\n' + st.to_string().replace('\n', '\n   '))
        sev = pd.cut(b.frac, [0, .1, .25, .5, 1.01], labels=['<10%', '10-25%', '25-50%', '>50%'])
        print('   severity (worst camera, share of rep frames):', sev.value_counts().sort_index().to_dict())
        print('   breaks by camera:', D[(D.set == S) & (D.broken > 0)].cam.value_counts().sort_index().to_dict())
        print()


if __name__ == '__main__':
    main()
