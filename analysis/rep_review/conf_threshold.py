#!/usr/bin/env python3
"""conf_threshold.py -- are OpenPose's low-confidence points still in the right place?

For every rep in a USABLE camera setup of a set (good calibration), and each camera c: triangulate each body joint
from the OTHER two cameras using only their confident points (conf > 0.5) and only where that pair agrees (both
residuals <= 15 px), project it into camera c, and measure how far camera c's own OpenPose point is from it --
binned by camera c's confidence.  The conf > 0.5 bins are the reference: the error a point the build already
trusts shows (it includes the 2-view triangulation's own error).  Also counts what a lower threshold would add:
joint-frames that would gain a view (2 -> 3) or become triangulable at all (1 -> 2).  Reads data only.

    python analysis/rep_review/conf_threshold.py --set B [--workers 6]
"""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeleton as K
import camcal as cc

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
JOINTS = {0: 'Nose', 1: 'Neck', 2: 'RSh', 3: 'RElb', 4: 'RWr', 5: 'LSh', 6: 'LElb', 7: 'LWr', 8: 'MidHip',
          9: 'RHip', 10: 'RKnee', 11: 'RAnk', 12: 'LHip', 13: 'LKnee', 14: 'LAnk'}
THRESH = (0.4, 0.3, 0.2, 0.1)


def subject(args):
    S, subj = args
    summ, cams = K._session(S, subj)
    if summ is None or cams is None or summ.get('status') != 'ok':
        return None
    errs, gains = [], []
    for rec in summ['reps']:
        loc, _ = K._locate(S, rec['stub'])
        if loc is None or not loc[1].get('usable'):
            continue
        cs, X, ok, start, lag, used, err, vis, conf, xy = K._triangulate(S, rec['stub'], loc)
        T = len(xy)
        present = (conf > 0) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))           # (T,3,25)
        for c in range(3):
            o = [k for k in range(3) if k != c]
            v = np.zeros_like(present)
            for k in o:
                v[:, k] = present[:, k] & (conf[:, k] > 0.5)
            Xp, okp = cc.triangulate(cs, xy, v)
            okp &= v[:, o[0]] & v[:, o[1]]
            e = cc.reprojection_per_view(cs, Xp, xy, v)                             # (T,3,25)
            okp &= (np.nan_to_num(e[:, o[0]], nan=1e9) <= 15) & (np.nan_to_num(e[:, o[1]], nan=1e9) <= 15)
            uv = cs.project(np.nan_to_num(Xp.reshape(-1, 3)), c).reshape(T, 25, 2)
            d = np.linalg.norm(uv - xy[:, c], axis=-1)
            m = okp & present[:, c]
            for j in JOINTS:
                sel = m[:, j]
                if sel.any():
                    errs.append(pd.DataFrame(dict(joint=JOINTS[j], cam=c + 1, conf=conf[sel, c, j], err=d[sel, j])))
        # what a lower threshold adds
        body = list(JOINTS)
        for t in THRESH:
            n05 = (present & (conf > 0.5))[:, :, body].sum(1)
            nt = (present & (conf > t))[:, :, body].sum(1)
            gains.append(dict(t=t, jf=int(n05.size), two_to_three=int(((n05 == 2) & (nt == 3)).sum()),
                              one_to_two=int(((n05 <= 1) & (nt >= 2)).sum())))
    if not errs:
        return None
    return pd.concat(errs), pd.DataFrame(gains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='B')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{K.ROOT}/{a.set}/GT3D/*/session_summary.json'))
    with Pool(a.workers) as pool:
        res = [r for r in pool.map(subject, [(a.set, s) for s in subs]) if r is not None]
    E = pd.concat([r[0] for r in res]); G = pd.concat([r[1] for r in res]).groupby('t').sum()
    E['bin'] = pd.cut(E.conf, BINS, right=True)
    q = lambda p: (lambda x: np.percentile(x, p))
    tab = E.groupby('bin', observed=True).err.agg(n='size', median='median', p75=q(75), p90=q(90),
                                                  within15=lambda x: (x <= 15).mean() * 100)
    print(f'set {a.set}, usable setups: camera c\'s OpenPose point vs the other two cameras\' triangulation, px (1080p)')
    print(tab.round(1).to_string())
    print('\nmedian px by joint and confidence (rows: joint; cols: conf bin)')
    E['b3'] = pd.cut(E.conf, [0, 0.2, 0.35, 0.5, 1.01], labels=['<=0.2', '0.2-0.35', '0.35-0.5', '>0.5'])
    print(E.pivot_table(index='joint', columns='b3', values='err', aggfunc='median', observed=True).round(1)
          .reindex(list(JOINTS.values())).to_string())
    print('\nshare of each joint\'s points with conf in (0.2, 0.5]:',
          (E.groupby('joint').conf.apply(lambda x: ((x > 0.2) & (x <= 0.5)).mean() * 100).round(0)
           .reindex(list(JOINTS.values())).to_dict()))
    G['two_to_three_%'] = (100 * G.two_to_three / G.jf).round(1); G['one_to_two_%'] = (100 * G.one_to_two / G.jf).round(1)
    print('\nlowering the threshold from 0.5 (body joint-frames in usable setups):')
    print(G[['jf', 'two_to_three', 'two_to_three_%', 'one_to_two', 'one_to_two_%']].sort_index(ascending=False).to_string())


if __name__ == '__main__':
    main()
