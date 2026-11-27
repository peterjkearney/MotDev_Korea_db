#!/usr/bin/env python3
"""view_agreement.py -- can camera agreement say which end of a bad bone is wrong?

For every rep of a set that reached a calibrated camera setup, re-triangulate (skeleton.py) and, for every frame
where a bone is more than 25% from the setup median (red: today's joint gate drops BOTH ends), classify its two
ends by camera agreement: 'strong' = triangulated from all 3 views with the worst residual <= tau px (1080p).
  one strong, one weak -> resolvable: keep the strong end, blame the weak one
  both strong          -> agreement can't explain it (OpenPose definition shift, shared error, calibration)
  neither strong       -> still ambiguous
Also: the residual distribution of 3-view joints (to choose tau), and how often the length-only rule (blame the
end whose OTHER bones are bad) picks the same end.  No change to any data.

    python analysis/rep_review/view_agreement.py --set C [--workers 6]
"""
import argparse, os, sys, glob, json
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skeleton as K

TAUS = (3.0, 5.0, 8.0)
RED = 0.25


def subject_rows(args):
    S, subj, tiers = args
    out, resid3 = [], []
    summ, cams = K._session(S, subj)
    if summ is None or cams is None or summ.get('status') != 'ok':
        return out, resid3
    for rec in summ['reps']:
        stub = rec['stub']
        loc, _ = K._locate(S, stub)
        if loc is None:
            continue
        st = loc[1]
        cs, X, ok, start, lag, used, err, vis, *_ = K._triangulate(S, stub, loc)
        med = st['session_bone_median']
        dev = K._devs(X, ok, med)                                   # (T, nb)
        views = np.where(ok, used.sum(1), 0)                       # (T, 25)
        worst = K._nanmax(err, axis=1)                             # (T, 25)
        r3 = worst[(views == 3)]
        resid3.append(r3[np.isfinite(r3)].astype(np.float32))
        # per joint: count of its bones that are red / present (for the length-only attribution)
        nb_bad = np.zeros_like(views); nb_good = np.zeros_like(views)
        for j, (a, b, k, sd) in enumerate(K.BONES):
            d = dev[:, j]; pres = np.isfinite(d)
            bad = pres & (np.abs(d) > RED)
            nb_bad[:, a] += bad; nb_bad[:, b] += bad
            nb_good[:, a] += pres & ~bad; nb_good[:, b] += pres & ~bad
        tier = tiers.get(stub, '?')
        for j, (a, b, k, sd) in enumerate(K.BONES):
            fr = np.where(np.isfinite(dev[:, j]) & (np.abs(dev[:, j]) > RED))[0]
            for t in fr:
                row = dict(tier=tier, stub=stub, frame=int(t), bone=k, side=sd, a=a, b=b, dev=float(dev[t, j]),
                           va=int(views[t, a]), vb=int(views[t, b]),
                           ra=float(worst[t, a]), rb=float(worst[t, b]),
                           # length-only rule: an end with another GOOD bone is cleared, an end without is blamed
                           len_a=bool(nb_good[t, a] == 0), len_b=bool(nb_good[t, b] == 0))
                out.append(row)
    return out, resid3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='C')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=None, help='optional CSV of every red-bone instance')
    a = ap.parse_args()
    S = a.set
    rv = pd.read_csv(f'{K.ROOT}/rep_review.csv', usecols=['set', 'stub', 'stage_failed'])
    tiers = dict(zip(rv[rv.set == S].stub, rv[rv.set == S].stage_failed))
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{K.ROOT}/{S}/GT3D/*/session_summary.json'))
    with Pool(a.workers) as pool:
        res = pool.map(subject_rows, [(S, s, tiers) for s in subs])
    R = pd.DataFrame([r for rr, _ in res for r in rr])
    r3 = np.concatenate([x for _, rr in res for x in rr])
    if a.out:
        R.to_csv(a.out, index=False)
    print(f'set {S}: {len(R)} red bone-frames in {R.stub.nunique()} reps')
    print('3-view joints, worst residual px: ' + ', '.join(f'p{q} {np.percentile(r3, q):.1f}' for q in (25, 50, 75, 90, 95, 99)))
    R['tier'] = R.tier.replace({'usable': 'usable', 'rep': 'rejected rep', 'setup': 'rejected setup'})
    for tau in TAUS:
        sa = (R.va == 3) & (R.ra <= tau); sb = (R.vb == 3) & (R.rb <= tau)
        R[f'cls{tau:g}'] = np.select([sa ^ sb, sa & sb], ['one strong -> resolvable', 'both strong'], 'neither strong')
    print('\nRed bones by camera-agreement class (share of red bone-frames):')
    tab = pd.concat({f'tau {t:g}px': R.groupby('tier')[f'cls{t:g}'].value_counts(normalize=True).unstack().round(3) * 100
                     for t in TAUS}, axis=1)
    print(tab.fillna(0).to_string())
    tau = 5.0
    sa = (R.va == 3) & (R.ra <= tau); sb = (R.vb == 3) & (R.rb <= tau)
    res = R[sa ^ sb].copy()
    res['blame_view'] = np.where(sa[sa ^ sb], 'b', 'a')
    lone = res.len_a ^ res.len_b
    res_len = np.where(res.len_a, 'a', 'b')
    agree = (res_len[lone.values] == res.blame_view.values[lone.values]).mean() if lone.any() else np.nan
    print(f'\nAt tau {tau:g}px: of the resolvable cases, the length-only rule also decides in {lone.mean():.0%}; '
          f'where both decide, they blame the same end {agree:.0%} of the time.')
    print('\nBy bone (tau 5px, all tiers): share resolvable / both strong / neither')
    by = R.groupby('bone')['cls5'].value_counts(normalize=True).unstack().fillna(0).round(3) * 100
    by['n'] = R.groupby('bone').size()
    print(by.to_string())
    # which end gets the blame, for the arm/leg chains
    names = {1: 'neck', 2: 'RSh', 3: 'RElb', 4: 'RWr', 5: 'LSh', 6: 'LElb', 7: 'LWr', 8: 'midhip', 9: 'RHip',
             10: 'RKnee', 11: 'RAnk', 12: 'LHip', 13: 'LKnee', 14: 'LAnk'}
    res['blamed'] = np.where(res.blame_view == 'a', res.a, res.b)
    print('\nResolvable cases (tau 5px): joint blamed, count:',
          res.blamed.map(names).value_counts().to_dict())
    print('Views of the weak end in resolvable cases:', res.apply(lambda r: r.va if r.blame_view == 'a' else r.vb, axis=1)
          .value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
