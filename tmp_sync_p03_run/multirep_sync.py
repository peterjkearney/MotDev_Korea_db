#!/usr/bin/env python3
"""multirep_sync.py -- sync two cameras over several reps of one subject, cameras fixed.

TEMPORARY experiment.  One end-on camera (runner along its line of sight) and one side-on camera.
Each rep's side footage is trimmed differently, so each rep has its own unknown lag, but the two
cameras -- and so the floor mapping between them -- are the same for every rep.

Per rep:
  1. runner = the track that travels furthest (bystanders stand still; the Korea tracker's
     "longest 3 tracks" can drop the runner, who is only in a side view for ~2 s),
  2. candidate lags = peaks of the Korea vertical-velocity cross-correlation (one per step),
     optionally only those where the elbow left-minus-right signal also correlates positively
     (--parity elbow: one per stride),
  3. foot contacts from the SIDE camera: a heel/big-toe midpoint that stops moving in the image
     (stance).  End-on contacts cannot be found the same way -- a swinging foot moves along the
     line of sight and looks still -- so for each side contact and candidate lag, the end-on
     camera's PLANTED foot is read over the same (lag-shifted) frames: the foot whose image row
     stays put (--foot still), or the side contact's labelled foot, which must stay put
     (--foot label).  A foot seen but moving is penalised; one out of view is skipped.
Per subject (all reps together):
  4. every combination of one candidate per rep is scored by how well ONE 1-D projective map
        x_side = (a * v_end + b) / (c * v_end + 1)
     (v_end = end-on image row of the contact, x_side = side image column) fits all reps'
     contacts (RMS px in the side image, after dropping the worst 10% of contacts).
  5. the best combination is the answer; the candidates are vertical-xcorr peaks, so they are
     already frame-exact (no separate fine stage).

The test: BioCV is hardware-synced (true lag 0), so each rep gets a random artificial trim
(--shift-range); the true lag is that trim.  Reported per trial: reps exactly right (+-1 frame),
whether the reps are right RELATIVE to each other (all off by the same amount), and the shared
offset.  A shared offset of a whole number of steps or strides is the ambiguity the mapping cannot
see (moving every contact by the same distance is itself a 1-D projective map).

    python multirep_sync.py                         # cams 7-5, 20 random trims
    python multirep_sync.py --pairs 7,5 7,1 3,5 3,1 --trials 20 --parity elbow
"""

import argparse
import itertools
import os

import numpy as np

import sync_two_cams as s2
import align_cameras as ac

ROOT = os.path.dirname(s2.DEFAULT_TRIAL)
FEET = {'L': (21, 19), 'R': (24, 22)}          # BODY_25 heel, big toe
CONF = s2.CONF


# --------------------------------------------------------------------------
# Runner, signals, candidates
# --------------------------------------------------------------------------

def travel(xy, sc):
    ok = (sc[:, 1] > CONF) & (sc[:, 8] > CONF)
    if ok.sum() < 20:
        return 0.0
    tor = np.linalg.norm(xy[ok, 1] - xy[ok, 8], axis=1)
    c = (xy[ok, 1] + xy[ok, 8]) / 2
    rng = lambda v: np.percentile(v, 95) - np.percentile(v, 5)
    return (rng(c[:, 0]) + rng(c[:, 1])) / np.median(tor) + 3 * rng(np.log(tor))


def runner(frames):
    tracks = [t for t in ac.build_tracks(frames) if (t[1] > 0).any(1).sum() >= 30]
    return max(tracks, key=lambda t: travel(*t))


def peaks(curve, min_r=0.3):
    """Local maxima of a [(lag, r)] curve above min_r."""
    l = [x[0] for x in curve]
    r = np.array([x[1] for x in curve])
    out = []
    for i in range(len(r)):
        lo, hi = r[i - 1] if i else -np.inf, r[i + 1] if i + 1 < len(r) else -np.inf
        if r[i] >= lo and r[i] > hi and r[i] >= min_r:
            out.append(l[i])
    return out


def lr_elbow(xy, sc):
    from lr_signals import lr_signal
    return lr_signal(xy, sc, 6, 3)


# --------------------------------------------------------------------------
# Foot contacts
# --------------------------------------------------------------------------

def torso(xy, sc):
    ok = (sc[:, 1] > CONF) & (sc[:, 8] > CONF)
    return np.where(ok, np.linalg.norm(xy[:, 1] - xy[:, 8], axis=1), np.nan)


def foot_points(xy, sc):
    """{'L'/'R': (T,2) heel/big-toe midpoint, NaN where unseen}."""
    out = {}
    for side, (h, t) in FEET.items():
        ok = (sc[:, h] > CONF) & (sc[:, t] > CONF)
        out[side] = np.where(ok[:, None], (xy[:, h] + xy[:, t]) / 2, np.nan)
    return out


def side_contacts(xy, sc, thr=0.04, min_len=4, merge_gap=15):
    """Stance phases in a side-on camera: (foot, t0, t1, x) with x the median image column."""
    tor = torso(xy, sc)
    segs = []
    for side, p in foot_points(xy, sc).items():
        sp = np.linalg.norm(np.diff(p, axis=0), axis=1) / tor[1:]
        still = np.r_[False, sp < thr]
        t, T = 0, len(still)
        while t < T:
            if still[t]:
                e = t
                while e + 1 < T and still[e + 1]:
                    e += 1
                segs.append([side, t - 1, e])
                t = e + 1
            else:
                t += 1
    # merge pieces of one stance split by a noisy frame
    segs.sort(key=lambda s: (s[0], s[1]))
    merged = []
    for s in segs:
        if merged and merged[-1][0] == s[0] and s[1] - merged[-1][2] <= merge_gap:
            # the same foot cannot land twice within half a stride: one stance split in pieces
            merged[-1][2] = s[2]
            continue
        merged.append(list(s))
    fp = foot_points(xy, sc)
    out = []
    for side, t0, t1 in merged:
        if t1 - t0 + 1 >= min_len:
            out.append(dict(foot=side, t0=t0, t1=t1, x=float(np.nanmedian(fp[side][t0:t1 + 1, 0]))))
    return sorted(out, key=lambda d: d['t0'])


def endon_row(xy, sc, t0, t1, foot=None, min_frac=0.5, still_frac=0.25):
    """Image row of the end-on camera's PLANTED foot over frames t0..t1.

    A planted foot keeps its image row; a swinging one moves it (it is lifted and, nearer the
    camera, drawn lower -- so "the lower foot" is often the wrong one).  foot=None takes whichever
    foot is stiller; foot='L'/'R' requires that labelled foot (the side camera's contact foot).
    Returns (row, status): status 'ok', 'unseen' (outside the footage or not detected -- neutral)
    or 'moving' (the foot is seen but not still -- evidence against this lag).
    """
    if t0 < 0 or t1 >= len(xy):
        return np.nan, 'unseen'
    # judge the middle of the stance: heel strike and toe-off at the ends move the foot point
    cut = int(round(0.2 * (t1 - t0 + 1)))
    t0, t1 = t0 + cut, t1 - cut
    fp = foot_points(xy[t0:t1 + 1], sc[t0:t1 + 1])
    tor = np.nanmedian(torso(xy[t0:t1 + 1], sc[t0:t1 + 1]))
    stats = {}
    for f in (['L', 'R'] if foot is None else [foot]):
        rows = fp[f][:, 1]
        if np.isfinite(rows).mean() >= min_frac:
            stats[f] = (np.nanmax(rows) - np.nanmin(rows), float(np.nanmedian(rows)))
    if not stats or not np.isfinite(tor):
        return np.nan, 'unseen'
    rng, row = min(stats.values())
    if rng > still_frac * tor:
        return np.nan, 'moving'
    return row, 'ok'


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------

def fit_map(v, x):
    """x = (a v + b) / (c v + 1) by linear least squares on x = a v + b - c v x."""
    A = np.c_[v, np.ones_like(v), -v * x]
    p, *_ = np.linalg.lstsq(A, x, rcond=None)
    return p


def apply_map(p, v):
    a, b, c = p
    return (a * v + b) / (c * v + 1)


CAP_PX = 60.0              # a contact costs at most this (px in the side image) -- a badly fitting one,
                           # or one whose end-on foot is seen but not planted ('moving')
MIN_SEEN = 2               # contacts a rep must have read (ok or moving) at a candidate lag


def map_cost(v, x, n_moving=0, iters=3):
    """Robust RMS side-image error (px) of one shared map: residuals capped at CAP_PX, each
    'moving' contact counted as CAP_PX.  Refitted on the uncapped points a few times so a
    wrong rep cannot drag the map; NOT trimmed -- dropping points let a wrong rep with one
    terrible contact cost nothing."""
    if len(v) < 5:
        return np.inf
    keep = np.ones(len(v), bool)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        p = fit_map(v[keep], x[keep])
        r = np.abs(apply_map(p, v) - x)
        new = r < CAP_PX
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    rc = np.minimum(r, CAP_PX)
    return float(np.sqrt((np.sum(rc ** 2) + n_moving * CAP_PX ** 2) / (len(rc) + n_moving)))


# --------------------------------------------------------------------------
# One rep, one subject
# --------------------------------------------------------------------------

def prepare_rep(trial, a, b):
    """Untrimmed runner tracks for end-on camera a and side camera b."""
    fa, _, _ = s2.load_openpose(trial, a)
    fb, _, _ = s2.load_openpose(trial, b)
    return runner(fa), runner(fb)


def rep_candidates(track_a, track_b, shift, max_lag, parity, foot_mode='still'):
    """Apply the artificial trim, then candidates and per-candidate contact pairs."""
    (xa, sa), (xb, sb) = track_a, track_b
    if shift > 0:
        xb, sb = xb[shift:], sb[shift:]
    elif shift < 0:
        xa, sa = xa[-shift:], sa[-shift:]
    ga, va = ac.normalise(*ac.SIGNALS['vertical'](xa, sa))
    gb, vb = ac.normalise(*ac.SIGNALS['vertical'](xb, sb))
    curve = ac.masked_xcorr(ga, gb, va, vb, max_lag)
    cands = peaks(curve)
    vcorr = dict(curve)
    if parity == 'elbow':
        ea, eva = lr_elbow(xa, sa)
        eb, evb = lr_elbow(xb, sb)
        ecorr = dict(ac.masked_xcorr(ea, eb, eva, evb, max_lag, min_overlap=30))
        cands = [l for l in cands if ecorr.get(l, 0) > 0]
    contacts = side_contacts(xb, sb)
    pairs = {}
    for L in cands:
        got = [endon_row(xa, sa, c['t0'] + L, c['t1'] + L, c['foot'] if foot_mode == 'label' else None)
               for c in contacts]
        v = np.array([g[0] for g in got])
        x = np.array([c['x'] for c in contacts])
        ok = np.isfinite(v)
        n_moving = sum(g[1] == 'moving' for g in got)
        if ok.sum() + n_moving >= MIN_SEEN:
            pairs[L] = (v[ok], x[ok], n_moving)
    cands = [L for L in cands if L in pairs]
    best_v = max(curve, key=lambda q: q[1])[0] if curve else np.nan
    return dict(cands=cands, pairs=pairs, vertical_best=best_v, n_contacts=len(contacts), vcorr=vcorr)


def joint_solve(reps, max_combos=200000):
    """Every combination of one candidate per rep, scored by the shared-map fit."""
    lists = [r['cands'] for r in reps]
    if any(len(l) == 0 for l in lists):
        return []
    n = int(np.prod([len(l) for l in lists]))
    if n > max_combos:
        raise SystemExit(f'{n} combinations -- narrow --max-lag or use --parity elbow')
    results = []
    for combo in itertools.product(*lists):
        v = np.concatenate([r['pairs'][L][0] for r, L in zip(reps, combo)])
        x = np.concatenate([r['pairs'][L][1] for r, L in zip(reps, combo)])
        mv = sum(r['pairs'][L][2] for r, L in zip(reps, combo))
        results.append((map_cost(v, x, mv), combo))
    results.sort(key=lambda q: q[0])
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--reps', nargs='+', default=[f'P03_RUN_0{i}' for i in range(1, 6)])
    ap.add_argument('--pairs', nargs='+', default=['7,5'], help='END-ON,SIDE camera pairs')
    ap.add_argument('--trials', type=int, default=20, help='random trim sets per camera pair')
    ap.add_argument('--shift-range', type=int, nargs=2, default=[-60, 30],
                    help='true lag per rep drawn from this range (frames); + trims the side camera')
    ap.add_argument('--max-lag', type=int, default=90)
    ap.add_argument('--parity', choices=['none', 'elbow'], default='none')
    ap.add_argument('--foot', choices=['still', 'label'], default='still',
                    help="end-on contact foot: the stiller one, or the side contact's labelled foot")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(s2.HERE, 'out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows = []
    for pair in args.pairs:
        a, b = map(int, pair.split(','))
        tracks = {rep: prepare_rep(os.path.join(ROOT, rep), a, b) for rep in args.reps}
        print(f'\ncameras {a} (end-on) - {b} (side), {len(args.reps)} reps, parity={args.parity}, foot={args.foot}')
        for k in range(args.trials):
            shifts = rng.integers(args.shift_range[0], args.shift_range[1] + 1, len(args.reps))
            reps = [rep_candidates(*tracks[rep], int(s), args.max_lag, args.parity, args.foot)
                    for rep, s in zip(args.reps, shifts)]
            res = joint_solve(reps)
            if not res:
                print(f'  trial {k}: a rep has no candidates')
                continue
            cost, combo = res[0]
            err = np.array(combo) - shifts
            exact = np.abs(err) <= 1
            relative = np.ptp(err) <= 2
            # best cost of a combination with the TRUE relative structure but another shared offset,
            # and of one with the reps NOT all off by the same amount
            def same_shift(c):
                e = np.array(c) - shifts
                return np.ptp(e) <= 2
            true_cost = next((c for c, cb in res if np.all(np.abs(np.array(cb) - shifts) <= 1)), np.nan)
            other_rel = next((c for c, cb in res if not same_shift(cb)), np.nan)
            other_shared = next((c for c, cb in res if same_shift(cb) and np.abs(np.median(np.array(cb) - shifts)) > 2),
                                np.nan)
            vert = np.array([r['vertical_best'] for r in reps]) - shifts
            rows.append(dict(cams=f'{a}-{b}', trial=k, parity=args.parity, foot=args.foot,
                             n_exact=int(exact.sum()), n_reps=len(reps), relative_ok=bool(relative),
                             shared_offset=int(np.median(err)), cost=cost, cost_true=true_cost,
                             best_not_relative=other_rel, best_other_shared=other_shared,
                             vertical_only_exact=int((np.abs(vert) <= 1).sum()),
                             n_combos=len(res), contacts='/'.join(str(r['n_contacts']) for r in reps)))
            print(f'  trial {k:2d}: true {list(shifts)} -> {list(combo)}  exact {exact.sum()}/{len(reps)}'
                  f'  relative {"ok" if relative else "WRONG"} (shared {int(np.median(err)):+d})'
                  f'  cost {cost:.1f}px; truth {true_cost:.1f}; best mixed {other_rel:.1f}; best other shared'
                  f' {other_shared:.1f}  | vertical-only exact {(np.abs(vert) <= 1).sum()}/{len(reps)}')

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, f'multirep_{args.parity}_{args.foot}.csv')
    df.to_csv(path, index=False)
    print('\nsummary per camera pair:')
    g = df.groupby('cams')
    summ = pd.DataFrame({
        'trials': g.size(),
        'all_reps_exact': g.apply(lambda d: (d.n_exact == d.n_reps).mean()),
        'relative_ok': g.relative_ok.mean(),
        'shared_offsets_seen': g.shared_offset.apply(lambda s: sorted(set(s))),
        'median_cost_px': g.cost.median(),
        'median_best_mixed_px': g.best_not_relative.median(),
        'median_best_other_shared_px': g.best_other_shared.median(),
        'vertical_only_rep_exact': g.apply(lambda d: d.vertical_only_exact.sum() / d.n_reps.sum()),
    })
    print(summ.to_string())
    print(f'-> {path}')


if __name__ == '__main__':
    main()
