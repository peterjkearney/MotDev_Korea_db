#!/usr/bin/env python3
"""worked_example.py -- the multi-rep sync method, one step at a time, with charts and tables.

TEMPORARY.  Lane (end-on) camera A and side camera B, a few reps of one subject.  Each rep's SIDE video
is cut to a random segment that still contains the runner; cutting k frames off its start makes the
true offset +k (convention: lane frame i + offset shows the same instant as side frame i).

  Step 1  bounce signal per camera, and its cross-correlation       -> out/example_1_bounce.png
  Step 2  candidate offsets = bounce-correlation peaks                 (table)
  Step 3  elbow and ankle left-minus-right signals                  -> out/example_3_left_right.png
  Step 4  keep candidates where mean(elbow r, ankle r) > --keep-r      (table)
  Step 5  every combination of kept candidates: one curve through all
          reps' footprints, mean error                                 (table)
                                                                    -> out/example_5_footprints.png

    python worked_example.py
    python worked_example.py --reps P03_RUN_02 P03_RUN_04 --seed 3
"""

import argparse
import itertools
import os

import numpy as np

import multirep_sync as m            # first: sync_two_cams puts korea_gt/ on the path
import align_cameras as ac
from lr_signals import lr_signal, raw_lr, LR_PAIRS

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID, CURVE, DROP = '#1f1f1e', '#6b6a64', '#e4e3de', '#3a3a38', '#b8b6ae'
MISS_PX = 60.0       # a footprint whose lane-camera foot is not planted counts as this error


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def bounce(xy, sc):
    """Step 1 signal, exactly as align_cameras.signal_vertical + normalise (see the printed formula)."""
    sig, ok = ac.SIGNALS['vertical'](xy, sc)
    return ac.normalise(sig, ok)


def random_segment(xb, sb, rng, max_start):
    """A random cut of the side video that keeps every frame the runner is seen in.  The start (= the
    true offset) is kept below max_start so the truth lies inside the search range -- in practice the
    search range is set from how much the clips could have been trimmed."""
    seen = np.where((sb > m.CONF).any(1))[0]
    first, last = seen.min(), seen.max()
    start = int(rng.integers(0, max(1, min(first - 5, max_start))))
    end = int(rng.integers(min(last + 5, len(xb) - 1), len(xb))) + 1
    return start, end


def fit_mean_error(v, x):
    """Least-squares curve x = (a v + b)/(c v + 1) through all points; mean |error| (px)."""
    p = m.fit_map(v, x)
    return float(np.mean(np.abs(m.apply_map(p, v) - x))), p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--reps', nargs='+', default=['P03_RUN_01', 'P03_RUN_03', 'P03_RUN_05'])
    ap.add_argument('--pair', default='7,5', help='LANE,SIDE cameras')
    ap.add_argument('--max-lag', type=int, default=90)
    ap.add_argument('--keep-r', type=float, default=0.3, help='keep a candidate if mean(elbow r, ankle r) > this')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.join(m.s2.HERE, 'out'))
    args = ap.parse_args()
    a, b = map(int, args.pair.split(','))
    rng = np.random.default_rng(args.seed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---------------- data: runner tracks, random side segments ----------------
    reps = {}
    for rep in args.reps:
        (xa, sa), (xb, sb) = m.prepare_rep(os.path.join(m.ROOT, rep), a, b)
        start, end = random_segment(xb, sb, rng, args.max_lag - 10)
        reps[rep] = dict(xa=xa, sa=sa, xb=xb[start:end], sb=sb[start:end], true=start, seg=(start, end - 1),
                         n_side_orig=len(xb))
    print(f'Cameras: {a} = lane (end-on), {b} = side.  Side videos cut to random segments:')
    for rep, r in reps.items():
        print(f'  {rep}: side frames {r["seg"][0]}-{r["seg"][1]} of 0-{r["n_side_orig"] - 1} kept '
              f'-> TRUE offset {r["true"]:+d} (hidden from the method)')

    print('''
STEP 1  bounce signal (align_cameras.signal_vertical), per camera:
  y[t,j], c[t,j]   image row and OpenPose confidence of joint j (all 25 BODY_25 joints) in frame t
  valid[t,j]       c[t,j] > 0.3 and the joint is not at (0,0), in BOTH frame t and frame t-1
  w[t,j]           = c[t,j] if valid[t,j], else 0
  v[t]             = sum_j w[t,j] * (y[t,j] - y[t-1,j])  /  sum_j w[t,j]      (px per frame; + = moving DOWN the image)
  s[t]             = (v[t-1] + v[t] + v[t+1]) / 3                              (3-frame centred mean)
  bounce[t]        = (s[t] - mean(s)) / SD(s)                                  (over the frames with a value)
cross-correlation at offset L:
  r(L) = Pearson correlation of lane_bounce[i + L] with side_bounce[i], over every i where both have a
         value (at least 60 frames), for L = -90 ... +90''')

    # ---------------- step 1/2: bounce + candidates ----------------
    fig, axes = plt.subplots(len(reps), 2, figsize=(13, 2.8 * len(reps) + 0.6),
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    axes = np.atleast_2d(axes)
    print('\nSTEP 2  candidate offsets = peaks of r(L) with r >= 0.3')
    for k, (rep, r) in enumerate(reps.items()):
        ga, va = bounce(r['xa'], r['sa'])
        gb, vb = bounce(r['xb'], r['sb'])
        curve = ac.masked_xcorr(ga, gb, va, vb, args.max_lag)
        r['bcurve'] = dict(curve)
        r['cands'] = sorted(m.peaks(curve))
        ax = axes[k, 0]
        ax.plot(np.where(va, ga, np.nan), color=SERIES[0], lw=1.4, label=f'camera {a} (lane), own frame numbers')
        ax.plot(np.where(vb, gb, np.nan), color=SERIES[1], lw=1.4, label=f'camera {b} (side, cut), own frame numbers')
        ax.set_title(f'{rep}: bounce signal, NOT aligned (true offset {r["true"]:+d})', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('frame number in each video', color=MUTED, fontsize=8)
        ax.invert_yaxis()
        style(ax)
        ax.legend(frameon=False, fontsize=7, loc='lower left')
        ax = axes[k, 1]
        l, rv = zip(*curve)
        ax.plot(l, rv, color=CURVE, lw=1.4)
        for c in r['cands']:
            ax.plot(c, r['bcurve'][c], 'o', color=SERIES[2], ms=7, mec='white', zorder=3)
            ax.annotate(f'{c:+d}', (c, r['bcurve'][c]), textcoords='offset points', xytext=(0, 6),
                        ha='center', fontsize=7, color=INK)
        ax.axhline(0.3, color=MUTED, ls=':', lw=0.8)
        ax.set_ylim(-1, 1.15)
        ax.set_title('bounce correlation r(L); dots = candidates', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('offset L (frames)', color=MUTED, fontsize=8)
        style(ax)
        print(f'  {rep}:  ' + '   '.join(f'{c:+d} (r {r["bcurve"][c]:.2f})' for c in r['cands']))
    fig.suptitle('Step 1-2: bounce signal (up on the chart = moving up the image) and its correlation between the cameras',
                 color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    p1 = os.path.join(args.out, 'example_1_bounce.png')
    fig.savefig(p1, dpi=115)
    plt.close(fig)

    # ---------------- step 3/4: elbow + ankle ----------------
    print(f'\nSTEP 3-4  elbow and ankle left-minus-right: r at each candidate; keep if mean(elbow r, ankle r) > {args.keep_r:+.1f}')
    print(f'  {"rep":11s} {"offset":>7s} {"bounce r":>9s} {"elbow r":>8s} {"ankle r":>8s} {"mean":>6s}  verdict')
    fig, axes = plt.subplots(len(reps), 3, figsize=(15, 2.8 * len(reps) + 0.6),
                             gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    axes = np.atleast_2d(axes)
    for k, (rep, r) in enumerate(reps.items()):
        rr = {}
        for col, joint in enumerate(('elbow', 'ankle')):
            jl, jr = LR_PAIRS[joint]
            A, VA = lr_signal(r['xa'], r['sa'], jl, jr)
            B, VB = lr_signal(r['xb'], r['sb'], jl, jr)
            rr[joint] = dict(ac.masked_xcorr(A, B, VA, VB, args.max_lag, min_overlap=30))
            ax = axes[k, col]
            ax.plot(raw_lr(r['xa'], r['sa'], jl, jr), color=SERIES[0], lw=1.2, label=f'camera {a} (lane)')
            ax.plot(raw_lr(r['xb'], r['sb'], jl, jr), color=SERIES[1], lw=1.2, label=f'camera {b} (side, cut)')
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f'{rep}: {joint} (left height - right height) / torso, NOT aligned', fontsize=9,
                         color=INK, loc='left')
            ax.set_xlabel('frame number in each video', color=MUTED, fontsize=8)
            style(ax)
            if k == 0:
                ax.legend(frameon=False, fontsize=7, loc='lower left')
        ax = axes[k, 2]
        for joint, colr in (('elbow', SERIES[3]), ('ankle', SERIES[4])):
            l, rv = zip(*sorted(rr[joint].items()))
            ax.plot(l, rv, color=colr, lw=1.4, label=joint)
        kept = []
        for c in r['cands']:
            re, ra = rr['elbow'].get(c, np.nan), rr['ankle'].get(c, np.nan)
            mean = np.nanmean([re, ra])
            keep = mean > args.keep_r
            if keep:
                kept.append(c)
            ax.axvline(c, color=SERIES[2] if keep else DROP, lw=2 if keep else 1, ls='-' if keep else '--', zorder=0)
            print(f'  {rep:11s} {c:+7d} {r["bcurve"][c]:9.2f} {re:+8.2f} {ra:+8.2f} {mean:+6.2f}  '
                  f'{"KEEP" if keep else "drop"}')
        r['kept'] = kept
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title('r(L): solid = candidate kept, dashed = dropped', fontsize=9, color=INK, loc='left')
        ax.set_xlabel('offset L (frames)', color=MUTED, fontsize=8)
        style(ax)
        ax.legend(frameon=False, fontsize=7, loc='lower left')
    fig.suptitle('Step 3-4: which arm / which foot is higher flips every step, so at a one-step offset the cameras disagree',
                 color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    p3 = os.path.join(args.out, 'example_3_left_right.png')
    fig.savefig(p3, dpi=115)
    plt.close(fig)

    # ---------------- step 5: footprints, every combination ----------------
    print('\nSTEP 5  footprints.  Side camera: each planted foot -> (frames, side image x).  For a candidate offset L, the')
    print('lane camera at those frames + L -> image row of its planted foot (or "not planted" / "out of view").')
    for rep, r in reps.items():
        r['contacts'] = m.side_contacts(r['xb'], r['sb'])
        r['pairs'] = {}
        for L in r['kept']:
            got = [m.endon_row(r['xa'], r['sa'], c['t0'] + L, c['t1'] + L) for c in r['contacts']]
            r['pairs'][L] = got
        print(f'  {rep}: {len(r["contacts"])} side footprints: ' +
              ', '.join(f'{c["foot"]} frames {c["t0"]}-{c["t1"]} x={c["x"]:.0f}' for c in r['contacts']))
        for L in r['kept']:
            def word(c, g):
                if np.isfinite(g[0]):
                    return f'{g[0]:.0f}'
                if g[1] == 'moving':
                    return 'not planted'
                if c['t0'] + L < 0 or c['t1'] + L >= len(r['xa']):
                    return 'outside lane video'
                return 'feet not visible'
            print(f'      offset {L:+d}: lane rows ' + ', '.join(word(c, g) for c, g in zip(r['contacts'], r['pairs'][L])))

    names = list(reps)
    combos = list(itertools.product(*[reps[n]['kept'] for n in names]))
    print(f'\n  candidates kept per rep: ' + ' x '.join(str(len(reps[n]['kept'])) for n in names) +
          f' = {len(combos)} combinations')
    rows = []
    for combo in combos:
        v, x, miss = [], [], 0
        for n, L in zip(names, combo):
            for c, (row, st) in zip(reps[n]['contacts'], reps[n]['pairs'][L]):
                if np.isfinite(row):
                    v.append(row); x.append(c['x'])
                elif st == 'moving':
                    miss += 1
        v, x = np.array(v), np.array(x)
        err, p = fit_mean_error(v, x) if len(v) >= 4 else (np.inf, None)
        score = (err * len(v) + MISS_PX * miss) / max(len(v) + miss, 1)
        truth = all(abs(L - reps[n]['true']) <= 1 for n, L in zip(names, combo))
        rows.append(dict(combo=combo, n=len(v), miss=miss, err=err, score=score, truth=truth, p=p, v=v, x=x))
    rows.sort(key=lambda d: d['score'])
    print(f'\n  {"rank":>4s}  ' + '  '.join(f'{n.replace("P03_", ""):>7s}' for n in names) +
          f'  {"footprints":>10s}  {"not planted":>11s}  {"mean error":>10s}  {"score*":>7s}')
    for i, d in enumerate(rows, 1):
        print(f'  {i:4d}  ' + '  '.join(f'{L:+7d}' for L in d['combo']) +
              f'  {d["n"]:10d}  {d["miss"]:11d}  {d["err"]:8.1f}px  {d["score"]:5.1f}px' + ('   <- TRUE offsets' if d['truth'] else ''))
    print(f'  * score = mean error with each "not planted" footprint counted as {MISS_PX:.0f} px')
    print('\n  true offsets: ' + ', '.join(f'{n.replace("P03_", "")} {reps[n]["true"]:+d}' for n in names))

    # chart: best combination vs the next two
    show = rows[:3]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 4.6), sharex=True, sharey=True)
    for ax, d in zip(np.atleast_1d(axes), show):
        i0 = 0
        for k, (n, L) in enumerate(zip(names, d['combo'])):
            vv = np.array([g[0] for g in reps[n]['pairs'][L]])
            xx = np.array([c['x'] for c in reps[n]['contacts']])
            ok = np.isfinite(vv)
            if d['p'] is not None:
                for a_, b_ in zip(vv[ok], xx[ok]):
                    ax.plot([a_, a_], [b_, m.apply_map(d['p'], a_)], color=SERIES[k], lw=1, alpha=0.8)
            ax.plot(vv[ok], xx[ok], 'o', ms=8, color=SERIES[k], mec='white', mew=1.5, zorder=3,
                    label=f'{n.replace("P03_", "")} offset {L:+d}')
        if d['p'] is not None and len(d['v']):
            vs = np.linspace(d['v'].min() - 15, d['v'].max() + 15, 300)
            xs = m.apply_map(d['p'], vs)
            den = d['p'][2] * vs + 1
            xs[np.r_[False, np.sign(den[1:]) != np.sign(den[:-1])]] = np.nan
            xs[(xs < -400) | (xs > 2400)] = np.nan
            ax.plot(vs, xs, color=CURVE, lw=2, zorder=2)
        rank = rows.index(d) + 1
        ax.set_title(f'rank {rank}{" (TRUE offsets)" if d["truth"] else ""}\nmean error {d["err"]:.1f} px, '
                     f'{d["miss"]} not planted', fontsize=9, color=INK, loc='left')
        ax.set_xlabel(f'lane camera {a}: planted-foot image row (px)', color=MUTED, fontsize=8)
        style(ax)
        ax.legend(frameon=False, fontsize=7, loc='upper right')
        ax.set_ylim(-150, 2250)
    np.atleast_1d(axes)[0].set_ylabel(f'side camera {b}: footprint image x (px)', color=MUTED, fontsize=8)
    fig.suptitle('Step 5: one curve through every rep\'s footprints -- the three best-fitting combinations',
                 color=INK, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    p5 = os.path.join(args.out, 'example_5_footprints.png')
    fig.savefig(p5, dpi=115)
    plt.close(fig)
    print(f'\ncharts: {p1}\n        {p3}\n        {p5}')


if __name__ == '__main__':
    main()
