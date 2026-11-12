#!/usr/bin/env python3
"""plot_mapping.py -- the footprint pairs and the shared curve, at correct and wrong offsets.

TEMPORARY.  For one end-on / side camera pair: every rep's (end-on row, side x) footprint pairs and
the one curve x = (a*row + b) / (c*row + 1) that fits them best, in four panels:
  every rep at its true offset | every rep one step off | every rep one stride off |
  one rep one step off, the rest right.
Thin lines join each point to the curve (its error).  Panel titles give the robust RMS error that
multirep_sync.py scores combinations by.

    python plot_mapping.py                  # cameras 7 (end-on) - 5 (side)
    python plot_mapping.py --pair 7,1 --wrong-rep P03_RUN_02
"""

import argparse
import os

import numpy as np

import multirep_sync as m

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID, CURVE = '#1f1f1e', '#6b6a64', '#e4e3de', '#3a3a38'


def robust_params(v, x, iters=3):
    """The curve multirep_sync.map_cost scores: refitted on points within CAP_PX."""
    keep = np.ones(len(v), bool)
    p = m.fit_map(v, x)
    for _ in range(iters):
        r = np.abs(m.apply_map(p, v) - x)
        new = r < m.CAP_PX
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
        p = m.fit_map(v[keep], x[keep])
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pair', default='7,5', help='END-ON,SIDE')
    ap.add_argument('--reps', nargs='+', default=[f'P03_RUN_0{i}' for i in range(1, 6)])
    ap.add_argument('--wrong-rep', default='P03_RUN_03', help='the rep put one step off in the last panel')
    ap.add_argument('--foot', choices=['still', 'label'], default='still')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    a, b = map(int, args.pair.split(','))
    out = args.out or os.path.join(m.s2.HERE, 'out', f'mapping_cam{a:02d}_cam{b:02d}.png')

    reps = {}
    for r in args.reps:
        tr = m.prepare_rep(os.path.join(m.ROOT, r), a, b)
        reps[r] = m.rep_candidates(*tr, 0, 90, 'none', args.foot)      # true lag 0 (hardware sync)

    def at_step(rep, k):
        """Candidate k steps from the true one (None if the rep has no such candidate)."""
        c = sorted(reps[rep]['cands'])
        i0 = min(range(len(c)), key=lambda i: abs(c[i]))
        return c[i0 + k] if 0 <= i0 + k < len(c) else None

    panels = [
        ('every rep at its true offset', {r: 0 for r in args.reps}),
        ('every rep one step off (+1 step)', {r: 1 for r in args.reps}),
        ('every rep one stride off (-2 steps)', {r: -2 for r in args.reps}),
        (f'only {args.wrong_rep.replace("P03_", "")} one step off', {r: (1 if r == args.wrong_rep else 0) for r in args.reps}),
    ]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, (title, steps) in zip(axes.ravel(), panels):
        V, X, mv, used = [], [], 0, {}
        for r, k in steps.items():
            L = at_step(r, k)
            if L is None or L not in reps[r]['pairs']:
                continue
            v, x, n_mv = reps[r]['pairs'][L]
            V.append(v); X.append(x); mv += n_mv; used[r] = (L, v, x)
        V, X = np.concatenate(V), np.concatenate(X)
        cost = m.map_cost(V, X, mv)
        wrong_one = [r for r, k in steps.items() if k] if len(set(steps.values())) > 1 else []
        if wrong_one:
            # mixed panel: the curve the correctly-offset reps agree on, so the wrong rep shows as off it
            ref = [r for r in used if r not in wrong_one]
            p = robust_params(np.concatenate([used[r][1] for r in ref]), np.concatenate([used[r][2] for r in ref]))
            curve_label = f'curve the other {len(ref)} reps agree on'
        else:
            p = robust_params(V, X)
            curve_label = 'best single curve'
        # draw over the data's row range only, broken where the curve's denominator crosses zero
        vv = np.linspace(V.min() - 15, V.max() + 15, 400)
        xx = m.apply_map(p, vv)
        den = p[2] * vv + 1
        xx[np.r_[False, np.sign(den[1:]) != np.sign(den[:-1])]] = np.nan
        xx[(xx < -400) | (xx > 2400)] = np.nan
        ax.plot(vv, xx, color=CURVE, lw=2, zorder=2, label=curve_label)
        for i, r in enumerate(args.reps):
            if r not in used:
                continue
            L, v, x = used[r]
            for vi, xi in zip(v, x):
                ax.plot([vi, vi], [xi, m.apply_map(p, vi)], color=SERIES[i], lw=1, alpha=0.8, zorder=1)
            ax.plot(v, x, 'o', ms=8, color=SERIES[i], mec='white', mew=1.5, zorder=3,
                    label=f'{r.replace("P03_", "")} (offset {L:+d})')
        extra = f', {mv} feet moving' if mv else ''
        capped = ' (the cap: no curve fits)' if cost >= m.CAP_PX - 0.05 else ''
        ax.set_title(f'{title}\nscored error {cost:.1f} px{capped}  ({len(V)} footprints{extra})',
                     fontsize=10, color=INK, loc='left')
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.legend(frameon=False, fontsize=7.5, loc='upper right')
        ax.set_ylim(-150, 2250)
    for ax in axes[1]:
        ax.set_xlabel(f'end-on camera {a}: planted foot image row (px; larger = nearer camera {a})', color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(f'side camera {b}: footprint image x (px)', color=MUTED, fontsize=9)
    fig.suptitle(f'Footprints seen by both cameras: one curve must fit every rep (true offset = 0 for every rep)',
                 color=INK, fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
