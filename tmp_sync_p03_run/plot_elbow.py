#!/usr/bin/env python3
"""plot_elbow.py -- a left-minus-right joint signal per rep, raw vs smoothed, and whether smoothing helps.

--joint elbow (default) | knee | ankle | wrist | shoulder | hip | heel | big toe (BODY_25 pairs as in lr_signals),
or --joint vertical: the Korea sync signal (align_cameras.signal_vertical: confidence-weighted mean signed
vertical velocity of all joints, 3-frame smoothed), rescaled per camera to mean 0, SD 1 for plotting.

TEMPORARY.  Signal: (y_right - y_left) / torso length, so positive = LEFT joint higher in
the image (lr_signals.raw_lr).  The --parity elbow filter in multirep_sync keeps a bounce peak only
where this signal's cross-correlation between the two cameras is > 0, so what matters is how far
above 0 it sits at the true offset and how far below 0 one step away.

  out/{joint}_signal_cam07_cam05.png   one panel per rep: both cameras, raw (thin) and smoothed (thick)
  printed table                      r at the truth and at +-1 step, raw vs smoothed, every camera pair

Smoothing = a centred moving average over the frames that have a value (needs half the window seen).

    python plot_elbow.py
    python plot_elbow.py --pair 7,1 --smooth 9
    python plot_elbow.py --joint knee
"""

import argparse
import os

import numpy as np

import multirep_sync as m
import align_cameras as ac
from lr_signals import raw_lr, LR_PAIRS

SERIES = ['#2a78d6', '#eb6834']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'
REPS = [f'P03_RUN_0{i}' for i in range(1, 6)]


def smooth(sig, win):
    """Centred moving average over valid (finite) frames; NaN where under half the window is valid."""
    if win <= 1:
        return sig.copy()
    ok = np.isfinite(sig)
    k = np.ones(win)
    s = np.convolve(np.where(ok, sig, 0.0), k, 'same')
    n = np.convolve(ok.astype(float), k, 'same')
    out = np.where(n >= win / 2, s / np.maximum(n, 1), np.nan)
    out[~ok] = np.nan                      # do not invent values inside gaps
    return out


def get_signal(xy, sc, joint):
    """The per-frame signal for `joint` (NaN where unusable)."""
    if joint == 'vertical':
        sig, ok = ac.SIGNALS['vertical'](xy, sc)
        sig, ok = ac.normalise(sig, ok)
        # a velocity: value i is the move from frame i to i+1 -- pad so it has one value per frame
        return np.r_[np.nan, np.where(ok, sig, np.nan)]
    return raw_lr(xy, sc, *LR_PAIRS[joint])


def xcorr_at(sa, sb, lags, max_lag=90):
    """Pearson r between camera A and B signals at given lags (A[i+lag] ~ B[i])."""
    va, vb = np.isfinite(sa), np.isfinite(sb)
    a, _ = ac.normalise(np.nan_to_num(sa), va)
    b, _ = ac.normalise(np.nan_to_num(sb), vb)
    c = dict(ac.masked_xcorr(a, b, va, vb, max_lag, min_overlap=30))
    return [c.get(l, np.nan) for l in lags]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pair', default='7,5', help='END-ON,SIDE pair to plot')
    ap.add_argument('--joint', default='elbow', choices=list(LR_PAIRS) + ['vertical'])
    ap.add_argument('--smooth', type=int, default=7, help='moving-average window (frames) for the thick line')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    a, b = map(int, args.pair.split(','))
    jname = args.joint.replace(' ', '_')
    out = args.out or os.path.join(m.s2.HERE, 'out', f'{jname}_signal_cam{a:02d}_cam{b:02d}.png')

    # ---- figure: the signal itself ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(REPS), 1, figsize=(11, 2.1 * len(REPS) + 0.8), sharey=True)
    for ax, rep in zip(axes, REPS):
        (xa, sa), (xb, sb) = m.prepare_rep(os.path.join(m.ROOT, rep), a, b)
        t = np.arange(len(xa))
        for col, (cam, xy, sc) in zip(SERIES, [(a, xa, sa), (b, xb, sb)]):
            raw = get_signal(xy, sc, args.joint)
            ax.plot(t, raw, '-', color=col, lw=0.8, alpha=0.55)
            ax.plot(t, smooth(raw, args.smooth), '-', color=col, lw=2.2,
                    label=f'camera {cam} ({"end-on" if cam == a else "side"})')
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(rep, fontsize=9, color=INK, loc='left')
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlim(0, 300)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    axes[-1].set_xlabel('frame (60 fps; true sync -- both cameras on the same clock)', color=MUTED, fontsize=9)
    ylab = ('moving down  <-  0  ->  moving up\n(vertical velocity, each camera rescaled to SD 1)'
            if args.joint == 'vertical' else
            f'right {args.joint} higher  <-  0  ->  left {args.joint} higher\n(height difference / torso length)')
    if args.joint == 'vertical':
        for ax in axes:
            ax.invert_yaxis()               # image y runs down: negative velocity = moving up the image
    fig.text(0.005, 0.5, ylab,
             rotation=90, va='center', color=MUTED, fontsize=8)
    what = ('Korea vertical-velocity signal (all joints)' if args.joint == 'vertical'
            else f'{args.joint.capitalize()} left-minus-right signal')
    gaps = 'body not seen' if args.joint == 'vertical' else f'both {args.joint}s not seen'
    fig.suptitle(f'{what}, cameras {a} and {b}: thin = raw, thick = {args.smooth}-frame '
                 f'moving average; gaps = {gaps}', color=INK, fontsize=10, x=0.02, ha='left')
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(out, dpi=120)
    print(f'-> {out}')

    # ---- table: does smoothing widen the margin the parity filter relies on? ----
    print(f'\n{args.joint}: r at the true offset vs one step either side (step = bounce period), raw vs smoothed')
    print(f'{"pair":6s} {"rep":11s} ' + '  '.join(f'{"win " + str(w):>22s}' for w in (1, 5, 9, 15)))
    worst = {w: [np.inf, -np.inf] for w in (1, 5, 9, 15)}
    for pa, pb in [(7, 5), (7, 1), (3, 5), (3, 1)]:
        for rep in REPS:
            (xa, sa), (xb, sb) = m.prepare_rep(os.path.join(m.ROOT, rep), pa, pb)
            ga, va = ac.normalise(*ac.SIGNALS['vertical'](xa, sa))
            gb, vb = ac.normalise(*ac.SIGNALS['vertical'](xb, sb))
            pk = sorted(m.peaks(ac.masked_xcorr(ga, gb, va, vb, 90)))
            i0 = min(range(len(pk)), key=lambda i: abs(pk[i]))
            truth = pk[i0]
            steps = [pk[i] for i in (i0 - 1, i0 + 1) if 0 <= i < len(pk)]
            cells = []
            for w in (1, 5, 9, 15):
                ea, eb = smooth(get_signal(xa, sa, args.joint), w), smooth(get_signal(xb, sb, args.joint), w)
                r0, *rs = xcorr_at(ea, eb, [truth] + steps)
                cells.append(f'{r0:+.2f} vs {", ".join(f"{r:+.2f}" for r in rs):>12s}')
                worst[w][0] = min(worst[w][0], r0)
                worst[w][1] = max(worst[w][1], np.nanmax(rs) if rs else -np.inf)
            print(f'{pa}-{pb:<4d} {rep:11s} ' + '  '.join(f'{c:>22s}' for c in cells))
    print('\nlowest r at the truth / highest r one step away, over all 20 rep x pair cases:')
    for w, (lo, hi) in worst.items():
        print(f'  window {w:2d}: truth >= {lo:+.2f}   one step away <= {hi:+.2f}')


if __name__ == '__main__':
    main()
