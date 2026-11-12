#!/usr/bin/env python3
"""lr_signals.py -- left-minus-right sync signals: which joint pair best tells the steps apart?

TEMPORARY experiment.  The Korea vertical signal averages every joint, so one step looks like the
next and the cross-correlation repeats every step.  A left-minus-right difference flips sign every
step and only repeats every stride (two steps), so it should rule out the one-step offsets.

Per joint pair and camera: (y_left - y_right) / torso length, per frame.  Vertical only: the cameras
are level, so up/down means the same in every view, while horizontal L-R separation does not (a
side-on camera sees fore-aft separation, an end-on camera sees sideways separation).  Divided by the
torso so the runner growing in the end-on camera does not grow the signal.

For every camera pair and joint pair: masked cross-correlation (align_cameras.masked_xcorr) over
+-max_lag frames.  Scores, with the true lag 0 (BioCV is hardware-synced):
  r0          correlation at lag 0 (should be clearly positive; negative = L/R labels disagree)
  best_lag    the lag of the highest correlation within +-window (window < one stride)
  margin      r0 minus the best correlation more than half a step from 0, within +-window --
              > 0 means lag 0 wins and by how much
  frames      frames where both cameras see both joints
Plus a combined signal: the mean of all joint pairs' correlation curves, and a two-stage result
(combined L/R within +-window picks the step, then the Korea vertical signal within +-half a step).

Writes out/lr_heatmaps.png (correlation vs lag, every joint pair, every camera pair),
out/lr_summary.png (margin per joint pair), out/lr_signals_raw.png / lr_signals_norm.png (the signals
themselves over time, one panel per camera; raw on one shared scale / each row rescaled) and
out/lr_scores.csv.

    python lr_signals.py                     # the four end-on/side pairs of P03_RUN_01
    python lr_signals.py --pairs 7,1 3,5 --window 45 --max-lag 90
"""

import argparse
import os

import numpy as np

import sync_two_cams as s2
import align_cameras as ac

# BODY_25 (left, right) indices
LR_PAIRS = {
    'shoulder': (5, 2), 'elbow': (6, 3), 'wrist': (7, 4), 'hip': (12, 9),
    'knee': (13, 10), 'ankle': (14, 11), 'heel': (21, 24), 'big toe': (19, 22),
}
CONF = s2.CONF

# reference palette (dataviz skill): categorical slots in fixed order, diverging blue <-> grey <-> red
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
DIV = ['#2a78d6', '#f0efec', '#e34948']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3de'


def lr_signal(xy, sc, left, right):
    """(y_left - y_right) / torso length, NaN-free with a validity mask."""
    ok = (sc[:, left] > CONF) & (sc[:, right] > CONF) & (sc[:, 1] > CONF) & (sc[:, 8] > CONF)
    torso = np.linalg.norm(xy[:, 1] - xy[:, 8], axis=1)
    ok &= torso > 1
    sig = np.zeros(len(xy))
    sig[ok] = (xy[ok, left, 1] - xy[ok, right, 1]) / torso[ok]
    return ac.normalise(sig, ok)


def curve_dict(curve):
    return dict(curve)


def score(curve, window, half_step):
    c = {l: r for l, r in curve.items() if abs(l) <= window}
    if not c:
        return dict(r0=np.nan, best_lag=np.nan, margin=np.nan)
    best = max(c, key=c.get)
    far = [r for l, r in c.items() if abs(l) > half_step]
    r0 = c.get(0, np.nan)
    return dict(r0=r0, best_lag=best, margin=r0 - max(far) if far else np.nan)


def analyse_pair(trial, a, b, max_lag, window):
    fa, _, _ = s2.load_openpose(trial, a)
    fb, _, _ = s2.load_openpose(trial, b)
    best_v, ta, tb = s2.motion_sync(fa, fb, 'vertical', max_lag)
    (xa, sa), (xb, sb) = ta[best_v['track'][0]], tb[best_v['track'][1]]
    (ga, va), (gb, vb) = best_v['signals']
    period = s2.step_period(ga, va)
    if not np.isfinite(period):
        period = s2.step_period(gb, vb)
    half = int(period // 2)
    rows, curves = [], {}
    for name, (l, r) in LR_PAIRS.items():
        A, VA = lr_signal(xa, sa, l, r)
        B, VB = lr_signal(xb, sb, l, r)
        cur = curve_dict(ac.masked_xcorr(A, B, VA, VB, max_lag))
        curves[name] = cur
        rows.append(dict(cams=f'{a}-{b}', joint=name, frames=int((VA & VB).sum()), **score(cur, window, half)))
    # combined: mean of the joint pairs' curves at each lag, wherever at least half of them reach it
    lags = sorted(set.union(*[set(c) for c in curves.values() if c]))
    comb = {}
    for l in lags:
        v = [c[l] for c in curves.values() if l in c]
        if len(v) >= len(LR_PAIRS) / 2:
            comb[l] = float(np.mean(v))
    curves['combined'] = comb
    rows.append(dict(cams=f'{a}-{b}', joint='combined', frames=np.nan, **score(comb, window, half)))
    # two-stage: combined L/R within +-window picks the step, vertical refines within +-half a step
    cw = {l: r for l, r in comb.items() if abs(l) <= window}
    coarse = max(cw, key=cw.get) if cw else np.nan
    near = [(l, r) for l, r in best_v['curve'] if abs(l - coarse) <= half]
    fine = max(near, key=lambda x: x[1])[0] if near else np.nan
    info = dict(cams=f'{a}-{b}', vertical_lag=best_v['lag'], step_period=period, lr_coarse=coarse, two_stage=fine,
                tracks={a: (xa, sa), b: (xb, sb)})
    return rows, curves, info


def plot_heatmaps(path, results, max_lag, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', DIV)
    names = list(LR_PAIRS) + ['combined']
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    lags = np.arange(-max_lag, max_lag + 1)
    for ax, (cams, curves, info) in zip(axes, results):
        M = np.full((len(names), len(lags)), np.nan)
        for i, nm in enumerate(names):
            for j, l in enumerate(lags):
                M[i, j] = curves[nm].get(int(l), np.nan)
        # no data (too little overlap to correlate) is hatched, so it is not mistaken for r = 0
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((lags[0] - .5, -.5), len(lags), len(names), facecolor='white',
                               edgecolor=GRID, hatch='////', lw=0, zorder=0))
        im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=-1, vmax=1, interpolation='nearest',
                       extent=[lags[0] - .5, lags[-1] + .5, len(names) - .5, -.5], zorder=1)
        ax.axhline(len(names) - 1.5, color='white', lw=2)          # separate the combined row
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color=INK)
        for x in (0,):
            ax.axvline(x, color=INK, ls=':', lw=1)
        for x in (-window, window):
            ax.axvline(x, color=MUTED, ls='--', lw=0.8)
        ax.set_title(f'cameras {cams}   (step {info["step_period"]:.0f} frames; dashed = +-{window} search window; '
                     f'dotted = true lag 0)', fontsize=9, color=INK, loc='left')
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xlabel('lag (frames, 60 fps)   +N: camera A frame i+N matches camera B frame i', color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('Pearson r', color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle('Left-minus-right vertical signals: correlation vs lag (red = positive, blue = negative, '
                 'hatched = under 60 overlapping frames)',
                 color=INK, fontsize=11, x=0.02, ha='left')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def raw_lr(xy, sc, left, right):
    """(y_right - y_left) / torso: positive = left joint HIGHER in the image (y runs down).  NaN where unusable."""
    ok = (sc[:, left] > CONF) & (sc[:, right] > CONF) & (sc[:, 1] > CONF) & (sc[:, 8] > CONF)
    torso = np.linalg.norm(xy[:, 1] - xy[:, 8], axis=1)
    ok &= torso > 1
    out = np.full(len(xy), np.nan)
    out[ok] = (xy[ok, right, 1] - xy[ok, left, 1]) / torso[ok]
    return out


def plot_signals(path, tracks, normalise_rows, step=None):
    """Signal value over time: one panel per camera, one row per joint pair, shared frame axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list('div', DIV)
    names = list(LR_PAIRS)
    cams = list(tracks)
    T = max(len(x) for x, _ in tracks.values())
    mats = {}
    for c in cams:
        xy, sc = tracks[c]
        M = np.vstack([raw_lr(xy, sc, *LR_PAIRS[n]) for n in names])
        if normalise_rows:
            M = (M - np.nanmean(M, 1, keepdims=True)) / np.nanstd(M, 1, keepdims=True)
        mats[c] = M
    if normalise_rows:
        vmax, label = 2.5, 'signal (each row: mean 0, SD 1)'
    else:
        vmax = np.nanpercentile(np.abs(np.hstack([m.ravel() for m in mats.values()])), 98)
        label = 'left minus right height / torso length'
    role = {7: 'runner approaching', 3: 'runner going away', 1: 'side-on', 5: 'side-on'}
    fig, axes = plt.subplots(len(cams), 1, figsize=(10, 2.3 * len(cams) + 0.6), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cams):
        ax.add_patch(Rectangle((-.5, -.5), T, len(names), facecolor='white', edgecolor=GRID,
                               hatch='////', lw=0, zorder=0))
        im = ax.imshow(mats[c], aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax, interpolation='nearest',
                       extent=[-.5, mats[c].shape[1] - .5, len(names) - .5, -.5], zorder=1)
        ax.set_xlim(-.5, T - .5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8, color=INK)
        ax.set_title(f'camera {c} ({role.get(c, "")})', fontsize=9, color=INK, loc='left')
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel('frame (60 fps, all cameras on the same clock)', color=MUTED, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(label, color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    what = 'each row rescaled' if normalise_rows else 'raw values, one shared scale'
    fig.suptitle(f'Left-minus-right vertical signals over time ({what}; red = left joint higher, '
                 f'blue = right joint higher, hatched = not usable)', color=INK, fontsize=10, x=0.02, ha='left')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary(path, rows, cams_list, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(LR_PAIRS) + ['combined']
    fig, ax = plt.subplots(figsize=(8, 5))
    dy = np.linspace(-0.27, 0.27, len(cams_list))
    for k, cams in enumerate(cams_list):
        for i, nm in enumerate(names):
            r = next(x for x in rows if x['cams'] == cams and x['joint'] == nm)
            if not np.isfinite(r['margin']):
                continue
            exact = abs(r['best_lag']) <= 1
            right_step = r['margin'] > 0
            ax.plot(r['margin'], i + dy[k], 'o' if right_step else 'X', ms=8 if right_step else 9,
                    mec=SERIES[k], mew=2, mfc=SERIES[k] if (exact or not right_step) else 'white', zorder=3,
                    label=f'cameras {cams}' if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=INK)
    ax.invert_yaxis()
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(f'margin: r at lag 0 minus best r more than half a step away (within +-{window})', color=MUTED)
    ax.set_title('How clearly each left/right pair picks the right step\n'
                 'filled dot = right step, within +-1 frame;  hollow dot = right step, 2+ frames off;  '
                 'cross = wrong step', color=INK, fontsize=9, loc='left')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], ls='', marker='s', ms=8, color=SERIES[k], label=f'cameras {c}')
                       for k, c in enumerate(cams_list)], frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--trial', default=s2.DEFAULT_TRIAL)
    ap.add_argument('--pairs', nargs='+', default=['7,1', '7,5', '3,1', '3,5'], help='camera pairs A,B')
    ap.add_argument('--max-lag', type=int, default=90, help='lags plotted')
    ap.add_argument('--window', type=int, default=45, help='lags searched for the scores (keep < one stride)')
    ap.add_argument('--out', default=os.path.join(s2.HERE, 'out'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_rows, results, infos = [], [], []
    for p in args.pairs:
        a, b = map(int, p.split(','))
        rows, curves, info = analyse_pair(args.trial, a, b, args.max_lag, args.window)
        all_rows += rows
        results.append((f'{a}-{b}', curves, info))
        infos.append(info)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out, 'lr_scores.csv'), index=False)
    wide = df.pivot(index='joint', columns='cams', values=['r0', 'best_lag', 'margin'])
    wide = wide.reindex(list(LR_PAIRS) + ['combined'])
    pd.set_option('display.width', 200)
    print(wide.round(2).to_string())
    print()
    print(pd.DataFrame([{k: v for k, v in i.items() if k != 'tracks'} for i in infos]).to_string(index=False))
    plot_heatmaps(os.path.join(args.out, 'lr_heatmaps.png'), results, args.max_lag, args.window)
    plot_summary(os.path.join(args.out, 'lr_summary.png'), all_rows, [r[0] for r in results], args.window)
    tracks = {}
    for info in infos:                          # first pair a camera appears in supplies its track
        for c, tr in info['tracks'].items():
            tracks.setdefault(c, tr)
    order = [c for c in (7, 3, 1, 5) if c in tracks] + [c for c in tracks if c not in (7, 3, 1, 5)]
    tracks = {c: tracks[c] for c in order}
    plot_signals(os.path.join(args.out, 'lr_signals_raw.png'), tracks, normalise_rows=False)
    plot_signals(os.path.join(args.out, 'lr_signals_norm.png'), tracks, normalise_rows=True)
    print(f'\n-> {args.out}/lr_heatmaps.png, lr_summary.png, lr_signals_raw.png, lr_signals_norm.png, lr_scores.csv')


if __name__ == '__main__':
    main()
