#!/usr/bin/env python3
"""analyse_results.py -- adults vs children through the MotionBERT pipeline.

Reads the two results tables run_batch.py wrote (one row per trial x camera,
scored against the leave-one-out triangulated-OpenPose target) and asks the
study's question: does the lifter's output survive the move from adult bodies
to 3-4.5 year olds?

Aggregation is camera -> trial -> subject (medians), then cohort medians with
the interquartile range across subjects.  Subjects are the unit: an adult
contributes ~16 trials x 9 cameras and a child ~10 reps x 3 cameras, so
anything pooled over rows would be dominated by whoever has more rows.

Read the numbers against two references:
  * the placement floor -- feeding the target itself back through PnP scores
    ~2% of stature (adults) and 3-5% (children), so nothing can beat that;
  * the adult row, not 1.0, for bone ratios -- the target's joints are where
    OpenPose puts them and the prediction's are SMPL's H36M regressor joints,
    two different anatomical definitions with the same offset in both cohorts.
    The child effect is child minus adult, bone by bone.

    python3 analyse_results.py                       # defaults under data/
    python3 analyse_results.py --adult a.csv --child b.csv --out data/analysis
"""
import argparse
import os
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Reference palette (dataviz skill): categorical slots 1 and 2, text/surface tokens.
COL = {'adult': '#2a78d6', 'child': '#eb6834'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

MIN_BETA_FRAMES = 30        # a body shape fitted from fewer frames is not a shape
FLOOR_PCT = {'adult': 1.8, 'child': 4.0}   # single-view PnP of a perfect skeleton

BONES = ['thigh', 'shank', 'upperarm', 'forearm', 'hip_width', 'shoulder_width']
JOINTS = ['RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
          'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
          'Hip', 'Thorax']          # Spine/Nose/Head are excluded from the target mask


def load(path, cohort):
    d = pd.read_csv(path)
    d['cohort'] = cohort
    d['camera'] = d['camera'].astype(str)
    # pool left/right: the question is about proportions, not side
    for b in ('thigh', 'shank', 'upperarm', 'forearm'):
        d[f'bone_{b}'] = d[[f'bone_l_{b}_ratio', f'bone_r_{b}_ratio']].mean(axis=1)
    d['bone_hip_width'] = d['bone_hip_width_ratio']
    d['bone_shoulder_width'] = d['bone_shoulder_width_ratio']
    d['placed_pct'] = d['err_placed_all_pct_stature']
    d['pa_pct'] = 100 * d['pa_mpjpe_mm'] / d['stature_mm']
    d['n_pct'] = 100 * d['n_mpjpe_mm'] / d['stature_mm']
    for j in JOINTS:
        d[f'j_{j}'] = 100 * d[f'errj_{j}_mm'] / d['stature_mm']
    d['low_beta'] = d['beta_n_frames'] < MIN_BETA_FRAMES
    d['bad'] = d['low_beta'] | d['placed_pct'].isna()
    return d


def per_subject(d, cols):
    """camera -> trial -> subject medians."""
    trial = d.groupby(['cohort', 'user', 'action'])[cols].median()
    return trial.groupby(['cohort', 'user']).median()


def cohort_table(subj, cols):
    rows = {}
    for cohort in ('adult', 'child'):
        s = subj.loc[cohort] if cohort in subj.index.get_level_values(0) else None
        if s is None or len(s) == 0:
            continue
        for c in cols:
            q = s[c].quantile([0.25, 0.5, 0.75])
            rows[(c, cohort)] = dict(median=q[0.5], q25=q[0.25], q75=q[0.75], n=int(s[c].notna().sum()))
    return pd.DataFrame(rows).T


def fmt(row):
    return f"{row['median']:.2f} [{row['q25']:.2f}-{row['q75']:.2f}] (n={int(row['n'])})"


def mann_whitney(subj, col):
    a = subj.loc['adult'][col].dropna()
    c = subj.loc['child'][col].dropna()
    if len(a) < 3 or len(c) < 3:
        return np.nan
    return stats.mannwhitneyu(a, c, alternative='two-sided').pvalue


# ---------------------------------------------------------------- figures

def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def dot_plot(ax, subj, cols, labels, title, xlabel, ref=None, floors=None):
    """One row per measure; per-subject dots, a thin bar at the cohort median."""
    y = np.arange(len(cols))[::-1]
    for k, cohort in enumerate(('adult', 'child')):
        if cohort not in subj.index.get_level_values(0):
            continue
        s = subj.loc[cohort]
        off = 0.18 if cohort == 'adult' else -0.18
        for yi, c in zip(y, cols):
            v = s[c].dropna().values
            ax.scatter(v, np.full(len(v), yi + off), s=22, color=COL[cohort], alpha=0.55,
                       edgecolor=SURFACE, linewidth=0.8, zorder=3)
            m = np.median(v)
            ax.plot([m, m], [yi + off - 0.14, yi + off + 0.14], color=COL[cohort], lw=2.2, zorder=4)
            ax.text(m, yi + off + 0.17, f'{m:.2f}', color=INK2, fontsize=7.5,
                    ha='center', va='bottom', zorder=5)
        ax.scatter([], [], s=22, color=COL[cohort], label=f'{cohort} (per subject; bar = median)')
    if ref is not None:
        ax.axvline(ref, color=INK2, lw=0.8, ls='--', zorder=2)
    if floors:
        for cohort, f in floors.items():
            ax.axvline(f, color=COL[cohort], lw=0.8, ls=':', zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK2)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adult', default=os.path.join(HERE, 'data', 'results_table_tri_BioCV.csv'))
    ap.add_argument('--child', default=os.path.join(HERE, 'data', 'results_table_tri_KOREA.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'analysis'))
    ap.add_argument('--keep-low-beta', action='store_true',
                    help=f'keep cameras whose shape came from < {MIN_BETA_FRAMES} frames')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = pd.concat([load(args.adult, 'adult'), load(args.child, 'child')], ignore_index=True)
    lines = []
    say = lines.append

    # ---- coverage and QA ------------------------------------------------
    say('# Adults vs children through the MotionBERT pipeline\n')
    say('Target: leave-one-out triangulated OpenPose, both cohorts. Unit of analysis: subject '
        '(camera -> trial -> subject medians; cohort medians with IQR across subjects).\n')
    say('## Coverage\n')
    say('| cohort | subjects | trials | camera rows | rows dropped (shape from <%d frames) |' % MIN_BETA_FRAMES)
    say('|---|---|---|---|---|')
    for cohort, g in d.groupby('cohort'):
        say(f"| {cohort} | {g.user.nunique()} | {g.groupby(['user','action']).ngroups} | {len(g)} | "
            f"{int(g.bad.sum())} ({100*g.bad.mean():.0f}%) |")
    if not args.keep_low_beta:
        d = d[~d.bad]
    say('')

    # ---- headline metrics -----------------------------------------------
    met = ['placed_pct', 'pa_pct', 'n_pct']
    met_labels = {'placed_pct': 'placement error (MPJPE, no alignment)',
                  'pa_pct': 'pose-shape error (PA-MPJPE)',
                  'n_pct': 'scale-free placement (N-MPJPE)'}
    subj = per_subject(d, met + [f'bone_{b}' for b in BONES] + [f'j_{j}' for j in JOINTS]
                       + [f'beta_{k:02d}' for k in range(10)])
    tab = cohort_table(subj, met)
    say('## Errors, % of stature (median [IQR] across subjects)\n')
    say('| measure | adult | child | floor adult / child |')
    say('|---|---|---|---|')
    for m in met:
        a = fmt(tab.loc[(m, 'adult')]) if (m, 'adult') in tab.index else '-'
        c = fmt(tab.loc[(m, 'child')]) if (m, 'child') in tab.index else '-'
        fl = f"{FLOOR_PCT['adult']} / {FLOOR_PCT['child']}" if m == 'placed_pct' else '0 / 0'
        say(f'| {met_labels[m]} | {a} | {c} | {fl} |')
    say('\nThe floor is what a *perfect* skeleton scores through single-view PnP; '
        'PA-MPJPE and bone ratios have no such floor.\n')

    # ---- bone ratios -----------------------------------------------------
    bcols = [f'bone_{b}' for b in BONES]
    tab_b = cohort_table(subj, bcols)
    say('## Bone length ratios, predicted / target (median [IQR] across subjects)\n')
    say('Read child against adult, not against 1.0: OpenPose joints and SMPL-regressed joints '
        'are different anatomical points, and that offset is the same in both cohorts.\n')
    say('| bone | adult | child | child - adult | Mann-Whitney p |')
    say('|---|---|---|---|---|')
    for b, c in zip(BONES, bcols):
        a = tab_b.loc[(c, 'adult')] if (c, 'adult') in tab_b.index else None
        k = tab_b.loc[(c, 'child')] if (c, 'child') in tab_b.index else None
        diff = f"{k['median'] - a['median']:+.3f}" if a is not None and k is not None else '-'
        p = mann_whitney(subj, c)
        say(f"| {b} | {fmt(a) if a is not None else '-'} | {fmt(k) if k is not None else '-'} | "
            f"{diff} | {p:.3g} |" if np.isfinite(p) else
            f"| {b} | {fmt(a) if a is not None else '-'} | {fmt(k) if k is not None else '-'} | {diff} | - |")
    say('')

    # ---- per joint -------------------------------------------------------
    jcols = [f'j_{j}' for j in JOINTS]
    tab_j = cohort_table(subj, jcols)
    variant = d.joint_variant.iloc[0]
    say(f'## Per-joint error, % of stature (variant `{variant}` -- the per-joint columns follow '
        f'run_batch\'s --joint-variant, so these are the *smoothed* skeleton unless it was changed)\n')
    say('| joint | adult | child |')
    say('|---|---|---|')
    for j, c in zip(JOINTS, jcols):
        a = tab_j.loc[(c, 'adult')]['median'] if (c, 'adult') in tab_j.index else np.nan
        k = tab_j.loc[(c, 'child')]['median'] if (c, 'child') in tab_j.index else np.nan
        say(f'| {j} | {a:.1f} | {k:.1f} |')
    say('')

    # ---- per camera ------------------------------------------------------
    say('## Placement error by camera, % of stature (median over trials)\n')
    cam = d.groupby(['cohort', 'camera'])['placed_pct'].median().unstack(0).round(1)
    say(cam.to_markdown())
    say('\nKorea camera 2 was the least stable calibration parameter; BioCV cameras differ in '
        'distance and angle.\n')

    # ---- betas -----------------------------------------------------------
    say('## SMPL shape parameters (betas), per-subject medians\n')
    bet = subj[[f'beta_{k:02d}' for k in range(10)]]
    say('| | ' + ' | '.join(f'b{k}' for k in range(10)) + ' | mean abs |')
    say('|---|' + '---|' * 11)
    for cohort in ('adult', 'child'):
        if cohort in bet.index.get_level_values(0):
            m = bet.loc[cohort].median()
            say(f'| {cohort} | ' + ' | '.join(f'{v:+.2f}' for v in m) +
                f' | {bet.loc[cohort].abs().mean(axis=1).median():.2f} |')
    say('\nbeta_0 is broadly overall size/height in SMPL; large magnitudes mean the fit is pushed '
        'toward the edge of the adult shape space.\n')

    # ---- figures ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    dot_plot(ax, subj, bcols, [b.replace('_', ' ') for b in BONES],
             'Bone length: predicted / triangulated target', 'ratio (1.0 = same length)', ref=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'bone_ratios.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor=SURFACE)
    dot_plot(ax, subj, met, ['placement (MPJPE)', 'pose shape (PA-MPJPE)', 'scale-free (N-MPJPE)'],
             'Error as % of stature', '% of stature', floors=FLOOR_PCT)
    ax.text(FLOOR_PCT['child'], ax.get_ylim()[1], ' dotted = PnP floor', color=INK2, fontsize=7.5, va='top')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'errors_pct_stature.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
    dot_plot(ax, subj, jcols, JOINTS, f'Per-joint error, % of stature ({variant})', '% of stature')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'per_joint_pct_stature.png'), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---- write -----------------------------------------------------------
    subj.round(4).to_csv(os.path.join(args.out, 'per_subject.csv'))
    report = '\n'.join(lines)
    with open(os.path.join(args.out, 'summary.md'), 'w') as f:
        f.write(report + '\n')
    print(report)
    print(f'\nwrote {args.out}/summary.md, per_subject.csv, bone_ratios.png, '
          f'errors_pct_stature.png, per_joint_pct_stature.png')


if __name__ == '__main__':
    main()
