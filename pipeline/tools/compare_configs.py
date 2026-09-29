#!/usr/bin/env python3
"""compare_configs.py -- the configurations side by side, from the long results table.

Aggregation is camera -> trial (mean) -> subject (median) -> cohort (median, IQR across
subjects): the subject is the unit of evidence.  Writes, next to the table:

    configs_wide_<metric>.csv     one row per trial, one column per configuration (mean over cameras),
                                  for placed MPJPE (mm and % of stature), PA-MPJPE and N-MPJPE
    configs_by_subject.csv        subject x configuration medians of the three metrics
    configs_summary.txt           cohort medians per configuration, the chain read top to bottom,
                                  and per-subject paired differences of every configuration against
                                  --baseline (default C8, the child recipe): how many subjects it
                                  beats, mean and median difference

    python3 tools/compare_configs.py                                   # {OUT_DIR}/results_table_tri.csv
    python3 tools/compare_configs.py --table .../results_table_tri_KOREA.csv --baseline C3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import OUT_DIR  # noqa: E402
import configs  # noqa: E402

METRICS = {'placed_mm': 'err_placed_all_mm', 'placed_pct': 'err_placed_all_pct_stature',
           'pa_mm': 'pa_mpjpe_mm', 'n_mm': 'n_mpjpe_mm'}


def order_key(c):
    return (c != 'step4', configs.ORDER.index(c) if c in configs.ORDER else 99)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--table', default=os.path.join(OUT_DIR, 'results_table_tri.csv'))
    ap.add_argument('--baseline', default='C8', help='configuration the paired differences are taken against')
    ap.add_argument('--out', default=None, help='output folder (default: the table\'s)')
    args = ap.parse_args()
    d = pd.read_csv(args.table)
    out = args.out or os.path.dirname(os.path.abspath(args.table))
    os.makedirs(out, exist_ok=True)
    cfgs = sorted(d['config'].unique(), key=order_key)
    for k, col in METRICS.items():
        if col not in d:
            continue
        d[col] = pd.to_numeric(d[col], errors='coerce')
    # camera -> trial
    trial = d.groupby(['user', 'action', 'config'])[[c for c in METRICS.values() if c in d]].mean().reset_index()
    for k, col in METRICS.items():
        if col not in trial:
            continue
        wide = trial.pivot_table(index=['user', 'action'], columns='config', values=col)[cfgs].round(1)
        wide.to_csv(os.path.join(out, f'configs_wide_{k}.csv'))
    # trial -> subject
    subj = trial.groupby(['user', 'config'])[[c for c in METRICS.values() if c in trial]].median().reset_index()
    subj.round(1).to_csv(os.path.join(out, 'configs_by_subject.csv'), index=False)

    lines = [f'{os.path.basename(args.table)}: {d["user"].nunique()} subjects, {len(trial.groupby(["user", "action"]))} trials, '
             f'{len(cfgs)} configurations', '',
             'cohort median [IQR across subjects] of the subject medians; placed = MPJPE with no alignment, PA = after Procrustes', '',
             f'{"config":8}{"placed mm":>22}{"placed %stat":>22}{"PA mm":>22}{"N mm":>22}   note']
    for c in cfgs:
        s = subj[subj['config'] == c]
        cells = []
        for k in ('placed_mm', 'placed_pct', 'pa_mm', 'n_mm'):
            col = METRICS[k]
            if col in s and s[col].notna().any():
                q = s[col].quantile([0.25, 0.5, 0.75]).values
                cells.append(f'{q[1]:6.1f} [{q[0]:5.1f}-{q[2]:5.1f}]')
            else:
                cells.append(f'{"-":>22}')
        note = configs.CONFIGS[c]['note'] if c in configs.CONFIGS else "step_4's PnP (hips as run)"
        lines.append(f'{c:8}' + ''.join(f'{x:>22}' for x in cells) + f'   {note}')
    base = args.baseline
    if base in cfgs:
        lines += ['', f'paired against {base} per subject (config minus {base}; negative = better than {base}):', '',
                  f'{"config":8}{"n":>4}{"placed: wins":>14}{"mean":>8}{"median":>8}{"PA: wins":>12}{"mean":>8}{"median":>8}']
        piv = {k: subj.pivot_table(index='user', columns='config', values=METRICS[k]) for k in ('placed_mm', 'pa_mm')}
        for c in cfgs:
            if c == base:
                continue
            cells = [f'{c:8}']
            n = None
            for k in ('placed_mm', 'pa_mm'):
                p = piv[k]
                if c not in p or base not in p:
                    cells.append(f'{"-":>30}')
                    continue
                diff = (p[c] - p[base]).dropna()
                n = len(diff)
                cells.append(f'{(diff < 0).sum():>7}/{n:<6}{diff.mean():8.1f}{diff.median():8.1f}')
            lines.append(cells[0] + f'{n if n is not None else 0:>4}' + ''.join(cells[1:]))
    lines += ['', 'wide views: configs_wide_<metric>.csv (trial x configuration); subject medians: configs_by_subject.csv']
    text = '\n'.join(lines)
    with open(os.path.join(out, 'configs_summary.txt'), 'w') as f:
        f.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
