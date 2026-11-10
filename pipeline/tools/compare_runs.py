#!/usr/bin/env python3
"""compare_runs.py -- the same trials and configurations scored in two OUT_DIRs, paired (B - A).

compare_configs compares configurations within one run; this compares two runs of the pipeline that differ
upstream (e.g. step_2a --clip-len 81 vs 16; datasets/korea/clone_out_dir.py), configuration by configuration.
Rows are matched on (user, action, camera, config); only the cameras scored in both runs count.  'step4' is the
unrefined MotionBERT skeleton (PnP placed), so its PA-MPJPE is MotionBERT's own pose error.

    per configuration: cohort median of the subject medians in A and B; per subject paired B - A:
    subjects where B is lower ("B wins"), mean and median difference; and the median trial-level difference.

    python3 tools/compare_runs.py --a {OUT_A}/dev/results_table_tri.csv --b {OUT_B}/dev/results_table_tri.csv \\
        --labels clip81,clip16
writes  {--out}/runs_{A}_vs_{B}.txt  and  runs_{A}_vs_{B}_by_subject.csv  (default --out: B's table folder)
"""
import argparse
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from compare_configs import METRICS, order_key  # noqa: E402

KEY = ['user', 'action', 'camera', 'config']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--a', required=True, help="run A's results_table_tri.csv (the reference)")
    ap.add_argument('--b', required=True, help="run B's results_table_tri.csv")
    ap.add_argument('--labels', default='A,B', help='two names for the runs, e.g. clip81,clip16')
    ap.add_argument('--configs', default=None, help='comma-separated; default every configuration in both')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    la, lb = args.labels.split(',')
    cols = list(METRICS.values())
    tabs = []
    for path, lab in ((args.a, la), (args.b, lb)):
        d = pd.read_csv(path)
        d = d[KEY + [c for c in cols if c in d]].copy()
        for c in cols:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        if args.configs:
            d = d[d['config'].isin(args.configs.split(','))]
        tabs.append(d)
    m = tabs[0].merge(tabs[1], on=KEY, suffixes=(f'_{la}', f'_{lb}'))
    if m.empty:
        raise SystemExit('no (user, action, camera, config) rows in both tables')
    only = [len(t) - len(m) for t in tabs]

    trial = m.groupby(['user', 'action', 'config']).mean(numeric_only=True).reset_index()
    subj = trial.groupby(['user', 'config']).median(numeric_only=True).reset_index()
    cfgs = sorted(m['config'].unique(), key=order_key)

    lines = [f'A = {la}: {args.a}', f'B = {lb}: {args.b}',
             f'matched {len(m)} camera rows, {trial.groupby(["user", "action"]).ngroups} trials, '
             f'{subj["user"].nunique()} subjects; unmatched rows A {only[0]}, B {only[1]}',
             'cohort median of subject medians; d = B - A per subject (negative = B better); '
             'wins = subjects with B lower; trial d = median over trials', '']
    hdr = f'{"config":8}{"metric":>12}{la:>9}{lb:>9}{"wins":>8}{"mean d":>9}{"med d":>9}{"trial d":>9}'
    lines.append(hdr)
    for c in cfgs:
        s, t = subj[subj['config'] == c], trial[trial['config'] == c]
        for k, col in METRICS.items():
            a, b = f'{col}_{la}', f'{col}_{lb}'
            if a not in s or s[a].isna().all():
                continue
            d, dt = (s[b] - s[a]).dropna(), (t[b] - t[a]).dropna()
            lines.append(f'{c:8}{k:>12}{s[a].median():9.1f}{s[b].median():9.1f}'
                         f'{f"{int((d < 0).sum())}/{len(d)}":>8}{d.mean():9.1f}{d.median():9.1f}{dt.median():9.1f}')
        lines.append('')
    out = args.out or os.path.dirname(os.path.abspath(args.b))
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f'runs_{la}_vs_{lb}')
    with open(stem + '.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    subj.round(2).to_csv(stem + '_by_subject.csv', index=False)
    print('\n'.join(lines) + f'\nwrote {stem}.txt, {stem}_by_subject.csv')


if __name__ == '__main__':
    main()
