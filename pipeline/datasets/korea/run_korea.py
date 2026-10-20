#!/usr/bin/env python3
"""run_korea.py -- the Korea children end to end: layout, MotionBERT, shapes, every configuration, tables.

Every step batches all trials itself and skips what is done, so this is only the ORDER and the
flags, in one place, resumable after a Colab disconnect by running it again:

    1. step_1_korea_2d      lay out the usable reps of the usable subjects (per-joint gated target)
    2. step_2a, 2b, 3, 4    MotionBERT (GPU), per-camera betas, skeleton, PnP
    3. step_2c              adult and kid shapes per subject, then the cohort-median kid shape
    4. step_4b              every configuration (formulations.py), GPU when there is one -- subject by subject,
                            each subject scored (step_8) and the tables refreshed as soon as it is done, so
                            {OUT_DIR}/results_table_tri.csv and configs_summary.txt grow while the run goes
    5. step_8               a final scoring pass (anything the per-subject loop missed)
    6. tools/results_table  the long table;  tools/compare_configs  the wide views and paired differences

    %env BIOCV_OUT=/content/drive/MyDrive/MotorDevelopment/Data/Korea
    %env MB_DIR=/content/drive/MyDrive/.../MotionBERT
    python3 datasets/korea/run_korea.py --gt3d /content/drive/MyDrive/.../Korea/B/GT3D
    python3 datasets/korea/run_korea.py --gt3d ... --configs chain --users B010,B011     # a subset
    python3 datasets/korea/run_korea.py --gt3d ... --from step_4b                        # resume from a stage
    python3 datasets/korea/run_korea.py --gt3d ... --until step_4                        # MotionBERT side only

Development subset (datasets/korea/pick_dev_subset.py writes {OUT_DIR}/dev_subset.json after step_1):
--subset runs step_2a-4, step_4b and step_8 on the subset's trials only (step_2c still fits each
subset subject's shape from all of that subject's reps) and writes its tables to {OUT_DIR}/{name}/,
apart from the full set's.  The subjects left out are the held-out set: run them once, at the end.

    python3 datasets/korea/run_korea.py --gt3d ... --from step_2c --subset dev_subset --configs C8,C13
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PIPE)
from config import OUT_DIR, require_out_dir, read_subset, subset_dir  # noqa: E402

STAGES = ['step_1', 'step_2a', 'step_2b', 'step_3', 'step_4', 'step_2c', 'step_4b', 'step_8', 'tables']


def run(script, *cli, cwd=_PIPE):
    cmd = [sys.executable, os.path.join(_PIPE, script), *cli]
    print(f'\n$ {os.path.relpath(cmd[1], _PIPE)} ' + ' '.join(cli), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd)
    print(f'  [{"ok" if r.returncode == 0 else "FAILED rc=" + str(r.returncode)}, {(time.time() - t0) / 60:.1f} min]', flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt3d', required=True, help="build_child_gt.py's output root (rep files + session summaries)")
    ap.add_argument('--users', default=None, help='comma-separated subjects; default every usable one')
    ap.add_argument('--subset', default=None,
                    help='pick_dev_subset.py JSON (or its name under OUT_DIR): only those trials; tables to {OUT_DIR}/{name}/')
    ap.add_argument('--configs', default='all', help="formulations.py selection for step_4b and step_8 ('all', 'chain', 'loo', ids)")
    ap.add_argument('--from', dest='start', default='step_1', choices=STAGES, help='resume from this stage')
    ap.add_argument('--until', default='tables', choices=STAGES, help='stop after this stage')
    ap.add_argument('--device', default='auto', help='step_4b: cpu | cuda | auto')
    ap.add_argument('--stop-on-fail', action='store_true')
    args = ap.parse_args()
    require_out_dir()
    if args.subset and args.users:
        raise SystemExit('--subset and --users: give one or the other')
    name, subset = read_subset(args.subset) if args.subset else (None, None)
    users = sorted(subset) if subset else (args.users.split(',') if args.users else None)

    def units():
        """(user, action) per call: the subset's trials, each user, or one call over everything."""
        if subset:
            return [(u, a) for u in users for a in subset[u]]
        return [(u, None) for u in users] if users else [(None, None)]

    def sel(u, a=None):
        return [*(['--user', u] if u else []), *(['--action', a] if a else [])]

    table_args = ['--subset', args.subset] if subset else []
    compare_args = ['--table', os.path.join(subset_dir(name), 'results_table_tri.csv')] if subset else []
    i0, i1 = STAGES.index(args.start), STAGES.index(args.until)
    if i1 < i0:
        raise SystemExit(f'--until {args.until} comes before --from {args.start}')
    stages = STAGES[i0:i1 + 1]
    print(f'OUT_DIR {OUT_DIR}\nstages  {", ".join(stages)}' +
          (f'\nsubset  {name}: {len(units())} trials of {len(users)} subjects' if subset else
           f'\nusers   {", ".join(users)}' if users else ''))
    ok = True
    for st in stages:
        if st == 'step_1':
            ok = run('datasets/korea/step_1_korea_2d.py', '--gt3d', args.gt3d, *(['--subjects', ','.join(users)] if users else []))
        elif st in ('step_2a', 'step_2b', 'step_3', 'step_4'):
            script = {'step_2a': 'step_2a_extract_betas.py', 'step_2b': 'step_2b_finalise_betas.py',
                      'step_3': 'step_3_extract_3d.py', 'step_4': 'step_4_PnP.py'}[st]
            extra = ['--exclude-joints', 'Hip,RHip,LHip'] if st == 'step_4' else []
            ok = all([run(script, *sel(u, a), *extra) for u, a in units()])
        elif st == 'step_2c':
            ok = all([run('step_2c_fit_shape.py', *sel(u)) for u in (users or [None])]) \
                and run('step_2c_fit_shape.py', '--cohort')
        elif st == 'step_4b':
            # subject by subject, scoring and refreshing the tables after each, so progress can be
            # read off results_table_tri.csv and configs_summary.txt while the run is going
            subjects = users or sorted(d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d)) and d[:1] in 'ABC')
            for i, u in enumerate(subjects, 1):
                print(f'\n##### subject {u} ({i}/{len(subjects)})', flush=True)
                ok = True
                for a in (subset[u] if subset else [None]):
                    ok = run('step_4b_refine.py', *sel(u, a), '--configs', args.configs, '--device', args.device) and ok
                    ok = run('step_8_spider_error.py', *sel(u, a), '--gt', 'triangulated') and ok
                    ok = run('step_8_spider_error.py', *sel(u, a), '--gt', 'triangulated', '--config', 'all') and ok
                run('tools/results_table.py', '--gt', 'triangulated', *table_args)
                run('tools/compare_configs.py', *compare_args)
                if not ok and args.stop_on_fail:
                    break
        elif st == 'step_8':
            ok = all([run('step_8_spider_error.py', *sel(u, a), '--gt', 'triangulated')
                      and run('step_8_spider_error.py', *sel(u, a), '--gt', 'triangulated', '--config', 'all')
                      for u, a in units()])
        elif st == 'tables':
            ok = run('tools/results_table.py', '--gt', 'triangulated', *table_args) and run('tools/compare_configs.py', *compare_args)
        if not ok and args.stop_on_fail:
            print(f'stopping at {st}')
            break


if __name__ == '__main__':
    main()
