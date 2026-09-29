#!/usr/bin/env python3
"""run_korea.py -- the Korea children end to end: layout, MotionBERT, shapes, every configuration, tables.

Every step batches all trials itself and skips what is done, so this is only the ORDER and the
flags, in one place, resumable after a Colab disconnect by running it again:

    1. step_1_korea_2d      lay out the usable reps of the usable subjects (per-joint gated target)
    2. step_2a, 2b, 3, 4    MotionBERT (GPU), per-camera betas, skeleton, PnP
    3. step_2c              adult and kid shapes per subject, then the cohort-median kid shape
    4. step_4b              every configuration (configs.py), GPU when there is one
    5. step_8               score step_4's placement and every configuration
    6. tools/results_table  the long table;  tools/compare_configs  the wide views and paired differences

    %env BIOCV_OUT=/content/drive/MyDrive/MotorDevelopment/Data/Korea
    %env MB_DIR=/content/drive/MyDrive/.../MotionBERT
    python3 tools/run_korea.py --gt3d /content/drive/MyDrive/.../Korea/B/GT3D
    python3 tools/run_korea.py --gt3d ... --configs chain --users B010,B011     # a subset
    python3 tools/run_korea.py --gt3d ... --from step_4b                        # resume from a stage
"""
import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
sys.path.insert(0, _PIPE)
from config import OUT_DIR, require_out_dir  # noqa: E402

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
    ap.add_argument('--configs', default='all', help="configs.py selection for step_4b and step_8 ('all', 'chain', 'loo', ids)")
    ap.add_argument('--from', dest='start', default='step_1', choices=STAGES, help='resume from this stage')
    ap.add_argument('--device', default='auto', help='step_4b: cpu | cuda | auto')
    ap.add_argument('--stop-on-fail', action='store_true')
    args = ap.parse_args()
    require_out_dir()
    users = ['--user', args.users] if args.users and ',' not in args.users else []
    if args.users and ',' in args.users:
        print('note: the steps take one --user; with several, step_1 lays out those subjects and the later steps '
              'run every subject already laid out under OUT_DIR')
    stages = STAGES[STAGES.index(args.start):]
    print(f'OUT_DIR {OUT_DIR}\nstages  {", ".join(stages)}')
    ok = True
    for st in stages:
        if st == 'step_1':
            ok = run('step_1_korea_2d.py', '--gt3d', args.gt3d, *(['--subjects', args.users] if args.users else []))
        elif st in ('step_2a', 'step_2b', 'step_3', 'step_4'):
            script = {'step_2a': 'step_2a_extract_betas.py', 'step_2b': 'step_2b_finalise_betas.py',
                      'step_3': 'step_3_extract_3d.py', 'step_4': 'step_4_PnP.py'}[st]
            ok = run(script, *users, *(['--exclude-joints', 'Hip,RHip,LHip'] if st == 'step_4' else []))
        elif st == 'step_2c':
            ok = run('step_2c_fit_shape.py', *users) and run('step_2c_fit_shape.py', '--cohort')
        elif st == 'step_4b':
            ok = run('step_4b_refine.py', *users, '--configs', args.configs, '--device', args.device)
        elif st == 'step_8':
            ok = run('step_8_spider_error.py', *users, '--gt', 'triangulated') \
                and run('step_8_spider_error.py', *users, '--gt', 'triangulated', '--config', 'all')
        elif st == 'tables':
            ok = run('tools/results_table.py', '--gt', 'triangulated') and run('tools/compare_configs.py')
        if not ok and args.stop_on_fail:
            print(f'stopping at {st}')
            break


if __name__ == '__main__':
    main()
