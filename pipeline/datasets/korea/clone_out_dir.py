#!/usr/bin/env python3
"""clone_out_dir.py -- copy a subset's MotionBERT-independent inputs into a fresh OUT_DIR.

For re-running the pipeline from step_2a with something MotionBERT-side changed (e.g. step_2a --clip-len)
without touching the original run.  Steps skip by file existence, so the new run needs a tree in which
nothing downstream of MotionBERT exists yet.  Copied:

    {src}/{name}.json                                   the subset itself
    {src}/*.npz, {src}/*.json                           root files (cohort_shapes.npz, ...); not the old tables
    {src}/{user}/<files>                                shapes.npz (step_2c, from the triangulated target), user_meta.json
    {src}/{user}/{action}/...                           everything step_1 laid out (2D, triangulated target, ...)

and NOT copied: Analysis/{mesh, PnP, diagnostics, features} (step_2a onwards).  step_2c's shapes come from the
triangulated bone lengths, not MotionBERT, so copying them keeps C9's cohort shape identical across the two runs
-- run the new OUT_DIR from step_2a to step_4, then from step_4b (skip step_2c).

    python3 datasets/korea/clone_out_dir.py --src /content/drive/.../Korea --dst /content/drive/.../Korea_clip16 --subset dev
    python3 datasets/korea/clone_out_dir.py --src ... --dst ... --subset dev --dry-run
    python3 datasets/korea/clone_out_dir.py --src ... --dst ... --subset dev --all-reps   # every rep of the dev children
"""
import argparse
import glob
import json
import os
import shutil

DOWNSTREAM = {'mesh', 'PnP', 'diagnostics', 'features'}     # under Analysis/: made by step_2a onwards


def read_subset(src, subset):
    path = subset if os.path.exists(subset) else os.path.join(src, subset + '.json')
    with open(path) as f:
        s = json.load(f)
    return path, s['name'], s['trials']


def ignore_downstream(d, names):
    return [n for n in names if os.path.basename(d) == 'Analysis' and n in DOWNSTREAM]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--src', required=True, help='the existing OUT_DIR')
    ap.add_argument('--dst', required=True, help='the new OUT_DIR (created; must not already hold any of the trials)')
    ap.add_argument('--subset', required=True, help='pick_dev_subset.py JSON, or its name under --src')
    ap.add_argument('--all-reps', action='store_true',
                    help="every rep step_1 laid out for the subset's SUBJECTS (e.g. the full dev split), not just its trials")
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if dst == src or dst.startswith(src + os.sep) or src.startswith(dst + os.sep):
        raise SystemExit('--dst must be a separate folder, not inside (or containing) --src')
    spath, name, trials = read_subset(src, args.subset)
    if args.all_reps:      # every laid-out rep (a folder with Analysis/keypoints) of the subset's subjects
        trials = {u: sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                            for p in glob.glob(os.path.join(src, u, '*', 'Analysis', 'keypoints')))
                  for u in trials}

    plan = [(spath, os.path.join(dst, name + '.json'))]
    plan += [(p, os.path.join(dst, os.path.basename(p)))
             for p in sorted(glob.glob(os.path.join(src, '*.npz')) + glob.glob(os.path.join(src, '*.json')))
             if os.path.abspath(p) != os.path.abspath(spath)]
    trees, missing = [], []
    for user, actions in sorted(trials.items()):
        udir = os.path.join(src, user)
        plan += [(p, os.path.join(dst, user, os.path.basename(p)))
                 for p in sorted(glob.glob(os.path.join(udir, '*'))) if os.path.isfile(p)]
        for a in actions:
            (trees if os.path.isdir(os.path.join(udir, a)) else missing).append((user, a))
    if missing:
        raise SystemExit(f'{len(missing)} subset trial(s) not under --src, e.g. {"/".join(missing[0])}')
    clash = [t for t in trees if os.path.exists(os.path.join(dst, *t))]
    if clash:
        raise SystemExit(f'{len(clash)} trial(s) already exist under --dst, e.g. {"/".join(clash[0])} -- use a fresh folder')
    if not any(os.path.basename(p) == 'shapes.npz' for p, _ in plan):
        print('WARNING: no {user}/shapes.npz under --src -- run step_2c in the new OUT_DIR (or on --src first)')

    print(f'subset {name}: {len(trials)} subjects, {len(trees)} trials; {len(plan)} files + {len(trees)} trial trees '
          f'(without Analysis/{{{",".join(sorted(DOWNSTREAM))}}})\n  {src}\n  -> {dst}')
    if args.dry_run:
        for s, d in plan:
            print(f'  {os.path.relpath(s, src)}')
        return
    for s, d in plan:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(s, d)
    for i, (user, a) in enumerate(trees, 1):
        shutil.copytree(os.path.join(src, user, a), os.path.join(dst, user, a), ignore=ignore_downstream)
        if i % 20 == 0 or i == len(trees):
            print(f'  {i}/{len(trees)} trials', flush=True)
    print('done')


if __name__ == '__main__':
    main()
