#!/usr/bin/env python3
"""pick_dev_subset.py -- a development subset of the laid-out Korea trials, and the held-out rest.

Recipe work (new configurations, weights, joint definitions) is judged on a few subjects; the rest
are held out and run once, when the recipe is settled, so the final numbers are a test rather than
the numbers the recipe was tuned on.  The split is by SUBJECT: a child's shape (step_2c) is fitted
from all their reps, so splitting reps would leak it.

Subjects are drawn at random (--seed) from those with at least --reps usable reps, with at least
--moved of them from sessions whose cameras were recalibrated part-way (several camera setups),
since those are where placement has failed before.  Each subject contributes up to --reps reps,
from different GMS tasks where it can (B010_GMS_<task>_<trial>).  A rep is usable when at least
--min-frames frames keep >= 10 of the 12 limb joints in the all-camera target (build_child_gt.py's rule).

    reads   {OUT_DIR}/{subject}/{rep}/Analysis/H36M/openpose_tri_h36m.npz, {rep}/*.mp4-mocAligned.calib
    writes  {OUT_DIR}/{name}.json   {"name", "seed", "criteria", "trials": {subject: [reps]}, "held_out": [subjects]}

    python3 datasets/korea/pick_dev_subset.py                       # 10 subjects x 3 reps -> dev_subset.json
    python3 datasets/korea/pick_dev_subset.py --subjects 8 --reps 2 --seed 1 --name dev_small
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from config import OUT_DIR, tri_target_path, require_out_dir  # noqa: E402

LIMBS = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]          # H36M limb joints (utils.openpose.H36M_CORE)


def survey(subject):
    """[(rep, usable frames, camera-setup signature)] for every laid-out rep of the subject."""
    out = []
    for p in sorted(glob.glob(tri_target_path(subject, '*'))):
        rep = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p))))
        X = np.load(p, allow_pickle=True)['kps3d']
        usable = int((np.isfinite(X[:, LIMBS]).all(-1).sum(1) >= 10).sum())
        h = hashlib.sha1()
        for c in sorted(glob.glob(os.path.join(OUT_DIR, subject, rep, '*.mp4-mocAligned.calib'))):
            with open(c, 'rb') as f:
                h.update(f.read())
        out.append((rep, usable, h.hexdigest()))
    return out


def task(rep):
    parts = rep.split('_')
    return parts[2] if len(parts) >= 4 else rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', type=int, default=10, help='development subjects')
    ap.add_argument('--reps', type=int, default=3, help='reps per development subject')
    ap.add_argument('--moved', type=int, default=2, help='at least this many subjects with several camera setups')
    ap.add_argument('--min-frames', type=int, default=60, help='usable frames for a rep to count')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--name', default='dev_subset')
    ap.add_argument('--force', action='store_true', help='overwrite an existing subset file')
    args = ap.parse_args()
    require_out_dir()
    path = os.path.join(OUT_DIR, f'{args.name}.json')
    if os.path.exists(path) and not args.force:
        raise SystemExit(f'{path} exists -- a subset should stay fixed once used; --force to replace it')

    subjects = sorted(d for d in os.listdir(OUT_DIR) if d.startswith('B') and os.path.isdir(os.path.join(OUT_DIR, d)))
    info = {}
    for s in subjects:
        reps = survey(s)
        good = [(r, n, sig) for r, n, sig in reps if n >= args.min_frames]
        if len(good) >= args.reps:
            info[s] = dict(good=good, setups=len({sig for _, _, sig in reps}))
    if len(info) < args.subjects:
        raise SystemExit(f'only {len(info)} subjects have {args.reps} reps with >= {args.min_frames} usable frames')

    rng = random.Random(args.seed)
    moved = sorted(s for s, v in info.items() if v['setups'] > 1)
    rest = sorted(s for s in info if s not in moved)
    rng.shuffle(moved)
    rng.shuffle(rest)
    k = min(args.moved, len(moved), args.subjects)
    chosen = sorted(moved[:k] + rng.sample(sorted(moved[k:] + rest), args.subjects - k))

    trials = {}
    for s in chosen:
        good = info[s]['good'][:]
        rng.shuffle(good)
        pick, tasks = [], set()
        for r, _, _ in good:                                  # one rep per task first, then fill
            if task(r) not in tasks:
                pick.append(r)
                tasks.add(task(r))
        pick += [r for r, _, _ in good if r not in pick]
        trials[s] = sorted(pick[:args.reps])

    held_out = [s for s in subjects if s not in trials]
    doc = dict(name=args.name, seed=args.seed,
               criteria=dict(subjects=args.subjects, reps=args.reps, moved=args.moved, min_frames=args.min_frames),
               trials=trials, held_out=held_out)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=1)
    n_moved = sum(info[s]['setups'] > 1 for s in trials)
    print(f'{len(subjects)} subjects laid out, {len(info)} eligible ({len(moved)} with several camera setups)')
    for s, reps in trials.items():
        print(f'  {s}: {", ".join(reps)}' + (f'   [{info[s]["setups"]} camera setups]' if info[s]['setups'] > 1 else ''))
    print(f'{sum(len(r) for r in trials.values())} trials of {len(trials)} subjects ({n_moved} with moved cameras); '
          f'{len(held_out)} subjects held out -> {path}')


if __name__ == '__main__':
    main()
