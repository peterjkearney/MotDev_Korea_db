#!/usr/bin/env python3
"""make_splits.py -- the fixed train / dev / test split of the Korea children, by subject.

One file decides, for every usable child, which job they do, and every tool that trains or scores reads it:

    train   fits anything learned from data: the OpenPose vertex regressor (vertex_regressor.py, which
            cross-validates WITHIN train), and any weight or threshold tuned by search
    dev     where formulations are compared (pick_dev_subset.py --splits -> run_korea --subset); looked at often
    test    run once, when the recipe is settled -- its numbers are the ones reported

The split is by SUBJECT (a child's shape is fitted from all their reps, so splitting reps would leak it),
per set (B, C), and within a set stratified by camera setups (sessions whose cameras were moved part-way are
where placement has failed before, so each split gets its share).  The sample children (--force-dev, the ones
every earlier recipe decision was made on) go to dev: they are no longer unseen.  Set A (20-35 months) is
younger than the use case and is listed as excluded; the BioCV adults are not in the file and are training
data wherever they are used.

    reads   {root}/{set}/GT3D/{subject}/session_summary.json
    writes  datasets/korea/splits.json   {"seed", "criteria", "sets": {set: {"train", "dev", "test"}},
                                           "excluded": {set: reason}, "train"/"dev"/"test": [all sets]}

    python3 datasets/korea/make_splits.py                      # B and C, 20% test, 5 dev per set
"""
import argparse
import glob
import json
import os
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
KOREA_ROOT = '/Volumes/Sandisk_SSD/MotorDevelopment/Korea'
SPLITS_PATH = os.path.join(_HERE, 'splits.json')


def survey(gt3d):
    """{subject: (usable reps, camera setups)} for every usable subject of a set."""
    out = {}
    for p in sorted(glob.glob(os.path.join(gt3d, '*', 'session_summary.json'))):
        with open(p) as f:
            s = json.load(f)
        if s.get('usable'):
            out[s['subject']] = (int(s.get('n_reps_usable', 0)), int(s.get('n_setups', 1)))
    return out


def load_splits(path=SPLITS_PATH):
    """{subject: 'train' | 'dev' | 'test'} for every child in the file."""
    with open(path) as f:
        doc = json.load(f)
    return {s: k for k in ('train', 'dev', 'test') for s in doc[k]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=KOREA_ROOT)
    ap.add_argument('--sets', default='B,C')
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--dev', type=int, default=5, help='dev subjects per set (forced ones included)')
    ap.add_argument('--force-dev', default='B010,B011,B014,B023', help='subjects that must be dev (already looked at)')
    ap.add_argument('--min-reps', type=int, default=3, help='usable reps for a subject to be eligible for dev')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=SPLITS_PATH)
    ap.add_argument('--force', action='store_true', help='overwrite an existing split')
    args = ap.parse_args()
    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f'{args.out} exists -- a split must stay fixed once used; --force to replace it')

    rng = random.Random(args.seed)
    forced = set(args.force_dev.split(',')) if args.force_dev else set()
    doc = dict(seed=args.seed, criteria=dict(test_frac=args.test_frac, dev_per_set=args.dev, force_dev=sorted(forced),
                                             min_reps_dev=args.min_reps, stratified_by='camera setups (1 vs several)'),
               sets={}, excluded={'A': '20-35 months: younger than the use case'}, train=[], dev=[], test=[])
    for st in args.sets.split(','):
        info = survey(os.path.join(args.root, st, 'GT3D'))
        missing = sorted(s for s in forced if s[0] == st and s not in info)
        if missing:
            raise SystemExit(f'forced dev subjects not usable in set {st}: {missing}')
        split = dict(train=[], dev=sorted(s for s in forced if s in info), test=[])
        n_test = round(args.test_frac * len(info))
        for moved in (True, False):                                 # each stratum gets its share of test and dev
            stratum = sorted(s for s, (_, n) in info.items() if (n > 1) == moved and s not in split['dev'])
            rng.shuffle(stratum)
            share = len(stratum) / max(1, len(info) - len(split['dev']))
            k_test = round(n_test * share)
            split['test'] += stratum[:k_test]
            rest = stratum[k_test:]
            free = args.dev - len(split['dev'])
            k_dev = max(1, round(free * share)) if moved else free  # at least one moved session in dev
            dev = [s for s in rest if info[s][0] >= args.min_reps][:k_dev]
            split['dev'] += dev
            split['train'] += [s for s in rest if s not in dev]
        split = {k: sorted(v) for k, v in split.items()}
        doc['sets'][st] = split
        for k in ('train', 'dev', 'test'):
            doc[k] += split[k]
        moved = {k: sum(info[s][1] > 1 for s in v) for k, v in split.items()}
        print(f'set {st}: {len(info)} usable -> ' + ', '.join(f'{k} {len(v)} ({moved[k]} moved)' for k, v in split.items()))
        print(f'  dev:  {", ".join(split["dev"])}\n  test: {", ".join(split["test"])}')
    with open(args.out, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
