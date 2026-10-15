#!/usr/bin/env python3
"""run_override.py -- step_4b with formulations.py settings overridden, for experiments that should not
(yet) be configurations.  Replaces the 2026-10-15 scratch runners run_contact.py / run_weight.py.

    --set KEY=VALUE[,KEY=VALUE]   applied to every --configs entry (floats parsed; 'BALANCE' adds the
                                  formulations.BALANCE terms)
                                  dict values as joint:scale|joint:scale (pull_scale=Spine1:0.1|L_Hip:0.1)
    --contact planted             floor_contact_m = the subject's median planted-ankle height (its targets)

Point BIOCV_OUT at a COPY of the output tree: results are written under the configuration's own id, so they
overwrite that configuration's outputs.  Run from pipeline/:

    BIOCV_OUT=<copy> python3 ../analysis/floor_depth/run_override.py --users B010,B023 --configs C12 \\
        --contact planted --set w_floor_contact=3e4,placement=ray
"""
import argparse
import glob
import os
import runpy
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formulations  # noqa: E402
from config import tri_target_path  # noqa: E402
from floorlib import grounded_ankles  # noqa: E402


def parse_set(spec):
    out = {}
    for item in filter(None, (s.strip() for s in (spec or '').split(','))):
        if item == 'BALANCE':
            out.update(formulations.BALANCE)
            continue
        k, v = item.split('=', 1)
        if ':' in v:                                                  # dict: joint:scale|joint:scale
            out[k] = {kk: float(vv) for kk, vv in (pair.split(':') for pair in v.split('|'))}
            continue
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--users', required=True, help='comma-separated subjects')
    ap.add_argument('--configs', default='C12')
    ap.add_argument('--set', default='', help='KEY=VALUE overrides, comma-separated; BALANCE adds the balance terms')
    ap.add_argument('--contact', choices=['config', 'planted'], default='config')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    over = parse_set(args.set)
    ids = [c.strip() for c in args.configs.split(',')]
    for user in args.users.split(','):
        extra = dict(over)
        if args.contact == 'planted':
            P = np.concatenate([grounded_ankles(np.load(p)['kps3d'].astype(float))
                                for p in sorted(glob.glob(tri_target_path(user, '*')))])
            extra['floor_contact_m'] = float(np.median(P[:, 2])) / 1000.0
        for c in ids:
            formulations.CONFIGS[c].update(extra)
        print(f'##### {user}: {args.configs} with {extra}', flush=True)
        sys.argv = ['step_4b_refine.py', '--user', user, '--configs', args.configs, '--force', '--device', args.device] + \
            (['--verbose'] if args.verbose else [])
        runpy.run_path('step_4b_refine.py', run_name='__main__')


if __name__ == '__main__':
    main()
