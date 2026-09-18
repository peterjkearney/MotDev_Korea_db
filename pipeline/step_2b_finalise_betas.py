#!/usr/bin/env python3
"""step_2b_finalise_betas.py -- one SMPL shape per CAMERA from pass 1.

Body shape (the 10 betas) is fixed per camera as the median of step_2a's
per-frame betas over confident frames (mean 2D confidence over the 17 joints
above 0.7).

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
            {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz
    writes  {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_final_betas.npz

This used to average the per-camera medians across every camera present.
That made the "single-camera" result depend on shape evidence from up to nine
views, which is not the deployment condition and is exactly the kind of help a
child body would need most.  Each camera now stands alone; nothing from any
other view enters its skeleton.

With no --user / --action it does every camera with pass-1 betas from
--detector; cameras already done are skipped (--force redoes them).

    python3 step_2b_finalise_betas.py                       # everything, OpenPose
    python3 step_2b_finalise_betas.py --detector yolo --user User03
"""
import argparse
import os

import numpy as np

from config import (DETECTORS, OUT_DIR as _OUT_DIR, twod_path as _twod_path, betas_path as _betas_path,
                    final_betas_path as _final_path, find_cameras, require_out_dir)

_CONF_GATE = 0.7   # mean 2D confidence a frame needs for its betas to count


def finalise(user, action, camera, args):
    """One camera -> {cam}_final_betas.npz.  Returns a summary line."""
    conf = np.load(_twod_path(user, action, args.detector, camera))['h36m_2d'][:, :, 2]
    all_betas = np.load(_betas_path(user, action, args.detector, camera))['betas']
    gate = conf.mean(axis=1) > args.conf_gate
    if not gate.any():
        # An empty median would be NaN and poison every downstream frame.
        raise ValueError(f'no frame has mean 2D confidence above {args.conf_gate}')
    betas = np.median(all_betas[gate], axis=0)
    out = _final_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, betas=betas, n_frames_used=int(gate.sum()), n_frames=len(gate),
             conf_gate=args.conf_gate, camera=camera, detector=np.array(args.detector))
    return f'betas from {gate.sum()}/{len(gate)} frames'


def main():
    ap = argparse.ArgumentParser(description='One body shape per camera from pass 1. With no --user/--action, '
                                             'every camera with betas from --detector; cameras already done are skipped.')
    ap.add_argument('--detector', choices=DETECTORS, default='openpose')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with betas')
    ap.add_argument('--conf-gate', type=float, default=_CONF_GATE)
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be run')
    args = ap.parse_args()

    require_out_dir()
    jobs = find_cameras(_betas_path, args.detector, args.user, args.action,
                        set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {args.detector} {{cam}}_betas.npz under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_2a_extract_betas.py first')
    todo = [j for j in jobs if args.force or not os.path.exists(_final_path(j[0], j[1], args.detector, j[2]))]
    print(f'{args.detector}: {len(jobs)} camera(s) with betas under {_OUT_DIR}: {len(todo)} to run, '
          f'{len(jobs) - len(todo)} already done')
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return
    n_done, failed = 0, []
    for user, action, cam in todo:
        try:
            print(f'{user}/{action} {cam}: {finalise(user, action, cam, args)}', flush=True)
            n_done += 1
        except Exception as e:
            failed.append((user, action, cam))
            print(f'{user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
