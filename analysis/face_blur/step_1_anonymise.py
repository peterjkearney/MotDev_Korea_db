#!/usr/bin/env python3
"""step_1_anonymise.py -- write each condition's TARGET_FPS video, and every blur region.

For each camera, the raw video's kept frames are passed through each method
and written with identical x264 settings (common.write_video):
  reenc   unchanged -- the baseline, so compression is the same everywhere
  haar    blur.Haar   (the ethics proposal)
  deface  blur.Deface (deface 1.5 defaults)
  oracle  blur.Oracle -- needs head.npz, so run it after step_3_head:
          python step_1_anonymise.py --methods oracle
faces/{method}/{cam}.npz keeps each frame's regions (for step_5's recall and
false-positive counts) and the time each frame took.

    python step_1_anonymise.py [--action P03_CMJM_01] [--methods reenc haar deface] [--workers 3]
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import time

import numpy as np

from common import Trial, iter_frames, faces_path, head_path, video_path, write_video, TARGET_FPS, CAMS
import blur


def run_camera(action, cam, methods, force):
    trial = Trial(action)
    methods = [m for m in methods if force or not os.path.exists(video_path(action, cam, m))]
    if not methods:
        return f'{cam}: done already'
    finders = {}
    for m in methods:
        if m == 'haar':
            finders[m] = blur.Haar()
        elif m == 'deface':
            finders[m] = blur.Deface()
        elif m == 'oracle':
            finders[m] = blur.Oracle(trial.cams[cam], np.load(head_path(action))['centre'])
    cam_obj = trial.cams[cam]
    pipes = {m: write_video(video_path(action, cam, m) + '.part.mp4', TARGET_FPS, cam_obj.w, cam_obj.h)
             for m in methods}
    regions = {m: [] for m in finders}
    secs = {m: [] for m in finders}
    n = 0
    try:
        for f, frame in enumerate(iter_frames(trial, cam, 'raw')):
            for m in methods:
                out = frame
                if m in finders:
                    t = time.time()
                    r = finders[m].find(frame, f) if m == 'oracle' else finders[m].find(frame)
                    out = finders[m].apply(frame.copy(), r)
                    secs[m].append(time.time() - t)
                    regions[m].append(r)
                pipes[m].stdin.write(np.ascontiguousarray(out).tobytes())
            n += 1
    finally:
        for p in pipes.values():
            p.stdin.close()
            p.wait()
    if n != trial.T:
        raise RuntimeError(f'{cam}: read {n} frames, expected {trial.T}')
    for m in methods:
        os.replace(video_path(action, cam, m) + '.part.mp4', video_path(action, cam, m))
    for m in finders:
        per_frame = np.empty(len(regions[m]), dtype=object)    # filled one by one: np.array() would stack
        for i, r in enumerate(regions[m]):                     # equal-shaped frames into one 3D array
            per_frame[i] = r
        os.makedirs(os.path.dirname(faces_path(action, m, cam)), exist_ok=True)
        np.savez(faces_path(action, m, cam),
                 regions=per_frame,                            # per frame (n,5) x1 y1 x2 y2 score
                 shape=np.array(finders[m].shape), seconds=np.array(secs[m]))
    return f'{cam}: ' + ', '.join(f'{m} {np.mean([len(r) > 0 for r in regions[m]]):.0%} of frames with a region'
                                  for m in finders) + (' ' if finders else '') + f'({n} frames)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--methods', nargs='+', default=['reenc', 'haar', 'deface'],
                    choices=['reenc', 'haar', 'deface', 'oracle'])
    ap.add_argument('--cams', nargs='+', default=CAMS)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if 'oracle' in args.methods and not os.path.exists(head_path(args.action)):
        raise SystemExit(f'oracle needs {head_path(args.action)} -- run step_3_head.py first')
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futs = {pool.submit(run_camera, args.action, c, args.methods, args.force): c for c in args.cams}
        for fut in cf.as_completed(futs):
            print(fut.result(), flush=True)


if __name__ == '__main__':
    main()
