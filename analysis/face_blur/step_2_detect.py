#!/usr/bin/env python3
"""step_2_detect.py -- every 2D detector on every condition's video.

Per (condition, detector, camera), 2d/{cond}/{det}/{cam}.npz holds the
subject's keypoints, picked per frame as the detection closest to the
projected mocap (common.pick_subject):
  coco      (T,17,3) x, y, conf -- zeros where no detection matched
  match_px  (T,)  median limb distance of the pick to the mocap, inf if none
  n_people  (T,)  detections in the frame
  world     (T,33,3) MediaPipe's world landmarks (mediapipe only), NaN if none

MediaPipe tracks one person and cannot be pointed at the subject: where it
locked on to someone else the frame counts as undetected.

    python step_2_detect.py [--conds reenc haar deface] [--dets yolo rtmpose mediapipe] [--workers 2]
"""
import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os

import numpy as np

from common import Trial, iter_frames, pick_subject, det_path, video_path, CAMS, DETECTORS


def run_camera(action, cam, conds, dets, force, threads):
    import torch
    torch.set_num_threads(threads)
    import detectors
    trial = Trial(action)
    m2d = trial.mocap2d(cam)
    jobs = [(c, d) for c in conds for d in dets if force or not os.path.exists(det_path(action, c, d, cam))]
    jobs = [(c, d) for c, d in jobs if os.path.exists(video_path(action, cam, c))]
    models = {}
    lines = []
    for cond in sorted({c for c, _ in jobs}, key=conds.index):
        todo = [d for c, d in jobs if c == cond]
        for d in todo:
            if d == 'mediapipe' or d not in models:      # MediaPipe tracks: a fresh one per video
                models[d] = detectors.make(d)
        T = trial.T
        out = {d: dict(coco=np.zeros((T, 17, 3), np.float32), match_px=np.full(T, np.inf),
                       n_people=np.zeros(T, int)) for d in todo}
        if 'mediapipe' in todo:
            out['mediapipe']['world'] = np.full((T, 33, 3), np.nan, np.float32)
        n = 0
        for f, frame in enumerate(iter_frames(trial, cam, cond)):
            for d in todo:
                k = models[d](frame)
                i, dist = pick_subject(k, m2d[f])
                o = out[d]
                o['n_people'][f] = len(k)
                if i >= 0:
                    o['coco'][f] = k[i]
                    o['match_px'][f] = dist
                    if d == 'mediapipe' and models[d].world is not None:
                        o['world'][f] = models[d].world
            n = f + 1
        if n != trial.T:
            raise RuntimeError(f'{cond}/{cam}: read {n} frames, expected {trial.T}')
        for d in todo:
            p = det_path(action, cond, d, cam)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, **out[d], frame_idx=trial.frame_idx)
            lines.append(f'{cond}/{d}/{cam}: subject found in {np.isfinite(out[d]["match_px"]).mean():.0%} of frames')
        for d in todo:
            if d == 'mediapipe':
                models.pop(d).close()
    return '\n'.join(lines) or f'{cam}: done already'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--conds', nargs='+', default=['reenc', 'haar', 'deface'],
                    choices=['raw', 'reenc', 'haar', 'deface', 'oracle'])
    ap.add_argument('--dets', nargs='+', default=list(DETECTORS), choices=DETECTORS)
    ap.add_argument('--cams', nargs='+', default=CAMS)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--threads', type=int, default=3, help='torch threads per worker')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn')) as pool:
        futs = [pool.submit(run_camera, args.action, c, args.conds, args.dets, args.force, args.threads)
                for c in args.cams]
        for fut in cf.as_completed(futs):
            print(fut.result(), flush=True)


if __name__ == '__main__':
    main()
