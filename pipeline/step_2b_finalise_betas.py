#!/usr/bin/env python3
"""step_2b_finalise_betas.py -- one SMPL shape per CAMERA from pass 1.

Body shape (the 10 betas) is fixed per camera as the median of step_2a's
per-frame betas over confident frames, and saved as {cam}_final_betas.npz.

This used to average the per-camera medians across every camera present.
That made the "single-camera" result depend on shape evidence from up to nine
views, which is not the deployment condition and is exactly the kind of help a
child body would need most.  Each camera now stands alone; nothing from any
other view enters its skeleton.

    python3 step_2b_finalise_betas.py --user P08 --action P08_CMJM_01 --cameras 00,06,07
"""
import argparse
import glob
import os

import numpy as np

from config import TRIAL_DIR as _TRIAL_DIR

_CONF_GATE = 0.7   # mean 2D confidence a frame needs for its betas to count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cameras', default=None, help='default: every camera with betas')
    args = ap.parse_args()

    kp_dir = os.path.join(_TRIAL_DIR, args.user, args.action, 'Analysis', 'keypoints')
    beta_dir = os.path.join(kp_dir, 'mesh')
    cameras = (args.cameras.split(',') if args.cameras else
               sorted(os.path.basename(p)[:-10] for p in glob.glob(os.path.join(beta_dir, '*_betas.npz'))
                      if not p.endswith('_final_betas.npz')))

    n_done = 0
    for camera in cameras:
        path_2d = os.path.join(kp_dir, f'{camera}_2d.npz')
        path_betas = os.path.join(beta_dir, f'{camera}_betas.npz')
        if not os.path.exists(path_2d) or not os.path.exists(path_betas):
            print(f'=== camera {camera}: missing 2d or betas, skipping ===')
            continue
        conf = np.load(path_2d)['h36m_2d'][:, :, 2]
        all_betas = np.load(path_betas)['betas']
        gate = conf.mean(axis=1) > _CONF_GATE
        if not gate.any():
            # An empty median would be NaN and poison every downstream frame.
            print(f'=== camera {camera}: no frame above conf {_CONF_GATE}, skipping ===')
            continue
        betas = np.median(all_betas[gate], axis=0)
        out = os.path.join(beta_dir, f'{camera}_final_betas.npz')
        np.savez(out, betas=betas, n_frames_used=int(gate.sum()), n_frames=len(gate),
                 conf_gate=_CONF_GATE, camera=camera)
        print(f'=== camera {camera}: betas from {gate.sum()}/{len(gate)} frames -> {out}')
        n_done += 1
    if n_done == 0:
        raise SystemExit('no camera produced usable betas')


if __name__ == '__main__':
    main()
