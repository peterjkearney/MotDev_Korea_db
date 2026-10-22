#!/usr/bin/env python3
"""step_2b_openpose.py -- OpenPose's per-frame JSONs (run on Colab) -> the same
2d/{cond}/openpose/{cam}.npz that step_2_detect writes for the local detectors.

OpenPose cannot run on this Mac.  On Colab it is run straight on the condition
videos step_1 wrote (already at TARGET_FPS, one JSON per frame):
    openpose.bin --video {cond}/{cam}.mp4 --write_json {cond}/{cam}/ --model_pose BODY_25
                 --number_people_max 3 --display 0 --render_pose 0
and the folders copied back to {OUT_ROOT}/{action}/openpose_json/{cond}/{cam}/.
BODY_25 is reduced to COCO-17 and the subject picked per frame by proximity to
the projected mocap, exactly as for the other detectors.

    python step_2b_openpose.py [--conds reenc haar deface oracle]
"""
import argparse
import glob
import json
import os

import numpy as np

from common import Trial, pick_subject, det_path, out_dir, CAMS

# BODY_25 index for each COCO-17 joint
BODY25_TO_COCO = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--conds', nargs='+', default=['reenc', 'haar', 'deface', 'oracle'])
    ap.add_argument('--cams', nargs='+', default=CAMS)
    args = ap.parse_args()
    trial = Trial(args.action)
    for cond in args.conds:
        for cam in args.cams:
            files = sorted(glob.glob(out_dir(args.action, 'openpose_json', cond, cam, '*_keypoints.json')))
            if not files:
                print(f'{cond}/{cam}: no JSONs -- skipped')
                continue
            if len(files) != trial.T:
                raise SystemExit(f'{cond}/{cam}: {len(files)} JSONs, expected {trial.T}')
            m2d = trial.mocap2d(cam)
            coco = np.zeros((trial.T, 17, 3), np.float32)
            match = np.full(trial.T, np.inf)
            n_people = np.zeros(trial.T, int)
            for f, p in enumerate(files):          # names end _{frame:012d}_keypoints.json, so sorted = frame order
                with open(p) as fh:
                    people = json.load(fh).get('people', [])
                k = np.array([np.reshape(q['pose_keypoints_2d'], (25, 3))[BODY25_TO_COCO] for q in people],
                             np.float32).reshape(-1, 17, 3)
                n_people[f] = len(k)
                i, d = pick_subject(k, m2d[f])
                if i >= 0:
                    coco[f], match[f] = k[i], d
            out = det_path(args.action, cond, 'openpose', cam)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, coco=coco, match_px=match, n_people=n_people, frame_idx=trial.frame_idx)
            print(f'{cond}/openpose/{cam}: subject found in {np.isfinite(match).mean():.0%} of frames')


if __name__ == '__main__':
    main()
