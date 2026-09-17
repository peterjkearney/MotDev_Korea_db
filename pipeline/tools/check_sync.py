#!/usr/bin/env python3
"""check_sync.py -- are the 2D rows, the video, the mocap and PnP on one clock?

Per trial and camera, prints the frame counts and rates every stage assumed,
and flags the two ways they come apart:
  * more 2D rows than OpenPose JSONs (rows built from the mocap's length while
    the JSONs came from the video's) -- the second half of the clip is empty;
  * a video whose duration disagrees with its mocap -- either the rates
    differ or the fps metadata is wrong -- so the two run at different speeds.

    python3 tools/check_sync.py                 # every trial under TRIAL_DIR
    python3 tools/check_sync.py --users User08
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from config import TRIAL_DIR, mocap_path, openpose_json_path
from run_batch import discover
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users')
    ap.add_argument('--actions')
    args = ap.parse_args()
    trials, _ = discover([u for u in args.users.split(',') if u] if args.users else None,
                         [a for a in args.actions.split(',') if a] if args.actions else None)
    print(f"{'trial':26s} {'cam':>3} {'video n@fps':>13} {'mocap n@fps':>13} {'JSONs':>5} "
          f"{'2D rows':>7} {'2D fps':>6} {'PnP':>5}  flags")
    for user, action in trials:
        trial = os.path.join(TRIAL_DIR, user, action)
        kp = os.path.join(trial, 'Analysis', 'keypoints')
        mp = mocap_path(user, action)
        mn, mf = (len(np.load(mp)['kps3d']), float(np.load(mp)['fps'])) if os.path.exists(mp) else (0, 0)
        for p in sorted(glob.glob(os.path.join(kp, '*_2d.npz'))):
            cam = os.path.basename(p)[:-7]
            d = np.load(p)
            rows, fps2 = d['h36m_2d'].shape[0], float(d['fps'])
            vpath = os.path.join(trial, f'{cam}.mp4')
            vn = vf = 0
            if os.path.exists(vpath):
                cap = cv2.VideoCapture(vpath)
                vn, vf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
                cap.release()
            jp, nj = openpose_json_path(user, action, cam), 0
            if os.path.exists(jp):
                with open(jp) as f:
                    nj = len(json.load(f)['frames'])
            pp = os.path.join(kp, 'PnP', f'{cam}_pnp.npz')
            npnp = np.load(pp)['kps_H36M_placed'].shape[0] if os.path.exists(pp) else 0
            flags = []
            if nj and rows > nj + 2:
                flags.append(f'2D rows > JSONs: {rows - nj} empty rows at the end')
            if vn and mn and abs(vn / vf - mn / mf) > 0.25:
                flags.append(f'video {vn / vf:.1f}s vs mocap {mn / mf:.1f}s')
            if npnp and npnp != rows:
                flags.append('PnP rows != 2D rows')
            print(f"{user + '/' + action:26s} {cam:>3} {f'{vn}@{vf:g}':>13} {f'{mn}@{mf:g}':>13} {nj:5d} "
                  f"{rows:7d} {fps2:6g} {npnp:5d}  {'; '.join(flags) or 'ok'}")


if __name__ == '__main__':
    main()
