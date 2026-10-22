#!/usr/bin/env python3
"""step_3_head.py -- where the head is, from the unblurred videos, for the oracle
blur and for scoring the anonymisers' recall.

BioCV's marker set has no head markers, so the head comes from the baseline
(reenc) 2D: each detector's five face points (nose, eyes, ears) triangulated
across the cameras (utils/triangulate.triangulate_robust, which drops
disagreeing views), then the median over detectors.  The other views outvote any
one camera, so a camera's own detection is not what places its oracle blur.

head.npz:
  face      (T,5,3) nose, l_eye, r_eye, l_ear, r_ear, lab mm (NaN where unsolved)
  centre    (T,3) midway between the nose and the ear midpoint; gaps
            interpolated, ends held.  blur.Oracle's sphere is centred here
  forward   (T,3) unit vector ear midpoint -> nose (where the face points)
  facing    (T,9) per camera, degrees between `forward` and the direction to
            the camera: 0 = looking straight at it, > 90 = face turned away
"""
import argparse

import numpy as np

from common import Trial, det_path, head_path, CAMS, FACE, BASELINE
from utils.triangulate import triangulate_robust

CONF_MIN = {'yolo': 0.5, 'rtmpose': 0.5, 'mediapipe': 0.5}


def _fill(x):
    """Linear interpolation over NaN frames, holding the ends."""
    x = x.copy()
    t = np.arange(len(x))
    for k in range(x.shape[1]):
        ok = np.isfinite(x[:, k])
        if ok.sum() >= 2:
            x[~ok, k] = np.interp(t[~ok], t[ok], x[ok, k])
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--cond', default=BASELINE)
    ap.add_argument('--dets', nargs='+', default=['yolo', 'rtmpose', 'mediapipe'])
    ap.add_argument('--thresh-px', type=float, default=20.0)
    args = ap.parse_args()

    trial = Trial(args.action)
    cams = [trial.cams[c].as_tuple() for c in CAMS]
    per_det = []
    for d in args.dets:
        xy = np.zeros((trial.T, len(CAMS), len(FACE), 2))
        vis = np.zeros((trial.T, len(CAMS), len(FACE)), bool)
        for ci, c in enumerate(CAMS):
            k = np.load(det_path(args.action, args.cond, d, c))['coco'][:, FACE]
            xy[:, ci] = k[..., :2]
            vis[:, ci] = k[..., 2] >= CONF_MIN[d]
        X, used, err = triangulate_robust(xy, vis, cams, thresh_px=args.thresh_px)
        print(f'{d}: face points solved in {np.isfinite(X[..., 0]).mean():.0%} of frame-joints, '
              f'median {np.nanmedian(used.sum(1)[np.isfinite(X[..., 0])]):.0f} views, '
              f'reprojection {np.nanmedian(err):.1f} px')
        per_det.append(X)
    face = np.nanmedian(np.stack(per_det), axis=0)                         # (T,5,3)

    ear_mid = np.nanmean(face[:, 3:5], axis=1)
    eye_mid = np.nanmean(face[:, 1:3], axis=1)
    nose = np.where(np.isfinite(face[:, 0]), face[:, 0], eye_mid)
    centre = _fill((nose + ear_mid) / 2)
    fwd = _fill(nose - ear_mid)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    facing = np.zeros((trial.T, len(CAMS)))
    for ci, c in enumerate(CAMS):
        to_cam = trial.cams[c].centre - centre
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing[:, ci] = np.degrees(np.arccos(np.clip((fwd * to_cam).sum(1), -1, 1)))

    np.savez(head_path(args.action), face=face, centre=centre, forward=fwd, facing=facing,
             frame_idx=trial.frame_idx, cams=np.array(CAMS))
    gap = ~np.isfinite((nose + ear_mid)[:, 0])
    print(f'head centre: {gap.mean():.0%} of frames interpolated; '
          f'nose-to-ear-midpoint {np.nanmedian(np.linalg.norm(nose - ear_mid, axis=1)):.0f} mm')
    print('frames facing each camera (< 90 deg): ' +
          ' '.join(f'{c}:{(facing[:, i] < 90).mean():.0%}' for i, c in enumerate(CAMS)))
    print(f'-> {head_path(args.action)}')


if __name__ == '__main__':
    main()
