#!/usr/bin/env python3
"""sagittal.py -- which segments / joint angles the placed skeleton under-reproduces, front to back.

Per frame, every segment's forward tilt from vertical in the subject's sagittal plane (target body frame:
up = lab Z, lateral = RHip->LHip horizontal, forward = lateral x up): trunk Hip->Thorax, thigh hip->knee,
shank knee->ankle, upper arm shoulder->elbow (left and right averaged).  Joint angles from the tilts:
hip flexion = trunk - thigh, knee flexion = shank - thigh (sign: + = flexed), and the regression of the
model's value on the target's (slope < 1 = under-reproduced) per camera, with the error at the target's
10th / 90th percentile.  Frames matched through source_frame_idx.

    python3 ../analysis/floor_depth/sagittal.py <OUT_DIR> <config> [--target mocap|tri]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from utils.smpl_lite import J  # noqa: E402


def tilts(Y, fwd):
    """Forward tilt from vertical (deg) of each segment, sagittal plane; + = top end / distal end forward."""
    def seg(a, b):                                                    # from a (proximal) to b
        v = Y[:, J[b]] - Y[:, J[a]]
        return v
    def tilt_up(v):                                                   # segment pointing up: tilt of its top forward
        return np.degrees(np.arctan2((v * fwd).sum(1), v[:, 2]))
    def tilt_down(v):                                                 # segment pointing down: tilt of its bottom forward
        return np.degrees(np.arctan2((v * fwd).sum(1), -v[:, 2]))
    out = {'trunk': tilt_up(seg('Hip', 'Thorax'))}
    for name, (a, b), f in (('thigh', ('Hip', 'Knee'), tilt_down), ('shank', ('Knee', 'Ankle'), tilt_down),
                            ('upper arm', ('Shoulder', 'Elbow'), tilt_down)):
        out[name] = np.nanmean([f(seg(s + a if a != 'Hip' else s + 'Hip', s + b)) for s in 'RL'], 0)
    out['hip flex'] = out['trunk'] + out['thigh']                     # thigh forward + trunk forward = closing the hip
    out['knee flex'] = out['thigh'] - out['shank']
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_dir')
    ap.add_argument('config')
    ap.add_argument('--target', choices=['mocap', 'tri'], default='tri')
    args = ap.parse_args()
    tname = 'mocap_h36m.npz' if args.target == 'mocap' else 'openpose_tri_h36m.npz'
    pooled = {}
    for p in sorted(glob.glob(f'{args.out_dir}/*/*/Analysis/PnP/openpose/{args.config}/*_pnp.npz')):
        u, a = p[len(args.out_dir) + 1:].split(os.sep)[:2]
        cam = os.path.basename(p).split('_')[0]
        base = f'{args.out_dir}/{u}/{a}/Analysis'
        d = np.load(p, allow_pickle=True)
        tz = np.load(f'{base}/H36M/{tname}')
        src = np.load(f'{base}/keypoints/openpose/{cam}_2d.npz')['source_frame_idx']
        R, t = d['cam_L_ext'][:3, :3], d['cam_L_ext'][:3, 3]
        X = (d['kps_H36M_placed'].astype(float) * 1000.0 - t) @ R
        n = min(len(X), len(src))
        _, ix, it = np.intersect1d(src[:n], tz['source_frame_idx'], return_indices=True)
        X, T = X[ix], tz['kps3d'][it].astype(float)
        lat = T[:, J['LHip']] - T[:, J['RHip']]
        lat[:, 2] = 0
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        fwd = np.cross(lat, [0, 0, 1.0])
        view = R[2]
        rel = np.degrees(np.arccos(np.clip(-(np.nanmedian(fwd, 0) @ view), -1, 1)))
        tm, tt = tilts(X, fwd), tilts(T, fwd)
        cells = []
        for k in tm:
            ok = np.isfinite(tm[k]) & np.isfinite(tt[k])
            if ok.sum() < 20:
                cells.append(f'{k}: -')
                continue
            s, c = np.polyfit(tt[k][ok], tm[k][ok], 1)
            lo, hi = np.percentile(tt[k][ok], [10, 90])
            cells.append(f'{k} {s:4.2f} ({(s - 1) * lo + c:+3.0f}/{(s - 1) * hi + c:+3.0f})')
            pooled.setdefault((k, 'front' if rel < 60 else 'side'), []).append((tt[k][ok], tm[k][ok]))
        print(f'{a} cam{cam} ({rel:3.0f} deg off front): ' + ' | '.join(cells))
    print('\npooled: slope model-on-target, error (deg) at target 10th / 90th pct, target range')
    for (k, where), v in sorted(pooled.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tt = np.concatenate([a for a, _ in v]); tm = np.concatenate([b for _, b in v])
        s, c = np.polyfit(tt, tm, 1)
        lo, hi = np.percentile(tt, [10, 90])
        print(f'  {where:5s} {k:10s} slope {s:4.2f}  err {(s - 1) * lo + c:+5.1f} / {(s - 1) * hi + c:+5.1f} deg   target {lo:+5.1f}..{hi:+5.1f} deg')


if __name__ == '__main__':
    main()
