#!/usr/bin/env python3
"""sample_conf_overlay.py -- step_5's left panel with OpenPose's confidence beside every joint.

Temporary diagnostic for the sample_hip_ablation.py output: the same layers step_5 draws on
its blank canvas (the detector's BODY_25 2D in yellow, the smoothed PnP-placed H36M skeleton
in blue, the triangulated target projected in orange), zoomed in on the child and slowed
down, plus the OpenPose confidence written next to each 2D joint -- red where it is under
the PnP gate (0.4, step_4), so a joint that did NOT count as a correspondence is obvious.
The hips (MidHip, RHip, LHip) get their names as well, since they are the joints under
suspicion.  A status line gives the three hip confidences and, for the hips, how far the
2D point moved since the previous frame (px) -- a jump with steady confidence is OpenPose
relocating the joint, not the child moving.

    reads   <root>/<variant>/<subject>/<stub>/Analysis/keypoints/openpose/{cam}_2d.npz
            <root>/<variant>/<subject>/<stub>/Analysis/PnP/openpose/{cam}_pnp.npz
            <root>/<variant>/<subject>/<stub>/Analysis/H36M/openpose_tri_h36m.npz
    writes  <root>/videos/<stub>_cam{cam}_<variant>_conf.mp4

With --videos DIR (BioCV: the folder holding {cam}.mp4) the layers go over the footage,
zoomed the same way; without it (Korea, no footage) they go on a blank canvas.

    ~/anaconda3/envs/motEnv/bin/python sample_conf_overlay.py --root data/sampleKorea/out/B023_GMS_2_2
    ~/anaconda3/envs/motEnv/bin/python sample_conf_overlay.py --root ... --variant pnp_all_joints --cameras 1 --slow 4
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'pipeline'))
from utils.calibration import reproject                                  # noqa: E402
from utils.openpose import H36M_NAMES                       # noqa: E402
from utils.topdown import H36M_LIMBS, hex_bgr                             # noqa: E402
from step_5_mocap_comparison import BODY25_LIMBS, DET_HEX, _CANVAS, _COL_2D, _TRI_HEX, _MIN_DEPTH_M   # noqa: E402

PNP_CONF = 0.4                # step_4's gate
HIPS_B25 = {8: 'MidHip', 9: 'RHip', 12: 'LHip'}
FACE_FEET = {0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
SIDE = {2: 'R', 3: 'R', 4: 'R', 9: 'R', 10: 'R', 11: 'R', 15: 'R', 17: 'R', 22: 'R', 23: 'R', 24: 'R',
        5: 'L', 6: 'L', 7: 'L', 12: 'L', 13: 'L', 14: 'L', 16: 'L', 18: 'L', 19: 'L', 20: 'L', 21: 'L'}
SIDE_COL = {'R': (255, 200, 120), 'L': (200, 120, 255)}        # BGR: R light blue, L pink
RED, WHITE, GREY = (40, 40, 230), (255, 255, 255), (160, 160, 160)


def find_trial(root, variant):
    hits = glob.glob(os.path.join(root, variant, '*', '*', 'Analysis'))
    if len(hits) != 1:
        raise SystemExit(f'expected one trial under {os.path.join(root, variant)}, found {len(hits)}')
    a = hits[0]
    return os.path.dirname(os.path.dirname(a)), os.path.basename(os.path.dirname(a)), a


def render(root, variant, cam, args):
    _, stub, adir = find_trial(root, variant)
    twod = np.load(os.path.join(adir, 'keypoints', 'openpose', f'{cam}_2d.npz'))
    pnp = np.load(os.path.join(adir, 'PnP', 'openpose', f'{cam}_pnp.npz'))
    tri = np.load(os.path.join(adir, 'H36M', 'openpose_tri_h36m.npz'))
    kp = twod['body25_2d']                                   # (T,25,3) x, y, conf
    sfi = twod['source_frame_idx'].astype(int)
    fps = float(twod['fps'])
    smooth = pnp['kps_H36M_smooth']
    K, dist_cv, L = pnp['cam_K'], pnp['cam_dist_cv'], pnp['cam_L_ext']
    W, H = int(pnp['cam_w']), int(pnp['cam_h'])
    used = pnp['pnp_joints_used'] if 'pnp_joints_used' in pnp.files else np.ones(17, bool)
    world = tri['kps3d']
    if args.tri == 'loo':
        cams = [str(c) for c in tri['cameras']]
        if cam in cams:
            world = tri['kps3d_loo'][cams.index(cam)]
    tri_cam = (world @ L[:3, :3].T + L[:3, 3]) / 1000.0     # (T,17,3) camera metres
    T = min(len(kp), len(smooth), len(sfi))

    # zoom: the box every seen 2D joint and the projected skeleton ever occupy, at the output aspect
    seen = kp[:T, :, 2] > 0
    pts = [kp[:T][seen][:, :2]]
    for t in range(T):
        P = smooth[t]
        if np.isfinite(P).all() and (P[:, 2] > _MIN_DEPTH_M).all():
            pts.append(reproject(P, K, dist_cv))
    pts = np.concatenate(pts)
    x0, y0 = np.percentile(pts, 1, axis=0) - 80
    x1, y1 = np.percentile(pts, 99, axis=0) + 80
    out_h = args.height
    out_w = int(round(out_h * 16 / 9))
    scale = min(out_w / (x1 - x0), out_h / (y1 - y0))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def tf(p):
        return (int(round((p[0] - cx) * scale + out_w / 2)), int(round((p[1] - cy) * scale + out_h / 2)))

    def text(img, s, org, col, size=0.5, thick=1):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, col, thick, cv2.LINE_AA)

    cap = None
    if args.videos:
        vp = os.path.join(args.videos, f'{cam}.mp4')
        cap = cv2.VideoCapture(vp) if os.path.exists(vp) else None
        if cap is None:
            print(f'  no {vp}: blank canvas')
    M = np.array([[scale, 0, out_w / 2 - cx * scale], [0, scale, out_h / 2 - cy * scale]])   # the same zoom, for the footage
    next_frame = 0

    os.makedirs(os.path.join(root, 'videos'), exist_ok=True)
    out_path = os.path.join(root, 'videos', f'{stub}_cam{cam}_{variant}_conf.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / args.slow, (out_w, out_h))
    prev_hips = None
    for t in range(T):
        fi = sfi[t]
        img = None
        if cap is not None:                                       # decode forward to native frame fi
            while next_frame < fi and cap.grab():
                next_frame += 1
            ok, frame = cap.read()
            if ok:
                next_frame = fi + 1
                img = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=_CANVAS)
                img = (img * 0.6).astype(np.uint8)                # dimmed, so the labels read
        if img is None:
            img = np.full((out_h, out_w, 3), _CANVAS, np.uint8)
        # triangulated target, projected (orange)
        if fi < len(tri_cam):
            T3 = tri_cam[fi]
            okj = np.isfinite(T3).all(-1) & (T3[:, 2] > _MIN_DEPTH_M)
            if okj.any():
                uv = np.full((17, 2), np.nan)
                uv[okj] = reproject(T3[okj], K, dist_cv)
                for a, b in H36M_LIMBS:
                    if okj[a] and okj[b]:
                        cv2.line(img, tf(uv[a]), tf(uv[b]), hex_bgr(_TRI_HEX), 2, cv2.LINE_AA)
                for j in np.where(okj)[0]:
                    cv2.circle(img, tf(uv[j]), 7, hex_bgr(_TRI_HEX), 2, cv2.LINE_AA)
        # the smoothed PnP-placed skeleton (blue); its hips ringed so they stand out
        P = smooth[t]
        placed = np.isfinite(P).all() and (P[:, 2] > _MIN_DEPTH_M).all()
        if placed:
            uv = reproject(P, K, dist_cv)
            for a, b in H36M_LIMBS:
                cv2.line(img, tf(uv[a]), tf(uv[b]), hex_bgr(DET_HEX['openpose']), 2, cv2.LINE_AA)
            for j in range(17):
                cv2.circle(img, tf(uv[j]), 4, hex_bgr(DET_HEX['openpose']), -1, cv2.LINE_AA)
            for j in (0, 1, 4):                                   # ringed: the H36M hips, the joints under suspicion
                cv2.circle(img, tf(uv[j]), 10, hex_bgr(DET_HEX['openpose']), 1, cv2.LINE_AA)
        # OpenPose BODY_25 (yellow) with the confidence beside every joint
        k = kp[t]
        seen_t = k[:, 2] > 0
        for a, b in BODY25_LIMBS:
            if seen_t[a] and seen_t[b]:
                cv2.line(img, tf(k[a]), tf(k[b]), _COL_2D, 2, cv2.LINE_AA)
        for j in np.where(seen_t)[0]:
            if j in FACE_FEET and not args.all_joints:
                continue
            p = tf(k[j])
            c = float(k[j, 2])
            cv2.circle(img, p, 5, _COL_2D, -1, cv2.LINE_AA)
            col = RED if c < PNP_CONF else WHITE
            label = f'{c:.2f}' + (f' {HIPS_B25[j]}' if j in HIPS_B25 else '')
            text(img, label, (p[0] + 7, p[1] - 6), col, 0.48, 1)
            side = SIDE.get(j)                                    # R / L above the value, so a swap is visible
            if side:
                text(img, side, (p[0] + 7, p[1] - 22), SIDE_COL[side], 0.5, 1)
        # status lines
        hips = k[[8, 9, 12]]
        jump = ''
        if prev_hips is not None:
            d = np.linalg.norm(hips[:, :2] - prev_hips[:, :2], axis=1)
            d[(hips[:, 2] <= 0) | (prev_hips[:, 2] <= 0)] = np.nan
            jump = '   moved since last frame: ' + '  '.join(f'{n} {v:.0f}px' if np.isfinite(v) else f'{n} -'
                                                              for n, v in zip(('MidHip', 'RHip', 'LHip'), d))
        prev_hips = hips
        left_out = [H36M_NAMES[j] for j in np.where(~used)[0]]
        text(img, f'cam {cam}   frame {fi}   t={t / fps:.2f}s   {stub}   PnP ' + ('on all joints' if not left_out else 'without ' + ', '.join(left_out)),
             (12, 28), WHITE, 0.7, 2)
        text(img, 'yellow: OpenPose BODY_25 + confidence (red: under the PnP gate %.1f; R/L = the side OpenPose assigned)   blue: H36M from MotionBERT, PnP-placed, smoothed (rings: its Hip, RHip, LHip)   orange: triangulated target' % PNP_CONF,
             (12, 56), GREY, 0.5, 1)
        text(img, 'hip confidence  MidHip %.2f  RHip %.2f  LHip %.2f' % tuple(hips[:, 2]) + jump, (12, 84), WHITE, 0.55, 1)
        if not placed:
            text(img, 'no PnP skeleton this frame', (12, 112), RED, 0.6, 2)
        writer.write(img)
    writer.release()
    if cap is not None:
        cap.release()
    return f'{T} frames -> {out_path}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.join(HERE, 'data', 'sampleKorea', 'out', 'B023_GMS_2_2'),
                    help="sample_hip_ablation.py's output folder for the rep")
    ap.add_argument('--variant', default='pnp_no_hips', help='which PnP variant to draw the blue skeleton from')
    ap.add_argument('--cameras', default='1,2,3')
    ap.add_argument('--videos', default=None, help='folder holding {cam}.mp4, to draw over the footage (BioCV)')
    ap.add_argument('--tri', choices=['all', 'loo'], default='all')
    ap.add_argument('--slow', type=float, default=3.0, help='slow-down factor (3 = one third speed)')
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--all-joints', action='store_true', help='label the face and foot joints too')
    args = ap.parse_args()
    for cam in args.cameras.split(','):
        print(f'cam {cam}: {render(os.path.abspath(args.root), args.variant, cam.strip(), args)}', flush=True)


if __name__ == '__main__':
    main()
