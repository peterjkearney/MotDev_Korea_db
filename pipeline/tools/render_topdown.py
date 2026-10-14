#!/usr/bin/env python3
"""render_topdown.py -- step_5's video + top-down layout, for several step_4b configurations.

LEFT   the source video with the detector's 2D keypoints (subject's left blue, right red, midline white)
RIGHT  top-down (X vs depth, this camera's frame, metres), this frame only, fixed axes: the mocap
       skeleton (thick, near-black) and each configuration's placed H36M skeleton in its own colour,
       with its clip MPJPE against mocap in the legend.  Same scoring as render_mesh.py.
       --split: one top-down panel per configuration (each with mocap, shared axes), the video
       cropped about the subject so the output keeps the single-panel width.

    reads   {OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{config}/{cam}_pnp.npz
            {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
            {OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz
            {TRIAL_DIR}/{user}/{action}/{cam}.mp4
    writes  {OUT_DIR}/{user}/{action}/Analysis/diagnostics/{detector}/{cam}_topdown_{configs}.mp4

    python3 tools/render_topdown.py --user P03 --action P03_CMJM_01 --cam 07 --detector yolo --configs C0,C5,C6 --smooth C0
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import TRIAL_DIR, analysis_dir, twod_path, pnp_path, mocap_path  # noqa: E402
from utils.metrics import mpjpe  # noqa: E402
from utils.topdown import TopDown, H36M_LIMBS, robust_limits  # noqa: E402

ROW_HEX = ['#db6154', '#ebad3d', '#4fa1db', '#73bf73', '#b380d9']      # render_mesh.py's row colours
CONFIG_HEX = {'C0': ROW_HEX[0], 'C5': ROW_HEX[1], 'C6': ROW_HEX[2]}     # as in the C0/C5/C6 mesh video
MOCAP_HEX = '#14181d'                                                  # near-black on the white panel
LEFT, RIGHT = {4, 5, 6, 11, 12, 13}, {1, 2, 3, 14, 15, 16}
KP_BGR = {'L': (230, 140, 40), 'R': (60, 60, 230), 'C': (235, 235, 235)}
TD_PANEL_W = 640


def side(j):
    return 'L' if j in LEFT else 'R' if j in RIGHT else 'C'


def hex_to_bgr(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def draw_2d(img, kp, th):
    seen = kp[:, 2] > 0
    for a, b in H36M_LIMBS:
        if seen[a] and seen[b]:
            cv2.line(img, tuple(np.round(kp[a, :2]).astype(int)), tuple(np.round(kp[b, :2]).astype(int)),
                     KP_BGR[side(b)], th, cv2.LINE_AA)
    for j in np.where(seen)[0]:
        cv2.circle(img, tuple(np.round(kp[j, :2]).astype(int)), th * 2, KP_BGR[side(j)], -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cam', required=True)
    ap.add_argument('--configs', default='C0,C5,C6', help='comma-separated step_4b configurations')
    ap.add_argument('--detector', default='yolo')
    ap.add_argument('--smooth', default='C0', help="configurations drawn from their RTS-smoothed skeleton ('pnp' placement)")
    ap.add_argument('--zoom', action='store_true',
                    help='frame the axes on the subject (camera off-panel) instead of from the camera out')
    ap.add_argument('--split', action='store_true', help='one top-down panel per configuration')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    smooth = {c.strip() for c in args.smooth.split(',') if c.strip()}

    tw = np.load(twod_path(args.user, args.action, args.detector, args.cam))
    sfi, kp2d, fps = tw['source_frame_idx'].astype(int), tw['h36m_2d'], float(tw['fps'])
    rows = []
    for cid in configs:
        d = np.load(pnp_path(args.user, args.action, args.detector, args.cam, cid))
        cfg = json.loads(str(d['config_json'])) if 'config_json' in d.files else {}
        rows.append(dict(cid=cid, H=d['kps_H36M_smooth' if cid in smooth else 'kps_H36M_placed'].astype(np.float64),
                         note=cfg.get('note', ''), pnp=d, hex=CONFIG_HEX.get(cid, ROW_HEX[len(rows) % 5])))
    T = min(len(sfi), *(len(r['H']) for r in rows))
    ref = rows[0]['pnp']
    L_ext, K, W, H = ref['cam_L_ext'], ref['cam_K'], int(ref['cam_w']), int(ref['cam_h'])

    g = np.load(mocap_path(args.user, args.action))
    gt_lab, gt_ok = g['kps3d'].astype(np.float64), g['valid_joint_mask'].astype(bool)
    gt_cam = np.full((T, 17, 3), np.nan)
    inside = sfi[:T] < len(gt_lab)
    gt_cam[inside] = (gt_lab[sfi[:T][inside]] @ L_ext[:3, :3].T + L_ext[:3, 3]) / 1000.0
    for r in rows:
        r['mpjpe'] = 1000 * float(np.nanmean(mpjpe(r['H'][:T], gt_cam, np.broadcast_to(gt_ok, gt_cam.shape[:2]))[0]))
        print(f'{r["cid"]}: MPJPE vs mocap {r["mpjpe"]:.1f} mm')

    xz = [gt_cam[..., [0, 2]]] + [r['H'][:T, :, [0, 2]] for r in rows]
    xlim, zlim = robust_limits(xz)
    if args.zoom:                                                        # equal aspect, filling the tall panel
        allxz = np.concatenate([a.reshape(-1, 2) for a in xz])
        allxz = allxz[np.isfinite(allxz).all(1)]
        (x0, z0), (x1, z1) = np.percentile(allxz, 0.5, 0), np.percentile(allxz, 99.5, 0)
        zspan = max(z1 - z0 + 1.0, (x1 - x0 + 0.6) / 0.6)
        zc, xc = (z0 + z1) / 2, (x0 + x1) / 2
        zlim, xlim = (zc - zspan / 2, zc + zspan / 2), (xc - 0.3 * zspan, xc + 0.3 * zspan)
    entry = lambda r: (f'{r["cid"]}: {r["note"].split(" (")[0]}  ({r["mpjpe"]:.0f} mm)', r['hex'], 1.0)
    half_fov = float(np.arctan(W / (2 * K[0, 0])))
    title = f'top-down, camera {args.cam}  ({args.detector} 2D)\n{args.user} / {args.action}'
    groups = [[r] for r in rows] if args.split else [rows]               # the configurations in each panel
    panels = [TopDown(TD_PANEL_W, H, half_fov, xlim, zlim,
                      (f'{grp[0]["cid"]} vs mocap, camera {args.cam}  ({args.detector} 2D)\n{args.user} / {args.action}'
                       if args.split else title), [('mocap', MOCAP_HEX, 1.0)] + [entry(r) for r in grp])
              for grp in groups]
    # the video, cropped about the subject when there are extra panels: output width stays W + 6 + panel
    crop_w = W - (len(panels) - 1) * (TD_PANEL_W + 6)
    vis = kp2d[:T, :, 2] > 0
    cx = float(np.median([kp2d[i, vis[i], 0].mean() for i in range(T) if vis[i].any()]))
    cx0 = int(np.clip(round(cx - crop_w / 2), 0, W - crop_w))

    video = os.path.join(TRIAL_DIR, args.user, args.action, f'{args.cam}.mp4')
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'cannot open {video}'
    out = args.out or os.path.join(analysis_dir(args.user, args.action), 'diagnostics', args.detector,
                                   f'{args.cam}_topdown_{"split_" if args.split else ""}{"_".join(configs)}{"_zoom" if args.zoom else ""}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Wout = crop_w + len(panels) * (6 + TD_PANEL_W)
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{Wout}x{H}', '-r', f'{fps:g}', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                           '-crf', '18', out], stdin=subprocess.PIPE) if shutil.which('ffmpeg') else None
    writer = None if ff else cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (Wout, H))
    sep = np.zeros((H, 6, 3), np.uint8)
    fs, th = max(0.5, W / 1920), max(2, round(H / 360))
    fi = 0
    for i in range(T):
        while fi < sfi[i]:                                               # decode only the analysis frames
            cap.grab(); fi += 1
        ok, canvas = cap.read(); fi += 1
        if not ok:
            break
        draw_2d(canvas, kp2d[i], th)
        canvas = np.ascontiguousarray(canvas[:, cx0:cx0 + crop_w])
        cv2.putText(canvas, f'{args.detector.upper()} 2D keypoints', (20, int(45 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1 * fs, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f'cam {args.cam}  frame {sfi[i]}  t={i / fps:.2f}s', (20, H - int(20 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (255, 255, 255), 2, cv2.LINE_AA)
        parts = [canvas]
        for panel, grp in zip(panels, groups):
            skels = []
            if np.isfinite(gt_cam[i]).any():                             # thick: the configurations sit on it
                skels.append((gt_cam[i][:, [0, 2]], gt_ok, hex_to_bgr(MOCAP_HEX), 5))
            for r in grp:
                skels.append((r['H'][i][:, [0, 2]], np.ones(17, bool), hex_to_bgr(r['hex']), 2))
            parts += [sep, panel.render(skels)]
        frame = np.concatenate(parts, axis=1)
        if ff:
            ff.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
    cap.release()
    if ff:
        ff.stdin.close(); ff.wait()
    else:
        writer.release()
    print('wrote', out)


if __name__ == '__main__':
    main()
