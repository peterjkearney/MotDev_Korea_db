#!/usr/bin/env python3
"""render_compare.py -- animated MotionBERT-vs-ground-truth comparison, one camera.

Left panel: the camera's view.  OpenPose's own 2D detections (BODY_25 from
the {cam}_2d.npz, i.e. straight from the JSON), with the placed MotionBERT
skeleton and the ground-truth skeleton both projected into the same image
through this camera's intrinsics.  The source video is used as the background
when it exists next to the trial; otherwise a dark canvas.

Right panels: the two 3D skeletons seen from the side (the camera's view
rotated 90 degrees about the vertical axis, so depth runs left-right) and from
above (camera's right across, depth up the page).  Axes are fixed over the
clip and centred on the ground truth's mean position, so motion is real.

The ground truth is what step_8 scores against: the leave-one-out
triangulated target (--gt triangulated, Korea and BioCV) or mocap (--gt
mocap, BioCV).  Everything is drawn in the lab frame: mocap's frame on BioCV,
the fitted floor frame on Korea (z up in both).

    python3 tools/render_compare.py --user B010 --action B010_GMS_1_1 --camera 1
    python3 tools/render_compare.py --user User08 --action P08_CMJM_01 --camera 07 --gt mocap
"""
import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import TRIAL_DIR, mocap_path
from utils.calibration import reproject

H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
BODY25_LIMBS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
                (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
                (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
EVAL = [1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 0]      # joints scored (no Spine/Nose/Head)

# reference palette: categorical slots 1-3
C_MB, C_OP, C_GT = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def draw_2d(img, pts, limbs, col, valid, r=4, t=2):
    for a, b in limbs:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), col, t, cv2.LINE_AA)
    for p in pts[valid]:
        cv2.circle(img, tuple(p.astype(int)), r, col, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--gt', choices=['triangulated', 'mocap'], default='triangulated')
    ap.add_argument('--gt-source', choices=['all', 'loo'], default='all',
                    help='triangulated only: the all-camera skeleton (complete on more frames; '
                         'default, for looking) or the leave-one-out one (what step_8 scores)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--dpi', type=int, default=100)
    args = ap.parse_args()
    cam = args.camera

    trial = os.path.join(TRIAL_DIR, args.user, args.action)
    kp = os.path.join(trial, 'Analysis', 'keypoints')
    d2 = np.load(os.path.join(kp, f'{cam}_2d.npz'))
    pnp = np.load(os.path.join(kp, 'PnP', f'{cam}_pnp.npz'))
    gtf = np.load(os.path.join(kp, 'openpose_tri_h36m.npz') if args.gt == 'triangulated'
                  else mocap_path(args.user, args.action))

    sfi = d2['source_frame_idx']
    fps = float(d2['fps']) if np.isfinite(float(d2['fps'])) else 30.0
    h36m_2d = d2['h36m_2d']
    body25 = d2['body25_2d'] if 'body25_2d' in d2.files else None
    K, L, dist = pnp['cam_K'], pnp['cam_L_ext'], pnp['cam_dist_cv'].reshape(1, -1)
    w, h = int(pnp['cam_w']), int(pnp['cam_h'])
    R, t = L[:3, :3], L[:3, 3]
    mb_cam = pnp['kps_H36M_placed'].astype(float)                 # (T,17,3) m, camera frame

    if args.gt == 'triangulated' and args.gt_source == 'loo':
        cams = [str(c) for c in gtf['cameras']]
        gt_lab = gtf['kps3d_loo'][cams.index(cam)]                # mm, lab
    else:
        gt_lab = gtf['kps3d']
    ok = sfi < len(gt_lab)                                         # video may outrun the target
    g = np.full((len(sfi), 17, 3), np.nan)
    g[ok] = gt_lab[sfi[ok]]
    gt_lab = g / 1000.0                                            # (T,17,3) m
    T = min(len(mb_cam), len(gt_lab), len(h36m_2d))
    if args.max_frames:
        T = min(T, args.max_frames)
    valid_gt = np.isfinite(gt_lab).all(-1)
    valid_mb = np.isfinite(mb_cam).all(-1)

    # both skeletons in the lab frame; MotionBERT: X_lab = R^T (X_cam - t)
    mb_lab = ((mb_cam * 1000.0 - t) @ R) / 1000.0
    gt_cam = (gt_lab * 1000.0 @ R.T + t) / 1000.0

    # view basis from the camera: heading = optical axis on the floor, right = image x on the floor
    up = np.array([0.0, 0.0, 1.0])
    fwd = R.T @ np.array([0.0, 0.0, 1.0]); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
    right = R.T @ np.array([1.0, 0.0, 0.0]); right[2] = 0; right /= np.linalg.norm(right)

    def side(X):   # depth across, height up
        return np.stack([X @ fwd, X @ up], -1)

    def top(X):    # right across, depth up the page
        return np.stack([X @ right, X @ fwd], -1)

    # axes from the ground truth only: a failed PnP frame must not blow the view up
    ref = gt_lab[valid_gt] if valid_gt.any() else mb_lab[valid_mb]
    centre = np.nanmean(ref.reshape(-1, 3), axis=0)
    span = 1.15 * float(np.nanpercentile(np.abs(ref - centre), 99)) + 0.15

    # per-frame error, same joints step_8 scores
    m = valid_gt[:, EVAL] & valid_mb[:, EVAL]
    per_frame = np.array([np.linalg.norm(mb_cam[i, EVAL][m[i]] - gt_cam[i, EVAL][m[i]], axis=1).mean() * 1000
                          if m[i].any() else np.nan for i in range(T)])

    video = os.path.join(trial, str(d2['video'])) if 'video' in d2.files else ''
    if not os.path.isfile(video):
        video = str(d2['video']) if 'video' in d2.files and os.path.isfile(str(d2['video'])) else ''
    cap = cv2.VideoCapture(video) if video else None
    if cap is not None and not cap.isOpened():
        cap = None
    # which video frame each row is: Korea rows are sync-offset from the video
    # (video_frame_idx); BioCV rows index the native video directly (source_frame_idx)
    vidx = d2['video_frame_idx'] if 'video_frame_idx' in d2.files else sfi
    korea = not os.path.exists(mocap_path(args.user, args.action))
    if cap is not None and korea and 'video_frame_idx' not in d2.files:
        # 2D file predates the video index: rows are sync-offset from the video, so
        # a background drawn from them would be the wrong frame.  Re-run
        # step_1_korea_2d.py (it rewrites the 2D files, leaves PnP/ alone) to get it.
        print('note: no video_frame_idx in the 2D file -- background video skipped')
        cap.release(); cap = None
    vf = {}
    if cap is not None:
        want = set(int(x) for x in vidx[:T] if x >= 0)
        i = 0
        while i <= max(want):
            if i in want:
                ok, fr = cap.read()
                if not ok:
                    break
                vf[i] = fr
            elif not cap.grab():
                break
            i += 1
        cap.release()

    def frame_image(i):
        img = vf.get(int(vidx[i]))
        img = img.copy() if img is not None else np.full((h, w, 3), 24, np.uint8)
        if img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        if body25 is not None:
            p = body25[i, :, :2]; v = body25[i, :, 2] > 0.3
            draw_2d(img, p, BODY25_LIMBS, hex_bgr(C_OP), v, 4, 2)
        else:
            p = h36m_2d[i, :, :2]; v = h36m_2d[i, :, 2] > 0.3
            draw_2d(img, p, H36M_LIMBS, hex_bgr(C_OP), v, 4, 2)
        for X, v, col in ((gt_cam[i], valid_gt[i], hex_bgr(C_GT)), (mb_cam[i], valid_mb[i], hex_bgr(C_MB))):
            if v.any() and (X[v, 2] > 0.05).all():
                uv = np.full((17, 2), np.nan)
                uv[v] = reproject(X[v], K, dist)
                draw_2d(img, uv, H36M_LIMBS, col, v & np.isfinite(uv).all(-1), 5, 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---- figure --------------------------------------------------------------
    fig = plt.figure(figsize=(16, 7), facecolor=SURFACE, dpi=args.dpi)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.2, 1], figure=fig, wspace=0.12, hspace=0.32,
                           left=0.02, right=0.98, top=0.93, bottom=0.07)
    axL = fig.add_subplot(gs[:, 0]); axS = fig.add_subplot(gs[0, 1]); axT = fig.add_subplot(gs[1, 1])
    im = axL.imshow(frame_image(0))
    axL.set_axis_off()
    axL.set_title(f'{args.user} / {args.action} / camera {cam}   '
                  f'OpenPose 2D (orange)   MotionBERT (blue)   ground truth (green)',
                  color=INK, fontsize=10, loc='left')
    txt = axL.text(0.01, 0.02, '', transform=axL.transAxes, color='white', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.5))

    lines = {}
    for ax, name, xl, yl in ((axS, 'side', 'depth from camera (m)', 'height (m)'),
                             (axT, 'top', 'camera right (m)', 'depth from camera (m)')):
        ax.set_facecolor(SURFACE)
        ax.set_aspect('equal')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        ax.set_xlabel(xl, color=INK2, fontsize=8); ax.set_ylabel(yl, color=INK2, fontsize=8)
        ax.set_title({'side': 'side view (camera view rotated 90 deg about vertical)',
                      'top': 'top-down view'}[name], color=INK, fontsize=9, loc='left')
        for key, col in (('gt', C_GT), ('mb', C_MB)):
            lines[(name, key)] = [ax.plot([], [], color=col, lw=2, solid_capstyle='round', alpha=0.9)[0]
                                  for _ in H36M_LIMBS]
        if name == 'side':
            ax.axhline(0, color=INK2, lw=0.8, ls='--')     # the floor
            ax.plot([], [], color=C_GT, lw=2, label='ground truth')
            ax.plot([], [], color=C_MB, lw=2, label='MotionBERT (placed)')
            ax.legend(frameon=False, fontsize=8, loc='upper right', labelcolor=INK2)
    proj = {'side': side, 'top': top}
    c2 = {'side': side(centre[None])[0], 'top': top(centre[None])[0]}
    axS.set_xlim(c2['side'][0] - span, c2['side'][0] + span); axS.set_ylim(-0.05, 2 * span)
    axT.set_xlim(c2['top'][0] - span, c2['top'][0] + span); axT.set_ylim(c2['top'][1] - span, c2['top'][1] + span)

    def update(i):
        im.set_data(frame_image(i))
        e = per_frame[i]
        txt.set_text(f'frame {i}/{T - 1}' + (f'   MotionBERT vs GT: {e:.0f} mm' if np.isfinite(e) else ''))
        for name in ('side', 'top'):
            for key, X, v in (('gt', gt_lab[i], valid_gt[i]), ('mb', mb_lab[i], valid_mb[i])):
                P = proj[name](X)
                for ln, (a, b) in zip(lines[(name, key)], H36M_LIMBS):
                    if v[a] and v[b]:
                        ln.set_data([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]])
                    else:
                        ln.set_data([], [])
        return [im, txt] + [ln for k in lines for ln in lines[k]]

    out = args.out or os.path.join(trial, 'Analysis', 'diagnostics',
                                   f'compare_{cam}{"" if args.gt == "mocap" else "_tri"}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=T, blit=True)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=3000), dpi=args.dpi)
    print(f'{T} frames @ {fps:g} fps, ground truth present on {int(valid_gt.any(1).sum())} frames, '
          f'median MotionBERT-vs-GT {np.nanmedian(per_frame):.0f} mm -> {out}')


if __name__ == '__main__':
    main()
