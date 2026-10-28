#!/usr/bin/env python3
"""blazepose_vs_ours.py -- MediaPipe BlazePose's own 3D against our best lifted
skeleton, both Procrustes-aligned to mocap frame by frame (what PA-MPJPE
scores), animated from the front and the side.

Ours: step_4b's kps_H36M_placed for each config under PnP/{detector}/ (YOLO by
default -- see the best-result convention); the config with the lowest mean
PA-MPJPE over the cameras is drawn unless --config names one.
BlazePose: heavy model (model_complexity=2), video mode with smoothing, as the
face-blur study runs it, fed the same source frames step_4b used (its 2d.npz
source_frame_idx); its 33 world landmarks -> COCO-17 -> H36M-17.
Scored on mocap's 15 marker-backed joints; joint-convention differences are
left in, this is a visual comparison.

Layout per frame: camera image | frontal view | sagittal view, PA-MPJPE trace
underneath.  The views are in mocap's body frame (lab Z up, facing direction
from the mean hip axis); each skeleton is the per-frame Procrustes fit, centred
horizontally on the mocap pelvis, so the jump height stays visible.

    python blazepose_vs_ours.py [--cams 00 05 07] [--render-cam 07] [--config C6] [--align pa|root]

Outputs in {trial out}/blazepose/: mp_{cam}.npz (cached BlazePose), scores.txt,
{cam}_{config}_{align}.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(REPO, 'pipeline'))
sys.path.insert(0, os.path.join(REPO, 'analysis', 'face_blur'))

from common import coco_to_h36m, procrustes_fit, write_video   # noqa: E402
from utils.calibration import load_calib                        # noqa: E402
from utils.triangulate import project                           # noqa: E402

DATA = os.path.join(REPO, 'data', 'sampleBiocv')
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]   # detectors.py's map
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
COL = {'gt': '#3a3a3a', 'ours': '#1f6fd1', 'mp': '#e8711a'}


def paths(user, action, detector):
    out = os.path.join(DATA, 'out', action)
    ana = os.path.join(out, 'step4b', user, action, 'Analysis')
    return dict(trial=os.path.join(DATA, action), ana=ana,
                calib=lambda c: os.path.join(DATA, f'{c}.mp4-mocAligned.calib'),
                twod=lambda c: os.path.join(ana, 'keypoints', detector, f'{c}_2d.npz'),
                pnp=lambda cfg, c: os.path.join(ana, 'PnP', detector, cfg, f'{c}_pnp.npz'),
                out=os.path.join(out, 'blazepose'))


def run_mediapipe(video, frames, out_path):
    """BlazePose heavy over the given source frames of one video -> world (T,17,3) H36M mm, NaN undetected."""
    if os.path.exists(out_path):
        return np.load(out_path)['world']
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video)
    world = np.full((len(frames), 17, 3), np.nan, np.float32)
    want = {int(f): i for i, f in enumerate(frames)}
    for f in range(int(frames[-1]) + 1):
        if f not in want:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        r = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if r.pose_world_landmarks is not None:
            w = np.array([(p.x, p.y, p.z, 1.0) for p in r.pose_world_landmarks.landmark], np.float32)
            world[want[f]] = coco_to_h36m(w[MP_TO_COCO][None])[0, :, :3] * 1000.0
        if want[f] % 50 == 0:
            print(f'  {os.path.basename(video)} {want[f]}/{len(frames)}', flush=True)
    cap.release()
    pose.close()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, world=world, source_frame_idx=frames)
    return world


def align(pred, gt, mask, mode):
    """Per-frame alignment of pred onto gt over the masked joints; returns aligned (T,17,3) and error (T,)."""
    A = np.full_like(gt, np.nan)
    err = np.full(len(gt), np.nan)
    for f in range(len(gt)):
        m = mask & np.isfinite(pred[f]).all(-1) & np.isfinite(gt[f]).all(-1)
        if m.sum() < 4:
            continue
        if mode == 'pa':
            s, R, t = procrustes_fit(pred[f, m], gt[f, m])
            A[f] = s * pred[f] @ R + t
        else:                                   # root: translation only, pelvis on pelvis
            A[f] = pred[f] + (gt[f, 0] - pred[f, 0])
        err[f] = np.linalg.norm(A[f, m] - gt[f, m], axis=-1).mean()
    return A, err


def body_axes(gt):
    """Lab Z up; right = mean RHip-LHip in the floor plane; forward = up x right."""
    up = np.array([0., 0., 1.])
    r = np.nanmean(gt[:, 1] - gt[:, 4], axis=0)
    r[2] = 0
    r /= np.linalg.norm(r)
    return r, np.cross(up, r), up


def config_ids(ana, detector):
    d = os.path.join(ana, 'PnP', detector)
    return sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', default='P03')
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--detector', default='yolo')
    ap.add_argument('--cams', nargs='+', default=['00', '05', '07'])
    ap.add_argument('--render-cam', default='07', help='07 is front-on for the jump')
    ap.add_argument('--config', default=None, help='step_4b config to draw (default: best mean PA-MPJPE)')
    ap.add_argument('--align', default='pa', choices=['pa', 'root'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    P = paths(args.user, args.action, args.detector)

    m = np.load(os.path.join(P['trial'], 'Analysis', 'H36M', 'mocap_h36m.npz'))
    mocap, mask = m['kps3d'].astype(float), np.asarray(m['valid_joint_mask'], bool)

    gt, mp_w, ours, sfi = {}, {}, {}, {}
    for c in args.cams:
        s = np.load(P['twod'](c))['source_frame_idx'].astype(int)
        sfi[c], gt[c] = s, mocap[s]
        # both predictions are in camera axes (BlazePose world: x right, y down, z away, like the camera);
        # rotate them into lab axes so root alignment means something -- PA is unaffected
        R = load_calib(P['calib'](c))[3][:3, :3]
        mp_w[c] = run_mediapipe(os.path.join(P['trial'], f'{c}.mp4'), s, os.path.join(P['out'], f'mp_{c}.npz')) @ R
        for cfg in config_ids(P['ana'], args.detector):
            if os.path.exists(P['pnp'](cfg, c)):
                ours[cfg, c] = np.load(P['pnp'](cfg, c))['kps_H36M_placed'].astype(float) * 1000.0 @ R

    # scores, every config, both alignments
    lines = [f'{args.user}/{args.action}, {args.detector}, mean error over the 15 mocap joints (mm)', '',
             f'{"method":>10} {"align":>5} ' + ' '.join(f'{c:>7}' for c in args.cams) + '    mean']
    table = {}
    for mode in ('pa', 'root'):
        rows = [('BlazePose', {c: mp_w[c] for c in args.cams})]
        rows += [(cfg, {c: ours[cfg, c] for c in args.cams})
                 for cfg in config_ids(P['ana'], args.detector) if all((cfg, c) in ours for c in args.cams)]
        for name, preds in rows:
            e = [np.nanmean(align(preds[c][:len(gt[c])], gt[c][:len(preds[c])], mask, mode)[1]) for c in args.cams]
            table[name, mode] = np.mean(e)
            lines.append(f'{name:>10} {mode:>5} ' + ' '.join(f'{x:7.1f}' for x in e) + f' {np.mean(e):7.1f}')
        lines.append('')
    cfgs = [k[0] for k in table if k[1] == 'pa' and k[0] != 'BlazePose']
    best = args.config or min(cfgs, key=lambda k: (round(table[k, 'pa'], 1), k))
    lines.append(f'drawn: {best}' + ('' if args.config else ' (lowest mean PA-MPJPE)'))
    os.makedirs(P['out'], exist_ok=True)
    with open(os.path.join(P['out'], 'scores.txt'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    render(args, P, best, gt[args.render_cam], ours[best, args.render_cam], mp_w[args.render_cam],
           sfi[args.render_cam], mask)


def render(args, P, best, gt, ours, mp_w, sfi, mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = min(len(gt), len(ours), len(mp_w))
    gt, ours, mp_w, sfi = gt[:T], ours[:T], mp_w[:T], sfi[:T]
    A_o, e_o = align(ours, gt, mask, args.align)
    A_m, e_m = align(mp_w, gt, mask, args.align)
    fps = float(np.load(P['twod'](args.render_cam))['fps'])
    t = np.arange(T) / fps

    r, f, up = body_axes(gt)
    def views(X, pelvis):                                       # noqa: E306
        Xc = X - pelvis * np.array([1, 1, 0])
        return {'front': np.stack([Xc @ -r, Xc @ up], -1), 'side': np.stack([Xc @ f, Xc @ up], -1)}
    floor = np.nanmin(gt[:, [3, 6], 2]) - 80
    top = np.nanmax(gt[:, mask, 2]) + 350
    half = (top - floor) / 2 * 0.75

    # camera crop around the projected mocap over the whole trial
    cw, chh, K, L, dist = load_calib(P['calib'](args.render_cam))
    uv = project(gt[:, mask].reshape(-1, 3), K, dist, L[:3, :3], L[:3, 3]).reshape(T, -1, 2)
    lo, hi = np.nanmin(uv, (0, 1)), np.nanmax(uv, (0, 1))
    pad = 0.12 * (hi - lo).max()
    x0, y0 = int(max(lo[0] - pad, 0)), int(max(lo[1] - pad, 0))
    x1, y1 = int(min(hi[0] + pad, cw)), int(min(hi[1] + pad, chh))

    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], hspace=0.22, wspace=0.08,
                          left=0.05, right=0.98, top=0.93, bottom=0.07)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_v = {'front': fig.add_subplot(gs[0, 1]), 'side': fig.add_subplot(gs[0, 2])}
    ax_e = fig.add_subplot(gs[1, :])
    lab = 'Procrustes-aligned (PA)' if args.align == 'pa' else 'root-aligned (translation only)'
    fig.suptitle(f'{args.action}  cam {args.render_cam}  |  mocap (grey) vs ours {best} {args.detector} (blue) '
                 f'vs BlazePose world (orange)  |  {lab}', fontsize=13)

    img_h = ax_img.imshow(np.zeros((y1 - y0, x1 - x0, 3), np.uint8))
    ax_img.set_axis_off()
    ax_img.set_title(f'camera {args.render_cam} (input to both)', fontsize=11)
    lines = {}
    for v, ax in ax_v.items():
        ax.set_xlim(-half, half)
        ax.set_ylim(floor, top)
        ax.set_aspect('equal')
        ax.axhline(floor + 80, color='#bbbbbb', lw=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('frontal (subject facing you)' if v == 'front' else 'sagittal (subject facing right)',
                     fontsize=11)
        for k, lw, z in (('gt', 5, 1), ('ours', 2.5, 3), ('mp', 2.5, 2)):
            lines[v, k] = [ax.plot([], [], color=COL[k], lw=lw, alpha=0.55 if k == 'gt' else 0.95,
                                   solid_capstyle='round', zorder=z)[0] for _ in BONES]
    ax_e.plot(t, e_o, color=COL['ours'], lw=1.6, label=f'ours {best}: mean {np.nanmean(e_o):.0f} mm')
    ax_e.plot(t, e_m, color=COL['mp'], lw=1.6, label=f'BlazePose: mean {np.nanmean(e_m):.0f} mm')
    ax_e.set_xlim(0, t[-1])
    ax_e.set_ylim(0, np.nanpercentile(np.r_[e_o, e_m], 99.5) * 1.1)
    ax_e.set_ylabel(('PA-MPJPE' if args.align == 'pa' else 'root-aligned MPJPE') + ' (mm)')
    ax_e.set_xlabel('time (s)')
    ax_e.legend(loc='upper left', frameon=False, fontsize=10)
    ax_e.spines[['top', 'right']].set_visible(False)
    ax_e.grid(axis='y', color='#e5e5e5')
    cursor = ax_e.axvline(0, color='#555555', lw=1)
    txt = {k: ax_e.text(0.995, 0.95 - 0.16 * i, '', transform=ax_e.transAxes, color=COL[k], ha='right',
                                 fontsize=11, va='top', family='monospace')
           for i, k in enumerate(('ours', 'mp'))}

    out = args.out or os.path.join(P['out'], f'{args.render_cam}_{best}_{args.align}.mp4')
    W, H = fig.canvas.get_width_height()
    proc = write_video(out, fps, W, H)
    cap = cv2.VideoCapture(os.path.join(P['trial'], f'{args.render_cam}.mp4'))
    want = {int(s): i for i, s in enumerate(sfi)}
    for src in range(int(sfi[-1]) + 1):
        if src not in want:
            cap.grab()
            continue
        ok, img = cap.read()
        if not ok:
            break
        i = want[src]
        img_h.set_data(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        pel = gt[i, 0]
        for k, X in (('gt', gt[i]), ('ours', A_o[i]), ('mp', A_m[i])):
            V = views(X, pel)
            for v in ax_v:
                for ln, (a, b) in zip(lines[v, k], BONES):
                    ln.set_data(V[v][[a, b], 0], V[v][[a, b], 1])
        cursor.set_xdata([t[i], t[i]])
        txt['ours'].set_text(f'ours      {e_o[i]:5.0f} mm' if np.isfinite(e_o[i]) else 'ours        --')
        txt['mp'].set_text(f'BlazePose {e_m[i]:5.0f} mm' if np.isfinite(e_m[i]) else 'BlazePose   --')
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        proc.stdin.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR).tobytes())
    cap.release()
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
