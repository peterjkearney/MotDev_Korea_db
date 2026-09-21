#!/usr/bin/env python3
"""step_5_mocap_comparison.py -- see the result: both detectors' skeletons on the video, and
how their DEPTH compares with the ground truths, which a 2D overlay alone cannot show (a
reprojection can look perfect while the skeleton is a metre too deep).

LEFT   the source video with the smoothed, PnP-placed H36M skeleton from each detector
       projected through the camera: OpenPose in blue, YOLO in green.
RIGHT  top-down (X vs depth, this camera's frame, metres), this frame only, fixed axes:
       mocap (red), the triangulated-OpenPose target (orange), and the same two smoothed
       skeletons.  Clip-level MPJPE against mocap is in the legend.

    reads   {TRIAL_DIR}/{user}/{action}/{cam}.mp4                          (or --videos DIR)
            {OUT_DIR}/.../Analysis/PnP/{openpose,yolo}/{cam}_pnp.npz        (step_4; also K, dist, L_ext)
            {OUT_DIR}/.../Analysis/keypoints/{openpose,yolo}/{cam}_2d.npz   (frame numbers)
            {OUT_DIR}/.../Analysis/H36M/mocap_h36m.npz, openpose_tri_h36m.npz
    writes  {OUT_DIR}/.../Analysis/diagnostics/{cam}_pnp_depth_vs_mocap.mp4
            (rendered on local disk, then copied)

A detector with no PnP result for a camera is simply left out; so is the triangulated target
if step_1b has not run.  The triangulated skeleton drawn is the all-camera solution, which
exists on the most frames; --tri loo draws the leave-one-out target step_8 scores against.

Frames.  Mocap is at the native 200 Hz and both 2D files carry the native frame number of
every row, so everything is looked up by that number; the two detectors' lists can differ by
a row or two at the end (YOLO's comes from the mocap's length, OpenPose's from the video's),
and the video covers their union.

With no --user / --action it does every camera that has a PnP result and whose video is in
the local copy; cameras already rendered are skipped (--force redoes them).

    python3 step_5_mocap_comparison.py --user User03 --action P03_CMJM_01 --cameras 06,07
    python3 step_5_mocap_comparison.py --user User03 --dry-run
"""
import argparse
import os
import shutil
import tempfile

import cv2
import numpy as np

from config import (DETECTORS, TRIAL_DIR as _TRIAL_DIR, OUT_DIR as _OUT_DIR, mocap_path as _mocap_path,
                    tri_target_path as _tri_path, twod_path as _twod_path, pnp_path as _pnp_path,
                    depth_video_path as _video_out, find_cameras, require_out_dir)
from utils.calibration import reproject
from utils.metrics import mpjpe
from utils.topdown import TopDown, H36M_LIMBS, hex_bgr, robust_limits

DET_LABEL = {'openpose': 'OpenPose', 'yolo': 'YOLO'}
DET_HEX = {'openpose': '#2a78d6', 'yolo': '#1baf7a'}
_MOCAP_HEX, _TRI_HEX = '#e0322b', '#eb6834'
_MIN_DEPTH_M = 0.05        # a joint at/behind the camera plane blows up under projection
_TD_PANEL_W = 640


def draw_skeleton(img, pts2d, color, radius=3, thickness=2):
    for a, b in H36M_LIMBS:
        cv2.line(img, tuple(pts2d[a].astype(int)), tuple(pts2d[b].astype(int)), color, thickness, cv2.LINE_AA)
    for p in pts2d:
        cv2.circle(img, tuple(p.astype(int)), radius, color, -1, cv2.LINE_AA)


def render(user, action, cam, video, args):
    """One camera -> the comparison video.  Returns a summary line."""
    dets = {}
    for d in DETECTORS:
        pp, tp = _pnp_path(user, action, d, cam), _twod_path(user, action, d, cam)
        if os.path.exists(pp) and os.path.exists(tp):
            pnp, twod = np.load(pp), np.load(tp)
            n = min(len(pnp['kps_H36M_smooth']), len(twod['source_frame_idx']))
            dets[d] = dict(smooth=pnp['kps_H36M_smooth'][:n], sfi=twod['source_frame_idx'][:n].astype(int),
                           fps=float(twod['fps']), pnp=pnp)
    if not dets:
        raise FileNotFoundError('no PnP result from either detector')
    ref = next(iter(dets.values()))['pnp']                   # K, dist and L_ext are the camera's, same in both
    K, dist_cv, L_ext = ref['cam_K'], ref['cam_dist_cv'], ref['cam_L_ext']
    R_ext, t_ext = L_ext[:3, :3], L_ext[:3, 3]
    to_cam_m = lambda world_mm: (world_mm @ R_ext.T + t_ext) / 1000.0

    mocap = np.load(_mocap_path(user, action))
    assert np.array_equal(mocap['source_frame_idx'], np.arange(len(mocap['source_frame_idx']))), \
        'mocap_h36m.npz looks decimated, not native-resolution -- re-run step_0_load_mocap.py'
    mocap_cam, mocap_joint_ok = to_cam_m(mocap['kps3d']), mocap['valid_joint_mask'].astype(bool)
    tri_cam = tri_joint_ok = None
    if os.path.exists(_tri_path(user, action)):
        tri = np.load(_tri_path(user, action))
        world = tri['kps3d']
        if args.tri == 'loo':
            cams = [str(c) for c in tri['cameras']]
            if cam in cams:
                world = tri['kps3d_loo'][cams.index(cam)]
        tri_cam, tri_joint_ok = to_cam_m(world), np.ones(17, bool)      # draw every joint it solved

    # clip-level error vs mocap, for the legend
    err = {}
    for d, r in dets.items():
        gt = np.full_like(r['smooth'], np.nan)
        inr = r['sfi'] < len(mocap_cam)
        gt[inr] = mocap_cam[r['sfi'][inr]]
        err[d] = 1000 * np.nanmean(mpjpe(r['smooth'], gt, np.broadcast_to(mocap_joint_ok, gt.shape[:2]))[0])

    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native = np.unique(np.concatenate([r['sfi'] for r in dets.values()]))
    row = {d: {int(n): i for i, n in enumerate(r['sfi'])} for d, r in dets.items()}
    shown = [mocap_cam[native[native < len(mocap_cam)]][..., [0, 2]]] + [r['smooth'][..., [0, 2]] for r in dets.values()]
    if tri_cam is not None:
        shown.append(tri_cam[native[native < len(tri_cam)]][..., [0, 2]])
    xlim, zlim = robust_limits(shown)
    legend = [('mocap', _MOCAP_HEX, 1.0)]
    if tri_cam is not None:
        legend.append(('triangulated OpenPose' + (' (leave-one-out)' if args.tri == 'loo' else ' (all cameras)'), _TRI_HEX, 1.0))
    legend += [(f'H36M from {DET_LABEL[d]}, smoothed: MPJPE {err[d]:.0f} mm', DET_HEX[d], 1.0) for d in dets]
    panel = TopDown(_TD_PANEL_W, H, float(np.arctan(W / (2 * K[0, 0]))), xlim, zlim,
                    f'top-down, camera {cam}\n{user} / {action}', legend)

    fps = max(r['fps'] for r in dets.values())
    tmp = os.path.join(tempfile.gettempdir(), f'{action}_{cam}_pnp_depth_vs_mocap.mp4')
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W + 6 + _TD_PANEL_W, H))
    sep = np.zeros((H, 6, 3), dtype=np.uint8)
    fs = max(0.5, W / 1920)
    want, fi, n_written = set(int(n) for n in native), 0, 0
    last = int(native[-1])
    # decode only the frames needed, and write each one immediately (a buffered 1080p clip is GBs)
    while fi <= last:
        if fi not in want:
            if not cap.grab():
                break
            fi += 1
            continue
        ok, canvas = cap.read()
        if not ok:
            break
        skels = []
        if fi < len(mocap_cam):
            skels.append((mocap_cam[fi][:, [0, 2]], mocap_joint_ok & np.isfinite(mocap_cam[fi]).all(-1), hex_bgr(_MOCAP_HEX), 5))   # thick: the target sits on it
        if tri_cam is not None and fi < len(tri_cam):
            skels.append((tri_cam[fi][:, [0, 2]], tri_joint_ok & np.isfinite(tri_cam[fi]).all(-1), hex_bgr(_TRI_HEX), 2))
        lost = []
        for d, r in dets.items():
            i = row[d].get(fi)
            P = r['smooth'][i] if i is not None else None
            if P is not None and np.isfinite(P).all() and (P[:, 2] > _MIN_DEPTH_M).all():
                draw_skeleton(canvas, reproject(P, K, dist_cv), hex_bgr(DET_HEX[d]))
                skels.append((P[:, [0, 2]], np.ones(17, bool), hex_bgr(DET_HEX[d]), 2))
            else:
                lost.append(DET_LABEL[d])
        for k, d in enumerate(dets):
            cv2.putText(canvas, f'H36M from {DET_LABEL[d]} (smoothed)', (20, int(40 * fs * (k + 1))),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fs, hex_bgr(DET_HEX[d]), 2, cv2.LINE_AA)
        if lost:
            cv2.putText(canvas, 'no skeleton this frame: ' + ', '.join(lost), (20, H - int(55 * fs)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f'cam {cam}  native frame {fi}  t={n_written / fps:.2f}s', (20, H - int(20 * fs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * fs, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(np.concatenate([canvas, sep, panel.render(skels)], axis=1))
        n_written += 1
        fi += 1
    cap.release()
    writer.release()
    out_path = _video_out(user, action, cam)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(tmp, out_path)               # cv2 writing through the Drive mount is slow and can truncate
    os.remove(tmp)
    return (f'{n_written} frames, ' + ', '.join(f'{DET_LABEL[d]} {err[d]:.0f} mm' for d in dets)
            + ('' if tri_cam is not None else ', no triangulated target') + f' -> {out_path}')


def main():
    ap = argparse.ArgumentParser(description='Video + top-down comparison of both detectors against the ground truths. '
                                             'With no --user/--action, every camera with a PnP result and a local video; '
                                             'cameras already rendered are skipped.')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with a PnP result')
    ap.add_argument('--videos', default=None, help="dir holding {cam}.mp4 if not {TRIAL_DIR}/{user}/{action} (one trial only)")
    ap.add_argument('--tri', choices=['all', 'loo'], default='all',
                    help="triangulated skeleton drawn: all-camera (default) or this camera's leave-one-out target")
    ap.add_argument('--force', action='store_true', help='redo cameras already rendered')
    ap.add_argument('--dry-run', action='store_true', help='list what would be rendered')
    args = ap.parse_args()

    require_out_dir()
    want = set(args.cameras.split(',')) if args.cameras else None
    jobs = sorted({j for d in DETECTORS for j in find_cameras(_pnp_path, d, args.user, args.action, want)})
    if not jobs:
        raise SystemExit(f'no {{cam}}_pnp.npz from either detector under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_4_PnP.py first')
    video_of = lambda u, a, c: os.path.join(args.videos or os.path.join(_TRIAL_DIR, u, a), f'{c}.mp4')
    have_video = [j for j in jobs if os.path.exists(video_of(*j))]
    todo = [j for j in have_video if args.force or not os.path.exists(_video_out(*j))]
    print(f'{len(jobs)} camera(s) with PnP results under {_OUT_DIR}: {len(todo)} to render, '
          f'{len(have_video) - len(todo)} already done, {len(jobs) - len(have_video)} with no local video')
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return
    n_done, failed = 0, []
    for i, (user, action, cam) in enumerate(todo, 1):
        try:
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: {render(user, action, cam, video_of(user, action, cam), args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\nrendered {n_done}, failed {len(failed)}')


if __name__ == '__main__':
    main()
