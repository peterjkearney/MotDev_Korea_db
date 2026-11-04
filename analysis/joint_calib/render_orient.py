#!/usr/bin/env python3
"""render_orient.py -- what MotionBERT's pelvis-orientation error looks like on one camera.

LEFT   the camera's video with
         OpenPose's H36M input as MotionBERT got it (dots; joints missing on this frame listed),
         mocap's skeleton projected (white),
         MotionBERT's 3D skeleton (op input, mb_input_defs.py's cache) turned into camera axes, scaled to
         mocap's bone lengths and put at mocap's pelvis -- so any gap is ORIENTATION (and pose), not placement (orange),
         and each one's pelvis axes: hip line, trunk, forward (thick = mocap, thin = MotionBERT).
RIGHT  top: from above (lab vertical), pelvis-centred: both hip lines + forward arrows, this camera's line of sight.
       bottom: total / heading / lean / tilt error (mb_input_defs.orientation_parts) over the trial, cursor at now.

    python3 analysis/joint_calib/render_orient.py --cam 05
    writes analysis/joint_calib/out/render/orient_{trial}_cam{cam}.mp4
"""
import argparse
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mb_input_defs as mb                                                       # noqa: E402
from utils.calibration import load_calib                                         # noqa: E402

LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
SCORE_LIMBS = [(a, b) for a, b in LIMBS if a not in (9, 10) and b not in (9, 10)]
MOCAP, MBC = (255, 255, 255), (40, 150, 255)                 # BGR: white, orange
AXIS = [(60, 60, 230), (80, 200, 80), (230, 140, 40)]        # hip line red, trunk green, forward blue
SERIES = [('total', (235, 235, 235)), ('lean', (40, 150, 255)), ('tilt', (80, 200, 80)), ('heading', (230, 140, 40))]  # orientation_parts' column order


def proj(P, K, dist):
    out = np.full(P.shape[:-1] + (2,), np.nan)
    ok = np.isfinite(P).all(-1) & (P[..., 2] > 1)
    if ok.any():
        p, _ = cv2.projectPoints(P[ok].reshape(-1, 1, 3).astype(np.float64), np.zeros(3), np.zeros(3), K, dist)
        out[ok] = p.reshape(-1, 2)
    return out


def line(img, a, b, col, w):
    if np.isfinite(a).all() and np.isfinite(b).all():
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, w, cv2.LINE_AA)


def bone_scale(P, G):
    """Median over limbs and frames of mocap/MotionBERT bone length."""
    r = [np.linalg.norm(G[:, a] - G[:, b], axis=-1) / np.linalg.norm(P[:, a] - P[:, b], axis=-1) for a, b in SCORE_LIMBS]
    return float(np.nanmedian(np.concatenate(r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default=os.path.join(mb._ROOT, 'data', 'sampleBiocv', 'P03_CMJM_01'))
    ap.add_argument('--calib-dir', default=None)
    ap.add_argument('--cam', default='05')
    ap.add_argument('--variant', default='op', help='which cached MotionBERT run to draw')
    ap.add_argument('--fps', type=float, default=30.0, help='output fps (input rows are 60 fps)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    trial = os.path.basename(os.path.abspath(args.trial))
    calib_dir = args.calib_dir or os.path.dirname(os.path.abspath(args.trial))
    cache = os.path.join(mb._ROOT, 'analysis', 'joint_calib', 'out', 'mb_cache', f'{trial}_{args.cam}_{args.variant}.npz')
    out = args.out or os.path.join(mb._ROOT, 'analysis', 'joint_calib', 'out', 'render',
                                   f'orient_{trial}_cam{args.cam}' + ('' if args.variant == 'op' else f'_{args.variant}') + '.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    v, gt, have, _ = mb.inputs(args.trial, args.cam, calib_dir)
    tw = np.load(os.path.join(args.trial, 'Analysis', 'openpose', f'{args.cam}_2d.npz'))
    raw, sfi = tw['h36m_2d'], tw['source_frame_idx'].astype(int)
    _, _, K, L, dist = load_calib(os.path.join(calib_dir, f'{args.cam}.mp4-mocAligned.calib'))
    dist = np.asarray(dist, float).reshape(1, -1)
    kp = np.load(cache)['kp']
    T = min(len(kp), len(gt))

    # MotionBERT -> camera axes (the flip mb_input_defs picks), mocap's scale, at mocap's pelvis
    Fg_all = np.full((T, 3, 3), np.nan)
    Fg_all[have[:T]] = mb.body_frame(gt[:T][have[:T]])
    Rray = np.tile(np.eye(3), (T, 1, 1))
    Rray[have[:T]] = mb.ray_rot(gt[:T][have[:T]][:, 0])            # MotionBERT's orientation is about the ray to the person
    Fop = mb.body_frame(kp[:T][have[:T]])
    flip = min(mb.FLIPS, key=lambda S: np.nanmedian(mb.geodesic_deg(mb.to_ray(Fop @ S, Rray[have[:T]]), Fg_all[have[:T]])))
    s = bone_scale((kp[:T] @ flip)[have[:T]], gt[:T][have[:T]])
    P = np.einsum('tij,tkj->tki', Rray, (kp[:T] - kp[:T, :1]) @ flip) * s + gt[:T, :1]   # NaN where mocap is missing
    Fp_all = mb.body_frame(P)
    parts = np.full((T, 4), np.nan)
    ok = have[:T]
    parts[ok] = np.stack(mb.orientation_parts(Fp_all[ok], Fg_all[ok]), 1)
    print(f'cam {args.cam}: flip diag{np.diag(flip).astype(int).tolist()}, scale {s:.3f}, median total '
          f'{np.nanmedian(parts[:, 0]):.1f} deg, heading mean |{np.nanmean(np.abs(parts[:, 3])):.1f}|')

    # lab frame for the top-down view: vertical from mocap (thorax above mid-ankles), camera centre
    R, t = L[:3, :3], L[:3, 3]
    to_lab = lambda X: (X - t) @ R                                   # camera mm -> lab mm
    G_lab, P_lab = to_lab(gt[:T]), to_lab(P)
    up = np.nanmedian(G_lab[:, 8] - 0.5 * (G_lab[:, 3] + G_lab[:, 6]), 0)
    up /= np.linalg.norm(up)
    e1 = np.cross(up, [0, 0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = lambda X: np.stack([X @ e1, X @ e2], -1)
    cam_c = flat(-R.T @ t)

    cap = cv2.VideoCapture(os.path.join(args.trial, f'{args.cam}.mp4'))
    vw, vh = int(cap.get(3)), int(cap.get(4))
    H = 720
    VW = int(round(vw * H / vh))
    PW, TD_H = 560, 400
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (VW + PW, H))
    want = {int(f): i for i, f in enumerate(sfi[:T])}
    plot_h, pad = H - TD_H, 30
    ylim = 60.0

    f = 0
    while True:
        okf, img = cap.read()
        if not okf or f > sfi[T - 1]:
            break
        i = want.get(f)
        f += 1
        if i is None:
            continue
        # ---- video panel
        mo2, mb2 = proj(gt[i], K, dist), proj(P[i], K, dist)
        for a, b in LIMBS:
            line(img, mo2[a], mo2[b], MOCAP, 5)
            line(img, mb2[a], mb2[b], MBC, 4)
        for j in range(17):
            if raw[i, j, 2] > 0:
                cv2.circle(img, tuple(np.round(raw[i, j, :2]).astype(int)), 7, (0, 255, 255), -1, cv2.LINE_AA)
        if np.isfinite(Fg_all[i]).all():
            for F, w, origin in ((Fg_all[i], 7, gt[i, 0]), (Fp_all[i], 3, P[i, 0])):
                o2 = proj(origin, K, dist)
                for k in range(3):
                    line(img, o2, proj(origin + 250 * F[k], K, dist), AXIS[k], w)
        img = cv2.resize(img, (VW, H), interpolation=cv2.INTER_AREA)
        missing = [mb.H36M[j] for j in range(17) if raw[i, j, 2] <= 0]
        txt = [f'cam {args.cam}  frame {sfi[i]}  ({i / 60:.2f} s)',
               'white = mocap   orange = MotionBERT (mocap scale + pelvis)   yellow = OpenPose input',
               'missing in input: ' + (', '.join(missing) if missing else 'none')]
        for k, s_ in enumerate(txt):
            cv2.putText(img, s_, (15, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, s_, (15, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- top-down panel
        td = np.full((TD_H, PW, 3), 30, np.uint8)
        c0 = flat(G_lab[i, 0]) if np.isfinite(G_lab[i, 0]).all() else None
        sc = (TD_H / 2 - 30) / 600.0                                   # px per mm, +-600 mm
        mid = np.array([PW / 2, TD_H / 2])
        to_px = lambda q: mid + (q - c0) * [sc, -sc]
        if c0 is not None:
            d = cam_c - c0
            d = d / np.linalg.norm(d)
            line(td, to_px(c0 + 560 * d), to_px(c0 + 200 * d), (200, 200, 200), 2)
            cv2.putText(td, 'to camera', tuple(np.round(to_px(c0 + 520 * d) + [5, -5]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            for X, col, w in ((G_lab[i], MOCAP, 4), (P_lab[i], MBC, 3)):
                for a, b in LIMBS:
                    line(td, to_px(flat(X[a])), to_px(flat(X[b])), tuple(int(0.45 * c) for c in col), 1)
                fwd = np.cross(X[4] - X[1], X[8] - X[0])                # lab: x cross y of the pelvis frame
                fwd = flat(fwd / np.linalg.norm(fwd))
                fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
                line(td, to_px(flat(X[1])), to_px(flat(X[4])), col, w)
                hc = flat(0.5 * (X[1] + X[4]))
                cv2.arrowedLine(td, tuple(np.round(to_px(hc)).astype(int)), tuple(np.round(to_px(hc + 300 * fwd)).astype(int)),
                                col, w, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(td, f'from above   heading error {parts[i, 3]:+.0f} deg   total {parts[i, 0]:.0f} deg',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- time series panel
        pl = np.full((plot_h, PW, 3), 18, np.uint8)
        x_of = lambda k: pad + (PW - 2 * pad) * k / max(T - 1, 1)
        y_of = lambda val: plot_h / 2 - (plot_h / 2 - pad) * np.clip(val, -ylim, ylim) / ylim
        for g in (-45, -30, -15, 0, 15, 30, 45):
            y = int(y_of(g))
            cv2.line(pl, (pad, y), (PW - pad, y), (70, 70, 70) if g else (130, 130, 130), 1)
            cv2.putText(pl, f'{g}', (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        for k, (name, col) in enumerate(SERIES):
            pts = np.stack([x_of(np.arange(T)), y_of(parts[:, k])], 1)
            good = np.isfinite(pts).all(1)
            for seg in np.split(np.arange(T), np.where(~good)[0]):
                seg = seg[good[seg]]
                if len(seg) > 1:
                    cv2.polylines(pl, [np.round(pts[seg]).astype(np.int32)], False, col, 1, cv2.LINE_AA)
            val = parts[i, k]
            cv2.putText(pl, f'{name} {val:+.0f}' if np.isfinite(val) else f'{name} -', (pad + 130 * k, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cx = int(x_of(i))
        cv2.line(pl, (cx, pad), (cx, plot_h - 5), (255, 255, 255), 1)
        cv2.putText(pl, 'deg, MotionBERT minus mocap (lean + = top of trunk away from camera)', (pad, plot_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        writer.write(np.hstack([img, np.vstack([td, pl])]))
    writer.release()
    cap.release()
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
