#!/usr/bin/env python3
"""
step_7_animate.py — feature animation: LEFT the source video with YOLO (orange),
placed H36M-17 (blue) and placed SMPL-24 (green) overlaid, exactly as in
step_5a/5b; RIGHT a grid of feature time-series from step_6_extract_features.py
with a playhead (vertical line + a dot on every series) marking the current
video frame.

Reads:
    Analysis/keypoints/PnP/{cam}_pnp.npz        (step_4_PnP.py)
    Analysis/keypoints/{cam}_2d.npz             (step_1_extract_2d.py)
    Analysis/features/{cam}_features.npz              (step_6_extract_features.py)
    {user}/{action}/{cam}.mp4
Writes:
    Analysis/diagnostics/{cam}_features.mp4

The chart grid is rendered by matplotlib ONCE; the per-frame playhead is
composited onto that cached image with cv2 using each axes' data->pixel
transform, so the per-frame cost is a couple of line draws rather than a full
figure re-render (the original make_feature_animation.py re-saved the figure
every frame). Frames step_6 marked invalid (PnP failed / person not in shot /
carried-forward detection) get no skeleton overlay and a red label; their
feature values are NaN so the dots simply vanish there.

Edit _PANELS to choose what is plotted; any key in the features npz works.
Requires only cv2, numpy, matplotlib -- host or container.
    docker run --rm --runtime nvidia \
          -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
          -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
          motor-dev:latest \
          python3 step_7_animate.py --user User28 --action P28_CMJM_01
    
"""

import os
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import TRIAL_DIR as _TRIAL_DIR

# (chart title, [(feature key, legend label, matplotlib color), ...])
_PANELS = [
    ('Knee flexion (deg)',
     [('knee_flexion_rot_L_noisy', 'L', 'tab:blue'), ('knee_flexion_rot_R_noisy', 'R', 'tab:orange')]),
    ('Hip flexion, sagittal (deg, +ve = thigh forward)',
     [('hip_sagittal_angle_L_noisy', 'L', 'tab:blue'), ('hip_sagittal_angle_R_noisy', 'R', 'tab:orange')]),
    ('Elbow flexion (deg)',
     [('elbow_flexion_rot_L_noisy', 'L', 'tab:blue'), ('elbow_flexion_rot_R_noisy', 'R', 'tab:orange')]),
    ('Upper arm vs spine, sagittal (deg, +ve = forward)',
     [('shoulder_sagittal_angle_L_noisy', 'L', 'tab:blue'), ('shoulder_sagittal_angle_R_noisy', 'R', 'tab:orange')]),
    ('Ankle height above lab floor (m)',
     [('ankle_height_L_smooth', 'L_s', 'tab:red'), ('ankle_height_R_smooth', 'R_s', 'tab:green')]),
    ('Ankle fwd/back offset vs pelvis (m)',
     [('ankle_sagittal_fwd_L_noisy', 'L', 'tab:blue'), ('ankle_sagittal_fwd_R_noisy', 'R', 'tab:orange')]),
    ('Ankle forward velocity (m/s, smoothed)',
     [('ankle_sagittal_velocity_L_smooth', 'L_s', 'tab:red'), ('ankle_sagittal_velocity_R_smooth', 'R_s', 'tab:green')]),
    ('Pelvis forward velocity (m/s)',
     [('hip_forward_velocity_noisy', 'Noisy', 'tab:orange'),('hip_forward_velocity_smooth', 'Smooth', 'tab:red')]),
]

H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL24_BONES = [(i, p) for i, p in enumerate(_SMPL_PARENTS) if p != -1]
_COL_YOLO, _COL_H36M, _COL_SMPL24 = (0, 140, 255), (255, 60, 0), (60, 200, 60)   # BGR
_CONF_THRESH, _MIN_DEPTH_M = 0.4, 0.05


def draw_points_and_bones(img, pts2d, bones, color, radius=3, thickness=2):
    for a, b in bones:
        cv2.line(img, tuple(pts2d[a].astype(int)), tuple(pts2d[b].astype(int)), color, thickness, cv2.LINE_AA)
    for p in pts2d:
        cv2.circle(img, tuple(p.astype(int)), radius, color, -1, cv2.LINE_AA)


def reproject(pts3d_cam, K, dist_cv):
    p, _ = cv2.projectPoints(pts3d_cam.astype(np.float64), np.zeros(3), np.zeros(3), K, dist_cv)
    return p.reshape(-1, 2)


class ChartPanel:
    """Feature grid rendered once; playhead + current-value dots composited per frame."""

    def __init__(self, feats, width, height, dpi=100):
        self.w, self.h = width, height
        t = feats['time_s']
        ncols = 2
        nrows = int(np.ceil(len(_PANELS) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.06, hspace=0.75, wspace=0.28)
        self.axes = axes.reshape(-1)
        self.series = []                    # per axis: [(values, bgr color), ...]
        for ax, (title, series) in zip(self.axes, _PANELS):
            entries = []
            for key, label, color in series:
                if key not in feats:
                    print(f'  [warn] feature {key!r} not in features file, skipping')
                    continue
                ax.plot(t, feats[key], color=color, lw=1.1, label=label)
                rgb = matplotlib.colors.to_rgb(color)
                entries.append((np.asarray(feats[key], float), tuple(int(255 * c) for c in rgb[::-1])))
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlim(t[0], t[-1])
            ax.margins(y=0.15)
            ax.grid(True, lw=0.4, alpha=0.5)
            if any(label for _, label, _ in series):
                ax.legend(fontsize=6, loc='upper right', framealpha=0.5)
            self.series.append(entries)
        for ax in self.axes[len(_PANELS):]:
            ax.axis('off')
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
        self.base = cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
        # data->pixel transforms, captured once (figure is never redrawn)
        self.tf = [ax.transData for ax in self.axes[:len(_PANELS)]]
        self.ylims = [ax.get_ylim() for ax in self.axes[:len(_PANELS)]]
        self.fig_h = img.shape[0]
        self.t = t
        plt.close(fig)

    def _px(self, i, x, y):
        X, Y = self.tf[i].transform((x, y))
        return int(round(X)), int(round(self.fig_h - Y))      # display origin is bottom-left

    def render(self, frame_i):
        img = self.base.copy()
        t_now = self.t[frame_i]
        for i, entries in enumerate(self.series):
            y0, y1 = self.ylims[i]
            x_top, y_top = self._px(i, t_now, y1)
            _, y_bot = self._px(i, t_now, y0)
            cv2.line(img, (x_top, y_top), (x_top, y_bot), (0, 0, 0), 1, cv2.LINE_AA)
            for values, bgr in entries:
                v = values[frame_i]
                if np.isfinite(v):
                    cv2.circle(img, self._px(i, t_now, v), 4, bgr, -1, cv2.LINE_AA)
                    cv2.circle(img, self._px(i, t_now, v), 4, (0, 0, 0), 1, cv2.LINE_AA)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--video-scale', type=float, default=2 / 3, help='left panel scale of source video (1080p*2/3 = 720p)')
    ap.add_argument('--chart-width', type=int, default=900)
    args = ap.parse_args()

    root = os.path.join(_TRIAL_DIR, args.user, args.action)
    kc = os.path.join(root, 'Analysis', 'keypoints')
    feat_dir = os.path.join(root, 'Analysis', 'features')
    out_dir = os.path.join(root, 'Analysis', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    for cam in args.cameras.split(','):
        paths = {'pnp': os.path.join(kc, 'PnP', f'{cam}_pnp.npz'), '2d': os.path.join(kc, f'{cam}_2d.npz'),
                 'feat': os.path.join(feat_dir, f'{cam}_features.npz'), 'video': os.path.join(root, f'{cam}.mp4')}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            print(f'=== camera {cam}: missing {missing}, skipping ===')
            continue
        print(f'=== camera {cam} ===')
        pnp, d2, feats = np.load(paths['pnp']), np.load(paths['2d']), np.load(paths['feat'])
        H, S, K, dist = pnp['kps_H36M_placed'], pnp['kps_SMPL24_placed'], pnp['cam_K'], pnp['cam_dist_cv']
        h2d, sfi, fps = d2['h36m_2d'], d2['source_frame_idx'], float(d2['fps'])
        valid = feats['frame_valid']
        T = min(H.shape[0], h2d.shape[0], sfi.shape[0], len(feats['time_s']))

        cap = cv2.VideoCapture(paths['video'])
        W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        wanted = set(int(i) for i in sfi[:T]); frames = {}; fi = 0; max_idx = max(wanted)
        while fi <= max_idx:
            if fi in wanted:
                ok, fr = cap.read()
                if not ok: break
                frames[fi] = fr
            else:
                if not cap.grab(): break
            fi += 1
        cap.release()

        vw, vh = int(round(W * args.video_scale)), int(round(Hh * args.video_scale))
        chart = ChartPanel({k: feats[k] for k in feats.files if feats[k].ndim == 1 and feats[k].shape[0] >= T}, args.chart_width, vh)
        out_path = os.path.join(out_dir, f'{cam}_features.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw + 6 + args.chart_width, vh))
        sep = np.zeros((vh, 6, 3), dtype=np.uint8)

        for tt in range(T):
            frame = frames.get(int(sfi[tt]))
            if frame is None:
                continue
            canvas = frame.copy()
            yolo = h2d[tt, :, :2]; yv = h2d[tt, :, 2] > _CONF_THRESH
            for a, b in H36M_LIMBS:
                if yv[a] and yv[b]:
                    cv2.line(canvas, tuple(yolo[a].astype(int)), tuple(yolo[b].astype(int)), _COL_YOLO, 2, cv2.LINE_AA)
            for p in yolo[yv]:
                cv2.circle(canvas, tuple(p.astype(int)), 3, _COL_YOLO, -1, cv2.LINE_AA)
            if valid[tt]:
                if np.isfinite(H[tt]).all() and (H[tt][:, 2] > _MIN_DEPTH_M).all():
                    draw_points_and_bones(canvas, reproject(H[tt], K, dist), H36M_LIMBS, _COL_H36M)
                if np.isfinite(S[tt]).all() and (S[tt][:, 2] > _MIN_DEPTH_M).all():
                    draw_points_and_bones(canvas, reproject(S[tt], K, dist), SMPL24_BONES, _COL_SMPL24)
            else:
                cv2.putText(canvas, 'invalid frame (no PnP / not in shot)', (20, Hh - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            for i, (txt, col) in enumerate([('YOLO', _COL_YOLO), ('H36M (placed)', _COL_H36M), ('SMPL-24 (placed)', _COL_SMPL24)]):
                cv2.putText(canvas, txt, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
            cv2.putText(canvas, f'frame {tt}  t={tt / fps:.2f}s', (20, Hh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            left = cv2.resize(canvas, (vw, vh), interpolation=cv2.INTER_AREA)
            writer.write(np.concatenate([left, sep, chart.render(tt)], axis=1))
            if tt % 100 == 0:
                print(f'  {tt}/{T}')
        writer.release()
        print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
