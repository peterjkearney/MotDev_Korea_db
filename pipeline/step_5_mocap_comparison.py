#!/usr/bin/env python3
"""
step_5b_mocap_comparison.py — step_4_render_overlay.py's front/video overlay
panel (YOLO + placed H36M-17 + placed SMPL-24, unchanged), with a second
panel added on the right: a top-down (X vs Z) comparison of step_3_PnP.py's
placed H36M-17 markers against the real mocap ground truth
(step_0_load_mocap.py), both drawn for the SAME frame -- to see directly how
accurate the model's solved DEPTH (Z, into the scene) actually is, which the
front panel's 2D reprojection can't reveal (a reprojection can look perfect
in 2D while still being wrong in depth).

Reads:
    Analysis/keypoints/PnP/{cam}_pnp.npz       (step_3_PnP.py; also
                                                       carries this camera's
                                                       K / dist_cv / L_ext)
    Analysis/keypoints/{cam}_2d.npz            (step_1_extract_2d.py)
    Analysis/H36M/mocap_h36m.npz               (step_0_load_mocap.py; under OUT_DIR)
    {user}/{action}/{cam}.mp4                        (source video)

Mocap's kps3d is in the c3d file's own WORLD-space millimetres (see
step_0_load_mocap.py's docstring). Converted into THIS camera's real
camera-space via cam_L_ext (R_ext, t_ext) -- the same transform
render_avatar_overlay.py uses for its Hip-only mocap overlay
(R_ext @ p_world + t_ext, then mm -> m), applied here to every joint at
once rather than just the root.

Frame-index alignment: mocap is kept at native (200Hz) resolution by
step_0_load_mocap.py -- no decimation, so mocap_h36m.npz's source_frame_idx
is just arange(n_native_frames) (checked below). Video and mocap share the
identical native capture rate (confirmed: both 200Hz for every trial), so a
camera frame's own native index (src_idx, from step_1_extract_2d.py) already
means the same real instant in mocap's stream too -- mocap is looked up by
indexing directly with src_idx, bounds-checked against mocap's own native
length, rather than by matching two independently-decimated index arrays
(which, empirically, are not guaranteed to agree frame-for-frame).

Saved to Analysis/diagnostics/{cam}_pnp_depth_vs_mocap.mp4.

Run inside the motor-dev container:
    docker run --rm --runtime nvidia \
      -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
      -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
      motor-dev:latest \
      python3 step_5_mocap_comparison.py --user User28 --action P28_CMJM_01
"""

import os
import argparse

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.calibration import reproject

from config import TRIAL_DIR as _TRIAL_DIR, mocap_path as _mocap_path

# H36M-17 kinematic bone pairs -- matches diagnose_frame.py's LIMBS and
# step_4_render_overlay.py's H36M_LIMBS. Mocap's kps3d uses this SAME joint
# order (step_0_load_mocap.py's own design), so the same pairs apply to both.
H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
              (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

# Standard SMPL-24 kinematic tree -- matches _SMPL_PARENTS in main_3d_mb_mesh_window.py.
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                 12, 13, 14, 16, 17, 18, 19, 20, 21]
SMPL24_BONES = [(i, p) for i, p in enumerate(_SMPL_PARENTS) if p != -1]

_COL_YOLO = (0, 140, 255)      # orange, BGR -- real 2D detections
_COL_H36M = (255, 60, 0)       # blue, BGR -- reprojected placed H36M-17
_COL_SMPL24 = (60, 200, 60)    # green, BGR -- reprojected placed SMPL-24
_PNP_CONF_THRESH = 0.4
# A joint at/behind the camera plane (depth <= this) blows up to inf/huge
# values under cv2.projectPoints' pinhole division, which then overflows
# cv2.line's int32 parsing (not caught by an np.isnan check, since inf/huge
# values are finite) -- see conversation history for the direct repro.
_MIN_DEPTH_M = 0.05

_TD_PANEL_W = 640   # top-down panel width in px; height matches the video's own h
_TD_COL_H36M = '#4b5563'     # grey -- placed MotionBERT H36M
_TD_COL_H36M_SMOOTH = '#2a78d6'     # blue -- smooth MotionBERT H36M
_TD_COL_MOCAP = '#FF0000'    # red -- mocap ground truth
_TD_COL_GRID = '#c7cdd4'
_TD_COL_INK = '#4b5563'


def draw_points_and_bones(img, pts2d, bones, color, radius=3, thickness=2):
    for a, b in bones:
        pa = tuple(pts2d[a].astype(int))
        pb = tuple(pts2d[b].astype(int))
        cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for p in pts2d:
        cv2.circle(img, tuple(p.astype(int)), radius, color, -1, cv2.LINE_AA)


def compute_topdown_limits(h36m_xz, mocap_xz):
    """Fixed axis limits from the whole clip's valid range of both datasets,
    computed once so the panel doesn't rescale frame to frame."""
    x_vals = [h36m_xz[..., 0][~np.isnan(h36m_xz[..., 0])],
              mocap_xz[..., 0][~np.isnan(mocap_xz[..., 0])]]
    z_vals = [h36m_xz[..., 1][~np.isnan(h36m_xz[..., 1])],
              mocap_xz[..., 1][~np.isnan(mocap_xz[..., 1])]]
    all_x = np.concatenate(x_vals)
    all_z = np.concatenate(z_vals)
    if all_x.size == 0 or all_z.size == 0:
        return (-1.0, 1.0), (-0.3, 2.0)
    x_half = max(float(np.abs(all_x).max()) * 1.15, 0.5)
    z_max = max(float(all_z.max()) * 1.15, 1.0)
    return (-x_half, x_half), (-0.3, z_max)


def _hex_to_bgr(hex_col, fade_towards_white=0.0):
    """matplotlib colour -> cv2 BGR tuple; fade_towards_white in [0,1) emulates
    alpha on the white panel background (cv2 primitives have no alpha)."""
    r, g, b = matplotlib.colors.to_rgb(hex_col)
    f = fade_towards_white
    return tuple(int(255 * (c * (1 - f) + f)) for c in (b, g, r))


class TopDownPanel:
    """Static top-down panel (FOV cone, camera marker, grid, labels, legend)
    rendered by matplotlib ONCE per camera; the three per-frame skeletons are
    composited onto a copy with cv2 using the captured data->pixel transform,
    so the per-frame cost is a few dozen line draws instead of a full figure
    re-render (same approach as step_7_animate.py's ChartPanel)."""

    def __init__(self, width, height, half_fov_x_rad, xlim, zlim, dpi=100):
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.10)
        ax.set_facecolor('white')

        cone_len = zlim[1] * 1.05
        for sign in (-1, 1):
            ax.plot([0, sign * cone_len * np.sin(half_fov_x_rad)],
                     [0, cone_len * np.cos(half_fov_x_rad)],
                     color=_TD_COL_INK, lw=1.1, ls='--', alpha=0.6)
        ax.plot(0, 0, marker='^', markersize=8, color=_TD_COL_INK)
        ax.text(0, -0.22, 'camera', color=_TD_COL_INK, fontsize=7.5, ha='center')

        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect('equal')
        ax.grid(True, color=_TD_COL_GRID, lw=0.6, alpha=0.7)
        ax.set_xlabel('X, camera-space (m)', color=_TD_COL_INK, fontsize=8)
        ax.set_ylabel('Z, depth (m)', color=_TD_COL_INK, fontsize=8)
        ax.set_title('top-down — H36M (placed) vs mocap', color='#14181d', fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_color(_TD_COL_GRID)
        ax.tick_params(colors=_TD_COL_INK, labelsize=7)
        handles = [
            plt.Line2D([0], [0], color=_TD_COL_H36M, marker='o', lw=2, markersize=6, label='H36M (placed)'),
            plt.Line2D([0], [0], color=_TD_COL_H36M_SMOOTH, marker='o', lw=2, markersize=6, label='H36M (smooth)'),
            plt.Line2D([0], [0], color=_TD_COL_MOCAP, marker='o', lw=2, markersize=6, label='mocap'),
        ]
        ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=7, labelcolor=_TD_COL_INK)

        fig.canvas.draw()      # transform is only valid AFTER the draw (set_aspect moves the axes box)
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
        self.base = cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
        self.tf = ax.transData
        self.fig_h = img.shape[0]
        plt.close(fig)

        self.col_raw = _hex_to_bgr(_TD_COL_H36M, fade_towards_white=0.8)     # emulates alpha=0.2
        self.col_smooth = _hex_to_bgr(_TD_COL_H36M_SMOOTH)
        self.col_mocap = _hex_to_bgr(_TD_COL_MOCAP)

    def _px(self, xz):
        X, Y = self.tf.transform(xz)
        return int(round(X)), int(round(self.fig_h - Y))     # display origin is bottom-left

    def _skeleton(self, img, xz, valid, bgr):
        finite = valid & np.isfinite(xz).all(axis=-1)
        for a, b in H36M_LIMBS:
            if finite[a] and finite[b]:
                cv2.line(img, self._px(xz[a]), self._px(xz[b]), bgr, 2, cv2.LINE_AA)
        for j in np.where(finite)[0]:
            cv2.circle(img, self._px(xz[j]), 3, bgr, -1, cv2.LINE_AA)

    def render(self, h36m_valid, h36m_xz, h36m_xz_smooth, mocap_valid, mocap_xz):
        img = self.base.copy()
        self._skeleton(img, mocap_xz, mocap_valid, self.col_mocap)
        self._skeleton(img, h36m_xz, h36m_valid, self.col_raw)
        self._skeleton(img, h36m_xz_smooth, h36m_valid, self.col_smooth)
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--user', default='User03')
    ap.add_argument('--action', default='P03_CMJM_01')
    args = ap.parse_args()
    cameras = args.cameras.split(',')

    trial_root = os.path.join(_TRIAL_DIR, args.user, args.action)
    pnp_dir = os.path.join(trial_root, 'Analysis', 'keypoints', 'PnP')
    yolo_2d_dir = os.path.join(trial_root, 'Analysis', 'keypoints')
    out_dir = os.path.join(trial_root, 'Analysis', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    mocap_path = _mocap_path(args.user, args.action)
    if not os.path.exists(mocap_path):
        print(f"no {mocap_path} -- run step_0_load_mocap.py first")
        return
    mocap_data = np.load(mocap_path)
    mocap_world_mm = mocap_data['kps3d']                 # (T,17,3) mm, world-space, NaN where invalid
    mocap_valid_joint_mask = mocap_data['valid_joint_mask']   # (17,) -- False for Nose/Head
    mocap_source_frame_idx = mocap_data['source_frame_idx']
    assert np.array_equal(mocap_source_frame_idx, np.arange(len(mocap_source_frame_idx))), \
        "mocap_h36m.npz looks decimated, not native-resolution -- re-run step_0_load_mocap.py"

    for camera in cameras:
        print(f"=== camera {camera} ===")

        pnp_path = os.path.join(pnp_dir, f'{camera}_pnp.npz')
        if not os.path.exists(pnp_path):
            print(f"  no {pnp_path} -- run step_3_PnP.py --cameras {camera} first, skipping")
            continue
        pnp_data = np.load(pnp_path)
        kps_H36M_placed = pnp_data['kps_H36M_placed']      # (T,17,3) metres, real camera-space
        kps_SMPL24_placed = pnp_data['kps_SMPL24_placed']  # (T,24,3)
        kps_H36M_smooth = pnp_data['kps_H36M_smooth']      # (T,17,3) metres, real camera-space
        kps_SMPL24_smooth = pnp_data['kps_SMPL24_smooth'] 
        K = pnp_data['cam_K']
        dist_cv = pnp_data['cam_dist_cv']
        L_ext = pnp_data['cam_L_ext']

        yolo_2d_path = os.path.join(yolo_2d_dir, f'{camera}_2d.npz')
        if not os.path.exists(yolo_2d_path):
            print(f"  no {yolo_2d_path} -- run step_1_extract_2d.py --cameras {camera} first, skipping")
            continue
        yolo_data = np.load(yolo_2d_path)
        h36m_2d = yolo_data['h36m_2d']                 # (T,17,3) x_px, y_px, conf
        source_frame_idx = yolo_data['source_frame_idx']
        fps = float(yolo_data['fps'])

        n_frames = kps_H36M_placed.shape[0]
        assert h36m_2d.shape[0] == n_frames, "PnP/2d frame count mismatch"


        # mocap world-space (mm) -> this camera's real camera-space (m)
        R_ext, t_ext = L_ext[:3, :3], L_ext[:3, 3]
        mocap_cam_mm = np.einsum('ij,tkj->tki', R_ext, mocap_world_mm) + t_ext
        mocap_cam_m = mocap_cam_mm / 1000.0
        
        video_path = os.path.join(trial_root, f'{camera}.mp4')
        if not os.path.exists(video_path):
            print(f"  no {video_path}, skipping")
            continue

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # top-down data + fixed axis limits, computed once over the whole clip
        h36m_xz_all = kps_H36M_placed[:, :, [0, 2]]
        h36m_xz_all_smooth = kps_H36M_smooth[:, :, [0, 2]]
        mocap_xz_all = mocap_cam_m[:, :, [0, 2]]
        xlim, zlim = compute_topdown_limits(h36m_xz_all, mocap_xz_all)
        half_fov_x_rad = float(np.arctan(w / (2 * K[0, 0])))

        panel = TopDownPanel(_TD_PANEL_W, h, half_fov_x_rad, xlim, zlim)

        out_path = os.path.join(out_dir, f'{camera}_pnp_depth_vs_mocap_SCALED_640.mp4')
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                  (w + 6 + _TD_PANEL_W, h))
        sep = np.zeros((h, 6, 3), dtype=np.uint8)

        # Decode only the frames we actually need (same grab/read pattern as
        # step_1_extract_2d.py) and process/write each one immediately in the
        # same pass, rather than buffering the whole decoded clip in memory
        # first -- with source_frame_idx currently close to dense (see
        # step_0_load_mocap.py's decimation), buffering the full clip is
        # multiple GB and OOMs on the Jetson's 7.4GB RAM (no swap headroom).
        n_ok = 0
        tt = 0
        frame_i = 0
        max_idx = int(source_frame_idx[-1]) if len(source_frame_idx) else -1
        while frame_i <= max_idx and tt < n_frames:
            src_idx = int(source_frame_idx[tt])
            if frame_i != src_idx:
                ret = cap.grab()
                if not ret:
                    break
                frame_i += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # --- panel 1: front/video overlay, same as step_4_render_overlay.py ---
            canvas = frame

            yolo_pts = h36m_2d[tt, :, :2]
            yolo_valid = h36m_2d[tt, :, 2] > _PNP_CONF_THRESH
            yolo_bones = [(a, b) for a, b in H36M_LIMBS if yolo_valid[a] and yolo_valid[b]]
            draw_points_and_bones(canvas, yolo_pts[yolo_valid], [], _COL_YOLO)
            for a, b in yolo_bones:
                cv2.line(canvas, tuple(yolo_pts[a].astype(int)), tuple(yolo_pts[b].astype(int)),
                         _COL_YOLO, 2, cv2.LINE_AA)

            has_h36m = (not np.isnan(kps_H36M_placed[tt]).any()
                        and (kps_H36M_placed[tt][:, 2] > _MIN_DEPTH_M).all())
            has_smpl24 = (not np.isnan(kps_SMPL24_placed[tt]).any()
                          and (kps_SMPL24_placed[tt][:, 2] > _MIN_DEPTH_M).all())

            if has_h36m:
                h36m_2d_pts = reproject(kps_H36M_placed[tt], K, dist_cv)
                draw_points_and_bones(canvas, h36m_2d_pts, H36M_LIMBS, _COL_H36M)
            # if has_smpl24:
            #     smpl24_2d_pts = reproject(kps_SMPL24_placed[tt], K, dist_cv)
            #     draw_points_and_bones(canvas, smpl24_2d_pts, SMPL24_BONES, _COL_SMPL24)

            if has_h36m or has_smpl24:
                n_ok += 1
            else:
                cv2.putText(canvas, "PnP failed this frame", (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(canvas, "YOLO", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COL_YOLO, 2, cv2.LINE_AA)
            cv2.putText(canvas, "H36M", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COL_H36M, 2, cv2.LINE_AA)
            #cv2.putText(canvas, "SMPL-24 (placed)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _COL_SMPL24, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {tt}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # --- panel 2: top-down H36M (placed) vs mocap, this frame only ---
            h36m_valid_tt = ~np.isnan(kps_H36M_placed[tt]).any(axis=-1)
            if src_idx < mocap_cam_m.shape[0]:
                mocap_valid_tt = mocap_valid_joint_mask & ~np.isnan(mocap_cam_m[src_idx]).any(axis=-1)
                mocap_xz_tt = mocap_xz_all[src_idx]
            else:
                # mocap's native clip ran out before this camera frame --
                # draw nothing for mocap this frame, don't invent data.
                mocap_valid_tt = np.zeros(17, dtype=bool)
                mocap_xz_tt = np.zeros((17, 2), dtype=np.float32)

            topdown_img = panel.render(h36m_valid_tt, h36m_xz_all[tt], h36m_xz_all_smooth[tt],
                                        mocap_valid_tt, mocap_xz_tt)

            combined = np.concatenate([canvas, sep, topdown_img], axis=1)
            writer.write(combined)
            if tt % 100 == 0:
                print(f"  {tt}/{n_frames}")

            tt += 1
            frame_i += 1

        cap.release()
        writer.release()
        print(f"  {n_ok}/{n_frames} frames with a placed overlay")
        print(f"  -> {out_path}")


if __name__ == '__main__':
    main()
