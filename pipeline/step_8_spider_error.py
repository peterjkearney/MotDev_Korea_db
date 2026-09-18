#!/usr/bin/env python3
"""
step_8_spider_error.py — per-camera PnP-vs-mocap 3D error, plotted on a radar/spider
chart with each camera placed at its REAL azimuthal angle around the subject (not
evenly spaced), so angle-dependent error patterns are visible directly.

For each camera: mean 3D Euclidean distance (mm) between mocap ground truth and the
PnP-placed H36M skeleton, in both its raw ("placed") and RTS-smoothed ("smooth")
forms (step_4_PnP.py), each reported two ways --
  - "all joints"        : every mocap-valid joint, regardless of YOLO's 2D confidence
  - "conf>--conf-thresh" : restricted to joints where YOLO's own detection was
                            confident that frame -- the subset PnP had good 2D
                            evidence for, vs the full picture including occluded/
                            low-confidence joints.

Camera angle: found by inverting cam_L_ext (same cam_to_lab transform step_6_extract_
features.py uses) to get each camera's position in the lab frame, then taking its
azimuth around the subject's mean floor position in the lab's horizontal (X,Y) plane
(Z-up, per step_6's LAB_UP convention).

Run inside the motor-dev container:
    docker run --rm --runtime nvidia \
      -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
      -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
      motor-dev:latest \
      python3 step_8_spider_error.py --user User28 --action P28_CMJM_01
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

import json

from config import TRIAL_DIR as _TRIAL_DIR, mocap_path as _mocap_path, tri_target_path as _tri_path
from utils import metrics as M
LAB_UP = np.array([0.0, 0.0, 1.0])   # lab frame: Z up, floor z = 0 (mocap; the fitted floor for Korea)


def cam_to_lab(p_cam_m, L_ext):
    """Camera-space metres -> lab-frame metres. L_ext maps lab(mm) -> camera(mm)."""
    R, t = L_ext[:3, :3], L_ext[:3, 3]
    return (R.T @ (p_cam_m * 1000.0 - t).reshape(-1, 3).T).T.reshape(p_cam_m.shape) / 1000.0


def camera_lab_xy(L_ext):
    """Camera's own position in the lab's horizontal (X,Y) plane, metres."""
    return cam_to_lab(np.zeros((1, 3)), L_ext)[0][:2]


def camera_lab_heading_xy(L_ext):
    """Unit vector (lab X,Y) the camera is looking towards -- R.T @ [0,0,1] (the camera's
    +Z/forward axis in camera space) mapped into the lab frame and flattened to the floor."""
    R = L_ext[:3, :3]
    fwd_xy = (R.T @ np.array([0.0, 0.0, 1.0]))[:2]
    n = np.linalg.norm(fwd_xy)
    return fwd_xy / n if n > 1e-9 else np.array([0.0, 1.0])


def camera_azimuth_deg(cam_xy, subject_lab_xy):
    """Camera's angle (deg, 0-360) around the subject in the lab's horizontal (X,Y) plane."""
    dx, dy = cam_xy[0] - subject_lab_xy[0], cam_xy[1] - subject_lab_xy[1]
    return float(np.degrees(np.arctan2(dy, dx)) % 360)


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def facing_and_trajectory(mocap_world_mm, valid_joint_mask):
    """Per-frame pelvis position and floor-forward body heading, lab-frame metres/XY.
    Mirrors step_6_extract_features.py's build_body_frame, applied directly to the H36M
    mocap joints (0=pelvis, 1=r_hip, 4=l_hip, 7=spine) since mocap is already lab-frame
    -- no cam_to_lab needed here."""
    m = mocap_world_mm / 1000.0                    # metres
    pelvis, spine, r_hip, l_hip = m[:, 0, :], m[:, 7, :], m[:, 1, :], m[:, 4, :]

    y_hat = _unit(spine - pelvis)                   # up the lowest spine segment
    hip_vec = l_hip - r_hip
    x_hat = _unit(hip_vec - np.sum(hip_vec * y_hat, axis=-1, keepdims=True) * y_hat)
    z_hat = np.cross(x_hat, y_hat)
    up_sign = np.sign(np.nanmedian(np.sum(y_hat * LAB_UP, axis=-1)))
    if up_sign < 0:
        y_hat, z_hat = -y_hat, -z_hat
    z_fwd = _unit(z_hat - np.sum(z_hat * LAB_UP, axis=-1, keepdims=True) * LAB_UP)

    joints_needed = [0, 1, 4, 7]
    frame_ok = (~np.isnan(m[:, joints_needed, :]).any(axis=(1, 2))
                & valid_joint_mask[joints_needed].all())
    return pelvis[:, :2], z_fwd[:, :2], frame_ok


def _camera_marker_verts(pos_xy, heading_xy, size):
    """Trapezoid in lab XY: narrow edge at the camera's own position, long edge out front
    in its viewing direction -- reads like a little frustum/lens icon on the floor map."""
    fwd = heading_xy / (np.linalg.norm(heading_xy) + 1e-9)
    perp = np.array([-fwd[1], fwd[0]])
    back_half, front_half, depth = 0.35 * size, 0.9 * size, 1.1 * size
    back_l  = pos_xy - perp * back_half
    back_r  = pos_xy + perp * back_half
    front_r = pos_xy + fwd * depth + perp * front_half
    front_l = pos_xy + fwd * depth - perp * front_half
    return np.array([back_l, back_r, front_r, front_l])


def draw_floor_map(ax, traj_xy, heading_xy, frame_ok, cam_data_by_label, subject_lab_xy):
    """Top-down (lab X,Y) panel: subject's path, sparse facing-direction arrows, and the
    real camera positions/viewing directions -- spatial context for the polar error plot's
    camera angles. cam_data_by_label: {label: (position_xy, heading_xy)}."""
    traj = traj_xy[frame_ok]
    heading = heading_xy[frame_ok]
    if len(traj) == 0:
        ax.set_title('no valid mocap frames for floor map')
        return

    ax.plot(traj[:, 0], traj[:, 1], color='#9aa3ad', lw=1.2, zorder=1, label='subject path')
    ax.scatter(*traj[0], color='#2a9d5c', s=50, zorder=3, label='start')
    ax.scatter(*traj[-1], color='#d64545', s=50, zorder=3, label='end')

    # arrow/marker size relative to the WHOLE scene (path + camera positions), not just the
    # path's own span -- otherwise a near-stationary trial (e.g. a jump in place) gets
    # invisible, sub-centimetre arrows/markers next to cameras several metres away.
    cam_xy = np.array([xy for xy, _ in cam_data_by_label.values()]) if cam_data_by_label else traj
    all_x = np.concatenate([traj[:, 0], cam_xy[:, 0]])
    all_y = np.concatenate([traj[:, 1], cam_xy[:, 1]])
    span = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 1.0)
    arrow_len = 0.05 * span

    n_arrows = min(18, len(traj))
    idx = np.linspace(0, len(traj) - 1, n_arrows).astype(int)
    ax.quiver(traj[idx, 0], traj[idx, 1], heading[idx, 0], heading[idx, 1],
              color='#2a78d6', scale=1.0 / arrow_len, scale_units='xy',
              width=0.006, zorder=2, label='facing direction')

    marker_size = arrow_len * 0.3
    for label, (xy, cam_heading) in cam_data_by_label.items():
        verts = _camera_marker_verts(xy, cam_heading, marker_size)
        ax.add_patch(Polygon(verts, closed=True, facecolor='#e07b39',
                              edgecolor='#a85a25', lw=1.0, zorder=4))
        ax.annotate(f'cam {label}', xy, textcoords='offset points', xytext=(5, 5), fontsize=8,
                    zorder=5, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.75))

    ax.scatter(*subject_lab_xy, marker='+', s=80, color='k', zorder=4, label="subject mean pos")
    ax.set_xlabel('lab X (m)')
    ax.set_ylabel('lab Y (m)')
    ax.set_title('top-down: path, facing direction, camera positions')
    ax.set_aspect('equal')
    ax.margins(0.15)
    ax.grid(True, alpha=0.3)

    handles, labels_ = ax.get_legend_handles_labels()
    if cam_data_by_label:
        handles.append(Patch(facecolor='#e07b39', edgecolor='#a85a25',
                              label='camera (long edge = view dir.)'))
        labels_.append('camera (long edge = view dir.)')
    ax.legend(handles, labels_, loc='best', fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--conf-thresh', type=float, default=0.3)
    ap.add_argument('--gt', choices=['mocap', 'triangulated'], default='mocap',
                    help='ground truth: mocap_h36m.npz, or openpose_tri_h36m.npz '
                         '(triangulated OpenPose, leave-one-out per camera -- the only '
                         'option for Korea, and the like-for-like one for BioCV)')
    args = ap.parse_args()
    cameras = args.cameras.split(',')
    suffix = '' if args.gt == 'mocap' else '_tri'

    trial_root = os.path.join(_TRIAL_DIR, args.user, args.action)
    pnp_dir = os.path.join(trial_root, 'Analysis', 'keypoints', 'PnP')
    yolo_2d_dir = os.path.join(trial_root, 'Analysis', 'keypoints')
    out_dir = os.path.join(trial_root, 'Analysis', 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    gt_name = 'mocap_h36m.npz' if args.gt == 'mocap' else 'openpose_tri_h36m.npz'
    mocap_path = _mocap_path(args.user, args.action) if args.gt == 'mocap' else _tri_path(args.user, args.action)
    if not os.path.exists(mocap_path):
        print(f"no {mocap_path} -- run " + ('step_0_load_mocap.py' if args.gt == 'mocap'
              else 'step_1b_triangulate_2d.py / step_1_korea_2d.py') + " first")
        return
    mocap_data = np.load(mocap_path)
    mocap_world_mm = mocap_data['kps3d']                     # (T,17,3) mm, world/lab-space
    mocap_valid_joint_mask = mocap_data['valid_joint_mask']  # (17,) False for Nose/Head
    # leave-one-out targets: one per held-out camera, keyed by label
    gt_loo = None
    if args.gt == 'triangulated':
        gt_loo = {str(c): mocap_data['kps3d_loo'][i] for i, c in enumerate(mocap_data['cameras'])}

    stature_mm = None
    meta_path = os.path.join(_TRIAL_DIR, args.user, 'user_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stature_mm = 1000.0 * float(json.load(f)['stature_m'])

    # subject's mean floor position (lab X,Y), used as the angle-measurement origin
    subject_xyz_mm = np.nanmean(mocap_world_mm[:, mocap_valid_joint_mask, :], axis=(0, 1))
    subject_lab_xy = subject_xyz_mm[:2] / 1000.0

    traj_xy, heading_xy, frame_ok = facing_and_trajectory(mocap_world_mm, mocap_valid_joint_mask)

    angles, labels, cam_data_by_label, extras = [], [], {}, []
    err_placed_all, err_placed_conf, err_smooth_all, err_smooth_conf = [], [], [], []
    # same four series again, but kept per joint (C,17) for the batch results table
    pj_placed_all, pj_placed_conf, pj_smooth_all, pj_smooth_conf = [], [], [], []
    n_frames = []

    for camera in cameras:
        pnp_path = os.path.join(pnp_dir, f'{camera}_pnp.npz')
        yolo_2d_path = os.path.join(yolo_2d_dir, f'{camera}_2d.npz')
        if not os.path.exists(pnp_path) or not os.path.exists(yolo_2d_path):
            print(f"=== camera {camera}: missing pnp/2d data, skipping ===")
            continue

        pnp_data = np.load(pnp_path)
        kps_placed = pnp_data['kps_H36M_placed']      # (T,17,3) m, camera-space
        kps_smooth = pnp_data['kps_H36M_smooth']
        L_ext = pnp_data['cam_L_ext']

        yolo_data = np.load(yolo_2d_path)
        h36m_2d = yolo_data['h36m_2d']                 # (T,17,3) x_px,y_px,conf -- H36M order
        source_frame_idx = yolo_data['source_frame_idx']

        T = min(len(kps_placed), len(h36m_2d), len(source_frame_idx))

        R, t = L_ext[:3, :3], L_ext[:3, 3]
        if gt_loo is not None:
            if camera not in gt_loo:
                print(f"=== camera {camera}: no leave-one-out target for it, skipping ===")
                continue
            gt_world_mm = gt_loo[camera]               # triangulated WITHOUT this camera
        else:
            gt_world_mm = mocap_world_mm
        mocap_cam_mm = gt_world_mm[source_frame_idx[:T]] @ R.T + t
        mocap_cam_m = mocap_cam_mm / 1000.0            # (T,17,3)

        joint_ok = mocap_valid_joint_mask[None, :] & ~np.isnan(mocap_cam_m).any(axis=-1)  # (T,17)
        conf_ok = h36m_2d[:T, :, 2] > args.conf_thresh

        def mean_err(kps, mask):
            """(overall mean mm, per-joint mean mm (17,)) over the masked entries."""
            d = np.linalg.norm(kps[:T] - mocap_cam_m, axis=-1) * 1000.0   # mm
            m = mask & ~np.isnan(d)
            overall = float(d[m].mean()) if m.any() else float('nan')
            # Per-joint by explicit loop rather than nanmean: joints with no valid
            # frame at all (Nose/Head, or fully occluded under conf_ok) must come
            # out NaN, and nanmean warns on an all-NaN slice.
            per_joint = np.full(d.shape[1], np.nan)
            for j in range(d.shape[1]):
                if m[:, j].any():
                    per_joint[j] = d[m[:, j], j].mean()
            return overall, per_joint

        e_placed_all,  j_placed_all  = mean_err(kps_placed, joint_ok)
        e_placed_conf, j_placed_conf = mean_err(kps_placed, joint_ok & conf_ok)
        e_smooth_all,  j_smooth_all  = mean_err(kps_smooth, joint_ok)
        e_smooth_conf, j_smooth_conf = mean_err(kps_smooth, joint_ok & conf_ok)

        # Shape and proportion measures (mm, camera frame, all mocap-valid
        # joints).  PA removes placement and scale, so it isolates pose shape;
        # bone ratios isolate proportions -- the child question.
        pred_mm, gt_mm = kps_placed[:T] * 1000.0, mocap_cam_mm
        pa, pa_j = M.pa_mpjpe(pred_mm, gt_mm, joint_ok)
        nm, _ = M.n_mpjpe(pred_mm, gt_mm, joint_ok)
        bias = M.signed_bias(pred_mm, gt_mm, joint_ok)
        bones = M.bone_ratios(pred_mm, gt_mm, joint_ok, stature_mm) if stature_mm else {}
        extra = dict(pa_mpjpe=float(np.nanmean(pa)), n_mpjpe=float(np.nanmean(nm)),
                     perjoint_pa=pa_j, bias_xyz=bias,
                     bone_names=list(bones), bone_ratio=[bones[b]['ratio'] for b in bones],
                     bone_pred_frac=[bones[b]['pred_frac'] for b in bones],
                     bone_gt_frac=[bones[b]['gt_frac'] for b in bones],
                     bone_pred_cv=[bones[b]['pred_cv'] for b in bones])
        extras.append(extra)
        pct = (f"  = {100 * e_placed_all / stature_mm:.1f}% of stature" if stature_mm else '')
        print(f"    PA-MPJPE {extra['pa_mpjpe']:6.1f}mm  N-MPJPE {extra['n_mpjpe']:6.1f}mm{pct}")
        if bones:
            print('    bone length, predicted / ground truth: ' +
                  '  '.join(f"{b} {bones[b]['ratio']:.2f}" for b in bones
                            if np.isfinite(bones[b]['ratio'])))

        cam_xy = camera_lab_xy(L_ext)
        cam_heading = camera_lab_heading_xy(L_ext)
        angle = camera_azimuth_deg(cam_xy, subject_lab_xy)

        print(f"=== camera {camera}  angle={angle:6.1f}deg  "
              f"placed(all)={e_placed_all:6.1f}mm  placed(conf)={e_placed_conf:6.1f}mm  "
              f"smooth(all)={e_smooth_all:6.1f}mm  smooth(conf)={e_smooth_conf:6.1f}mm ===")

        angles.append(angle)
        labels.append(camera)
        cam_data_by_label[camera] = (cam_xy, cam_heading)
        err_placed_all.append(e_placed_all)
        err_placed_conf.append(e_placed_conf)
        err_smooth_all.append(e_smooth_all)
        err_smooth_conf.append(e_smooth_conf)
        pj_placed_all.append(j_placed_all)
        pj_placed_conf.append(j_placed_conf)
        pj_smooth_all.append(j_smooth_all)
        pj_smooth_conf.append(j_smooth_conf)
        n_frames.append(T)

    if not angles:
        print("no camera data found")
        return

    # sort by angle so the polygon's edges follow the cameras' real order around the room
    order = np.argsort(angles)
    angles = np.array(angles)[order]
    labels = [labels[i] for i in order]
    conf_label = f'conf>{args.conf_thresh:g}'
    series = {
        'placed (all joints)':  np.array(err_placed_all)[order],
        f'placed ({conf_label})': np.array(err_placed_conf)[order],
        'smooth (all joints)':  np.array(err_smooth_all)[order],
        f'smooth ({conf_label})': np.array(err_smooth_conf)[order],
    }
    styles = {
        'placed (all joints)':    dict(color='#9aa3ad', ls='--', marker='o'),
        f'placed ({conf_label})': dict(color='#4b5563', ls='-',  marker='o'),
        'smooth (all joints)':    dict(color='#8fb8e8', ls='--', marker='s'),
        f'smooth ({conf_label})': dict(color='#2a78d6', ls='-',  marker='s'),
    }

    # Rotate so the reference camera sits at the top and match the top-down panel's plain
    # Cartesian sense (counter-clockwise = increasing real-world angle) instead of the
    # polar default's own zero/direction -- otherwise the two panels rotate oppositely
    # for the same camera order, which is confusing to compare side by side.
    ref_camera = '07' if '07' in labels else labels[0]
    ref_angle = angles[labels.index(ref_camera)]
    theta = np.radians((angles - ref_angle) % 360)
    theta_closed = np.concatenate([theta, theta[:1]])

    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection='polar')
    ax_map = fig.add_subplot(1, 2, 2)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)

    for name, vals in series.items():
        vals_closed = np.concatenate([vals, vals[:1]])
        ax.plot(theta_closed, vals_closed, label=name, **styles[name])

    ax.set_xticks(theta)
    ax.set_xticklabels([f'cam {c}\n{a:.0f}\N{DEGREE SIGN}' for c, a in zip(labels, angles)])
    ax.set_ylabel('mean 3D joint error (mm)', labelpad=30)
    ax.set_title(f'PnP vs {"mocap" if args.gt == "mocap" else "triangulated OpenPose (LOO)"}: '
                 f'mean 3D joint error by camera\n{args.user} / {args.action}', pad=24)
    ax.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15), fontsize=7)

    draw_floor_map(ax_map, traj_xy, heading_xy, frame_ok, cam_data_by_label, subject_lab_xy)

    out_path = os.path.join(out_dir, f'spider_error_{args.action}{suffix}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n-> {out_path}")

    # Machine-readable twin of the plot, for tools/run_batch.py to aggregate
    # across trials. Rows follow the same angle-sorted camera order as the
    # chart, so index i means the same camera in both.
    metrics_path = os.path.join(out_dir, f'error_metrics{suffix}.npz')
    ex = [extras[i] for i in order]
    np.savez(metrics_path,
             gt=np.array(args.gt), stature_mm=np.array(stature_mm if stature_mm else np.nan),
             pa_mpjpe=np.array([e['pa_mpjpe'] for e in ex]),
             n_mpjpe=np.array([e['n_mpjpe'] for e in ex]),
             perjoint_pa=np.array([e['perjoint_pa'] for e in ex]),
             bias_xyz=np.array([e['bias_xyz'] for e in ex]),
             bone_names=np.array(ex[0]['bone_names']),
             bone_ratio=np.array([e['bone_ratio'] for e in ex]),
             bone_pred_frac=np.array([e['bone_pred_frac'] for e in ex]),
             bone_gt_frac=np.array([e['bone_gt_frac'] for e in ex]),
             bone_pred_cv=np.array([e['bone_pred_cv'] for e in ex]),
             cameras=np.array(labels),
             angles_deg=angles,
             n_frames=np.array(n_frames)[order],
             joint_names=mocap_data['joint_names'],
             conf_thresh=np.array(args.conf_thresh),
             err_placed_all=np.array(err_placed_all)[order],
             err_placed_conf=np.array(err_placed_conf)[order],
             err_smooth_all=np.array(err_smooth_all)[order],
             err_smooth_conf=np.array(err_smooth_conf)[order],
             perjoint_placed_all=np.array(pj_placed_all)[order],
             perjoint_placed_conf=np.array(pj_placed_conf)[order],
             perjoint_smooth_all=np.array(pj_smooth_all)[order],
             perjoint_smooth_conf=np.array(pj_smooth_conf)[order])
    print(f"-> {metrics_path}")


if __name__ == '__main__':
    main()
