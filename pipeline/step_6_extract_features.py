#!/usr/bin/env python3
"""
step_6_extract_features.py — per-camera movement features from the PnP-placed,
stature-scaled SMPL-24 joint centres (step_4_PnP.py) and the per-frame SMPL
joint rotations (step_3_extract_3d.py), written to
    {user}/{action}/Analysis/features/{cam}_features.npz

Using the SMPL-24 keypoints rather than H36M as SMPL-24 limbs are rigid, H36M markers are more
closely aligned with the skin, so joint angles are not represented correctly.

Rotation-based angles copied straight from rotmats: per-joint rotation
magnitude (axis-angle, deg) and an X-Y-Z Tait-Bryan decomposition in
SMPL's template axes. rotmats[0] (global orientation) is known-unreliable
for this checkpoint and is excluded from named features (kept in the
raw copy). NOTE on axes: SMPL joint rotations are relative to the parent
and expressed in the rest-pose axes, so for legs x = flexion axis,
y = long-axis twist, z = ab/adduction; mirrored joints (L vs R) share the
x sign but have opposite y/z signs. Arms hang along +-x in the T-pose, so
the same decomposition means different anatomical rotations there.

Validity: a frame is valid when PnP succeeded (no NaN in the placed points)
AND the 2D core-joint confidence (Hip, Thorax) clears --conf-thresh (the
person_in_shot rule from the original), AND -- if step_1 wrote a `detected`
array -- YOLO actually detected the person that frame (not a carried-forward
copy). Every feature is NaN on invalid frames; velocities use np.gradient and
are NaN next to invalid frames too.

Run on host (numpy only) or inside the container:
    docker run --rm --runtime nvidia \
        -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
        -w /ssd/MotorDevelopment/Python/PnP_depth_clean \
        motor-dev:latest \
        python3 step_6_extract_features.py --user User28 --action P28_CMJM_01
"""
import os
import argparse

import numpy as np

from config import TRIAL_DIR as _TRIAL_DIR

SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
    'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand',
]
S = {n: i for i, n in enumerate(SMPL_JOINT_NAMES)}
H36M_HIP, H36M_THORAX = 0, 8          # core joints for person_in_shot
LAB_UP = np.array([0.0, 0.0, 1.0])    # mocap lab frame: Z up, floor z = 0


def _unit(v, eps=1e-8):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def _angle_between(a, b):
    c = np.clip(np.sum(_unit(a) * _unit(b), axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def rot_angle_deg(R):
    """Axis-angle magnitude of a (...,3,3) rotation stack, in degrees."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def rot_to_euler_xyz_deg(R):
    """Intrinsic X-Y-Z Tait-Bryan angles (deg) of a (...,3,3) stack, R = Rx(a) Ry(b) Rz(c)."""
    b = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    a = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    c = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    return np.degrees(np.stack([a, b, c], axis=-1))


def smooth_nan(x, window):
    """
    NaN-aware centred Gaussian smoothing along axis 0 (sigma = window/4).
    Each output is the weighted mean of the VALID samples in its window, so
    gaps neither spread nor bias the neighbours; outputs stay NaN where the
    input was NaN. window <= 1 returns x unchanged.
    """
    if window <= 1:
        return x
    half = window // 2
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / (window / 4.0)) ** 2)
    valid = ~np.isnan(x)
    xf = np.where(valid, x, 0.0)
    num = np.zeros_like(xf); den = np.zeros_like(xf)
    T = x.shape[0]
    for off, w in zip(range(-half, half + 1), k):
        lo, hi = max(0, -off), min(T, T - off)
        num[lo:hi] += w * xf[lo + off:hi + off]
        den[lo:hi] += w * valid[lo + off:hi + off]
    out = num / np.where(den > 0, den, np.nan)
    out[~valid] = np.nan
    return out


def cam_to_lab(p_cam_m, L_ext):
    """Camera-space metres -> lab-frame metres. L_ext maps lab(mm) -> camera(mm)."""
    R, t = L_ext[:3, :3], L_ext[:3, 3]
    return (R.T @ (p_cam_m * 1000.0 - t).reshape(-1, 3).T).T.reshape(p_cam_m.shape) / 1000.0


def build_body_frame(J):
    """
    J : (T,24,3) SMPL-24 joints in the lab frame.
    Returns per-frame unit vectors:
      x_hat  subject's left (hip line, orthogonalised to the spine)
      y_hat  up the lowest spine segment (Pelvis -> Spine1)
      z_hat  perpendicular to the spine and the hip line ("spine-forward";
             pitches with the trunk -- only used for spine-relative angles)
      z_fwd  the same heading flattened into the lab horizontal plane
             ("floor-forward", parallel to the floor) -- used for every
             forward/backward position and velocity feature, so trunk
             lean and squat depth don't leak into them (see history:
             spine-forward read a vertical leg as -0.23 m "behind").
    """
    y_hat = _unit(J[:, S['Spine1']] - J[:, S['Pelvis']]) # yhat is lowest spine segment
    hip_vec = J[:, S['L_Hip']] - J[:, S['R_Hip']] # hip_vec is the vector pointing from right hip to left hip
    x_hat = _unit(hip_vec - np.sum(hip_vec * y_hat, axis=-1, keepdims=True) * y_hat) # finding projection of hip_vec on y_hat 
    z_hat = np.cross(x_hat, y_hat)          # SMPL: +x left, +y up, +z forward
    # up should agree with gravity; if the clip's median says otherwise, flip the whole frame
    up_sign = np.sign(np.nanmedian(np.sum(y_hat * LAB_UP, axis=-1)))
    if up_sign < 0:
        y_hat, z_hat = -y_hat, -z_hat
    z_fwd = _unit(z_hat - np.sum(z_hat * LAB_UP, axis=-1, keepdims=True) * LAB_UP)
    return x_hat, y_hat, z_hat, z_fwd


def compute_features(J, rotmats, fps, smooth_window=9):
    """
    J: (T,24,3) lab-frame metres (NaN rows on invalid frames). rotmats: (T,24,3,3).
    Velocities are differentiated from a Gaussian-smoothed copy of the
    positions (smooth_window frames); angles and positions are left raw.
    Per-frame PnP depth jitter (~0.1-0.2 m) differentiated at 60 fps is
    otherwise +-10 m/s of noise on a 1.5 m/s walk.
    """
    T = J.shape[0]
    x_hat, y_hat, z_hat, z_fwd = build_body_frame(J)
    pelvis = J[:, S['Pelvis']]
    Js = smooth_nan(J.reshape(T, -1), smooth_window).reshape(J.shape)   # for velocities only
    feats = {'frame': np.arange(T), 'time_s': np.arange(T) / fps}

    # 1) elbow internal angle (180 = straight)
    for side in ('L', 'R'):
        sh, el, wr = J[:, S[f'{side}_Shoulder']], J[:, S[f'{side}_Elbow']], J[:, S[f'{side}_Wrist']]
        feats[f'elbow_angle_{side}'] = _angle_between(sh - el, wr - el)

    # 2) upper arm vs spine, sagittal plane, signed (0 = hanging down, + = forward)
    for side in ('L', 'R'):
        arm = J[:, S[f'{side}_Elbow']] - J[:, S[f'{side}_Shoulder']]
        feats[f'shoulder_sagittal_angle_{side}'] = np.degrees(
            np.arctan2(np.sum(arm * z_hat, -1), -np.sum(arm * y_hat, -1)))

    # 3) pelvis velocity along body-forward (m/s), plus lab-frame speed for reference
    vel = np.gradient(Js[:, S['Pelvis']], axis=0) * fps
    feats['hip_forward_velocity'] = np.sum(vel * z_fwd, axis=-1)      # floor-parallel
    feats['hip_vertical_velocity'] = np.sum(vel * LAB_UP, axis=-1)    # + = up
    feats['hip_speed_lab'] = np.linalg.norm(vel, axis=-1)

    # 4) knee internal angle (180 = straight)
    for side in ('L', 'R'):
        hp, kn, an = J[:, S[f'{side}_Hip']], J[:, S[f'{side}_Knee']], J[:, S[f'{side}_Ankle']]
        feats[f'knee_angle_{side}'] = _angle_between(hp - kn, an - kn)

    # 5) ankle height above the lab floor (metres) -- true vertical, not
    #    screen-relative. The ankle joint centre sits ~0.07-0.09 m above the
    #    sole when standing. SMPL's L_Foot/R_Foot are deliberately NOT used:
    #    with no foot/toe keypoints in the H36M-17 input, the ankle rotation
    #    that places them is an unobserved network prior, so the foot joint
    #    is just the ankle plus a guessed 7 cm offset. Use ankle height with
    #    ankle_sagittal_velocity ~ 0 as the ground-contact criterion.
    for side in ('L', 'R'):
        feats[f'ankle_height_{side}'] = J[:, S[f'{side}_Ankle'], 2]

    # 6) ankle forward/back offset vs pelvis, sagittal (metres)
    for side in ('L', 'R'):
        rel = J[:, S[f'{side}_Ankle']] - pelvis
        feats[f'ankle_sagittal_fwd_{side}'] = np.sum(rel * z_fwd, axis=-1)   # floor-parallel

    # 6b) absolute ankle velocity along body-forward (m/s, ~0 when planted)
    for side in ('L', 'R'):
        av = np.gradient(Js[:, S[f'{side}_Ankle']], axis=0) * fps
        feats[f'ankle_sagittal_velocity_{side}'] = np.sum(av * z_fwd, axis=-1)   # floor-parallel

    # 6c) absolute ankle velocity  (m/s, ~0 when planted)
    for side in ('L', 'R'):
        av = np.gradient(Js[:, S[f'{side}_Ankle']], axis=0) * fps
        feats[f'ankle_velocity_{side}'] = np.linalg.norm(av, axis=-1)

    # 7) thigh vs spine, sagittal, signed (0 = hanging down, + = raised forward)
    for side in ('L', 'R'):
        thigh = J[:, S[f'{side}_Knee']] - J[:, S[f'{side}_Hip']]
        feats[f'hip_sagittal_angle_{side}'] = np.degrees(
            np.arctan2(np.sum(thigh * z_hat, -1), -np.sum(thigh * y_hat, -1)))

    # 8) rotation-based angles straight from the SMPL joint rotations
    mag = rot_angle_deg(rotmats)                      # (T,24)
    eul = rot_to_euler_xyz_deg(rotmats)               # (T,24,3)
    for side in ('L', 'R'):
        feats[f'knee_flexion_rot_{side}'] = mag[:, S[f'{side}_Knee']]
        feats[f'elbow_flexion_rot_{side}'] = mag[:, S[f'{side}_Elbow']]
        for jn, key in ((f'{side}_Hip', 'hip'), (f'{side}_Shoulder', 'shoulder')):
            feats[f'{key}_rot_x_{side}'] = eul[:, S[jn], 0]
            feats[f'{key}_rot_y_{side}'] = eul[:, S[jn], 1]
            feats[f'{key}_rot_z_{side}'] = eul[:, S[jn], 2]
    feats['spine_flexion_rot'] = mag[:, S['Spine1']] + mag[:, S['Spine2']] + mag[:, S['Spine3']]
    return feats, eul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cameras', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--conf-thresh', type=float, default=0.3,
                    help='2D Hip/Thorax confidence below this = person not in shot')
    ap.add_argument('--smooth-window', type=int, default=9,
                    help='Gaussian window (frames, odd) applied to positions before '
                         'differentiating velocities; 1 = no smoothing')
    args = ap.parse_args()

    analysis = os.path.join(_TRIAL_DIR, args.user, args.action, 'Analysis')
    kc, out_dir = os.path.join(analysis, 'keypoints'), os.path.join(analysis, 'features')
    os.makedirs(out_dir, exist_ok=True)

    for cam in args.cameras.split(','):
        paths = {'pnp': os.path.join(kc, 'PnP', f'{cam}_pnp.npz'),
                 'mesh': os.path.join(kc, 'mesh', f'{cam}_mesh_pose.npz'),
                 '2d': os.path.join(kc, f'{cam}_2d.npz')}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            print(f'=== camera {cam}: missing {missing}, skipping ===')
            continue
        print(f'=== camera {cam} ===')
        pnp, mesh, yolo = (np.load(paths[k]) for k in ('pnp', 'mesh', '2d'))
        J_cam_noisy, J_cam_smooth, rotmats, h2d = pnp['kps_SMPL24_placed'],pnp['kps_SMPL24_smooth'], mesh['rotmats'], yolo['h36m_2d']
        sfi, fps = yolo['source_frame_idx'], float(yolo['fps'])
        T = min(J_cam_noisy.shape[0], J_cam_smooth.shape[0],rotmats.shape[0], h2d.shape[0], sfi.shape[0])
        J_cam_noisy, J_cam_smooth, rotmats, h2d, sfi = J_cam_noisy[:T], J_cam_smooth[:T], rotmats[:T], h2d[:T], sfi[:T]

        # validity: PnP solved, person in shot (core 2D confidence), and -- if
        # step_1 recorded it -- an actual YOLO detection rather than a carried frame
        pnp_ok = ~np.isnan(J_cam_noisy).any(axis=(1, 2))
        core_conf = 0.5 * (h2d[:, H36M_HIP, 2] + h2d[:, H36M_THORAX, 2])
        in_shot = core_conf >= args.conf_thresh
        detected = yolo['detected'][:T].astype(bool) if 'detected' in yolo else np.ones(T, bool)
        valid = pnp_ok & in_shot & detected

        J_lab_noisy = cam_to_lab(J_cam_noisy, pnp['cam_L_ext'])
        J_lab_noisy[~valid] = np.nan
        J_lab_smooth = cam_to_lab(J_cam_smooth, pnp['cam_L_ext'])
        J_lab_smooth[~valid] = np.nan
        H_lab_noisy = cam_to_lab(pnp['kps_H36M_placed'][:T], pnp['cam_L_ext'])
        H_lab_noisy[~valid] = np.nan
        H_lab_smooth = cam_to_lab(pnp['kps_H36M_smooth'][:T], pnp['cam_L_ext'])
        H_lab_smooth[~valid] = np.nan

        feats_noisy, eul = compute_features(J_lab_noisy, rotmats, fps, smooth_window=args.smooth_window)
        feats_smooth, _ = compute_features(J_lab_smooth, rotmats, fps, smooth_window=args.smooth_window)

        for k in feats_noisy:
            if k not in ('frame', 'time_s'):
                feats_noisy[k] = feats_noisy[k].astype(np.float64)
                feats_noisy[k][~valid] = np.nan

        for k in feats_smooth:
                    if k not in ('frame', 'time_s'):
                        feats_smooth[k] = feats_smooth[k].astype(np.float64)
                        feats_smooth[k][~valid] = np.nan

        # merging noisy and smooth dicts
        shared_feats = ('frame', 'time_s')
        merged_feats = {k: feats_noisy[k] for k in shared_feats}
        merged_feats.update({f'{k}_noisy':  v for k, v in feats_noisy.items()  if k not in shared_feats})
        merged_feats.update({f'{k}_smooth': v for k, v in feats_smooth.items() if k not in shared_feats})


        out = os.path.join(out_dir, f'{cam}_features.npz')
        np.savez(out, **merged_feats,
                 frame_valid=valid, pnp_ok=pnp_ok, person_in_shot=in_shot, detected=detected,
                 core_confidence=core_conf, source_frame_idx=sfi, fps=fps,
                 kps_SMPL24_lab_noisy=J_lab_noisy.astype(np.float32), kps_SMPL24_lab_smooth=J_lab_smooth.astype(np.float32),
                 kps_H36M_lab_noisy=H_lab_noisy.astype(np.float32),kps_H36M_lab_smooth=H_lab_smooth.astype(np.float32),
                 rotmats=rotmats.astype(np.float32), joint_euler_xyz_deg=eul.astype(np.float32),
                 smpl_joint_names=np.array(SMPL_JOINT_NAMES),
                 units=np.array('positions/heights m, velocities m/s, angles deg, lab frame Z-up floor z=0'))
        # ka = feats['knee_angle_L'][valid]
        # print(f'  {valid.sum()}/{T} valid frames (pnp ok {pnp_ok.sum()}, in shot {in_shot.sum()}, detected {detected.sum()})'
        #       f' | knee_angle_L {np.nanmin(ka):.0f}..{np.nanmax(ka):.0f} deg'
        #       f' | ankle_height_L {np.nanmin(feats["ankle_height_L"]):.2f}..{np.nanmax(feats["ankle_height_L"]):.2f} m'
        #       f' | hip_forward_velocity mean {np.nanmean(feats["hip_forward_velocity"]):+.2f} m/s')
        print(f'  -> {out}')


if __name__ == '__main__':
    main()
