#!/usr/bin/env python3
"""sample_refine_2d.py -- refine MotionBERT's SMPL rotations against the 2D keypoints, shape fixed.

step_4 places MotionBERT's skeleton with a RIGID transform (PnP).  When the predicted pose
is more upright than the real one (trunk under-leaned, arms not swung, knees under-flexed
at the crouch), the only rigid escape is to push the whole body away from the camera.
This replaces that rigid fit with an articulated one, SMPLify-style, starting from what
the pipeline already has:

    variables, per frame   global rotation + translation (init: step_4's PnP), and an
                           axis-angle OFFSET on each of the 23 body joints (init: zero,
                           i.e. MotionBERT's rotation)
    objective              robust weighted reprojection of the SMPL H36M joints onto the
                           OpenPose 2D (confidence-weighted, pseudo-Huber)
                         + pull of every offset toward zero        (--w-pull)
                         + smoothness of the offsets over time      (--w-smooth)
                         + smoothness of the global trajectory      (--w-trans-acc)
    solver                 L-BFGS over the whole trial at once; stage 1 global pose only
                           (reproduces PnP), stage 2 everything

Shape (the betas) and the stature scale are frozen at the variant's values.  SMPL is
evaluated only at the ~330 vertices the joint regressors use, so a full-trial forward pass
is milliseconds.  At zero offsets the forward pass reproduces step_3's kps_H36M_scaled to
<0.1 mm (checked at start-up), so the baseline is the first iterate.

Reads an OUT_DIR variant that is through step_4 (default the MotionBERT baseline the betas
ablation made) and writes a sibling variant with refined mesh_pose (rotmats, joints) and
PnP (placed joints) files, then scores it with step_8 against mocap and the triangulated
target and prints the change against the source variant.  Nothing under pipeline/ changes.

    ~/anaconda3/envs/motEnv/bin/python sample_refine_2d.py
    ~/anaconda3/envs/motEnv/bin/python sample_refine_2d.py --w-pull 10 --out data/.../refine2d_pull10
"""
import argparse
import glob
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, 'pipeline')
sys.path.insert(0, HERE)
from sample_fit_betas_bones import MB_MESH  # noqa: E402  (locates MotionBERT/data/mesh)

DEFAULT_SRC = os.path.join(HERE, 'data', 'sampleBiocv', 'out', 'P03_CMJM_01', 'betas_ablation', 'motionbert')
H36M = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
SMPL24 = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
          'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
          'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']
GROUPS = {'trunk': ['Spine1', 'Spine2', 'Spine3', 'Neck', 'Head'], 'hips': ['L_Hip', 'R_Hip'],
          'knees': ['L_Knee', 'R_Knee'], 'ankles/feet': ['L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot'],
          'shoulders': ['L_Collar', 'R_Collar', 'L_Shoulder', 'R_Shoulder'], 'elbows': ['L_Elbow', 'R_Elbow'],
          'wrists/hands': ['L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']}


def load_h36m_regressor():
    """The H36M regressor step_3 actually uses.  utils_smpl.SMPL registers J_regressor_h36m as a buffer,
    so the copy stored in the mesh checkpoint overrides data/mesh/J_regressor_h36m_correct.npy when the
    state dict is loaded (strict=True).  The two differ by a left/right swap of the six leg rows; the
    checkpoint's version is the one consistent with the pipeline's OpenPose and mocap labels."""
    ckpt = os.path.join(MB_MESH, '..', '..', 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin')
    if os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location='cpu')['model']
        for k, v in sd.items():
            if k.endswith('head.smpl.J_regressor_h36m'):
                return v.numpy().astype(np.float64)
    warnings.warn('mesh checkpoint not found; using data/mesh/J_regressor_h36m_correct.npy (legs may be swapped)')
    return np.load(os.path.join(MB_MESH, 'J_regressor_h36m_correct.npy'))


# ----------------------------------------------------------------------------- SMPL on the regressor vertices
class MiniSMPL:
    """SMPL linear blend skinning evaluated only at the vertices J_regressor / J_regressor_h36m read."""

    def __init__(self, betas, device):
        with open(os.path.join(MB_MESH, 'SMPL_NEUTRAL.pkl'), 'rb') as f, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = pickle.load(f, encoding='latin1')
        Jr = m['J_regressor']
        Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr)
        Jh = load_h36m_regressor()
        idx = np.where((np.abs(Jr).sum(0) > 0) | (np.abs(Jh).sum(0) > 0))[0]
        v_full = np.asarray(m['v_template'], dtype=np.float64)
        v_t = v_full[idx]
        S = np.asarray(m['shapedirs'])[idx][:, :, :10]
        betas = np.asarray(betas, dtype=np.float64)
        if betas.size > 10:                                   # 11th entry: AGORA kid blend weight
            from sample_fit_betas_bones import kid_shapedir
            S = np.concatenate([S, kid_shapedir(v_full)[idx][:, :, None]], axis=2)
        v_shaped = v_t + S @ betas[:S.shape[-1]]
        f64 = dict(dtype=torch.float64, device=device)
        self.v_shaped = torch.tensor(v_shaped, **f64)                                   # (n,3)
        self.posedirs = torch.tensor(np.asarray(m['posedirs'])[idx].reshape(len(idx) * 3, 207).T, **f64)  # (207,n*3)
        self.W = torch.tensor(np.asarray(m['weights'])[idx], **f64)                      # (n,24)
        self.Jr = torch.tensor(Jr[:, idx], **f64)                                        # (24,n)
        self.Jh = torch.tensor(Jh[:, idx], **f64)                                        # (17,n)
        self.parents = np.asarray(m['kintree_table'])[0].astype(np.int64)
        self.parents[0] = -1
        self.J_rest = self.Jr @ self.v_shaped                                            # (24,3)
        self.n = len(idx)

    def forward(self, rotmats):
        """rotmats (T,24,3,3) -> SMPL-24 joints (T,24,3), H36M joints (T,17,3), in SMPL's own frame (metres)."""
        T = rotmats.shape[0]
        eye = torch.eye(3, dtype=rotmats.dtype, device=rotmats.device)
        pose_feat = (rotmats[:, 1:] - eye).reshape(T, 207)
        v_posed = self.v_shaped + (pose_feat @ self.posedirs).reshape(T, self.n, 3)
        # kinematic chain
        G = [None] * 24
        rel = self.J_rest.clone()
        rel[1:] = self.J_rest[1:] - self.J_rest[self.parents[1:]]
        for k in range(24):
            Tk = torch.zeros(T, 4, 4, dtype=rotmats.dtype, device=rotmats.device)
            Tk[:, :3, :3] = rotmats[:, k]
            Tk[:, :3, 3] = rel[k]
            Tk[:, 3, 3] = 1.0
            G[k] = Tk if k == 0 else G[self.parents[k]] @ Tk
        G = torch.stack(G, 1)                                                            # (T,24,4,4)
        # remove the rest-pose joint location so A maps rest verts -> posed verts
        Jh4 = torch.cat([self.J_rest, torch.zeros(24, 1, dtype=rotmats.dtype, device=rotmats.device)], 1)  # (24,4)
        A = G.clone()
        A[:, :, :3, 3] = G[:, :, :3, 3] - (G[:, :, :3, :] @ Jh4[None, :, :, None])[..., 0]
        Tv = torch.einsum('nk,tkij->tnij', self.W, A)                                    # (T,n,4,4)
        v = (Tv[:, :, :3, :3] @ v_posed[..., None])[..., 0] + Tv[:, :, :3, 3]            # (T,n,3)
        return torch.einsum('jn,tnc->tjc', self.Jr, v), torch.einsum('jn,tnc->tjc', self.Jh, v)


def rodrigues(aa):
    """(...,3) axis-angle -> (...,3,3) rotation matrices, differentiable at zero."""
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / theta
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def project(X_cam, K, dist):
    """OpenCV pinhole + radial/tangential distortion.  X_cam (...,3) metres -> (...,2) px."""
    x, y = X_cam[..., 0] / X_cam[..., 2], X_cam[..., 1] / X_cam[..., 2]
    k1, k2, p1, p2, k3 = [float(d) for d in dist[:5]]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)


def kabsch(P, Q):
    """Rigid R,t with Q = R P + t, over rows.  P,Q (n,3) numpy."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - R @ mp


# ----------------------------------------------------------------------------- one camera
def refine_camera(vdir, user, action, detector, cam, args, log):
    A = os.path.join(vdir, user, action, 'Analysis')
    mesh = dict(np.load(os.path.join(A, 'mesh', detector, f'{cam}_mesh_pose.npz'), allow_pickle=True))
    pnp = dict(np.load(os.path.join(A, 'PnP', detector, f'{cam}_pnp.npz'), allow_pickle=True))
    twod = np.load(os.path.join(A, 'keypoints', detector, f'{cam}_2d.npz'), allow_pickle=True)
    rot0 = mesh['rotmats'].astype(np.float64)                        # (T,24,3,3)
    scale = float(mesh['scale_factor'])
    kp2d = twod['h36m_2d'].astype(np.float64)                        # (T,17,3) px + conf
    T = min(rot0.shape[0], kp2d.shape[0], pnp['kps_H36M_placed'].shape[0])
    rot0, kp2d = rot0[:T], kp2d[:T]
    K, dist = pnp['cam_K'].astype(np.float64), np.asarray(pnp['cam_dist_cv']).ravel()
    dev = torch.device('cpu')
    smpl = MiniSMPL(mesh['betas'], dev)

    # --- check the forward pass against step_3's joints (metres, mesh frame, stature-scaled)
    with torch.no_grad():
        _, Jh0 = smpl.forward(torch.tensor(rot0))
    err = np.abs(Jh0.numpy() * scale - mesh['kps_H36M_scaled'][:T]).max() * 1000
    log(f'  {cam}: SMPL forward vs step_3 kps_H36M_scaled: max |diff| {err:.3f} mm over {T} frames')
    if err > 1.0:
        raise SystemExit('forward pass does not reproduce step_3 -- check SMPL conventions')

    # --- PnP init per frame (rigid map from the scaled mesh frame to the camera), frames where PnP solved
    placed = pnp['kps_H36M_placed'][:T].astype(np.float64)
    pnp_ok = np.isfinite(placed).all(axis=(1, 2))
    R0 = np.tile(np.eye(3), (T, 1, 1))
    t0 = np.zeros((T, 3))
    for i in np.where(pnp_ok)[0]:
        R0[i], t0[i] = kabsch(Jh0[i].numpy() * scale, placed[i])
    # fill frames PnP failed on from the nearest solved frame
    if (~pnp_ok).any() and pnp_ok.any():
        good = np.where(pnp_ok)[0]
        for i in np.where(~pnp_ok)[0]:
            j = good[np.argmin(np.abs(good - i))]
            R0[i], t0[i] = R0[j], t0[j]
    if args.fix_rotation in ('ray', 'ray_const'):
        # MotionBERT's frame is the camera frame of a virtual camera looking straight at the person.  For the real
        # camera, rotate by the angle between the optical axis and the ray through the person's 2D root.  Pure geometry.
        from scipy.spatial.transform import Rotation as Rot
        root_px = kp2d[:, H36M.index('Hip'), :2].copy()
        good = kp2d[:, H36M.index('Hip'), 2] > args.conf_thresh
        if not good.any():
            raise SystemExit(f'{cam}: no confident 2D root for the ray rotation')
        idx_good = np.where(good)[0]
        for i in np.where(~good)[0]:                                  # nearest confident frame
            root_px[i] = root_px[idx_good[np.argmin(np.abs(idx_good - i))]]
        Kinv = np.linalg.inv(K)
        wob, R_rays = [], []
        for i in range(T):
            d = Kinv @ np.array([root_px[i, 0], root_px[i, 1], 1.0])
            d /= np.linalg.norm(d)
            ax = np.cross([0.0, 0.0, 1.0], d)
            sn = np.linalg.norm(ax)
            R_ray = np.eye(3) if sn < 1e-9 else Rot.from_rotvec(ax / sn * np.arctan2(sn, d[2])).as_matrix()
            R_rays.append(R_ray)
        R_rays = np.array(R_rays)
        R_const = np.eye(3)
        if args.fix_rotation == 'ray_const':
            # what step_4's PnP rotation had beyond the ray correction, averaged over the trial (a per-trial constant:
            # MotionBERT's orientation bias for this viewpoint + any calibration pitch); no per-frame rotation is solved
            resid = Rot.from_matrix(np.array([R_rays[i].T @ R0[i] for i in np.where(pnp_ok)[0]]))
            R_const = resid.mean().as_matrix()
            log(f'  {cam}: constant residual rotation beyond the ray correction: '
                f'{np.round(np.degrees(Rot.from_matrix(R_const).as_rotvec()), 1)} deg')
        for i in range(T):
            R_new = R_rays[i] @ R_const
            if pnp_ok[i]:
                wob.append(np.degrees(Rot.from_matrix(R0[i] @ R_new.T).magnitude()))
                t0[i] = placed[i].mean(0) - R_new @ (Jh0[i].numpy() * scale).mean(0)
            R0[i] = R_new
        if (~pnp_ok).any() and pnp_ok.any():
            good_t = np.where(pnp_ok)[0]
            for i in np.where(~pnp_ok)[0]:
                t0[i] = t0[good_t[np.argmin(np.abs(good_t - i))]]
        wob = np.array(wob)
        log(f'  {cam}: per-frame ray rotation (MotionBERT orientation + off-axis correction'
            + (' + per-trial constant' if args.fix_rotation == 'ray_const' else '') + '), no rotation solved; '
            f"step_4's PnP rotation differed from it by median {np.median(wob):.1f} deg, max {wob.max():.1f}")
    elif args.fix_rotation != 'none':
        from scipy.spatial.transform import Rotation as Rot
        fps_ = float(twod['fps']) if 'fps' in twod.files else 60.0
        if not pnp_ok.any():
            raise SystemExit(f'{cam}: PnP solved no frame at all -- nothing to freeze the rotation to')
        sel = pnp_ok.copy()
        if args.fix_rotation == 'stance':
            # the first --stance-sec seconds OF FRAMES PNP SOLVED (the subject may enter the view late, e.g. a run)
            first = np.where(pnp_ok)[0][0]
            sel &= (np.arange(T) - first) / fps_ < args.stance_sec
            if sel.sum() < 5:
                log(f'  {cam}: fewer than 5 PnP frames in the stance window -- using the whole-trial mean rotation instead')
                sel = pnp_ok.copy()
        R_fix = Rot.from_matrix(R0[sel]).mean().as_matrix()
        wob = np.degrees((Rot.from_matrix(R0[pnp_ok]) * Rot.from_matrix(R_fix).inv()).magnitude())
        log(f'  {cam}: global rotation frozen to the {args.fix_rotation} PnP rotation ({sel.sum()} frames); '
            f'PnP per-frame rotation deviated from it by median {np.median(wob):.1f} deg, max {wob.max():.1f}')
        for i in range(T):
            R0[i] = R_fix
            t0[i] = placed[i].mean(0) - R_fix @ (Jh0[i].numpy() * scale).mean(0) if pnp_ok[i] else t0[i]
        if (~pnp_ok).any() and pnp_ok.any():
            good = np.where(pnp_ok)[0]
            for i in np.where(~pnp_ok)[0]:
                t0[i] = t0[good[np.argmin(np.abs(good - i))]]
    conf = kp2d[:, :, 2]
    w_obs = np.where(conf > args.conf_thresh, conf, 0.0)
    w_obs[~pnp_ok] = 0.0                                             # no 2D evidence used where PnP had none
    for name in [n for n in args.exclude_2d.split(',') if n]:
        w_obs[:, H36M.index(name)] = 0.0                             # joints left out of the reprojection term
    frame_ok = (w_obs > 0).sum(1) >= 6
    w_obs[~frame_ok] = 0.0
    log(f'  {cam}: {frame_ok.sum()}/{T} frames with >= 6 confident joints; '
        f'{int((w_obs > 0).sum())} 2D observations')

    # --- foot contact from the 2D alone: a planted foot is stationary in the image
    L_ext = pnp['cam_L_ext'].astype(np.float64)
    R_cam, t_cam = L_ext[:3, :3], L_ext[:3, 3]
    if np.abs(t_cam).max() > 50:                                     # calib in mm, placed joints in metres
        t_cam = t_cam / 1000.0
    segments = []                                                    # (ankle H36M index, frame indices)
    fps = float(twod['fps']) if 'fps' in twod.files else 60.0
    if args.w_foot > 0:
        for jn in ('RAnkle', 'LAnkle'):
            j = H36M.index(jn)
            xy = kp2d[:, j, :2].copy()
            good = (conf[:, j] > args.conf_thresh) & frame_ok
            xy[~good] = np.nan
            k = np.ones(5) / 5
            sm = np.stack([np.convolve(np.nan_to_num(xy[:, c]), k, 'same') /
                           np.maximum(np.convolve(good.astype(float), k, 'same'), 1e-9) for c in range(2)], 1)
            speed = np.linalg.norm(np.gradient(sm, axis=0), axis=1)          # mean velocity over the 5-frame window
            grounded = good & (speed < args.foot_speed_px)
            if args.foot_lowest_margin_px > 0:
                jo = H36M.index('LAnkle' if jn == 'RAnkle' else 'RAnkle')
                other_ok = conf[:, jo] > args.conf_thresh
                above = other_ok & (kp2d[:, j, 1] < kp2d[:, jo, 1] - args.foot_lowest_margin_px)   # image y grows downward
                grounded &= ~above
            grounded[~good] = False
            i = 0
            while i < T:
                if grounded[i]:
                    j0 = i
                    while i < T and grounded[i]:
                        i += 1
                    if i - j0 >= args.min_contact_frames:
                        segments.append((j, np.arange(j0, i)))
                else:
                    i += 1
        desc = ', '.join(f'{H36M[j]} {seg[0] / fps:.2f}-{seg[-1] / fps:.2f}s' for j, seg in segments) if segments else 'none'
        log(f'  {cam}: foot contact (ankle speed < {args.foot_speed_px} px/frame): {desc}')
    R_cam_t = torch.tensor(R_cam, dtype=torch.float64, device=dev)
    t_cam_t = torch.tensor(t_cam, dtype=torch.float64, device=dev)
    seg_t = [(j, torch.tensor(seg, device=dev)) for j, seg in segments]

    def foot_term(X_cam):
        """Sum over contact intervals of the ankle's horizontal lab-frame deviation from its interval mean."""
        if not seg_t:
            return torch.zeros((), dtype=torch.float64, device=dev)
        X_lab = (X_cam - t_cam_t) @ R_cam_t                          # X_cam = R X_lab + t  ->  X_lab = R^T (X_cam - t)
        e = torch.zeros((), dtype=torch.float64, device=dev)
        for j, seg in seg_t:
            xy = X_lab[seg, j, :2]
            e = e + ((xy - xy.mean(0, keepdim=True)) ** 2).sum()
        return args.w_foot * e

    ankle_idx = [H36M.index('RAnkle'), H36M.index('LAnkle')]
    ok_t = torch.tensor(frame_ok, device=dev)

    def floor_term(X_cam):
        """Hinge on any grounded-or-not ankle below the margin above the lab floor (z = 0), plus, with
        --w-floor-contact, a TWO-SIDED pull of every grounded ankle (the foot-contact intervals) to
        --floor-contact-m above the floor.  The hinge only stops the body sinking; moving it toward the
        camera along the viewing ray lifts the feet off the floor unpunished, which is exactly the
        single-view depth escape.  A grounded ankle at a known height above a calibrated floor pins depth."""
        e = torch.zeros((), dtype=torch.float64, device=dev)
        if args.w_floor <= 0 and not (args.w_floor_contact > 0 and seg_t):
            return e
        z_all = ((X_cam - t_cam_t) @ R_cam_t)[:, :, 2]
        if args.w_floor > 0:
            z = z_all[:, ankle_idx]
            pen = torch.clamp(args.floor_margin_m - z, min=0.0) ** 2
            e = e + args.w_floor * pen[ok_t].sum()
        if args.w_floor_contact > 0:
            for j, seg in seg_t:
                e = e + args.w_floor_contact * ((z_all[seg, j] - args.floor_contact_m) ** 2).sum()
        return e

    def foot_slide_mm(X_cam):
        if not seg_t:
            return float('nan')
        X_lab = (X_cam - t_cam_t) @ R_cam_t
        d = [((X_lab[seg, j, :2] - X_lab[seg, j, :2].mean(0, keepdim=True)) ** 2).sum(-1) for j, seg in seg_t]
        return 1000 * float(torch.sqrt(torch.cat(d).mean()))

    # --- tensors
    f64 = dict(dtype=torch.float64, device=dev)
    rot0_t = torch.tensor(rot0, **f64)
    R0_t, t0_t = torch.tensor(R0, **f64), torch.tensor(t0, **f64)
    obs = torch.tensor(kp2d[:, :, :2], **f64)
    W = torch.tensor(w_obs, **f64)
    K_t = torch.tensor(K, **f64)
    dg = torch.zeros(T, 3, **f64, requires_grad=True)                # global rotation offset
    dt = torch.zeros(T, 3, **f64, requires_grad=True)                # translation offset (m)
    db = torch.zeros(T, 23, 3, **f64, requires_grad=True)            # body joint offsets (axis-angle)
    sig, hub = args.sigma_px, args.huber

    def rotmats(db_):
        body = rodrigues(db_) @ rot0_t[:, 1:]
        return torch.cat([rot0_t[:, :1], body], 1)

    def place(Jh, dg_, dt_):
        R = rodrigues(dg_) @ R0_t
        return (R[:, None] @ (Jh * scale)[..., None])[..., 0] + (t0_t + dt_)[:, None]

    def objective(dg_, dt_, db_, use_body):
        _, Jh = smpl.forward(rotmats(db_) if use_body else rot0_t)
        X = place(Jh, dg_, dt_)
        r = (project(X, K_t, dist) - obs) / sig                        # (T,17,2)
        rho = hub ** 2 * (torch.sqrt(1 + (r / hub) ** 2) - 1)          # pseudo-Huber per coordinate
        e_rep = (W[..., None] * rho).sum()
        e_pull = args.w_pull * (db_ ** 2).sum() if use_body else torch.zeros((), **f64)
        e_sm = args.w_smooth * ((db_[1:] - db_[:-1]) ** 2).sum() if use_body else torch.zeros((), **f64)
        e_sm = e_sm + args.w_smooth * ((dg_[1:] - dg_[:-1]) ** 2).sum()
        acc = (t0_t + dt_)[2:] - 2 * (t0_t + dt_)[1:-1] + (t0_t + dt_)[:-2]
        e_tr = args.w_trans_acc * (acc ** 2).sum()
        e_ft = foot_term(X) + floor_term(X)
        return e_rep + e_pull + e_sm + e_tr + e_ft, (e_rep, e_pull, e_sm, e_tr, e_ft)

    def rms_px(dg_, dt_, db_, use_body):
        with torch.no_grad():
            _, Jh = smpl.forward(rotmats(db_) if use_body else rot0_t)
            r = project(place(Jh, dg_, dt_), K_t, dist) - obs
            d2 = (r ** 2).sum(-1)
            m = W > 0
            return float(torch.sqrt((d2[m]).mean()))

    def run_stage(params, use_body, iters, label):
        opt = torch.optim.LBFGS(params, max_iter=iters, history_size=30, tolerance_grad=1e-9,
                                tolerance_change=1e-12, line_search_fn='strong_wolfe')
        t_start = time.time()

        def closure():
            opt.zero_grad()
            loss, _ = objective(dg, dt, db, use_body)
            loss.backward()
            return loss
        opt.step(closure)
        loss, parts = objective(dg, dt, db, use_body)
        log(f'  {cam}: {label}: objective {float(loss):.1f} (reproj {float(parts[0]):.1f}, pull {float(parts[1]):.1f}, '
            f'smooth {float(parts[2]):.1f}, trans-acc {float(parts[3]):.1f}, foot {float(parts[4]):.1f}), reprojection RMS '
            f'{rms_px(dg, dt, db, use_body):.2f} px, {time.time() - t_start:.0f} s')

    with torch.no_grad():
        _, Jh_init = smpl.forward(rot0_t)
        slide0 = foot_slide_mm(place(Jh_init, dg, dt))
    log(f'  {cam}: start: reprojection RMS {rms_px(dg, dt, db, False):.2f} px (step_4 placement, MotionBERT pose)'
        + (f', grounded-ankle horizontal slide RMS {slide0:.0f} mm' if seg_t else ''))
    glob_params = [dt] if args.fix_rotation != 'none' else [dg, dt]
    run_stage(glob_params, False, args.iters_global, 'stage 1 (global ' + ('translation' if args.fix_rotation != 'none' else 'pose') + ' only)')
    if args.no_body:
        log(f'  {cam}: --no-body: MotionBERT rotations kept, placement only')
    else:
        run_stage(glob_params + [db], True, args.iters, 'stage 2 (global + body offsets)')

    # --- results
    with torch.no_grad():
        rot_new = rotmats(db)
        J24, Jh = smpl.forward(rot_new)
        R_new = rodrigues(dg) @ R0_t
        t_new = t0_t + dt
        H_cam = place(Jh, dg, dt).numpy()
        S_cam = ((R_new[:, None] @ (J24 * scale)[..., None])[..., 0] + t_new[:, None]).numpy()
        deg = np.degrees(torch.linalg.norm(db, dim=-1).numpy())      # (T,23)
    groups = {g: np.mean([deg[frame_ok][:, SMPL24.index(j) - 1].mean() for j in js]) for g, js in GROUPS.items()}
    log(f'  {cam}: mean |joint offset| over valid frames (deg): ' + ', '.join(f'{g} {v:.1f}' for g, v in groups.items()))
    if seg_t:
        log(f'  {cam}: grounded-ankle horizontal slide RMS after refinement: {foot_slide_mm(torch.tensor(H_cam)):.0f} mm')
    dz = (t_new - t0_t).numpy()[frame_ok]
    log(f'  {cam}: translation change vs PnP: mean {1000 * dz.mean(0).round(3)} mm, |max| {1000 * np.abs(dz).max():.0f} mm')

    H_cam[~frame_ok] = np.nan
    S_cam[~frame_ok] = np.nan
    mesh.update(rotmats=rot_new.numpy().astype(np.float32),
                kps_H36M_scaled=(Jh.numpy() * scale).astype(np.float32),
                kps_SMPL24_scaled=(J24.numpy() * scale).astype(np.float32),
                refined_2d=np.array(f'w_pull={args.w_pull} w_smooth={args.w_smooth} sigma_px={args.sigma_px} '
                                    f'huber={args.huber} exclude_2d={args.exclude_2d!r} w_foot={args.w_foot} '
                                    f'fix_rotation={args.fix_rotation} no_body={args.no_body} w_floor={args.w_floor} '
                                    f'w_floor_contact={args.w_floor_contact} floor_contact_m={args.floor_contact_m}'),
                rotmats_motionbert=rot0.astype(np.float32))
    np.savez(os.path.join(A, 'mesh', detector, f'{cam}_mesh_pose.npz'), **mesh)
    pnp.update(kps_H36M_placed=H_cam.astype(np.float32), kps_SMPL24_placed=S_cam.astype(np.float32),
               kps_H36M_smooth=H_cam, kps_SMPL24_smooth=S_cam, pnp_ok=frame_ok,
               refined_2d=np.array('articulated 2D refinement (sample_refine_2d.py); smooth == placed'))
    np.savez(os.path.join(A, 'PnP', detector, f'{cam}_pnp.npz'), **pnp)
    return groups


# ----------------------------------------------------------------------------- scoring
def run_step(script, out_dir, trial_root, *cli):
    env = dict(os.environ, BIOCV_OUT=out_dir, BIOCV_ROOT=trial_root)
    subprocess.run([sys.executable, os.path.join(PIPE, script), *cli], cwd=PIPE, env=env, check=True,
                   stdout=subprocess.DEVNULL)


def metrics(vdir, user, action, detector, gt):
    p = os.path.join(vdir, user, action, 'Analysis', 'diagnostics', detector,
                     'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def compare(src, dst, user, action, detector):
    lines = []
    for gt in ('mocap', 'triangulated'):
        a, b = metrics(src, user, action, detector, gt), metrics(dst, user, action, detector, gt)
        if a is None or b is None:
            lines.append(f'({gt}: not scored on both)')
            continue
        lines += [f'{user}/{action} against {gt} (mm): step_4 PnP placement  ->  2D-refined', '']
        lines.append(f'{"camera":>8} {"":>14} {"PnP":>10} {"refined":>10} {"change":>10}')
        for i, cam in enumerate(a['cameras']):
            j = list(b['cameras']).index(cam)
            for key, label in (('err_placed_all', 'MPJPE placed'), ('n_mpjpe', 'N-MPJPE'), ('pa_mpjpe', 'PA-MPJPE')):
                lines.append(f'{str(cam):>8} {label:>14} {a[key][i]:10.1f} {b[key][j]:10.1f} {b[key][j] - a[key][i]:+10.1f}')
        for key, label in (('err_placed_all', 'MPJPE placed'), ('n_mpjpe', 'N-MPJPE'), ('pa_mpjpe', 'PA-MPJPE')):
            lines.append(f'{"mean":>8} {label:>14} {a[key].mean():10.1f} {b[key].mean():10.1f} {b[key].mean() - a[key].mean():+10.1f}')
        lines += ['', f'{"joint":>10} {"PnP":>10} {"refined":>10} {"change":>10}   (placed, mean over cameras)']
        pa, pb = np.nanmean(a['perjoint_placed_all'], 0), np.nanmean(b['perjoint_placed_all'], 0)
        for k, nm in enumerate(a['joint_names']):
            if np.isfinite(pa[k]):
                lines.append(f'{str(nm):>10} {pa[k]:10.1f} {pb[k]:10.1f} {pb[k] - pa[k]:+10.1f}')
        lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=DEFAULT_SRC, help='OUT_DIR variant through step_4 to refine')
    ap.add_argument('--out', default=None, help='new OUT_DIR variant (default <src>/../refine2d)')
    ap.add_argument('--trial-root', default=None, help='BIOCV_ROOT for step_8 (default <src>/../../trial)')
    ap.add_argument('--detector', default='openpose')
    ap.add_argument('--cameras', default=None)
    ap.add_argument('--w-pull', type=float, default=30.0, help='weight on body offsets (rad^2); 30 ~ "10 deg costs 5 px"')
    ap.add_argument('--w-smooth', type=float, default=30.0, help='weight on frame-to-frame change of offsets (rad^2)')
    ap.add_argument('--w-trans-acc', type=float, default=1e4, help='weight on translation acceleration (m^2 per frame^2)')
    ap.add_argument('--sigma-px', type=float, default=5.0, help='reprojection residual scale')
    ap.add_argument('--huber', type=float, default=3.0, help='pseudo-Huber knee, in units of sigma')
    ap.add_argument('--conf-thresh', type=float, default=0.3)
    ap.add_argument('--exclude-2d', default='', help='H36M joints left out of the reprojection term, e.g. Hip,RHip,LHip')
    ap.add_argument('--fix-rotation', choices=['none', 'stance', 'mean', 'ray', 'ray_const'], default='none',
                    help="freeze the global rotation per camera: 'stance' = chordal mean of step_4's PnP rotations over the "
                         "first --stance-sec seconds (upright, MotionBERT reliable), 'mean' = over the whole trial; 'ray' = PER FRAME, "
                         "MotionBERT's own orientation with only the geometric off-axis correction (rotation taking the optical axis "
                         "to the ray through the 2D root), no rotation estimated at all; only the translation (and body offsets "
                         "unless --no-body) are then solved")
    ap.add_argument('--stance-sec', type=float, default=1.5)
    ap.add_argument('--no-body', action='store_true', help='keep MotionBERT rotations exactly: no body offsets, placement only')
    ap.add_argument('--w-floor', type=float, default=0.0,
                    help='hinge penalty (m^-2) on an ankle below --floor-margin-m above the lab floor z=0; 0 = off')
    ap.add_argument('--floor-margin-m', type=float, default=0.0)
    ap.add_argument('--w-floor-contact', type=float, default=0.0,
                    help='two-sided pull (m^-2) of a GROUNDED ankle (foot-contact intervals, so needs --w-foot > 0) to '
                         '--floor-contact-m above the lab floor: pins depth from the calibrated floor. 0 = off')
    ap.add_argument('--floor-contact-m', type=float, default=0.05,
                    help='height of a grounded ankle joint above the floor (OpenPose ankle on a ~1 m child: ~0.05 m)')
    ap.add_argument('--w-foot', type=float, default=0.0,
                    help='foot-contact weight (m^-2): while a foot is grounded its ankle keeps a constant horizontal '
                         'lab-frame position; 3000 makes a 10 mm slide cost like a 3 px reprojection residual. 0 = off')
    ap.add_argument('--foot-speed-px', type=float, default=1.5,
                    help='smoothed ankle pixel speed (px/frame) below which the foot is grounded (5-frame mean velocity, so '
                         'jitter averages out but a steady drift, e.g. a runner approaching the camera, does not)')
    ap.add_argument('--min-contact-frames', type=int, default=6, help='shorter grounded runs are ignored')
    ap.add_argument('--foot-lowest-margin-px', type=float, default=60.0,
                    help='a grounded ankle must not sit more than this many px ABOVE the other ankle in the image; 0 = off')
    ap.add_argument('--iters-global', type=int, default=60)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--no-score', action='store_true', help='skip step_8')
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    out = os.path.abspath(args.out or os.path.join(src, '..', 'refine2d'))
    trial_root = os.path.abspath(args.trial_root or os.path.join(src, '..', '..', 'trial'))
    hits = sorted(glob.glob(os.path.join(src, '*', '*', 'Analysis')))
    if len(hits) != 1:
        raise SystemExit(f'expected one {{user}}/{{action}}/Analysis under {src}, found {len(hits)}')
    action = os.path.basename(os.path.dirname(hits[0]))
    user = os.path.basename(os.path.dirname(os.path.dirname(hits[0])))
    pnp_dir = os.path.join(hits[0], 'PnP', args.detector)
    cams = args.cameras.split(',') if args.cameras else sorted(os.path.basename(p).split('_')[0]
                                                                for p in glob.glob(os.path.join(pnp_dir, '*_pnp.npz')))
    if not cams:
        raise SystemExit(f'no {{cam}}_pnp.npz under {pnp_dir} -- the source variant must be through step_4')

    def log(msg):
        print(msg, flush=True)
    log(f'{user}/{action} cameras {cams}\n  src {src}\n  out {out}')
    if os.path.isdir(out):
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns('diagnostics', 'features'))
    for cam in cams:
        log(f'\n### camera {cam}')
        refine_camera(out, user, action, args.detector, cam, args, log)
    if args.no_score:
        return
    log('\n### step_8')
    h36m = os.path.join(hits[0], 'H36M')
    for gt, fn in (('mocap', 'mocap_h36m.npz'), ('triangulated', 'openpose_tri_h36m.npz')):
        if not os.path.exists(os.path.join(h36m, fn)):
            log(f'  no {fn}: not scored against {gt}')       # Korea has no mocap
            continue
        run_step('step_8_spider_error.py', out, trial_root, '--user', user, '--action', action,
                 '--cameras', ','.join(cams), '--force', '--gt', gt, '--detector', args.detector)
    text = compare(src, out, user, action, args.detector)
    with open(os.path.join(out, 'comparison.txt'), 'w') as f:
        f.write(text + '\n')
    log('\n' + text)
    log(f'numbers in {os.path.join(out, "comparison.txt")}')


if __name__ == '__main__':
    main()
