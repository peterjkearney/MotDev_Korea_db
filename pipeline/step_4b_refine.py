#!/usr/bin/env python3
"""step_4b_refine.py -- place MotionBERT's pose in the camera under one or more CONFIGURATIONS.

step_4 is one recipe: MotionBERT's shape, rescaled to the stature, placed rigidly by PnP per
frame.  This runs the configurations in configs.py, each a different combination of

    shape        MotionBERT's betas | adult SMPL fitted to bone lengths | kid blend | cohort kid
    placement    rigid PnP | frozen ray rotation with the translation solved | free rotation
    body         MotionBERT's joint angles kept, or refined against the 2D (SMPLify-style,
                 L-BFGS over the whole trial: robust reprojection + pull toward MotionBERT +
                 temporal smoothness)
    2D joint set which H36M joints the reprojection term (or PnP) uses
    feet / floor foot pin while grounded, floor hinge, grounded ankle held at its floor height

and writes each result in step_4's file format one folder down, so step_8 --config scores it
and nothing else in the pipeline has to know:

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
            {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz         pass-1 rotations
            {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_final_betas.npz   shape 'motionbert'
            {OUT_DIR}/{user}/shapes.npz, {OUT_DIR}/cohort_shapes.npz                   step_2c shapes
            {user}/{cam}.mp4-mocAligned.calib, {user}/user_meta.json
    writes  {OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{config}/{cam}_pnp.npz

Every configuration computes its own PnP on its own skeleton (the frozen-rotation modes take
their per-trial constant from it), so step_4 need not have run.  The foot-contact detector
works from the 2D alone: an ankle whose smoothed pixel speed stays low for min_contact_s is
grounded.  The lab floor is z = 0 of the calib's lab frame (BioCV mocap frame, Korea fitted
floor).

    python3 step_4b_refine.py                            # every trial, every configuration; done ones skipped
    python3 step_4b_refine.py --configs C8 --user B023
    python3 step_4b_refine.py --configs chain --device cuda
    python3 configs.py                                   # the legend
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import configs
from config import (DETECTORS, OUT_DIR, twod_path, betas_path, final_betas_path, pnp_path, shapes_path,
                    cohort_shapes_path, calib_path, stature_path, find_cameras, require_out_dir)
from step_4_PnP import solve_root_pose, parse_joints, _PNP_CONF_THRESH
from utils.calibration import load_calib
from utils.rts_smoother import rts_smooth_3d
from utils.smpl_lite import H36M, SMPL24, MiniSMPL, load_linear, mesh_height_m, rodrigues, project, kabsch

GROUPS = {'trunk': ['Spine1', 'Spine2', 'Spine3', 'Neck', 'Head'], 'hips': ['L_Hip', 'R_Hip'],
          'knees': ['L_Knee', 'R_Knee'], 'ankles/feet': ['L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot'],
          'shoulders': ['L_Collar', 'R_Collar', 'L_Shoulder', 'R_Shoulder'], 'elbows': ['L_Elbow', 'R_Elbow'],
          'wrists/hands': ['L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']}


# ----------------------------------------------------------------------------- inputs
class Shapes:
    """Shape vectors and the step_3 scale for each configuration's 'shape'."""

    def __init__(self):
        self.lin = {10: load_linear(10), 11: load_linear(10, kid=True)}
        self._stature = {}

    def stature(self, user):
        if user not in self._stature:
            p = stature_path(user)
            if not os.path.exists(p):
                raise FileNotFoundError(f'{p} -- step_4b needs the stature for the MotionBERT shape')
            with open(p) as f:
                self._stature[user] = float(json.load(f)['stature_m'])
        return self._stature[user]

    def get(self, name, user, action, detector, cam):
        """-> (shape vector, scale factor, mesh T-pose height m, description)"""
        if name == 'motionbert':
            b = np.load(final_betas_path(user, action, detector, cam), allow_pickle=True)['betas'].ravel()[:10].astype(np.float64)
            h = mesh_height_m(self.lin[10], b)
            return b, self.stature(user) / h, h, f"MotionBERT betas, rescaled x{self.stature(user) / h:.3f} to {self.stature(user):.2f} m"
        if name in ('adult', 'kid'):
            p = shapes_path(user)
            if not os.path.exists(p):
                raise FileNotFoundError(f'{p} -- run step_2c_fit_shape.py first')
            b = np.load(p, allow_pickle=True)[name].astype(np.float64)
        elif name == 'cohort_kid':
            p = cohort_shapes_path()
            if not os.path.exists(p):
                raise FileNotFoundError(f'{p} -- run step_2c_fit_shape.py --cohort first')
            b = np.load(p, allow_pickle=True)['kid'].astype(np.float64)
        else:
            raise ValueError(f'unknown shape {name!r}')
        h = mesh_height_m(self.lin[len(b)], b)
        return b, 1.0, h, f'{name} shape (metric, T-pose {h:.3f} m' + (f', kid weight {b[-1]:.2f})' if len(b) > 10 else ')')


def pnp_init(Jh_m, kp2d, K, dist_cv, use_joint):
    """Rigid PnP per frame on this skeleton.  -> R (T,3,3), t (T,3), ok (T,), frames PnP failed on filled
    from the nearest solved one."""
    T = Jh_m.shape[0]
    R0, t0, ok = np.tile(np.eye(3), (T, 1, 1)), np.zeros((T, 3)), np.zeros(T, bool)
    for i in range(T):
        sol = solve_root_pose(Jh_m[i], kp2d[i, :, :2], kp2d[i, :, 2], K, dist_cv, use_joint)
        if sol is not None:
            R0[i], t0[i], ok[i] = sol[0].astype(np.float64), sol[1].astype(np.float64), True
    if ok.any() and not ok.all():
        good = np.where(ok)[0]
        for i in np.where(~ok)[0]:
            j = good[np.argmin(np.abs(good - i))]
            R0[i], t0[i] = R0[j], t0[j]
    return R0, t0, ok


# ----------------------------------------------------------------------------- one camera
def run_camera(user, action, detector, cam, cid, cfg, shapes, device, log):
    twod = np.load(twod_path(user, action, detector, cam), allow_pickle=True)
    kp2d = twod['h36m_2d'].astype(np.float64)
    fps = float(twod['fps']) if 'fps' in twod.files else 60.0
    rot0 = np.load(betas_path(user, action, detector, cam), allow_pickle=True)['rotmats'].astype(np.float64)
    T = min(rot0.shape[0], kp2d.shape[0])
    rot0, kp2d = rot0[:T], kp2d[:T]
    w, h, K, L_ext, dist = load_calib(calib_path(user, cam))
    K, L_ext = K.astype(np.float64), L_ext.astype(np.float64)
    dist_cv = np.asarray(dist, dtype=np.float64).reshape(1, 5)
    shape, scale, mesh_h, shape_desc = shapes.get(cfg['shape'], user, action, detector, cam)
    use_joint = parse_joints(cfg['exclude_2d'])
    log(f'  {cam}: {shape_desc}; {T} frames at {fps:.0f} fps')

    smpl = MiniSMPL(shape, device)
    f64 = dict(dtype=torch.float64, device=device)
    rot0_t = torch.tensor(rot0, **f64)
    with torch.no_grad():
        J24_0, Jh_0 = smpl.forward(rot0_t)
    Jh_m = Jh_0.cpu().numpy() * scale                       # (T,17,3) metres, mesh frame
    J24_m = J24_0.cpu().numpy() * scale

    R0, t0, pnp_ok = pnp_init(Jh_m, kp2d, K, dist_cv, use_joint)
    log(f'  {cam}: PnP solved {pnp_ok.sum()}/{T} frames' + (f' (without {cfg["exclude_2d"]})' if cfg['exclude_2d'] else ''))
    common = dict(pnp_ok=pnp_ok, detector=np.array(detector), pnp_joints_used=use_joint, pnp_conf_thresh=_PNP_CONF_THRESH,
                  cam_w=w, cam_h=h, cam_K=K, cam_L_ext=L_ext, cam_dist_cv=dist_cv,
                  config=np.array(cid), config_json=np.array(json.dumps(cfg)), shape=shape.astype(np.float32),
                  scale_factor=scale, mesh_height_m=mesh_h, fps=fps)

    if cfg['placement'] == 'pnp':
        H = np.full((T, 17, 3), np.nan, np.float32)
        S = np.full((T, 24, 3), np.nan, np.float32)
        for i in np.where(pnp_ok)[0]:
            H[i] = (R0[i] @ Jh_m[i].T).T + t0[i]
            S[i] = (R0[i] @ J24_m[i].T).T + t0[i]
        if pnp_ok.any():
            root_s, _ = rts_smooth_3d(H[:, 0, :], pnp_ok, H[:, 0, :], 1 / fps, sigma_along=0.15, sigma_perp=0.02,
                                      process_accel_std=10.0)
            adj = (root_s - H[:, 0, :])[:, None, :]
            Hs, Ss = H + adj, S + adj
        else:
            Hs, Ss = H, S
        return dict(common, kps_H36M_placed=H, kps_SMPL24_placed=S, kps_H36M_smooth=Hs, kps_SMPL24_smooth=Ss,
                    rotmats=rot0.astype(np.float32), refined=False)

    if not pnp_ok.any():
        raise RuntimeError('PnP solved no frame at all -- nothing to initialise the refinement from')

    # --- global rotation per the placement mode
    from scipy.spatial.transform import Rotation as Rot
    mode = cfg['placement']
    if mode in ('ray', 'ray_const'):
        root_px = kp2d[:, H36M.index('Hip'), :2].copy()
        good = kp2d[:, H36M.index('Hip'), 2] > cfg['conf_thresh']
        if not good.any():
            raise RuntimeError('no confident 2D root for the ray rotation')
        idx_good = np.where(good)[0]
        for i in np.where(~good)[0]:
            root_px[i] = root_px[idx_good[np.argmin(np.abs(idx_good - i))]]
        Kinv = np.linalg.inv(K)
        R_rays = []
        for i in range(T):
            d = Kinv @ np.array([root_px[i, 0], root_px[i, 1], 1.0])
            d /= np.linalg.norm(d)
            ax = np.cross([0.0, 0.0, 1.0], d)
            sn = np.linalg.norm(ax)
            R_rays.append(np.eye(3) if sn < 1e-9 else Rot.from_rotvec(ax / sn * np.arctan2(sn, d[2])).as_matrix())
        R_rays = np.array(R_rays)
        R_const = np.eye(3)
        if mode == 'ray_const':
            resid = Rot.from_matrix(np.array([R_rays[i].T @ R0[i] for i in np.where(pnp_ok)[0]]))
            R_const = resid.mean().as_matrix()
            log(f'  {cam}: constant residual rotation beyond the ray correction '
                f'{np.round(np.degrees(Rot.from_matrix(R_const).as_rotvec()), 1)} deg')
        for i in range(T):
            R_new = R_rays[i] @ R_const
            if pnp_ok[i]:
                t0[i] = ((R0[i] @ Jh_m[i].T).T + t0[i]).mean(0) - R_new @ Jh_m[i].mean(0)
            R0[i] = R_new
    elif mode in ('stance', 'mean'):
        sel = pnp_ok.copy()
        if mode == 'stance':
            first = np.where(pnp_ok)[0][0]
            sel &= (np.arange(T) - first) / fps < cfg['stance_sec']
            if sel.sum() < 5:
                sel = pnp_ok.copy()
        R_fix = Rot.from_matrix(R0[sel]).mean().as_matrix()
        for i in range(T):
            if pnp_ok[i]:
                t0[i] = ((R0[i] @ Jh_m[i].T).T + t0[i]).mean(0) - R_fix @ Jh_m[i].mean(0)
            R0[i] = R_fix
    elif mode != 'free':
        raise ValueError(f'unknown placement {mode!r}')
    if mode != 'free' and not pnp_ok.all():
        good_t = np.where(pnp_ok)[0]
        for i in np.where(~pnp_ok)[0]:
            t0[i] = t0[good_t[np.argmin(np.abs(good_t - i))]]

    # --- 2D observations
    conf = kp2d[:, :, 2]
    w_obs = np.where(conf > cfg['conf_thresh'], conf, 0.0)
    w_obs[~pnp_ok] = 0.0
    w_obs[:, ~use_joint] = 0.0
    frame_ok = (w_obs > 0).sum(1) >= 6
    w_obs[~frame_ok] = 0.0

    # --- foot contact from the 2D
    R_cam, t_cam = L_ext[:3, :3], L_ext[:3, 3]
    if np.abs(t_cam).max() > 50:                                    # calib in mm, skeleton in metres
        t_cam = t_cam / 1000.0
    min_contact = max(2, int(round(cfg['min_contact_s'] * fps)))
    speed_thr = cfg['foot_speed_px_s'] / fps
    segments = []
    if cfg['w_foot'] > 0 or cfg['w_floor_contact'] > 0:
        for jn in ('RAnkle', 'LAnkle'):
            j = H36M.index(jn)
            xy = kp2d[:, j, :2].copy()
            good = (conf[:, j] > cfg['conf_thresh']) & frame_ok
            xy[~good] = np.nan
            k = np.ones(5) / 5
            sm = np.stack([np.convolve(np.nan_to_num(xy[:, c]), k, 'same') /
                           np.maximum(np.convolve(good.astype(float), k, 'same'), 1e-9) for c in range(2)], 1)
            speed = np.linalg.norm(np.gradient(sm, axis=0), axis=1)
            grounded = good & (speed < speed_thr)
            if cfg['foot_lowest_margin_px'] > 0:
                jo = H36M.index('LAnkle' if jn == 'RAnkle' else 'RAnkle')
                above = (conf[:, jo] > cfg['conf_thresh']) & (kp2d[:, j, 1] < kp2d[:, jo, 1] - cfg['foot_lowest_margin_px'])
                grounded &= ~above
            i = 0
            while i < T:
                if grounded[i]:
                    j0 = i
                    while i < T and grounded[i]:
                        i += 1
                    if i - j0 >= min_contact:
                        segments.append((j, np.arange(j0, i)))
                else:
                    i += 1
        desc = ', '.join(f'{H36M[j]} {s[0] / fps:.2f}-{s[-1] / fps:.2f}s' for j, s in segments) if segments else 'none'
        log(f'  {cam}: foot contact (ankle speed < {speed_thr:.1f} px/frame, >= {min_contact} frames): {desc}')

    R_cam_t, t_cam_t = torch.tensor(R_cam, **f64), torch.tensor(t_cam, **f64)
    seg_t = [(j, torch.tensor(s, device=device)) for j, s in segments]
    ankle_idx = [H36M.index('RAnkle'), H36M.index('LAnkle')]
    ok_t = torch.tensor(frame_ok, device=device)
    R0_t, t0_t = torch.tensor(R0, **f64), torch.tensor(t0, **f64)
    obs, W, K_t = torch.tensor(kp2d[:, :, :2], **f64), torch.tensor(w_obs, **f64), torch.tensor(K, **f64)
    dg = torch.zeros(T, 3, **f64, requires_grad=True)
    dt = torch.zeros(T, 3, **f64, requires_grad=True)
    db = torch.zeros(T, 23, 3, **f64, requires_grad=True)
    scale_t = float(scale)
    sig, hub = cfg['sigma_px'], cfg['huber']

    def rotmats(db_):
        return torch.cat([rot0_t[:, :1], rodrigues(db_) @ rot0_t[:, 1:]], 1)

    def place_(Jh, dg_, dt_):
        R = rodrigues(dg_) @ R0_t
        return (R[:, None] @ (Jh * scale_t)[..., None])[..., 0] + (t0_t + dt_)[:, None]

    def floor_terms(X_cam):
        e = torch.zeros((), **f64)
        if cfg['w_foot'] <= 0 and cfg['w_floor'] <= 0 and cfg['w_floor_contact'] <= 0:
            return e
        X_lab = (X_cam - t_cam_t) @ R_cam_t
        if cfg['w_floor'] > 0:
            pen = torch.clamp(-X_lab[:, ankle_idx, 2], min=0.0) ** 2
            e = e + cfg['w_floor'] * pen[ok_t].sum()
        for j, seg in seg_t:
            if cfg['w_foot'] > 0:
                xy = X_lab[seg, j, :2]
                e = e + cfg['w_foot'] * ((xy - xy.mean(0, keepdim=True)) ** 2).sum()
            if cfg['w_floor_contact'] > 0:
                e = e + cfg['w_floor_contact'] * ((X_lab[seg, j, 2] - cfg['floor_contact_m']) ** 2).sum()
        return e

    def objective(use_body):
        _, Jh = smpl.forward(rotmats(db) if use_body else rot0_t)
        X = place_(Jh, dg, dt)
        r = (project(X, K_t, dist_cv) - obs) / sig
        rho = hub ** 2 * (torch.sqrt(1 + (r / hub) ** 2) - 1)
        e_rep = (W[..., None] * rho).sum()
        e_pull = cfg['w_pull'] * (db ** 2).sum() if use_body else torch.zeros((), **f64)
        e_sm = cfg['w_smooth'] * ((db[1:] - db[:-1]) ** 2).sum() if use_body else torch.zeros((), **f64)
        e_sm = e_sm + cfg['w_smooth'] * ((dg[1:] - dg[:-1]) ** 2).sum()
        tr = t0_t + dt
        e_tr = cfg['w_trans_acc'] * ((tr[2:] - 2 * tr[1:-1] + tr[:-2]) ** 2).sum()
        return e_rep + e_pull + e_sm + e_tr + floor_terms(X), (e_rep, e_pull, e_sm, e_tr)

    def rms_px(use_body):
        with torch.no_grad():
            _, Jh = smpl.forward(rotmats(db) if use_body else rot0_t)
            d2 = ((project(place_(Jh, dg, dt), K_t, dist_cv) - obs) ** 2).sum(-1)
            m = W > 0
            return float(torch.sqrt(d2[m].mean())) if m.any() else float('nan')

    def stage(params, use_body, iters, label):
        opt = torch.optim.LBFGS(params, max_iter=iters, history_size=30, tolerance_grad=1e-9,
                                tolerance_change=1e-12, line_search_fn='strong_wolfe')
        t_start = time.time()

        def closure():
            opt.zero_grad()
            loss, _ = objective(use_body)
            loss.backward()
            return loss
        opt.step(closure)
        loss, parts = objective(use_body)
        log(f'  {cam}: {label}: objective {float(loss):.1f} (reproj {float(parts[0]):.1f}, pull {float(parts[1]):.1f}, '
            f'smooth {float(parts[2]):.1f}, trans-acc {float(parts[3]):.1f}), reprojection RMS {rms_px(use_body):.2f} px, '
            f'{time.time() - t_start:.0f} s')

    log(f'  {cam}: start: reprojection RMS {rms_px(False):.2f} px ({mode} rotation, MotionBERT pose)')
    glob_params = [dg, dt] if mode == 'free' else [dt]
    stage(glob_params, False, cfg['iters_global'], 'stage 1 (global ' + ('pose' if mode == 'free' else 'translation') + ')')
    if cfg['body']:
        stage(glob_params + [db], True, cfg['iters'], 'stage 2 (global + body angles)')

    with torch.no_grad():
        rot_new = rotmats(db) if cfg['body'] else rot0_t
        J24, Jh = smpl.forward(rot_new)
        R_new = rodrigues(dg) @ R0_t
        t_new = t0_t + dt
        H = place_(Jh, dg, dt).cpu().numpy()
        S = ((R_new[:, None] @ (J24 * scale_t)[..., None])[..., 0] + t_new[:, None]).cpu().numpy()
        if cfg['body']:
            deg = np.degrees(torch.linalg.norm(db, dim=-1).cpu().numpy())
            groups = {g: np.mean([deg[frame_ok][:, SMPL24.index(j) - 1].mean() for j in js]) for g, js in GROUPS.items()}
            log(f'  {cam}: mean |joint offset| (deg): ' + ', '.join(f'{g} {v:.1f}' for g, v in groups.items()))
        dz = (t_new - t0_t).cpu().numpy()[frame_ok]
        log(f'  {cam}: translation change vs PnP: mean {1000 * dz.mean(0).round(3)} mm, |max| {1000 * np.abs(dz).max():.0f} mm')
    H[~frame_ok] = np.nan
    S[~frame_ok] = np.nan
    return dict(common, pnp_ok=frame_ok, kps_H36M_placed=H.astype(np.float32), kps_SMPL24_placed=S.astype(np.float32),
                kps_H36M_smooth=H.astype(np.float32), kps_SMPL24_smooth=S.astype(np.float32),
                rotmats=rot_new.cpu().numpy().astype(np.float32), refined=True)


# ----------------------------------------------------------------------------- batch
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detector', choices=DETECTORS, default='openpose')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with pass-1 rotations')
    ap.add_argument('--configs', default='all', help="'all', 'chain', 'loo', or comma-separated ids (configs.py)")
    ap.add_argument('--device', default='auto', help='cpu | cuda | auto')
    ap.add_argument('--force', action='store_true', help='redo cameras whose output exists')
    ap.add_argument('--verbose', action='store_true', help='print the solver log for every camera')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    require_out_dir()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    want = set(args.cameras.split(',')) if args.cameras else None
    work = find_cameras(betas_path, args.detector, args.user, args.action, want)
    if not work:
        raise SystemExit(f'no {args.detector} {{cam}}_betas.npz under {OUT_DIR} -- run step_2a first')
    sel = configs.select(args.configs)
    todo = [(u, a, c, cid, cfg) for (u, a, c) in work for cid, cfg in sel
            if args.force or not os.path.exists(pnp_path(u, a, args.detector, c, cid))]
    print(f'{len(work)} camera(s) x {len(sel)} configuration(s) under {OUT_DIR}: {len(todo)} to run, '
          f'{len(work) * len(sel) - len(todo)} already done; device {device}')
    if args.dry_run:
        for u, a, c, cid, _ in todo:
            print(f'  {u}/{a} cam {c} {cid}')
        return
    shapes = Shapes()
    log = print if args.verbose else (lambda *_: None)
    n_ok, failed, t_all = 0, [], time.time()
    for u, a, c, cid, cfg in todo:
        t0 = time.time()
        try:
            out = run_camera(u, a, args.detector, c, cid, cfg, shapes, device, log)
            p = pnp_path(u, a, args.detector, c, cid)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, **out)
            print(f'{u}/{a} cam {c} {cid}: ok, {int(out["pnp_ok"].sum())} frames, {time.time() - t0:.0f} s', flush=True)
            n_ok += 1
        except Exception as e:                                       # one bad camera must not stop the batch
            failed.append((u, a, c, cid))
            print(f'{u}/{a} cam {c} {cid}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_ok}, failed {len(failed)}, {(time.time() - t_all) / 60:.1f} min')
    for f in failed:
        print('  failed:', *f)


if __name__ == '__main__':
    main()
