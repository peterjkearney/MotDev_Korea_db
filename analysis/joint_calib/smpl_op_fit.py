#!/usr/bin/env python3
"""smpl_op_fit.py -- where OpenPose puts the hips and shoulders relative to SMPL's own joints.

The 'op' joint sets (smpl_lite.OP_OFFSETS_MM) were OpenPose minus MOCAP joint centres, added to SMPL-24 as if
SMPL-24 were mocap; it is not (op_offsets.py, C15-C17).  This calibrates against SMPL itself: per subject, one
shape and a pose per frame are fitted to triangulated OpenPose BODY_25 (metres, lab frame), with

    anchors       knees, ankles, elbows, wrists  = SMPL-24 joints (SMPLify's convention for OpenPose limbs);
                  nose, eyes, ears, toes, heels = SMPL surface vertices (smplx's vertex_ids)
    free offsets  RHip/LHip   = SMPL-24 hip      + R_pelvis @ d_hip        (one d per subject, L/R mirrored)
                  RShoulder/LShoulder = SMPL-24 shoulder + R_bone @ d_sh   (bone: Spine3 as smpl_lite, or collar)
                  Neck = SMPL-24 Neck + R_spine3 @ d_neck,  MidHip = SMPL-24 Pelvis + R_pelvis @ d_midhip

so the hip and shoulder keypoints constrain the pelvis and thorax only through how they MOVE with them, and the
constant part is read off as the offset.  Joints are skinned-vertex regressions, as MiniSMPL (step_4b) computes
them, so the offsets are relative to what the pipeline's model actually puts out.  Priors: ridge on the betas
(kid weight free in [0,1] for children), L2 on the spine/neck/collar rotations (offsets are relative to a
neutral collar and spine), a light L2 on the rest.

Offsets are reported as (lateral, up, forward) mm in smpl_lite._segment_frame of the fitted body's rest pose
(lateral away from the midline), i.e. directly comparable with OP_OFFSETS_MM, and scaled to OP_SIZE_MM by the
fitted body's op_size.  Also per-keypoint residuals, and the hip residual against hip flexion (does the hip
keypoint ride on the pelvis alone, or partly on the thigh?).

    python3 analysis/joint_calib/smpl_op_fit.py biocv data/sampleBiocv --subjects P03 --out-dir <dir>
    python3 analysis/joint_calib/smpl_op_fit.py korea data/sampleKorea/out_v4 --out-dir <dir>

BioCV: {root}/{P}_{ACTION}/Analysis/{keypoints/,}openpose/{cam}_2d.npz (body25_2d) with calibs from
--calib-root/{P}/ (the Expansion drive); frames gated against mocap_h36m.npz when present.
Korea: {root}/{user}/{rep}/Analysis/keypoints/openpose/{cam}_2d.npz with the rep's (else the user's) calibs;
frames kept only where the official target (openpose_tri_h36m.npz) is solved and agrees with this triangulation.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))
from utils.calibration import load_calib  # noqa: E402
from utils.openpose import BODY25_NAMES, present_mask  # noqa: E402
from utils.triangulate import triangulate_robust  # noqa: E402
from utils.smpl_lite import (SMPL24, OPENPOSE_FOOT_VERTICES, OP_SIZE_MM, OP_OFFSETS_MM, J as HJ, _load_pkl,  # noqa: E402
                             kid_shapedir, rodrigues, op_size, _segment_frame)

B = {n: i for i, n in enumerate(BODY25_NAMES)}
S24 = {n: i for i, n in enumerate(SMPL24)}
FACE_VERTICES = {'Nose': 332, 'REye': 6260, 'LEye': 2800, 'REar': 4071, 'LEar': 583}
SURFACE = {**FACE_VERTICES, **OPENPOSE_FOOT_VERTICES}
ANCHOR_JOINTS = {f'{s}{n}': f'{s[0]}_{n}' for s in ('R', 'L') for n in ('Knee', 'Ankle', 'Elbow', 'Wrist')}
# free-offset keypoints: (SMPL-24 base joint, carrying bone, offset group, mirrored side or '')
FREE = {'RHip': ('R_Hip', 'Pelvis', 'hip', 'R'), 'LHip': ('L_Hip', 'Pelvis', 'hip', 'L'),
        'RShoulder': ('R_Shoulder', 'Spine3', 'shoulder', 'R'), 'LShoulder': ('L_Shoulder', 'Spine3', 'shoulder', 'L'),
        'Neck': ('Neck', 'Spine3', 'neck', ''), 'MidHip': ('Pelvis', 'Pelvis', 'midhip', '')}
GROUPS = ('hip', 'shoulder', 'neck', 'midhip')
# --two-bone: a second offset carried by the child bone (thigh / upper arm), so a keypoint can move with the limb
CHILD = {'RHip': ('R_Hip', 'hip_c'), 'LHip': ('L_Hip', 'hip_c'), 'RShoulder': ('R_Shoulder', 'shoulder_c'),
         'LShoulder': ('L_Shoulder', 'shoulder_c')}
CHILD_GROUPS = ('hip_c', 'shoulder_c')
# group -> smpl_lite segment frame it is reported in
REPORT_FRAME = {'hip': 'pelvis', 'shoulder': 'thorax', 'neck': 'thorax', 'midhip': 'pelvis', 'hip_c': 'pelvis',
                'shoulder_c': 'thorax'}     # child offsets: in the rest pose the limb frames are the parent's
WEIGHT = {**{k: 1.0 for k in ANCHOR_JOINTS}, **{k: 1.0 for k in FREE}, **{k: 0.5 for k in SURFACE}}
TORSO_POSE = ['Spine1', 'Spine2', 'Spine3', 'Neck', 'L_Collar', 'R_Collar']
FIXED_POSE = ['L_Hand', 'R_Hand']                     # nothing observed beyond the wrist / hand


# ----------------------------------------------------------------------------------------------- data
def triangulate(xy, conf, calibs, conf_thresh, reproj):
    vis = (conf > conf_thresh) & present_mask(xy, conf)
    X, used, _ = triangulate_robust(xy, vis, calibs, reproj)
    return X, used.sum(1)


def biocv_trials(root, subject):
    out = []
    for d in sorted(glob.glob(os.path.join(root, f'{subject}_*'))):
        for sub in ('Analysis/keypoints/openpose', 'Analysis/openpose'):
            if len(glob.glob(os.path.join(d, sub, '*_2d.npz'))) >= 3:
                out.append((d, os.path.join(d, sub)))
                break
    return out


def load_views(kp_dir, calib_dir_list, frames=None):
    cams = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(kp_dir, '*_2d.npz')))
    xy, conf, calibs, sfi = [], [], [], None
    for c in cams:
        z = np.load(os.path.join(kp_dir, f'{c}_2d.npz'))
        s = z['source_frame_idx']
        sfi = s if sfi is None else sfi[:min(len(sfi), len(s))]
        xy.append(z['body25_2d'][..., :2])
        conf.append(z['body25_2d'][..., 2])
        cp = next(p for p in (os.path.join(r, f'{c}.mp4-mocAligned.calib') for r in calib_dir_list) if os.path.exists(p))
        _, _, K, L, dist = load_calib(cp)
        calibs.append((K, dist, L[:3, :3], L[:3, 3]))
    n = len(sfi)
    xy = np.stack([x[:n] for x in xy], 1).astype(float)
    conf = np.stack([x[:n] for x in conf], 1).astype(float)
    if frames is not None:
        xy, conf, sfi = xy[frames], conf[frames], sfi[frames]
    return xy, conf, calibs, sfi


CORE25 = ['RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'LShoulder', 'LElbow', 'LWrist',
          'RShoulder', 'RElbow', 'RWrist']


def load_biocv(root, subject, calib_root, every, max_frames, rng):
    out = []
    for d, kp in biocv_trials(root, subject):
        n = len(np.load(glob.glob(os.path.join(kp, '*_2d.npz'))[0])['source_frame_idx'])
        idx = np.arange(0, n, every)
        xy, conf, calibs, sfi = load_views(kp, [os.path.join(calib_root, subject)], idx)
        X, nv = triangulate(xy, conf, calibs, 0.3, 15.0)
        ok = np.isfinite(X[:, [B[k] for k in CORE25]]).all((1, 2))
        mp = os.path.join(d, 'Analysis', 'H36M', 'mocap_h36m.npz')
        if os.path.exists(mp):                         # wrong person / L-R swaps: > 150 mm from mocap
            M = np.load(mp)['kps3d']
            h36 = [HJ[k] for k in CORE25]
            m = np.full((len(sfi), 12, 3), np.nan)
            inr = sfi < len(M)
            m[inr] = M[sfi[inr]][:, h36]
            e = np.linalg.norm(X[:, [B[k] for k in CORE25]] - m, axis=-1)
            ok &= np.nan_to_num(e, nan=1e9).max(1) < 150
        out.append((os.path.basename(d), X[ok] / 1000.0, nv[ok]))
        print(f'  {os.path.basename(d)}: {ok.sum()}/{len(ok)} frames', flush=True)
    return pool(out, max_frames, rng)


def load_korea(root, user, every, max_frames, rng):
    out = []
    for tp in sorted(glob.glob(os.path.join(root, user, '*', 'Analysis', 'H36M', 'openpose_tri_h36m.npz'))):
        rep_dir = os.path.dirname(os.path.dirname(os.path.dirname(tp)))
        kp = os.path.join(rep_dir, 'Analysis', 'keypoints', 'openpose')
        G = np.load(tp)['kps3d']
        n = len(np.load(glob.glob(os.path.join(kp, '*_2d.npz'))[0])['source_frame_idx'])
        idx = np.arange(0, min(n, len(G)), every)
        xy, conf, calibs, sfi = load_views(kp, [rep_dir, os.path.join(root, user)], idx)
        X, nv = triangulate(xy, conf, calibs, 0.3, 20.0)
        g = G[idx][:, [HJ[k] for k in CORE25]]
        e = np.linalg.norm(X[:, [B[k] for k in CORE25]] - g, axis=-1)
        ok = np.isfinite(g).all((1, 2)) & (np.nan_to_num(e, nan=1e9).max(1) < 30)
        out.append((os.path.basename(rep_dir), X[ok] / 1000.0, nv[ok]))
        print(f'  {os.path.basename(rep_dir)}: {ok.sum()}/{len(ok)} frames', flush=True)
    return pool(out, max_frames, rng)


def load_gt3d(root, user, every, max_frames, rng):
    """build_child_gt.py's own output ({root}/{user}/{rep}.npz, GT3D_v4): the all-camera BODY_25 in camera 1's
    frame (index 0 of X_body25_cam), metres at the session's stature scale, frames its gates passed."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, user, f'{user}_*.npz'))):
        z = np.load(p)
        if not bool(z['usable']):
            continue
        X = z['X_body25_cam'][0].astype(float)
        X[~z['valid_body25']] = np.nan
        full = np.isfinite(X[:, [B[k] for k in CORE25 + ['Neck']]]).all((1, 2))
        keep = np.where(z['frame_usable'] & full)[0][::every]
        out.append((os.path.basename(p)[:-4], X[keep], z['n_views_body25'][keep]))
    print(f'  {len(out)} usable reps, {sum(len(x) for _, x, _ in out)} frames', flush=True)
    if not out:
        return np.zeros((0, 25, 3)), np.zeros((0, 25)), np.array([])
    return pool(out, max_frames, rng)


def pool(out, max_frames, rng):
    X = np.concatenate([x for _, x, _ in out])
    nv = np.concatenate([v for _, _, v in out])
    src = np.concatenate([[t] * len(x) for t, x, _ in out])
    if len(X) > max_frames:
        k = np.sort(rng.choice(len(X), max_frames, replace=False))
        X, nv, src = X[k], nv[k], src[k]
    return X, nv, src


# ----------------------------------------------------------------------------------------------- model
class SMPLFit(torch.nn.Module):
    """SMPL LBS at the vertices the keypoints need, with shape, pose and the offsets all free."""

    def __init__(self, T, kid, shoulder_bone, two_bone=False):
        super().__init__()
        m = _load_pkl()
        Jr = m['J_regressor']
        Jr = Jr.toarray() if hasattr(Jr, 'toarray') else np.asarray(Jr)
        surf = list(SURFACE.values())
        idx = np.unique(np.r_[np.where(np.abs(Jr).sum(0) > 0)[0], surf])
        self.surf_idx = torch.tensor([int(np.searchsorted(idx, v)) for v in surf])
        v_t = np.asarray(m['v_template'], np.float64)
        S = np.asarray(m['shapedirs'], np.float64)[:, :, :10]
        if kid:
            S = np.concatenate([S, kid_shapedir(v_t)[:, :, None]], 2)
        f64 = dict(dtype=torch.float64)
        self.v_t, self.S = torch.tensor(v_t[idx], **f64), torch.tensor(S[idx], **f64)
        self.posedirs = torch.tensor(np.asarray(m['posedirs'])[idx].reshape(len(idx) * 3, 207).T, **f64)
        self.W = torch.tensor(np.asarray(m['weights'])[idx], **f64)
        self.Jr = torch.tensor(Jr[:, idx], **f64)
        self.parents = np.asarray(m['kintree_table'])[0].astype(np.int64)
        self.kid = kid
        self.n = len(idx)
        self.betas = torch.nn.Parameter(torch.zeros(S.shape[-1], **f64))
        self.root = torch.nn.Parameter(torch.zeros(T, 3, **f64))
        self.trans = torch.nn.Parameter(torch.zeros(T, 3, **f64))
        self.pose = torch.nn.Parameter(torch.zeros(T, 23, 3, **f64))
        self.groups = GROUPS + (CHILD_GROUPS if two_bone else ())
        self.offsets = torch.nn.ParameterDict({g: torch.nn.Parameter(torch.zeros(3, **f64)) for g in self.groups})
        self.two_bone = two_bone
        self.free = dict(FREE)
        if shoulder_bone == 'collar':
            self.free['RShoulder'] = ('R_Shoulder', 'R_Collar', 'shoulder', 'R')
            self.free['LShoulder'] = ('L_Shoulder', 'L_Collar', 'shoulder', 'L')
        self.pose_mask = torch.ones(23, 1, **f64)
        for nm in FIXED_POSE:
            self.pose_mask[S24[nm] - 1] = 0

    def shape(self):
        b = self.betas
        if self.kid:
            b = torch.cat([b[:-1], b[-1:].clamp(0, 1)])
        return b

    def rest(self):
        v = self.v_t + self.S @ self.shape()
        return v, self.Jr @ v

    def forward(self, frames=None):
        sl = slice(None) if frames is None else frames
        v_sh, J_rest = self.rest()
        aa = torch.cat([self.root[sl, None], self.pose[sl] * self.pose_mask], 1)
        R = rodrigues(aa)
        T = R.shape[0]
        eye = torch.eye(3, dtype=R.dtype)
        v_posed = v_sh + ((R[:, 1:] - eye).reshape(T, 207) @ self.posedirs).reshape(T, self.n, 3)
        G = [None] * 24
        rel = torch.cat([J_rest[:1], J_rest[1:] - J_rest[self.parents[1:]]])
        for k in range(24):
            Tk = torch.zeros(T, 4, 4, dtype=R.dtype)
            Tk[:, :3, :3], Tk[:, :3, 3], Tk[:, 3, 3] = R[:, k], rel[k], 1.0
            G[k] = Tk if k == 0 else G[self.parents[k]] @ Tk
        G = torch.stack(G, 1)
        A = G.clone()
        A[:, :, :3, 3] = G[:, :, :3, 3] - (G[:, :, :3, :3] @ J_rest[None, :, :, None])[..., 0]
        Tv = torch.einsum('nk,tkij->tnij', self.W, A)
        v = (Tv[:, :, :3, :3] @ v_posed[..., None])[..., 0] + Tv[:, :, :3, 3]
        J24 = torch.einsum('jn,tnc->tjc', self.Jr, v)
        t = self.trans[sl][:, None]
        kp = {n: J24[:, S24[s]] for n, s in ANCHOR_JOINTS.items()}
        kp.update({n: v[:, i] for n, i in zip(SURFACE, self.surf_idx)})
        mirror = torch.tensor([-1.0, 1.0, 1.0], dtype=R.dtype)
        for n, (base, bone, g, side) in self.free.items():
            d = self.offsets[g] * (mirror if side == 'R' else 1.0)
            kp[n] = J24[:, S24[base]] + G[:, S24[bone], :3, :3] @ d
            if self.two_bone and n in CHILD:
                cb, cg = CHILD[n]
                kp[n] = kp[n] + G[:, S24[cb], :3, :3] @ (self.offsets[cg] * (mirror if side == 'R' else 1.0))
        return {n: x + t[:, 0] for n, x in kp.items()}, J24 + t, R


def init_root(X):
    """Per-frame root rotation (rest SMPL frame: x left, y up, z forward) and translation from the hips/neck."""
    L, Rh, mid, neck = X[:, B['LHip']], X[:, B['RHip']], X[:, B['MidHip']], X[:, B['Neck']]
    mid = np.where(np.isfinite(mid), mid, 0.5 * (L + Rh))
    x = L - Rh
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    up = neck - mid
    y = up - (up * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    Rm = np.stack([x, y, z], -1)                         # columns: rest axes in the lab
    from scipy.spatial.transform import Rotation
    aa = Rotation.from_matrix(Rm).as_rotvec()
    return aa, mid


def gm(r2, s):
    """Geman-McClure on squared residuals, scale s (m)."""
    return r2 / (r2 + s * s) * s * s


def fit_subject(X, kid, args, log):
    T = len(X)
    names = [n for n in WEIGHT]
    tgt = torch.tensor(np.stack([X[:, B[n]] for n in names], 1))            # (T,K,3)
    ok = torch.isfinite(tgt).all(-1)
    tgt = torch.nan_to_num(tgt)
    w = torch.tensor([WEIGHT[n] for n in names], dtype=torch.float64)[None] * ok
    model = SMPLFit(T, kid, args.shoulder_bone, args.two_bone)
    aa, mid = init_root(X)
    with torch.no_grad():
        model.root.copy_(torch.tensor(aa))
        if kid:
            model.betas[-1] = 0.6
        _, J_rest = model.rest()
        Rr = rodrigues(model.root)
        model.trans.copy_(torch.tensor(mid) - Rr @ J_rest[0])
        for nm, sgn in (('L_Shoulder', -1.0), ('R_Shoulder', 1.0)):       # arms down, as most frames are
            model.pose[:, S24[nm] - 1, 2] = sgn * np.radians(75.0)
    torso = torch.tensor([n in ('LHip', 'RHip', 'LShoulder', 'RShoulder', 'Neck', 'MidHip', 'LKnee', 'RKnee')
                          for n in names], dtype=torch.float64)
    torso_idx = torch.tensor([S24[n] - 1 for n in TORSO_POSE])

    def loss(scale_mm, torso_only):
        kp, _, _ = model()
        P = torch.stack([kp[n] for n in names], 1)
        r2 = ((P - tgt) ** 2).sum(-1)
        ww = w * (torso[None] if torso_only else 1.0)
        data = (ww * gm(r2, scale_mm / 1000.0)).sum() / ww.sum() * 1e6 * (args.robust_mm / scale_mm) ** 2
        b = model.betas
        prior = args.lam_beta * (b[:10] ** 2).sum()
        pose = model.pose
        prior = prior + args.lam_torso * (pose[:, torso_idx] ** 2).sum() / T
        prior = prior + args.lam_pose * (pose ** 2).sum() / T
        prior = prior + sum(((model.offsets[g] * 1000.0 / (args.sigma_child_mm if g in CHILD_GROUPS else
                                                           args.sigma_offset_mm)) ** 2).sum() for g in model.groups)
        return data + prior, data

    t0 = time.time()
    rt, rtp = [model.root, model.trans], [model.root, model.trans, model.pose]
    # coarse to fine: a wide robust scale first so far-off limbs still pull, then the working scale
    for stage, (params, iters, scale, torso_only) in enumerate(((rt, 60, 300.0, True), (rtp, 100, 300.0, False),
                                                                (rtp, 100, 100.0, False),
                                                                (list(model.parameters()), args.iters, args.robust_mm,
                                                                 False))):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            L, _ = loss(scale, torso_only)
            L.backward()
            return L
        opt.step(closure)
        with torch.no_grad():
            L, D = loss(scale, torso_only)
        log(f'    stage {stage}: loss {L.item():.1f} (data {D.item():.1f}) [{time.time() - t0:.0f}s]')
    return model, names


# ----------------------------------------------------------------------------------------------- report
def report(model, X, names, label, out_dir, extra):
    with torch.no_grad():
        kp, J24, R = model()
        v_sh, J_rest = model.rest()
        shape = model.shape().numpy()
        offs = {g: model.offsets[g].numpy() * 1000.0 for g in model.groups}
    J_rest = J_rest.numpy()
    # rest-pose H36M-17 from SMPL-24 (as smpl_lite.joint_regressor('smpl24')) for the reporting frames
    j17 = np.zeros((17, 3))
    for h, s in (('RHip', 'R_Hip'), ('LHip', 'L_Hip'), ('RShoulder', 'R_Shoulder'), ('LShoulder', 'L_Shoulder'),
                 ('RKnee', 'R_Knee'), ('LKnee', 'L_Knee'), ('RAnkle', 'R_Ankle'), ('LAnkle', 'L_Ankle'),
                 ('RElbow', 'R_Elbow'), ('LElbow', 'L_Elbow'), ('RWrist', 'R_Wrist'), ('LWrist', 'L_Wrist')):
        j17[HJ[h]] = J_rest[S24[s]]
    j17[HJ['Hip']] = 0.5 * (j17[HJ['RHip']] + j17[HJ['LHip']])
    j17[HJ['Thorax']] = 0.5 * (j17[HJ['RShoulder']] + j17[HJ['LShoulder']])
    size_mm = float(op_size(J_rest)) * 1000.0
    local = {}
    for g in model.groups:
        F = _segment_frame(j17, REPORT_FRAME[g], 'L')            # rows: frame axes in the rest frame (L side)
        local[g] = F @ offs[g]
    m = _load_pkl()                                              # full rest mesh for the height
    S = np.asarray(m['shapedirs'], np.float64)[:, :, :10]
    if model.kid:
        S = np.concatenate([S, kid_shapedir(np.asarray(m['v_template'], np.float64))[:, :, None]], 2)
    vf = np.asarray(m['v_template'], np.float64) + S @ shape
    height = float(vf[:, 1].max() - vf[:, 1].min()) * 1000.0
    res = {}
    for i, n in enumerate(names):
        d = np.linalg.norm(kp[n].numpy() - X[:, B[n]], axis=-1) * 1000.0
        res[n] = float(np.nanmedian(d))
    # hip residual (keypoint minus model keypoint) in the pelvis frame vs hip flexion
    Rp = R.numpy()
    lines = [f'{label}: {len(X)} frames, shape {"kid w=%.2f " % shape[-1] if model.kid else ""}'
             f'betas max |{np.abs(shape[:10]).max():.1f}|, mesh height {height:.0f} mm, op_size {size_mm:.0f} mm']
    lines.append('  offsets (lateral, up, forward) mm in smpl_lite segment frames; [scaled to OP_SIZE_MM '
                 f'{OP_SIZE_MM:.0f}]')
    for g in model.groups:
        sc = local[g] * OP_SIZE_MM / size_mm
        lines.append(f'    {g:8s} {REPORT_FRAME[g]:6s} ' + ','.join(f'{x:+6.1f}' for x in local[g])
                     + '   [' + ','.join(f'{x:+6.1f}' for x in sc) + ']')
    lines.append('  median residual mm: ' + ' '.join(f'{n} {res[n]:.0f}' for n in names))
    # thigh flexion vs hip residual along the pelvis forward/up axes (does the keypoint ride on the thigh too?)
    with torch.no_grad():
        flex = []
        for s, hip in (('L', 'L_Hip'), ('R', 'R_Hip')):
            pel_up = Rp[:, 0] @ np.array([0, 1.0, 0])
            thigh = J24[:, S24[f'{s}_Knee']].numpy() - J24[:, S24[hip]].numpy()
            cosang = (-(thigh * pel_up).sum(-1)) / np.linalg.norm(thigh, axis=-1)
            flex.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
            e = X[:, B[f'{s}Hip']] - kp[f'{s}Hip'].numpy()
            loc = np.einsum('tij,tj->ti', np.transpose(Rp[:, 0], (0, 2, 1)), e) * 1000.0    # in pelvis rest coords
            fl = flex[-1]
            ok = np.isfinite(loc).all(-1)
            if ok.sum() > 10:
                bins = [(0, 20), (20, 50), (50, 80), (80, 180)]
                cells = []
                for a, b in bins:
                    m = ok & (fl >= a) & (fl < b)
                    if m.sum() >= 5:
                        cells.append(f'{a}-{b}deg n={m.sum()}: up {np.median(loc[m, 1]):+.0f} fwd {np.median(loc[m, 2]):+.0f}')
                lines.append(f'  {s}Hip residual by hip flexion: ' + ' | '.join(cells))
    msg = '\n'.join(lines)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, f'{label}.npz'), shape=shape, size_mm=size_mm, height_mm=height,
                 groups=np.array(model.groups), offsets_local_mm=np.array([local[g] for g in model.groups]),
                 offsets_rest_mm=np.array([offs[g] for g in model.groups]),
                 residual_names=np.array(names), residual_mm=np.array([res[n] for n in names]),
                 n_frames=len(X), X=X.astype(np.float32), root=model.root.detach().numpy(),
                 trans=model.trans.detach().numpy(), pose=(model.pose * model.pose_mask).detach().numpy(),
                 betas_raw=model.betas.detach().numpy(), kid=model.kid, **extra)
    return msg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dataset', choices=('biocv', 'korea', 'gt3d'))
    ap.add_argument('root')
    ap.add_argument('--subjects', default=None, help='comma list; default every subject under root')
    ap.add_argument('--calib-root', default='/Volumes/Expansion/MotorDevelopment/BioCV')
    ap.add_argument('--every', type=int, default=4, help='frame step within a trial before pooling')
    ap.add_argument('--max-frames', type=int, default=300, help='frames per subject')
    ap.add_argument('--shoulder-bone', choices=('spine3', 'collar'), default='spine3')
    ap.add_argument('--robust-mm', type=float, default=30.0)
    ap.add_argument('--lam-beta', type=float, default=1.0)
    ap.add_argument('--lam-torso', type=float, default=100.0, help='L2 on spine/neck/collar axis-angles, per frame')
    ap.add_argument('--lam-pose', type=float, default=1.0, help='L2 on every body axis-angle, per frame')
    ap.add_argument('--sigma-offset-mm', type=float, default=100.0)
    ap.add_argument('--two-bone', action='store_true', help='hip / shoulder keypoints also ride on the thigh / upper arm')
    ap.add_argument('--sigma-child-mm', type=float, default=50.0, help='prior scale of the child-bone offsets')
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--exclude', default=None, help='comma list of subjects to leave out (dev / sample children)')
    ap.add_argument('--skip-done', action='store_true', help='skip subjects with {out_dir}/{subject}.npz')
    args = ap.parse_args()
    torch.set_num_threads(args.threads or max(1, os.cpu_count() - 1))
    rng = np.random.default_rng(args.seed)

    if args.dataset == 'biocv':
        subs = sorted({os.path.basename(d).split('_')[0] for d in glob.glob(os.path.join(args.root, 'P*_*'))})
    else:
        subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(args.root, '*', 'shapes.npz')))
    if args.dataset == 'gt3d':
        import json
        subs = []
        for d in sorted(glob.glob(os.path.join(args.root, '*', 'session_summary.json'))):
            if json.load(open(d)).get('usable'):
                subs.append(os.path.basename(os.path.dirname(d)))
    if args.subjects:
        subs = [s for s in subs if s in args.subjects.split(',')]
    if args.exclude:
        subs = [s for s in subs if s not in args.exclude.split(',')]
    if args.skip_done and args.out_dir:
        subs = [s for s in subs if not os.path.exists(os.path.join(args.out_dir, f'{s}.npz'))]
    summary = []
    for s in subs:
        print(f'{s}:', flush=True)
        if args.dataset == 'biocv':
            X, nv, src = load_biocv(args.root, s, args.calib_root, args.every, args.max_frames, rng)
        elif args.dataset == 'korea':
            X, nv, src = load_korea(args.root, s, args.every, args.max_frames, rng)
        else:
            X, nv, src = load_gt3d(args.root, s, args.every, args.max_frames, rng)
        if len(X) < 30:
            print(f'  only {len(X)} frames, skipped')
            continue
        try:
            model, names = fit_subject(X, args.dataset != 'biocv', args, print)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}', flush=True)
            continue
        msg = report(model, X, names, s, args.out_dir, dict(dataset=args.dataset, source=src,
                                                            shoulder_bone=args.shoulder_bone, two_bone=args.two_bone))
        print(msg, flush=True)
        summary.append(msg)
    print('\nreference OP_OFFSETS_MM (OpenPose - mocap): hip ' + str(OP_OFFSETS_MM['Hip'])
          + ' shoulder ' + str(OP_OFFSETS_MM['Shoulder']))


if __name__ == '__main__':
    main()
