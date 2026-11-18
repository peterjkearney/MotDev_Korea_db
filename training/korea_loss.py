"""korea_loss.py -- the fine-tuning loss on Korea clips, and the dev metrics.

Both the prediction and the pseudo-GT target are posed on the CHILD's body (KidBody: fit_v3 shape and keypoint
offsets), so MotionBERT's own betas play no part (the pipeline replaces them with step_2c's kid shape anyway).
Every term is masked to what the data determines (build_dataset.py, pseudo_gt.OBSERVE):

    rot   segments whose full global orientation is observed: mean |G_pred - G_tgt| over the 9 matrix entries
          (MotionBERT's pose loss, on global not local rotations: a local rotation carries its parents' split)
    dir   segments whose bone direction is observed: 1 - cos between the predicted and target global axis
    pos   BODY_25 keypoints the fit had data for, relative to the mid-hip: mean distance (mm) -- this is what holds
          the trunk, whose pelvis / spine split the keypoints do not fix
    vel   the same keypoints' frame-to-frame motion (mm/frame), on frame pairs where both are seen

frame_ok gates every frame.  Weights default to MotionBERT's (lambda_pose 1000, lambda_3d 0.5, lambda_3dv 10).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kid_body import B25  # noqa: E402

MID = B25['MidHip']
# H36M-17 from BODY_25 (as utils.openpose.H36M_FROM_BODY25; spine = mid(pelvis, thorax)), for the LOO score
H36M_FROM_B25 = {0: [9, 12], 1: [9], 2: [10], 3: [11], 4: [12], 5: [13], 6: [14], 8: [2, 5], 9: [0], 10: [17, 18],
                 11: [5], 12: [6], 13: [7], 14: [2], 15: [3], 16: [4]}


def axis_angle_to_matrix(aa):
    """(...,3) -> (...,3,3), Rodrigues; safe at zero."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = th.sin()[..., None], th.cos()[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + s * K + (1 - c) * (K @ K)


def pose_body(body, aa, gt):
    """aa (N,T,24,3) -> keypoints (M,25,3) mid-hip relative, G (M,24,3,3), axes (M,24,3); M = N*T."""
    N, T = aa.shape[:2]
    R = axis_angle_to_matrix(aa.reshape(N * T, 24, 3))
    shape = gt['shape'][:, None].expand(N, T, -1).reshape(N * T, -1)
    off = gt['offsets'][:, None].expand(N, T, -1, -1).reshape(N * T, 4, 3)
    kp, G, axes = body(R, shape, off)
    return kp - kp[:, MID:MID + 1], G, axes


def korea_loss(output, gt, body):
    theta = output[0]['theta']                                         # (N,T,82)
    N, T = theta.shape[:2]
    kp_p, G_p, ax = pose_body(body, theta[..., :72].reshape(N, T, 24, 3), gt)
    with torch.no_grad():
        kp_t, G_t, _ = pose_body(body, gt['pose'], gt)
    ok = gt['ok'].reshape(-1)
    full = gt['obs_full'].reshape(-1, 24) & ok[:, None]
    sw = gt['obs_swing'].reshape(-1, 24) & ok[:, None]
    km = gt['kp_mask'].reshape(-1, 25) & ok[:, None]
    km[:, MID] = False
    out = {}
    out['rot'] = (G_p - G_t).abs().mean((-1, -2))[full].mean() if full.any() else theta.sum() * 0
    d_p = (G_p @ ax[..., None])[..., 0]
    d_t = (G_t @ ax[..., None])[..., 0]
    out['dir'] = (1 - (d_p * d_t).sum(-1))[sw].mean() if sw.any() else theta.sum() * 0
    e = (kp_p - kp_t).norm(dim=-1) * 1000.0
    out['pos'] = e[km].mean() if km.any() else theta.sum() * 0
    v_p = (kp_p.reshape(N, T, 25, 3)[:, 1:] - kp_p.reshape(N, T, 25, 3)[:, :-1])
    v_t = (kp_t.reshape(N, T, 25, 3)[:, 1:] - kp_t.reshape(N, T, 25, 3)[:, :-1])
    kmv = km.reshape(N, T, 25)
    kmv = kmv[:, 1:] & kmv[:, :-1]
    ev = (v_p - v_t).norm(dim=-1) * 1000.0
    out['vel'] = ev[kmv].mean() if kmv.any() else theta.sum() * 0
    return out


def h36m_from_kp(kp):
    """(M,25,3) BODY_25 -> (M,17,3) H36M-17 as the Korea targets build it."""
    out = [None] * 17
    for k, idx in H36M_FROM_B25.items():
        out[k] = kp[:, idx].mean(1)
    out[7] = 0.5 * (out[0] + out[8])
    return torch.stack(out, 1)


def procrustes_err(P, Q, w):
    """mean distance after the best similarity transform of P onto Q, per frame; P,Q (M,J,3), w (M,J) bool."""
    errs = []
    for p, q, m in zip(P, Q, w):
        if m.sum() < 4:
            continue
        p, q = p[m], q[m]
        mp, mq = p.mean(0), q.mean(0)
        p0, q0 = p - mp, q - mq
        U, S, Vt = torch.linalg.svd(p0.T @ q0)
        d = torch.sign(torch.det(U @ Vt))
        D = torch.diag(torch.tensor([1.0, 1.0, float(d)], device=P.device))
        Rm = U @ D @ Vt
        s = (S * torch.diag(D)).sum() / (p0 ** 2).sum()
        errs.append(((s * p0 @ Rm) - q0).norm(dim=-1).mean())
    return torch.stack(errs) if errs else torch.zeros(0)


@torch.no_grad()
def korea_metrics(output, gt, body, eval_mask):
    """Per-batch sums for the dev report: keypoint error vs the pseudo-GT (mm), H36M error vs camera c's LOO
    target (root-relative and Procrustes, mm), and orientation error of observed segments (deg)."""
    theta = output[0]['theta']
    N, T = theta.shape[:2]
    kp_p, G_p, ax = pose_body(body, theta[..., :72].reshape(N, T, 24, 3), gt)
    kp_t, G_t, _ = pose_body(body, gt['pose'], gt)
    ok = gt['ok'].reshape(-1)
    km = gt['kp_mask'].reshape(-1, 25) & ok[:, None]
    km[:, MID] = False
    e = (kp_p - kp_t).norm(dim=-1)[km] * 1000.0
    # vs LOO: both relative to their own H36M pelvis
    H = h36m_from_kp(kp_p)
    L = gt['loo3d'].reshape(-1, 17, 3)
    lm = gt['loo_ok'].reshape(-1, 17) & ok[:, None] & eval_mask[None]
    have_root = gt['loo_ok'].reshape(-1, 17)[:, 0]
    lm &= have_root[:, None]
    Hr, Lr = H - H[:, :1], L - L[:, :1]
    el = (Hr - Lr).norm(dim=-1)[lm] * 1000.0
    pa = procrustes_err(H, L, lm) * 1000.0
    full = gt['obs_full'].reshape(-1, 24) & ok[:, None]
    tr = (G_p.transpose(-1, -2) @ G_t).diagonal(dim1=-2, dim2=-1).sum(-1)
    ang = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))[full]
    sw = gt['obs_swing'].reshape(-1, 24) & ok[:, None]
    d_p = (G_p @ ax[..., None])[..., 0]
    d_t = (G_t @ ax[..., None])[..., 0]
    dang = torch.rad2deg(torch.arccos((d_p * d_t).sum(-1).clamp(-1, 1)))[sw]
    return dict(kp=(e.sum().item(), e.numel()), loo=(el.sum().item(), el.numel()), loo_pa=(pa.sum().item(), pa.numel()),
                rot_deg=(ang.sum().item(), ang.numel()), dir_deg=(dang.sum().item(), dang.numel()))
