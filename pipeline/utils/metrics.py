"""metrics.py -- pose error measures for step_8, identical for both streams.

All take pred and gt as (T,17,3) in the same frame and units, with NaN for
missing joints, plus a (T,17) mask of joints to score.  Per-frame values are
NaN where fewer than `min_joints` scored joints are present.

  mpjpe      absolute error: mean joint distance, no alignment.  Tests
             placement, and depends on scale being right.
  n_mpjpe    after the single best scale factor (per frame).  Removes scale,
             keeps rotation/translation.
  pa_mpjpe   after Procrustes (rotation + translation + scale, per frame).
             Pose shape only.
  bone_ratio each bone's median length as a fraction of stature, predicted vs
             ground truth -- the direct test of whether limb proportions
             survive lifting.  Reported per bone so a systematic bias on one
             segment is visible, which is what an adult prior on a child body
             would produce.
"""
import numpy as np

H36M_BONES = {          # name: (a, b) -- direct-counterpart joints only
    'r_thigh': (1, 2), 'r_shank': (2, 3), 'l_thigh': (4, 5), 'l_shank': (5, 6),
    'l_upperarm': (11, 12), 'l_forearm': (12, 13),
    'r_upperarm': (14, 15), 'r_forearm': (15, 16),
    'hip_width': (1, 4), 'shoulder_width': (11, 14),
}


def _frame_mask(pred, gt, mask):
    ok = mask & np.isfinite(pred).all(-1) & np.isfinite(gt).all(-1)
    return ok


def mpjpe(pred, gt, mask, min_joints=4):
    ok = _frame_mask(pred, gt, mask)
    d = np.linalg.norm(pred - gt, axis=-1)
    out = np.full(len(pred), np.nan)
    per_joint = np.full(pred.shape[1], np.nan)
    for j in range(pred.shape[1]):
        if ok[:, j].any():
            per_joint[j] = d[ok[:, j], j].mean()
    n = ok.sum(1)
    good = n >= min_joints
    out[good] = np.array([d[f, ok[f]].mean() for f in np.where(good)[0]])
    return out, per_joint


def _procrustes(P, G, scale=True, rotate=True):
    """Align P onto G (both (n,3)), return aligned P."""
    mp, mg = P.mean(0), G.mean(0)
    P0, G0 = P - mp, G - mg
    if rotate:
        U, S, Vt = np.linalg.svd(P0.T @ G0)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        Rm = U @ D @ Vt
        P0r = P0 @ Rm
        s = (S * np.diag(D)).sum() / (P0 ** 2).sum() if scale else 1.0
    else:
        P0r = P0
        s = (P0 * G0).sum() / (P0 ** 2).sum() if scale else 1.0
    return s * P0r + mg


def aligned_mpjpe(pred, gt, mask, scale=True, rotate=True, min_joints=4):
    ok = _frame_mask(pred, gt, mask)
    out = np.full(len(pred), np.nan)
    pj_sum = np.zeros(pred.shape[1])
    pj_n = np.zeros(pred.shape[1])
    for f in range(len(pred)):
        m = ok[f]
        if m.sum() < min_joints:
            continue
        A = _procrustes(pred[f, m], gt[f, m], scale=scale, rotate=rotate)
        d = np.linalg.norm(A - gt[f, m], axis=1)
        out[f] = d.mean()
        pj_sum[m] += d
        pj_n[m] += 1
    per_joint = np.where(pj_n > 0, pj_sum / np.maximum(pj_n, 1), np.nan)
    return out, per_joint


def n_mpjpe(pred, gt, mask, **kw):
    return aligned_mpjpe(pred, gt, mask, scale=True, rotate=False, **kw)


def pa_mpjpe(pred, gt, mask, **kw):
    return aligned_mpjpe(pred, gt, mask, scale=True, rotate=True, **kw)


def bone_lengths(P, mask):
    """{bone: (T,) length, NaN where an end is missing}."""
    out = {}
    for name, (a, b) in H36M_BONES.items():
        l = np.linalg.norm(P[:, a] - P[:, b], axis=1)
        good = mask[:, a] & mask[:, b] & np.isfinite(l)
        out[name] = np.where(good, l, np.nan)
    return out


def bone_ratios(pred, gt, mask, stature):
    """Per bone: median length / stature for pred and gt, and their ratio.

    Medians over frames, since MotionBERT does not hold bone lengths fixed
    frame to frame.  ratio > 1 means the model makes that bone too long.
    """
    bp, bg = bone_lengths(pred, mask), bone_lengths(gt, mask)
    rows = {}
    for name in H36M_BONES:
        p, g = np.nanmedian(bp[name]), np.nanmedian(bg[name])
        rows[name] = dict(pred_frac=p / stature, gt_frac=g / stature,
                          ratio=p / g if g > 0 else np.nan,
                          pred_cv=float(np.nanstd(bp[name]) / p) if p > 0 else np.nan,
                          n=int(np.isfinite(bp[name] * bg[name]).sum()))
    return rows


def signed_bias(pred, gt, mask, min_joints=4):
    """Mean signed offset pred - gt per joint (3,) -- direction of the error."""
    ok = _frame_mask(pred, gt, mask)
    out = np.full((pred.shape[1], 3), np.nan)
    for j in range(pred.shape[1]):
        if ok[:, j].any():
            out[j] = (pred[ok[:, j], j] - gt[ok[:, j], j]).mean(0)
    return out
