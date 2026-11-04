#!/usr/bin/env python3
"""mb_input_defs.py -- how much do OpenPose's joint definitions bias MotionBERT's rotations?

step_2a gives MotionBERT the H36M-17 built from BODY_25 (utils/openpose.py): hips/knees/... are OpenPose's
points, Thorax = mid shoulders, Spine = mid(pelvis, thorax), Head = mid ears.  Here the same camera's input is
rebuilt several ways and MotionBERT-Mesh run on each; every variant keeps step_2a's OpenPose confidences
(missing joints filled as step_2a does) so only positions change:

    op            the pipeline's input (baseline)
    mocap         every mocap-covered joint replaced by mocap's joint centre projected through the calib
                  (Spine = mocap's T10-based point); Nose/Head stay OpenPose (no head markers)
    mocap_spmid   mocap, but Spine = mid(pelvis, thorax) as MotionBERT's own COCO->H36M builds it
    swap_hips     op with mocap's Hip/RHip/LHip (Spine re-made as the midpoint)
    swap_shld     op with mocap's Thorax/LShoulder/RShoulder (Spine re-made)
    swap_limbs    op with mocap's knees, ankles, elbows, wrists
    eyes          op with Head = mid eyes (MotionBERT's in-the-wild convention) instead of mid ears
    spine_c0      op with Spine confidence 0 (position kept)
    spine_zero    op with Spine at (0, 0), confidence 0 (what an unfilled missing joint looks like)

The definition-only variants keep this camera's OpenPose 2D (its noise) and move each limb/hip/shoulder keypoint
by OpenPose's DEFINITION offset from the mocap joint centre: a (lateral, up, forward) offset fixed in the body
segment's frame (op_offsets.py), rotated into the lab by that frame's segment orientation from the triangulated
OpenPose skeleton, and projected -- the 2D shift is proj(tri - offset) - proj(tri).  Hip, Thorax, Spine are
re-made from the moved points.

    defcorr       smpl_lite.OP_OFFSETS_MM (5 BioCV subjects INCLUDING P03), scaled to this body
    defcorr_own   this trial's own median offsets (an upper bound for a calibrated correction)
    headtop       op with Head = nose + 0.24 L up the 2D trunk (L = mid-hip -> mid-shoulder), the training data's
                  head-top; eyes / ears midpoint + 0.24 L when the nose is missing
    defcorr_h36m  defcorr + headtop + Thorax 0.13 L above mid-shoulders (H36M's neck-base thorax)

Reported per variant: the per-joint geodesic angle to op's SMPL local rotations (mean over frames and cameras),
and MotionBERT's own 3D H36M joints vs mocap, PA-MPJPE over the 13 mocap joints that share a definition.

    python3 analysis/joint_calib/mb_input_defs.py --trial data/sampleBiocv/P03_CMJM_01 --cams 00 05 07
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
from step_2a_extract_betas import load_model, _reflect_index, _CLIP_LEN          # noqa: E402
from lib.utils.utils_data import crop_scale                                       # noqa: E402
from utils.missing_joints import fill_missing                                     # noqa: E402
from utils.calibration import load_calib                                          # noqa: E402
from utils.triangulate import project                                             # noqa: E402
from utils.smpl_lite import OP_OFFSETS_MM, OP_SIZE_MM, OP_FRAME                   # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from op_offsets import frames, J as JN                                            # noqa: E402

H36M = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
SMPL24 = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
          'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder', 'l_elbow',
          'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']
SHOW = [0, 1, 2, 4, 5, 3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19]      # pelvis = global orientation
HIPS, SHLD = [0, 1, 4], [8, 11, 14]
LIMBS = [2, 3, 5, 6, 12, 13, 15, 16]
SCORE = [0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16]             # not Spine/Nose/Head
VARIANTS = ['op', 'mocap', 'mocap_spmid', 'swap_hips', 'swap_shld', 'swap_limbs', 'eyes', 'spine_c0', 'spine_zero',
            'defcorr', 'defcorr_own', 'headtop', 'defcorr_h36m']
HEAD_UP, THORAX_UP = 0.24, 0.13     # training 2D (mesh_det_h36m): above nose / above mid-shoulders, in L = mid-hip -> mid-shoulder


def spine_mid(x):
    x[:, 7, :2] = 0.5 * (x[:, 0, :2] + x[:, 8, :2])
    return x


def remake_mids(x):
    x[:, 0, :2] = 0.5 * (x[:, 1, :2] + x[:, 4, :2])
    x[:, 8, :2] = 0.5 * (x[:, 11, :2] + x[:, 14, :2])
    return spine_mid(x)


def own_offsets(M, O):
    """This trial's median OpenPose-minus-mocap offset per joint type, (lateral, up, forward) mm, frames from O."""
    F = frames(O)
    out = {}
    for jn, f in OP_FRAME.items():
        loc = []
        for s in 'LR':
            R = F[f, s if f not in ('pelvis', 'thorax') else '']
            d = np.einsum('tij,tj->ti', R, O[:, JN[s + jn]] - M[:, JN[s + jn]])
            if s == 'R':
                d[:, 0] *= -1
            loc.append(d)
        out[jn] = np.nanmedian(np.concatenate(loc), 0)
    return out


def body_size(O):
    """Rest Thorax-to-ankle height (trunk + thigh + shank) of a triangulated skeleton, mm (OP_SIZE_MM's measure)."""
    n = lambda a, b: np.nanmedian(np.linalg.norm(O[:, JN[a]] - O[:, JN[b]], axis=-1))
    return n('Thorax', 'Hip') + 0.5 * sum(n(s + 'Hip', s + 'Knee') + n(s + 'Knee', s + 'Ankle') for s in 'LR')


def def_shift(O, offsets, scale, K, dist, L):
    """(T,17,3) triangulated OpenPose (lab mm, NaN rows) -> (T,17,2) pixel shift moving each keypoint to the mocap
    definition: proj(O - offset) - proj(O); 0 where O is missing."""
    F = frames(O)
    C = O.copy()
    for jn, f in OP_FRAME.items():
        for s in 'LR':
            R = F[f, s if f not in ('pelvis', 'thorax') else '']
            loc = np.asarray(offsets[jn], float) * scale
            if s == 'R':
                loc = loc * [-1, 1, 1]
            C[:, JN[s + jn]] -= np.einsum('tji,j->ti', R, loc)          # R^T loc: segment -> lab
    T = len(O)
    p0 = project(O.reshape(-1, 3), K, dist, L[:3, :3], L[:3, 3]).reshape(T, 17, 2)
    p1 = project(C.reshape(-1, 3), K, dist, L[:3, :3], L[:3, 3]).reshape(T, 17, 2)
    return np.nan_to_num(p1 - p0)


def head_top(x, b25):
    """op Head -> nose (else eye / ear midpoint) + HEAD_UP * L up the 2D trunk; conf from that face point."""
    x = x.copy()
    ms, mh = 0.5 * (x[:, 11, :2] + x[:, 14, :2]), 0.5 * (x[:, 1, :2] + x[:, 4, :2])
    u = ms - mh
    up = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
    face = np.zeros((len(x), 3))
    for idx in ([0], [15, 16], [17, 18]):                # nose, eyes, ears: first available wins
        ok = (b25[:, idx, 2] > 0).all(1) & (face[:, 2] == 0)
        face[ok, :2] = b25[ok][:, idx, :2].mean(1)
        face[ok, 2] = b25[ok][:, idx, 2].mean(1)
    have = face[:, 2] > 0
    x[have, 10, :2] = face[have, :2] + HEAD_UP * u[have]
    x[have, 10, 2] = face[have, 2]
    return x


def inputs(trial, cam, calib_dir):
    tw = np.load(os.path.join(trial, 'Analysis', 'openpose', f'{cam}_2d.npz'))
    op = fill_missing(tw['h36m_2d'])
    b25 = tw['body25_2d']
    mo = np.load(os.path.join(trial, 'Analysis', 'H36M', 'mocap_h36m.npz'))
    sfi = tw['source_frame_idx'].astype(int)
    _, _, K, L, dist = load_calib(os.path.join(calib_dir, f'{cam}.mp4-mocAligned.calib'))
    T = len(op)
    have = sfi < len(mo['kps3d'])
    X = np.full((T, 17, 3), np.nan)
    X[have] = mo['kps3d'][sfi[have]]
    uv = project(X.reshape(-1, 3), K, dist, L[:3, :3], L[:3, 3]).reshape(T, 17, 2)
    uv[~have] = op[~have, :, :2]                        # rows past the mocap's end: OpenPose
    valid = mo['valid_joint_mask']

    v = {'op': op}
    m = op.copy()
    m[:, valid, :2] = uv[:, valid]
    v['mocap'] = m
    v['mocap_spmid'] = spine_mid(m.copy())
    for name, idx in (('swap_hips', HIPS), ('swap_shld', SHLD), ('swap_limbs', LIMBS)):
        s = op.copy()
        s[:, idx, :2] = uv[:, idx]
        v[name] = spine_mid(s) if name != 'swap_limbs' else s
    e = op.copy()
    eyes_ok = (b25[:, [15, 16], 2] > 0).all(1)
    e[eyes_ok, 10, :2] = b25[eyes_ok][:, [15, 16], :2].mean(1)
    v['eyes'] = e
    c0 = op.copy()
    c0[:, 7, 2] = 0.0
    v['spine_c0'] = c0
    z = op.copy()
    z[:, 7] = 0.0
    v['spine_zero'] = z

    tri = np.load(os.path.join(trial, 'Analysis', 'H36M', 'openpose_tri_h36m.npz'))['kps3d'].astype(float)
    O = np.full((T, 17, 3), np.nan)
    in_tri = sfi < len(tri)
    O[in_tri] = tri[sfi[in_tri]]
    n = min(len(tri), len(mo['kps3d']))
    own = own_offsets(mo['kps3d'][:n].astype(float), tri[:n])
    moved = LIMBS + [1, 4, 11, 14]
    for name, offs, sc in (('defcorr', OP_OFFSETS_MM, body_size(tri) / OP_SIZE_MM), ('defcorr_own', own, 1.0)):
        c = op.copy()
        c[:, moved, :2] += def_shift(O, offs, sc, K, dist, L)[:, moved]
        v[name] = remake_mids(c)
    v['headtop'] = head_top(op, b25)
    h = head_top(v['defcorr'], b25)
    h[:, 8, :2] += THORAX_UP * (h[:, 8, :2] - h[:, 0, :2])
    v['defcorr_h36m'] = spine_mid(h)

    limb_px = np.linalg.norm(uv[have][:, LIMBS] - op[have][:, LIMBS, :2], axis=-1)
    gt_cam = X @ L[:3, :3].T + L[:3, 3]                 # camera frame, mm
    shift = np.linalg.norm(v['defcorr'][:, moved, :2] - op[:, moved, :2], axis=-1)
    print(f'cam {cam}: defcorr shift median {np.median(shift):.1f} px (scale {body_size(tri) / OP_SIZE_MM:.3f}); own '
          + ' '.join(f'{k} {np.round(o).astype(int).tolist()}' for k, o in own.items()), flush=True)
    return v, gt_cam, have, float(np.median(limb_px))


def run_mb(model, h2d, batch=16):
    """(T,17,3) -> rotmats (T,24,3,3), kp_3d (T,17,3) mm, exactly step_2a's windowing, batched."""
    norm = crop_scale(h2d.astype(np.float32), scale_range=[1, 1])
    T, half = len(norm), _CLIP_LEN // 2
    rot, kp = np.zeros((T, 24, 3, 3)), np.zeros((T, 17, 3))
    for s in range(0, T, batch):
        fr = range(s, min(s + batch, T))
        win = np.stack([norm[[_reflect_index(f - half + j, T) for j in range(_CLIP_LEN)]] for f in fr])
        with torch.no_grad():
            feat = model.backbone.get_representation(torch.from_numpy(win))
            feat = feat.reshape(len(fr), _CLIP_LEN, model.feat_J, -1)[:, half:half + 1]
            out = model.head(feat)[0]
        th = out['theta'][:, :72].numpy().reshape(-1, 24, 3)
        rot[s:s + len(fr)] = Rotation.from_rotvec(th.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        kp[s:s + len(fr)] = out['kp_3d'].numpy().reshape(-1, 17, 3)
    return rot, kp


def geodesic_deg(Ra, Rb):
    tr = np.einsum('...ij,...ij->...', Ra, Rb)              # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def pa_mpjpe(P, G):
    """(T,J,3) each -> (T,) mm after similarity Procrustes per frame."""
    out = np.full(len(P), np.nan)
    for t in range(len(P)):
        p, g = P[t] - P[t].mean(0), G[t] - G[t].mean(0)
        U, S, Vt = np.linalg.svd(p.T @ g)
        d = np.sign(np.linalg.det(U @ Vt))
        D = np.diag([1, 1, d])
        R = U @ D @ Vt
        s = (S * np.diag(D)).sum() / (p ** 2).sum()
        out[t] = np.linalg.norm(s * p @ R - g, axis=-1).mean()
    return out


def body_frame(K):
    """(T,17,3) H36M joints -> (T,3,3) pelvis frame, rows x = RHip->LHip, y = Hip->Thorax (orthogonalised), z = x cross y."""
    x = K[:, 4] - K[:, 1]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = K[:, 8] - K[:, 0]
    y = y - (y * x).sum(-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], 1)


def ray_rot(p):
    """(T,3) camera-frame points -> (T,3,3) rotation taking the optical axis onto the ray through each point.
    MotionBERT sees a crop, so its orientation is relative to the ray to the person (step_4b's 'ray' placement)."""
    r = p / np.linalg.norm(p, axis=-1, keepdims=True)
    ax = np.cross([0.0, 0.0, 1.0], r)
    sn = np.linalg.norm(ax, axis=-1)
    rv = ax / np.maximum(sn, 1e-12)[:, None] * np.arctan2(sn, r[:, 2])[:, None]
    return Rotation.from_rotvec(rv).as_matrix()


def to_ray(F, Rray):
    """Frames (rows = axis vectors) rotated by Rray per frame."""
    return F @ np.transpose(Rray, (0, 2, 1))


# proper axis flips: MotionBERT's kp_3d axes vs OpenCV camera axes are a convention, picked once per run on op
FLIPS = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def orientation_parts(Fp, Fg):
    """Pelvis frames (camera axes: x right, y down, z away) -> per frame (total angle, lean toward camera,
    sideways tilt, heading), degrees, prediction minus truth.  Lean = trunk axis tipping along the line of sight
    (+ = top of trunk further from the camera), tilt = in the image plane, heading = hip axis turn about vertical."""
    tot = geodesic_deg(Fp, Fg)
    lean = lambda F: np.degrees(np.arctan2(F[:, 1, 2], -F[:, 1, 1]))      # y axis points up the trunk (camera -y)
    tilt = lambda F: np.degrees(np.arctan2(-F[:, 1, 0], -F[:, 1, 1]))
    head = lambda F: np.degrees(np.arctan2(F[:, 0, 2], F[:, 0, 0]))
    wrap = lambda a: (a + 180) % 360 - 180
    return tot, wrap(lean(Fp) - lean(Fg)), wrap(tilt(Fp) - tilt(Fg)), wrap(head(Fp) - head(Fg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default=os.path.join(_ROOT, 'data', 'sampleBiocv', 'P03_CMJM_01'))
    ap.add_argument('--calib-dir', default=None, help='default: the trial folder\'s parent')
    ap.add_argument('--cams', nargs='+', default=['00', '05', '07'])
    ap.add_argument('--variants', nargs='+', default=VARIANTS, help='op is always run (the reference)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache', default=os.path.join(_ROOT, 'analysis', 'joint_calib', 'out', 'mb_cache'),
                    help='MotionBERT outputs per trial/camera/variant; reused when present')
    args = ap.parse_args()
    variants = ['op'] + [n for n in args.variants if n != 'op']
    calib_dir = args.calib_dir or os.path.dirname(os.path.abspath(args.trial))
    trial = os.path.basename(os.path.abspath(args.trial))
    os.makedirs(args.cache, exist_ok=True)

    model = None
    ang = {v: [] for v in variants}                       # per cam (T,24)
    pa = {v: [] for v in variants}
    ori = {v: [] for v in variants}                       # per cam (T',4): total, lean, tilt, heading
    flip = None
    for cam in args.cams:
        v, gt, have, px = inputs(args.trial, cam, calib_dir)
        print(f'cam {cam}: {len(gt)} frames; mocap limbs vs OpenPose median {px:.1f} px (calib check)', flush=True)
        res = {}
        for name in variants:
            path = os.path.join(args.cache, f'{trial}_{cam}_{name}.npz')
            if os.path.isfile(path):
                z = np.load(path)
                res[name] = (z['rot'], z['kp'])
                continue
            model = model or load_model('cpu')
            t0 = time.time()
            res[name] = run_mb(model, v[name])
            np.savez(path, rot=res[name][0], kp=res[name][1])
            print(f'  {name:12s} {time.time() - t0:5.0f} s', flush=True)
        Fg = body_frame(gt[have])
        Rray = ray_rot(gt[have][:, 0])                    # ray to mocap's pelvis
        if flip is None:
            Fop = body_frame(res['op'][1][have])
            flip = min(FLIPS, key=lambda S: np.nanmedian(geodesic_deg(to_ray(Fop @ S, Rray), Fg)))
            print(f'  MotionBERT -> camera axes: diag{np.diag(flip).astype(int).tolist()}', flush=True)
        for name in variants:
            ang[name].append(geodesic_deg(res['op'][0], res[name][0]))
            pa[name].append(pa_mpjpe(res[name][1][have][:, SCORE], gt[have][:, SCORE]))
            ori[name].append(np.stack(orientation_parts(to_ray(body_frame(res[name][1][have]) @ flip, Rray), Fg), 1))

    print('\nmean angle to op\'s rotation (deg), all cameras; pelvis = global orientation')
    print(f'{"":12s}' + ''.join(f'{SMPL24[j][:8]:>9s}' for j in SHOW) + f'{"PA-MPJPE":>10s}')
    for name in variants:
        a = np.concatenate(ang[name]).mean(0)
        print(f'{name:12s}' + ''.join(f'{a[j]:9.1f}' for j in SHOW) + f'{np.concatenate(pa[name]).mean():10.1f}')
    print('\nPA-MPJPE (mm) per camera: ' + '  '.join(
        f'{n}: ' + '/'.join(f'{np.mean(p):.0f}' for p in pa[n]) for n in variants))

    print('\npelvis orientation vs mocap (deg; MotionBERT turned onto the ray to the pelvis, minus mocap, camera frame).'
          '  total = median geodesic;'
          '\nlean/tilt/heading = mean signed error (bias) | mean |error|; lean + = top of trunk away from camera')
    print(f'{"":12s}{"total":>7s}{"lean":>14s}{"tilt":>14s}{"heading":>14s}')
    for name in variants:
        o = np.concatenate(ori[name])
        print(f'{name:12s}{np.nanmedian(o[:, 0]):7.1f}' + ''.join(
            f'{np.nanmean(o[:, k]):7.1f} |{np.nanmean(np.abs(o[:, k])):5.1f}' for k in (1, 2, 3)))
    print('\nper camera: total median / lean bias')
    for name in variants:
        print(f'{name:12s}' + '  '.join(f'{c}: {np.nanmedian(o[:, 0]):4.1f}/{np.nanmean(o[:, 1]):+5.1f}'
                                        for c, o in zip(args.cams, ori[name])))
    if args.out:
        np.savez(args.out, cams=np.array(args.cams), variants=np.array(variants),
                 **{f'ang_{n}': np.stack([a.mean(0) for a in ang[n]]) for n in variants},
                 **{f'pa_{n}': np.array([np.mean(p) for p in pa[n]]) for n in variants},
                 **{f'ori_{n}': np.stack([np.nanmean(np.abs(o), 0) for o in ori[n]]) for n in variants},
                 **{f'oribias_{n}': np.stack([np.nanmean(o, 0) for o in ori[n]]) for n in variants})


if __name__ == '__main__':
    main()
