#!/usr/bin/env python3
"""build_dataset.py -- pseudo_gt.py fits -> per-camera training sequences for fine-tuning MotionBERT-Mesh (step 3).

A fit is one SMPL pose per frame in camera 1's frame.  Each camera c of the rep becomes one sequence:

    input    camera c's OpenPose H36M-17 2D (GT3D h36m_2d[c], 1080p px + confidence), and the same after
             step_2a's preparation (fill_missing, then crop_scale over the whole rep) -- what MotionBERT sees
    target   the fitted pose with ONLY the root (global orientation) turned into camera c: R_c R_1^T R_root.
             The 23 joint rotations are relative to their parents and do not depend on the viewer; translation
             is not a target (MotionBERT's output is root-relative)
    body     the child's fit_v3 shape (10 betas + kid weight) -- the loss poses this body, not MotionBERT's betas
    body     ... and its keypoint offsets (hip / shoulder / neck / mid-hip groups, pseudo_gt's keypoint model)
    masks    obs_swing / obs_full (T,24): which segments' GLOBAL orientation the keypoints determine, recomputed
             with pseudo_gt.OBSERVE (pelvis: its hip line only; spine / neck: none -- the fit's split of the
             trunk bend is its prior's); frame_ok = enough keypoints, small residual, hip line seen
    keypts   the gated triangulated BODY_25 the fit used, in camera c (metres; NaN where not used): the trunk's
             position target
    scoring  camera c's leave-one-out target (X_h36m_cam_loo[c], triangulated WITHOUT camera c) and the fit's own
             H36M joints, both in camera c's frame (metres)

Sequences are stored whole (clips are cut by the training loader).  Checks, per sequence:

    projection   the fit's H36M joints in camera c projected with K vs camera c's 2D (px) -- a wrong transform
                 or camera shows up as tens to hundreds of px
    orientation  the target root vs base MotionBERT's root for the same frames (--mb-root, a run tree holding
                 {child}/{rep}/Analysis/mesh/openpose/{cam}_betas.npz) -- expect ~10-20 deg, not 90 or 180
    root-only    the posed body with the turned root, root-relative, vs the camera-1 body turned rigidly (mm)

    python3 training/build_dataset.py --pgt training/out/pgt_dev --out training/out/dataset_dev.npz \\
        --mb-root /Volumes/Sandisk_SSD/MotorDevelopment/Korea/korea_dev
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'training'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
import pseudo_gt as PG  # noqa: E402
from config import MB_DIR  # noqa: E402
from utils.missing_joints import fill_missing  # noqa: E402
from utils.smpl_lite import MiniSMPL, rodrigues  # noqa: E402

sys.path.insert(0, MB_DIR)
from lib.utils.utils_data import crop_scale  # noqa: E402


def find_fits(pgt, split, root, splits_json, sets):
    """[(rep, fit path)]: a folder of prototype fits ({rep}_lam*.npz), or the full run's PseudoGT layout."""
    if split is None:
        out = []
        for p in sorted(glob.glob(os.path.join(pgt, '*.npz'))):
            out.append((os.path.basename(p).rsplit('_lam', 1)[0], p))
        return out
    sp = json.load(open(splits_json))
    out = []
    for st in sets:
        subs = sorted({u for nm in split.split(',') for u in sp['sets'][st][nm]}
                      | ({u for u in sp['criteria'].get('force_dev', []) if u[0] == st} if 'dev' in split else set()))
        for u in subs:
            for p in sorted(glob.glob(os.path.join(pgt or os.path.join(root, st, 'PseudoGT'), u, f'{u}_GMS_*.npz'))):
                if not p.endswith('.part.npz'):
                    out.append((os.path.basename(p)[:-4], p))
    return out


def split_of(subject, sp):
    for st in sp['sets'].values():
        for nm, subs in st.items():
            if subject in subs:
                return nm
    return 'dev' if subject in sp['criteria'].get('force_dev', []) else '?'


def angle_deg(Ra, Rb):
    """(T,3,3) x2 -> (T,) angle of Ra^T Rb."""
    tr = np.einsum('tij,tij->t', Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pgt', default=None, help='folder of pseudo_gt.py fits (prototype {rep}_lam*.npz, or with '
                                                '--split the PseudoGT root holding {child}/{rep}.npz)')
    ap.add_argument('--split', default=None, help="full mode: splits.json splits, e.g. 'train' (PseudoGT layout)")
    ap.add_argument('--sets', default='B,C')
    ap.add_argument('--root', default=PG.KORE_ROOT, help='holds {set}/GT3D/ (and {set}/PseudoGT/ in full mode)')
    ap.add_argument('--splits-json', default=os.path.join(_ROOT, 'pipeline', 'datasets', 'korea', 'splits.json'))
    ap.add_argument('--mb-root', default=None, help='run tree with base MotionBERT {cam}_betas.npz, for the check')
    ap.add_argument('--min-keypoints', type=int, default=12, help='frame_ok: keypoints the fit had on the frame')
    ap.add_argument('--max-residual-mm', type=float, default=30.0, help="frame_ok: the fit's median keypoint miss")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not (args.pgt or args.split):
        raise SystemExit('give --pgt (prototype fits) or --split')
    sp = json.load(open(args.splits_json))

    fits = find_fits(args.pgt, args.split, args.root, args.splits_json, args.sets.split(','))
    print(f'{len(fits)} fitted reps')
    seq = {k: [] for k in ('rep', 'subject', 'cam', 'split', 'start', 'len', 'shape', 'offsets_rest_mm',
                           'proj_px', 'orient_deg', 'rootonly_mm')}
    fr = {k: [] for k in ('x2d_raw', 'x2d', 'pose', 'obs_swing', 'obs_full', 'frame_ok', 'frame_residual_mm',
                          'x3d_body25', 'loo3d', 'loo_ok', 'fit3d')}
    smpl_cache = {}
    n = 0
    for rep, path in fits:
        f = np.load(path, allow_pickle=True)
        u = rep[:4]
        z = np.load(os.path.join(args.root, u[0], 'GT3D', u, f'{rep}.npz'))
        R, t, K = z['R_world_to_cam'], z['t_world_to_cam'], z['K_1080']
        T = len(f['root'])
        Rroot1 = Rotation.from_rotvec(f['root']).as_matrix()                      # (T,3,3) camera 1
        ok = (f['n_keypoints'] >= args.min_keypoints) & (f['frame_residual_mm'] <= args.max_residual_mm)
        H1 = f['H36M_model'].astype(float)                                        # fit's H36M, camera 1, m
        ev = z['eval_joint_mask_h36m']

        # posed body in camera 1, for the root-only check (a few frames)
        shape = f['shape'].astype(np.float64)
        key = shape.tobytes()
        if key not in smpl_cache:
            smpl_cache = {key: MiniSMPL(shape, full=False)}
        smpl = smpl_cache[key]
        chk = np.linspace(0, T - 1, min(T, 5)).round().astype(int)

        def joints(root_aa, pose):
            with torch.no_grad():
                Rm = rodrigues(torch.cat([torch.tensor(root_aa)[:, None], torch.tensor(pose)], 1))
                return smpl.forward(Rm)[0].numpy()                                # (n,24,3) SMPL-24 joints
        J1 = joints(f['root'][chk], f['pose'][chk])
        J1 = J1 - J1[:, :1]
        # observed masks recomputed with pseudo_gt.OBSERVE as it stands (global segment orientation), not read from
        # the fit: fits made before 2026-11-17's change hold the local-rotation version.  The keypoints the fit
        # used are the finite entries of its residual_mm.
        Xb, _ = PG.load_rep(os.path.join(args.root, u[0], 'GT3D'), rep)       # gated BODY_25, camera 1, m
        used = np.isfinite(f['residual_mm'])
        Xb[~used] = np.nan
        obs_swing, obs_full = PG.observed_mask(Xb, joints(f['root'], f['pose']), int(f['gap_frames']))
        ok &= obs_swing[:, PG.F.S24['Pelvis']]

        for c, cam in enumerate(z['cameras']):
            Rc1 = R[c] @ R[0].T
            Rroot_c = Rc1[None] @ Rroot1
            pose = np.concatenate([Rotation.from_matrix(Rroot_c).as_rotvec()[:, None], f['pose']], 1)
            # checks
            Hc = ((H1 - t[0]) @ R[0]) @ R[c].T + t[c]
            uv = Hc @ K[c].T
            uv = uv[..., :2] / uv[..., 2:]
            x2d = z['h36m_2d'][c]
            m = (x2d[..., 2] > 0) & np.isfinite(uv).all(-1) & ok[:, None]
            proj = float(np.median(np.linalg.norm(uv - x2d[..., :2], axis=-1)[m])) if m.any() else np.nan
            orient = np.nan
            if args.mb_root:
                bp = os.path.join(args.mb_root, u, rep, 'Analysis', 'mesh', 'openpose', f'{cam}_betas.npz')
                if os.path.exists(bp):
                    Rmb = np.load(bp)['rotmats'][:, 0].astype(float)
                    if len(Rmb) == T:
                        orient = float(np.median(angle_deg(Rroot_c[ok], Rmb[ok])))
            Jc = joints(pose[chk, 0], pose[chk, 1:])
            Jc = Jc - Jc[:, :1]
            rootonly = float(np.abs(Jc - J1 @ Rc1.T).max() * 1000.0)

            loo = z['X_h36m_cam_loo'][c].astype(np.float32)
            loo_ok = z['valid_h36m_loo_gated'][c] & ev[None]
            seq['rep'].append(rep)
            seq['subject'].append(u)
            seq['cam'].append(str(cam))
            seq['split'].append(split_of(u, sp))
            seq['start'].append(n)
            seq['len'].append(T)
            seq['shape'].append(shape.astype(np.float32))
            seq['offsets_rest_mm'].append(f['offsets_rest_mm'].astype(np.float32))
            seq['proj_px'].append(proj)
            seq['orient_deg'].append(orient)
            seq['rootonly_mm'].append(rootonly)
            fr['x2d_raw'].append(x2d.astype(np.float32))
            fr['x2d'].append(crop_scale(fill_missing(x2d), scale_range=[1, 1]).astype(np.float32))
            fr['pose'].append(pose.astype(np.float32))
            fr['obs_swing'].append(obs_swing)
            fr['obs_full'].append(obs_full)
            fr['x3d_body25'].append((((Xb - t[0]) @ R[0]) @ R[c].T + t[c]).astype(np.float32))
            fr['frame_ok'].append(ok)
            fr['frame_residual_mm'].append(f['frame_residual_mm'].astype(np.float32))
            fr['loo3d'].append(np.where(loo_ok[..., None], loo, np.nan).astype(np.float32))
            fr['loo_ok'].append(loo_ok)
            fr['fit3d'].append(Hc.astype(np.float32))
            n += T
            print(f'  {rep} cam {cam}: {T} fr, frame_ok {100 * ok.mean():.0f}%, projection {proj:.1f} px, '
                  f'root vs MotionBERT {orient:.1f} deg, root-only check {rootonly:.3f} mm', flush=True)

    out = {f'seq_{k}': np.array(v) for k, v in seq.items()}
    out.update({k: np.concatenate(v) for k, v in fr.items()})
    out['note'] = np.array('frames of every sequence concatenated; sequence s = frames [seq_start[s], +seq_len[s]). '
                           'pose: axis-angle (24,3), root in that camera (OpenCV axes); x2d: step_2a-prepared input; '
                           'loo3d / fit3d / x3d_body25: camera frame, metres (cohort stature scale); obs_*: GLOBAL '
                           'segment orientation observed (pseudo_gt.OBSERVE); offsets: groups hip, shoulder, neck, midhip')
    out['fps'] = np.array(30.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    p, o = out['seq_proj_px'], out['seq_orient_deg']
    print(f'\n{len(p)} sequences, {n} frames, frame_ok {100 * out["frame_ok"].mean():.0f}% -> {args.out}')
    print(f'projection px: median {np.nanmedian(p):.1f}, max {np.nanmax(p):.1f} | root vs MotionBERT deg: '
          f'median {np.nanmedian(o):.1f}, max {np.nanmax(o):.1f} | root-only check max '
          f'{out["seq_rootonly_mm"].max():.3f} mm')


if __name__ == '__main__':
    main()
