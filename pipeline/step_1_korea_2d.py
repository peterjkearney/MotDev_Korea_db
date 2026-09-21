#!/usr/bin/env python3
"""step_1_korea_2d.py -- lay out the Korea child ground truth as pipeline trials.

Stream 2 (children).  Reads the per-rep files build_child_gt.py wrote under
GT3D/ and writes, for every usable rep, the trial layout the rest of the
pipeline expects -- so steps 2a, 2b, 3, 4, 5, 6 and 8 run on children unchanged:

    {OUT_DIR}/{subject}/user_meta.json                                       stature (step_3)
    {OUT_DIR}/{subject}/{cam}.mp4-mocAligned.calib                           K, L_ext (step_4, step_8)
    {OUT_DIR}/{subject}/{rep}/Analysis/keypoints/openpose/{cam}_2d.npz       OpenPose 2D (step_2a)
    {OUT_DIR}/{subject}/{rep}/Analysis/H36M/openpose_tri_h36m.npz            target, mocap_h36m.npz layout

Here user = subject (B010), action = rep (B010_GMS_1_1), cameras = 1, 2, 3.
OUT_DIR is config's output root: point BIOCV_OUT at the KOREA folder on Drive
before running this and every later step, e.g.
    %env BIOCV_OUT=/content/drive/MyDrive/MotorDevelopment/Data/Korea
There is no local copy of anything for this dataset (no videos, no mocap), so
the calibs and stature live in OUT_DIR too, where config.calib_path and
config.stature_path look second.  Reps already laid out are skipped (--force).

Frames.  BioCV's "lab" frame is the mocap frame, Z up.  For Korea the lab
frame is the fitted floor frame: Z up, origin on the floor under camera 1.
L_ext maps lab (mm) -> camera (mm) exactly as BioCV's calibs do, so step_8's
camera azimuths and step_6's LAB_UP work as they do for adults.

Targets are leave-one-out (kps3d_loo[c] triangulated from the other two
cameras) and already carry the per-frame quality gates as NaN, so step_8's
NaN check excludes bad frames without knowing about them.

Units.  GT3D files are in metres scaled so the child's measured stature is
the cohort median; mm here, matching mocap.  Stature is written in the same
units, so the pipeline is fed the target's own height.

    python3 step_1_korea_2d.py --gt3d /content/data/Korea/GT3D
    python3 step_1_korea_2d.py --gt3d ... --subjects B010,B011
"""
import argparse
import glob
import json
import os

import numpy as np

from config import OUT_DIR as _OUT_DIR, twod_path as _twod_path, tri_target_path as _tri_path, require_out_dir
from utils.calibration import save_calib
from utils.openpose import H36M_NAMES, body25_to_h36m_2d

EVAL_EXCLUDE = ('Nose', 'Head', 'Spine')   # synthesised joints; see step_1b_triangulate_2d.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt3d', required=True, help='build_child_gt.py output root')
    ap.add_argument('--subjects', default=None)
    ap.add_argument('--include-unusable', action='store_true',
                    help='also lay out reps flagged unusable (still marked in the target)')
    ap.add_argument('--force', action='store_true', help='redo reps already laid out')
    args = ap.parse_args()
    require_out_dir()

    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.gt3d, 'B*'))
                  if os.path.isdir(p))
    if args.subjects:
        want = set(args.subjects.split(','))
        subs = [s for s in subs if s in want]

    n_reps = n_sub = n_skip = 0
    for sub in subs:
        summ_path = os.path.join(args.gt3d, sub, 'session_summary.json')
        if not os.path.exists(summ_path):
            continue
        with open(summ_path) as f:
            summ = json.load(f)
        if not summ.get('usable') and not args.include_unusable:
            continue
        reps = sorted(glob.glob(os.path.join(args.gt3d, sub, f'{sub}_*.npz')))
        if not reps:
            continue
        udir = os.path.join(_OUT_DIR, sub)
        os.makedirs(udir, exist_ok=True)
        wrote_session = False

        for rp in reps:
            d = np.load(rp)
            if not bool(d['usable']) and not args.include_unusable:
                continue
            stub = str(d['stub'])
            if os.path.exists(_tri_path(sub, stub)) and not args.force:      # the target is written last
                n_skip += 1
                continue
            cams = [str(c) for c in d['cameras']]
            T = d['h36m_2d'].shape[1]
            fps = float(d['fps'])
            Rw, org = d['R_world_to_floor'], d['floor_origin']
            R_wc, t_wc = d['R_world_to_cam'], d['t_world_to_cam']
            w, h = [int(v) for v in d['image_size_1080']]

            if not wrote_session:
                with open(os.path.join(udir, 'user_meta.json'), 'w') as f:
                    json.dump({'stature_m': round(float(d['stature_m']), 4),
                               'source': 'crown-to-sole measured from the triangulated '
                                         'reconstruction (build_child_gt.py)'}, f, indent=1)
                for c, cam in enumerate(cams):
                    # lab (floor) -> camera: X_cam = R_c (Rw^T X_floor + org) + t_c
                    R_lc = R_wc[c] @ Rw.T
                    t_lc = (R_wc[c] @ org + t_wc[c]) * 1000.0
                    L = np.eye(4)
                    L[:3, :3], L[:3, 3] = R_lc, t_lc
                    save_calib(os.path.join(udir, f'{cam}.mp4-mocAligned.calib'),
                               w, h, d['K_1080'][c], L, d['dist'][c])
                wrote_session = True
                n_sub += 1

            # 2D input per camera, in step_1's file layout
            for c, cam in enumerate(cams):
                xy = d['xy_openpose_1080'][c]
                cf = d['conf_openpose'][c]
                body25 = np.concatenate([xy, cf[..., None]], axis=-1).astype(np.float32)
                # source_frame_idx indexes the aligned rows (what the target uses);
                # video_frame_idx is the frame of the (sync-offset) source video.
                os.makedirs(os.path.dirname(_twod_path(sub, stub, 'openpose', cam)), exist_ok=True)
                np.savez(_twod_path(sub, stub, 'openpose', cam),
                         h36m_2d=d['h36m_2d'][c].astype(np.float32), body25_2d=body25,
                         source_frame_idx=np.arange(T), video=str(d['video_path'][c]),
                         video_frame_idx=d['video_frame_idx'][c],
                         fps=fps, detected=(d['h36m_2d'][c][:, :, 2] > 0).any(axis=1),
                         joint_names=np.array(H36M_NAMES), detector=np.array('openpose_body25'))

            # targets in the floor frame, mm, gated frames as NaN
            def to_floor_mm(X_cam, c):
                X_world = (X_cam - t_wc[c]) @ R_wc[c]          # R^T applied on the right
                return ((X_world - org) @ Rw.T) * 1000.0

            X3 = d['X_h36m_floor'].astype(float) * 1000.0
            X3[~(d['valid_h36m'] & d['frame_usable'][:, None])] = np.nan
            loo = np.full((len(cams), T, 17, 3), np.nan, np.float32)
            for c in range(len(cams)):
                Xc = to_floor_mm(d['X_h36m_cam_loo'][c].astype(float), c)
                Xc[~(d['valid_h36m_loo'][c] & d['frame_usable_loo'][c][:, None])] = np.nan
                loo[c] = Xc
                # round trip: the calib written above must map it back to camera c
                Lp = np.eye(4)
                Lp[:3, :3] = R_wc[c] @ Rw.T
                Lp[:3, 3] = (R_wc[c] @ org + t_wc[c]) * 1000.0
                back = (Xc @ Lp[:3, :3].T + Lp[:3, 3]) / 1000.0
                ref = d['X_h36m_cam_loo'][c]
                m = np.isfinite(back).all(-1) & np.isfinite(ref).all(-1)
                if m.any():
                    assert np.abs(back[m] - ref[m]).max() < 1e-4, 'calib/target frame mismatch'

            os.makedirs(os.path.dirname(_tri_path(sub, stub)), exist_ok=True)
            np.savez(_tri_path(sub, stub),
                     kps3d=X3.astype(np.float32), kps3d_loo=loo, cameras=np.array(cams),
                     source_frame_idx=np.arange(T), fps=fps, units=np.array('mm'),
                     valid_joint_mask=np.array([n not in EVAL_EXCLUDE for n in H36M_NAMES]),
                     joint_names=np.array(H36M_NAMES),
                     frame=np.array('floor: z up, origin under camera 1; X_cam = R @ X + t per calib'),
                     stature_mm=float(d['stature_m']) * 1000.0)
            n_reps += 1
        print(f'{sub}: laid out {n_reps} reps so far -> {udir}')
    print(f'\n{n_sub} subjects, {n_reps} reps laid out under {_OUT_DIR}, {n_skip} already there')


if __name__ == '__main__':
    main()
