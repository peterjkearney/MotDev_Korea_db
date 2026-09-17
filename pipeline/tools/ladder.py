#!/usr/bin/env python3
"""ladder.py -- why did the BioCV numbers get worse?  One trial, one change at a time.

Builds variant copies of a trial under a separate work root and runs each
through the unchanged pipeline steps (2a, 2b, 3, 4, 8), so every rung differs
from its neighbour in exactly one thing:

  A  YOLO 2D, betas AVERAGED over all cameras           (the original recipe)
  B  YOLO 2D, per-camera betas                          A->B: beta averaging
  C  OpenPose 2D as run: missing joints = (0,0,0)       B->C: the detector bundle
  D  OpenPose 2D with missing joints FILLED (carried    C->D: the zero-joint
     from the nearest seen frame, low confidence)              encoding alone
  E  D + betas gate on the 12 limb joints only          D->E: the confidence gate

Placed and smoothed are both reported for every rung, and every rung is
scored against mocap (the detector-independent truth); the triangulated
target is added where it exists.  Rung F needs no lifting at all: the 2D
detections themselves against projected mocap, YOLO vs OpenPose, same frames.

Needs, under TRIAL_DIR/{user}/{action}: the videos (for YOLO), the OpenPose
JSON dirs or the OpenPose {cam}_2d.npz files already in Analysis/keypoints,
and mocap_h36m.npz.  YOLO 2D is regenerated with step_1_extract_2d.py
(ultralytics + the model in MODELS_DIR), once, and shared by A and B.

    python3 tools/ladder.py --user User08 --action P08_CMJM_01 --camera 07
    python3 tools/ladder.py ... --rungs C,D,E        # skip the YOLO rungs
    python3 tools/ladder.py ... --prepare-only        # build inputs, run nothing on the GPU
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
sys.path.insert(0, _PIPE)
from config import TRIAL_DIR, RESULTS_DIR, mocap_path, twod_path
from utils.calibration import load_calib, reproject
from utils.missing_joints import fill_missing

H36M_CORE = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
H36M_NAMES = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax',
              'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
RUNGS = {
    'A': 'YOLO, averaged betas',
    'B': 'YOLO, per-camera betas',
    'C': 'OpenPose, zeros for missing joints',
    'D': 'OpenPose, missing joints filled',
    'E': 'OpenPose filled, core-joint betas gate',
}


# ------------------------------------------------------------- 2D transforms

def final_betas_from(betas_path, twod_path, out_path, joints=None, thresh=0.7):
    """Median betas over frames whose mean 2D confidence (over `joints`, or all)
    clears `thresh` -- what step_2b does, with the joint set as a knob."""
    conf = np.load(twod_path)['h36m_2d'][:, :, 2]
    if joints is not None:
        conf = conf[:, joints]
    b = np.load(betas_path)['betas']
    gate = conf.mean(axis=1) > thresh
    if not gate.any():
        return 0
    np.savez(out_path, betas=np.median(b[gate], axis=0), n_frames_used=int(gate.sum()),
             n_frames=len(gate), conf_gate=thresh)
    return int(gate.sum())


# ---------------------------------------------------------- variant plumbing

def run(cmd, env, log):
    with open(log, 'a') as f:
        f.write(' '.join(cmd) + '\n')
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=_PIPE, env=env)
    if p.returncode != 0:
        raise RuntimeError(f'{os.path.basename(cmd[1])} failed (rc={p.returncode}) -- see {log}')


def step(script, user, action, env, log, *extra):
    run([sys.executable, os.path.join(_PIPE, script), '--user', user, '--action', action, *extra],
        env, log)


def make_variant(work, user, action, rung, base_trial, base_kp_src, cams):
    """A trial dir with the base's videos (symlinks), mocap and target, no 2D yet."""
    vdir = os.path.join(work, user, f'{action}_{rung}')
    kp = os.path.join(vdir, 'Analysis', 'keypoints')
    if os.path.isdir(vdir):
        shutil.rmtree(vdir)
    os.makedirs(kp)
    for cam in cams:
        v = os.path.join(base_trial, f'{cam}.mp4')
        if os.path.exists(v):
            os.symlink(v, os.path.join(vdir, f'{cam}.mp4'))
    p = os.path.join(base_kp_src, 'openpose_tri_h36m.npz')
    if os.path.exists(p):
        shutil.copy2(p, kp)
    h36m = os.path.join(vdir, 'Analysis', 'H36M')          # config.mocap_path's layout, under the work root
    os.makedirs(h36m, exist_ok=True)
    shutil.copy2(mocap_path(user, action), h36m)
    return vdir, kp


def read_metrics(vdir, cam, gt):
    p = os.path.join(vdir, 'Analysis', 'diagnostics',
                     'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz')
    if not os.path.exists(p):
        return None
    m = np.load(p, allow_pickle=True)
    cams = [str(c) for c in m['cameras']]
    if cam not in cams:
        return None
    i = cams.index(cam)
    st = float(m['stature_mm']) if 'stature_mm' in m.files else np.nan
    out = dict(placed_mm=float(m['err_placed_all'][i]), smooth_mm=float(m['err_smooth_all'][i]),
               pa_mm=float(m['pa_mpjpe'][i]), n_mm=float(m['n_mpjpe'][i]),
               placed_pct=100 * float(m['err_placed_all'][i]) / st,
               n_frames=int(m['n_frames'][i]))
    names = [str(b) for b in m['bone_names']]
    r = m['bone_ratio'][i]
    for key in ('thigh', 'shank', 'upperarm', 'forearm'):
        out[f'{key}_ratio'] = float(np.nanmean([r[names.index(f'l_{key}')], r[names.index(f'r_{key}')]]))
    return out


def px_vs_mocap(twod_path, calib_path, mocap_path):
    """Per-joint pixel error of the 2D detections against projected mocap."""
    d = np.load(twod_path)
    h, sfi = d['h36m_2d'], d['source_frame_idx']
    m = np.load(mocap_path)
    K3 = np.full((len(sfi), 17, 3), np.nan)
    ok = sfi < len(m['kps3d'])
    K3[ok] = m['kps3d'][sfi[ok]]
    w, hh, K, L, dist = load_calib(calib_path)
    cam = K3 @ L[:3, :3].T + L[:3, 3]
    uv = reproject(cam.reshape(-1, 3), K, dist.reshape(1, 5)).reshape(len(sfi), 17, 2)
    uv[cam[..., 2] <= 0] = np.nan
    det = (h[:, :, 2] > 0.3) & np.isfinite(uv).all(-1) & m['valid_joint_mask'][None, :]
    e = np.linalg.norm(h[:, :, :2] - uv, axis=-1)
    per_joint = np.array([np.median(e[det[:, j], j]) if det[:, j].any() else np.nan for j in range(17)])
    seen = det[:, H36M_CORE].mean()
    return per_joint, float(np.nanmedian(per_joint[H36M_CORE])), float(seen)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--camera', required=True, help='the camera every rung is scored on')
    ap.add_argument('--rungs', default='A,B,C,D,E')
    ap.add_argument('--work-dir', default=None, help='default {TRIAL_DIR}/../ladder')
    ap.add_argument('--openpose-2d', default=None,
                    help="dir holding the OpenPose {cam}_2d.npz files; default the base trial's Analysis/keypoints")
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--accel-std', default='10')
    args = ap.parse_args()
    rungs = [r for r in args.rungs.split(',') if r in RUNGS]
    cam = args.camera
    user, action = args.user, args.action

    base = os.path.join(TRIAL_DIR, user, action)
    base_kp = os.path.join(base, 'Analysis', 'keypoints')
    # In a fresh Colab session only the videos are on local disk; the trial's
    # Analysis/ (mocap, OpenPose 2D, target) lives in RESULTS_DIR on Drive.
    if not os.path.exists(mocap_path(user, action)):
        raise SystemExit(f'no {mocap_path(user, action)} -- run step_0_load_mocap.py first')
    if not os.path.isdir(base_kp):
        alt = os.path.join(RESULTS_DIR, user, action, 'Analysis', 'keypoints')
        if os.path.isdir(alt):
            print(f'no local Analysis/ for this trial -- using the synced copy {alt}')
            base_kp = alt
    work = args.work_dir or os.path.abspath(os.path.join(TRIAL_DIR, '..', 'ladder'))
    os.makedirs(os.path.join(work, user), exist_ok=True)
    for p in glob.glob(os.path.join(TRIAL_DIR, user, '*.calib')) + [os.path.join(TRIAL_DIR, user, 'user_meta.json')]:
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(work, user))
    env = dict(os.environ, BIOCV_ROOT=work, BIOCV_OUT=work, BIOCV_RESULTS=os.path.join(work, 'results'))
    log = os.path.join(work, f'{user}_{action}_{cam}_ladder.log')
    open(log, 'w').close()

    cams = sorted(os.path.splitext(os.path.basename(v))[0] for v in glob.glob(os.path.join(base, '0*.mp4')))
    if not cams:
        cams = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(base_kp, '*_2d.npz')))
    have_yolo = all(os.path.exists(twod_path(user, action, 'yolo', c)) for c in cams)
    if any(r in ('A', 'B') for r in rungs) and not have_yolo and not glob.glob(os.path.join(base, '0*.mp4')):
        raise SystemExit(
            f'rungs A/B regenerate YOLO 2D from the trial\'s videos (00.mp4 .. 08.mp4), which '
            f'live in the BioCV data root -- but BIOCV_ROOT is {TRIAL_DIR}, which has none for '
            f'{user}/{action}. Set BIOCV_ROOT to the data root (e.g. /content/data/BioCV) with '
            f'the trial copied there, keep BIOCV_RESULTS on the results folder, or use --rungs C,D,E.')
    op_dir = args.openpose_2d or os.path.dirname(twod_path(user, action, 'openpose', cam))
    if cam not in cams:
        raise SystemExit(f'camera {cam} not among {cams}')
    print(f'trial {user}/{action}, camera {cam}, cameras {",".join(cams)}\nwork {work}\nlog  {log}\n')

    # ---- YOLO 2D once, shared by A and B ---------------------------------
    yolo_dir = os.path.dirname(twod_path(user, action, 'yolo', cam)) if have_yolo else None
    if any(r in ('A', 'B') for r in rungs) and not have_yolo:
        ydir, ykp = make_variant(work, user, action, 'yolo2d', base, base_kp, cams)
        yolo_dir = os.path.join(ykp, 'yolo')              # step_1 writes to keypoints/yolo/ under BIOCV_OUT = work
        existing = glob.glob(os.path.join(yolo_dir, '*_2d.npz'))
        if not existing:
            t = time.time()
            print('running YOLO (step_1_extract_2d.py) on all cameras ...', flush=True)
            step('step_1_extract_2d.py', user, f'{action}_yolo2d', env, log)
            print(f'  done ({time.time() - t:.0f}s)')

    # ---- rung F: raw 2D vs mocap, no lifting -----------------------------
    calib = os.path.join(work, user, f'{cam}.mp4-mocAligned.calib')
    mocap = mocap_path(user, action)
    print('F  2D detections vs projected mocap (median px over the clip, camera %s)' % cam)
    for name, p in (('YOLO', os.path.join(yolo_dir, f'{cam}_2d.npz') if yolo_dir else None),
                    ('OpenPose', os.path.join(op_dir, f'{cam}_2d.npz'))):
        if p and os.path.exists(p):
            pj, core, seen = px_vs_mocap(p, calib, mocap)
            print(f'   {name:9s} core-joint median {core:5.1f} px, detected {100 * seen:.0f}% of core joints | ' +
                  ' '.join(f'{H36M_NAMES[j]} {pj[j]:.0f}' for j in H36M_CORE))
    print()

    # ---- build every rung's inputs ---------------------------------------
    variants = {}
    for r in rungs:
        vdir, kp = make_variant(work, user, action, r, base, base_kp, cams)
        for c in cams:
            if r in ('A', 'B'):
                src = os.path.join(yolo_dir, f'{c}_2d.npz')
                if os.path.exists(src):
                    shutil.copy2(src, kp)
            else:
                src = os.path.join(op_dir, f'{c}_2d.npz')
                if not os.path.exists(src):
                    continue
                d = dict(np.load(src))
                if r in ('D', 'E'):
                    d['h36m_2d'] = fill_missing(d['h36m_2d'])
                np.savez(os.path.join(kp, f'{c}_2d.npz'), **d)
        variants[r] = vdir
    if args.prepare_only:
        print('prepared:', ', '.join(f'{r}: {v}' for r, v in variants.items()))
        return

    # ---- run each rung through 2a, 2b, 3, 4, 8 ---------------------------
    rows = {}
    for r in rungs:
        vaction = f'{action}_{r}'
        vdir = variants[r]
        kp = os.path.join(vdir, 'Analysis', 'keypoints')
        run_cams = cams if r == 'A' else [cam]          # averaging needs every camera's betas
        t = time.time()
        print(f'{r}  {RUNGS[r]}: 2a on {len(run_cams)} camera(s) ...', flush=True)
        # step_2a fills missing joints by default now; rung C is the old behaviour on purpose
        step('step_2a_extract_betas.py', user, vaction, env, log, '--cameras', ','.join(run_cams),
             *(['--missing', 'zero'] if r == 'C' else []))
        mesh = os.path.join(kp, 'mesh')
        if r == 'A':
            # the original step_2b: per-camera medians, then the mean over cameras
            meds = []
            for c in run_cams:
                p = os.path.join(mesh, f'{c}_final_betas.npz')
                if final_betas_from(os.path.join(mesh, f'{c}_betas.npz'), os.path.join(kp, f'{c}_2d.npz'), p):
                    meds.append(np.load(p)['betas'])
            avg = np.mean(meds, axis=0)
            np.savez(os.path.join(mesh, f'{cam}_final_betas.npz'), betas=avg,
                     n_frames_used=-1, n_cameras_averaged=len(meds))
            print(f'   betas averaged over {len(meds)} cameras')
        else:
            joints = H36M_CORE if r == 'E' else None
            n = final_betas_from(os.path.join(mesh, f'{cam}_betas.npz'), os.path.join(kp, f'{cam}_2d.npz'),
                                 os.path.join(mesh, f'{cam}_final_betas.npz'), joints=joints)
            print(f'   betas from {n} frames')
            if n == 0:
                print('   no frame passed the gate -- rung skipped')
                continue
        step('step_3_extract_3d.py', user, vaction, env, log, '--cameras', cam)
        step('step_4_PnP.py', user, vaction, env, log, '--cameras', cam, '--process-accel-std', args.accel_std)
        step('step_8_spider_error.py', user, vaction, env, log, '--cameras', cam, '--gt', 'mocap')
        row = read_metrics(vdir, cam, 'mocap') or {}
        if os.path.exists(os.path.join(kp, 'openpose_tri_h36m.npz')):
            step('step_8_spider_error.py', user, vaction, env, log, '--cameras', cam, '--gt', 'triangulated')
            tri = read_metrics(vdir, cam, 'triangulated')
            if tri:
                row['placed_mm_tri'] = tri['placed_mm']
        rows[r] = row
        print(f'   {time.time() - t:.0f}s')

    # ---- table ------------------------------------------------------------
    cols = ['placed_mm', 'smooth_mm', 'pa_mm', 'n_mm', 'placed_pct', 'thigh_ratio', 'shank_ratio',
            'upperarm_ratio', 'forearm_ratio', 'placed_mm_tri']
    print('\nscored against mocap, camera %s (mm; ratios = predicted / mocap bone length)' % cam)
    print(f"{'rung':4s} {'':40s} " + ' '.join(f'{c:>14s}' for c in cols))
    for r, row in rows.items():
        print(f'{r:4s} {RUNGS[r]:40s} ' + ' '.join(
            f"{row[c]:14.2f}" if c in row and np.isfinite(row.get(c, np.nan)) else f"{'-':>14s}" for c in cols))
    out = os.path.join(work, f'{user}_{action}_{cam}_ladder.json')
    with open(out, 'w') as f:
        json.dump({r: {k: (None if not np.isfinite(v) else float(v)) for k, v in row.items()}
                   for r, row in rows.items()}, f, indent=1)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
