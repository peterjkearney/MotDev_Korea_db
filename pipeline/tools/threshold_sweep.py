#!/usr/bin/env python3
"""threshold_sweep.py -- OpenPose vs YOLO through MotionBERT, one camera, over a
range of 2D confidence thresholds.  Standalone: it does not touch the trial's
Analysis/ outputs, and runs MotionBERT, SMPL, PnP and the smoother itself.

For every threshold (and every missing-joint encoding) one video is written:

    +---------------------------------------+------------------+
    | OpenPose 2D + the H36M derived from it |                  |
    | on the source video                    |  top-down (as    |
    +---------------------------------------+  step_5): mocap,  |
    | YOLO 2D + the H36M derived from it     |  H36M from       |
    | on the source video                    |  OpenPose, H36M  |
    +---------------------------------------+  from YOLO       |
                                             +------------------+

What the threshold does.  MotionBERT is fed (x, y, confidence) for 17 joints.
A joint whose 2D confidence is below the threshold is treated as MISSING
before it goes in; everything downstream (betas, SMPL skeleton, PnP,
smoother) follows the pipeline's steps 2a-4 with single-camera betas.  How a
missing joint is encoded is the other knob (--missing):
    zero  (x, y, conf) = (0, 0, 0), which is what the pipeline does now with
          OpenPose's missing joints.  crop_scale leaves these out of the
          bounding box but still normalises them, so MotionBERT sees the
          joint in the top-left corner of the crop.
    fill  the same joint's position from the nearest frame where it was kept,
          at confidence 0.1 (tools/ladder.py's rung D).
Threshold 0 feeds the detections as they are (in fill mode, joints the
detector itself left out are still filled).  PnP keeps its own 0.4 gate, so
dropped joints never anchor the placement either.

On the video panels: orange = 2D joints fed to MotionBERT, grey rings =
detected but dropped by the threshold, thin orange rings = filled
substitutes; blue / green = the placed H36M from OpenPose / YOLO.  Top-down:
mocap red, each detector's placed skeleton faded and its smoothed one solid.
Clip-level errors against mocap go in the panel legend, the console table
and summary.csv.

2D inputs: Analysis/keypoints/openpose/{cam}_2d.npz and .../yolo/{cam}_2d.npz
under the trial (TRIAL_DIR, then RESULTS_DIR), which is where
step_1_openpose_2d.py and step_1_extract_2d.py each keep their own copy; or
--openpose-2d / --yolo-2d.  To add the detector the trial was not run with:
    python3 step_1_extract_2d.py --user U --action A --pattern 06.mp4
Videos are rendered on local disk and copied to --out-dir (default on Drive,
{RESULTS_DIR}/threshold_sweep/{user}/{action}).

    python3 tools/threshold_sweep.py                               # User03 P03_CMJM_01 cam 06
    python3 tools/threshold_sweep.py --thresholds 0,0.3,0.5,0.7 --missing fill
    python3 tools/threshold_sweep.py --max-frames 120               # quick look
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
sys.path.insert(0, _PIPE)
from config import TRIAL_DIR, RESULTS_DIR, MB_DIR, mocap_path as _mocap_path, twod_path
from utils.calibration import load_calib, reproject
from utils.metrics import mpjpe, pa_mpjpe
from utils.missing_joints import fill_missing
from utils.rts_smoother import rts_smooth_3d
from utils.topdown import TopDown, hex_bgr

H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
DETECTORS = ('openpose', 'yolo')
DET_LABEL = {'openpose': 'OpenPose', 'yolo': 'YOLO'}
DET_HEX = {'openpose': '#2a78d6', 'yolo': '#1baf7a'}          # H36M derived from each detector
_MOCAP_HEX, _INK, _GRID = '#e0322b', '#4b5563', '#c7cdd4'
_COL_2D, _COL_DROPPED = (0, 140, 255), (170, 170, 170)        # BGR
_CLIP_LEN = 81
_PNP_CONF, _PNP_MIN_POINTS, _MIN_DEPTH_M = 0.4, 6, 0.05


def check_out_dir(out_dir):
    """A path under /content/drive with Drive not mounted is silently created on the VM's
    own disk and dies with the runtime -- refuse it rather than lose the videos."""
    if os.path.abspath(out_dir).startswith('/content/drive') and not os.path.ismount('/content/drive'):
        raise SystemExit(f'{out_dir} is on Google Drive but Drive is not mounted -- run '
                         f"drive.mount('/content/drive') first, or pass --out-dir")


def publish(tmp_path, out_path):
    """Rendered on local disk (cv2 writing through the Drive mount is slow and can truncate),
    then copied to where it is kept and the local copy removed."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(tmp_path, out_path)
    os.remove(tmp_path)


# ------------------------------------------------------------------ 2D inputs

def find_2d(det, explicit, cam, kp_dirs, user, action):
    """{cam}_2d.npz for one detector: the path given, else where its step writes it
    (config.twod_path: Analysis/keypoints/{yolo,openpose}/ under OUT_DIR)."""
    if explicit:
        p = explicit if explicit.endswith('.npz') else os.path.join(explicit, f'{cam}_2d.npz')
        if not os.path.exists(p):
            raise SystemExit(f'no {p}')
        return p
    cands = [twod_path(user, action, det, cam)]
    for p in cands:
        if os.path.exists(p):
            return p
    make = ('step_1_extract_2d.py --pattern {cam}.mp4' if det == 'yolo'
            else 'step_1_openpose_2d.py --cameras {cam}').format(cam=cam)
    raise SystemExit(f'no {DET_LABEL[det]} 2D: looked for ' + ' and '.join(cands)
                     + f'\nmake it with:  python3 {make} --user ... --action ...   (or pass --{det}-2d)')


def apply_threshold(h2d, thr, mode):
    """(fed, kept, filled): the array MotionBERT sees, and which joints were kept / substituted.

    kept = detected and confidence >= thr.  Everything else is (0, 0, 0) (zero), or takes the
    joint's position in the nearest frame where it was kept (fill; utils/missing_joints.py).
    """
    kept = (h2d[:, :, 2] > 0) & (h2d[:, :, 2] >= thr)
    fed = np.where(kept[:, :, None], h2d, 0.0).astype(np.float32)
    if mode == 'fill':
        fed = fill_missing(fed)
    return fed, kept, ~kept & (fed[:, :, 2] > 0)


# ------------------------------------------------------------------ MotionBERT, SMPL, PnP

def load_motionbert(device):
    import torch
    from config import require_motionbert
    require_motionbert()
    if MB_DIR not in sys.path:
        sys.path.insert(0, MB_DIR)
    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone
    from lib.model.model_mesh import MeshRegressor
    cfg = get_config(os.path.join(MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml'))
    cfg.data_root = os.path.join(MB_DIR, 'data', 'mesh')
    model = MeshRegressor(cfg, backbone=load_backbone(cfg), dim_rep=cfg.dim_rep,
                          hidden_dim=cfg.hidden_dim, dropout_ratio=cfg.dropout)
    ckpt = torch.load(os.path.join(MB_DIR, 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin'),
                      map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()}, strict=True)
    return model.eval().to(device)


def reflect_windows(n_frames, clip_len=_CLIP_LEN):
    """(T, clip_len) frame indices of the window centred on each frame, mirrored at the ends (step_2a)."""
    if n_frames == 1:
        return np.zeros((1, clip_len), int)
    half, period = clip_len // 2, 2 * (n_frames - 1)
    i = (np.arange(n_frames)[:, None] - half + np.arange(clip_len)[None, :]) % period
    return np.where(i >= n_frames, period - i, i)


def motionbert_pass(model, fed, device, batch):
    """Step 2a on one 2D array: per-frame SMPL rotations (T,24,3,3) and betas (T,10)."""
    import torch
    from scipy.spatial.transform import Rotation
    from lib.utils.utils_data import crop_scale
    motion = np.asarray(crop_scale(fed.astype(np.float32), scale_range=[1, 1]), np.float32)
    T = motion.shape[0]
    win = reflect_windows(T)
    theta = np.zeros((T, 82), np.float32)
    with torch.no_grad():
        for s in range(0, T, batch):
            x = torch.from_numpy(motion[win[s:s + batch]]).to(device)            # (B,81,17,3)
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    feat = model.backbone.get_representation(x)
            else:
                feat = model.backbone.get_representation(x)
            feat = feat.reshape(x.shape[0], _CLIP_LEN, model.feat_J, -1)[:, _CLIP_LEN // 2:_CLIP_LEN // 2 + 1]
            out = model.head(feat.float())
            theta[s:s + x.shape[0]] = out[0]['theta'].reshape(x.shape[0], -1)[:, :82].cpu().numpy()
    rot = Rotation.from_rotvec(theta[:, :72].reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3).astype(np.float32)
    return rot, theta[:, 72:]


def skeleton_from(model, rotmats, betas, stature, device, chunk=64):
    """Step 3: SMPL with fixed betas + the pass-1 rotations -> H36M joints, scaled to stature (m)."""
    import torch
    smpl = model.head.smpl
    b = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(1, 10)
    eye = torch.eye(3, device=device).repeat(1, 24, 1, 1)
    with torch.no_grad():
        v = smpl(betas=b, body_pose=eye[:, 1:], global_orient=eye[:, 0:1], pose2rot=False).vertices[0]
        mesh_h = float(v[:, 1].max() - v[:, 1].min())
        R = torch.from_numpy(rotmats).to(device)
        out = []
        for s in range(0, len(R), chunk):
            r = R[s:s + chunk]
            o = smpl(betas=b.expand(len(r), 10), body_pose=r[:, 1:], global_orient=r[:, 0:1], pose2rot=False)
            out.append((smpl.J_regressor_h36m @ o.vertices).cpu().numpy())
    scale = stature / mesh_h
    return np.concatenate(out).astype(np.float32) * scale, mesh_h, scale


def place(kps, fed, K, dist_cv, fps, accel_std):
    """Step 4: per-frame SQPnP onto the 2D that was fed (conf > 0.4), then the RTS smoother on the root."""
    placed = np.full_like(kps, np.nan)
    ok = np.zeros(len(kps), bool)
    for t in range(len(kps)):
        m = fed[t, :, 2] > _PNP_CONF
        if m.sum() < _PNP_MIN_POINTS:
            continue
        try:
            good, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(kps[t][m], np.float64),
                                            np.ascontiguousarray(fed[t, m, :2], np.float64),
                                            K, dist_cv, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if good:
            R, _ = cv2.Rodrigues(rvec)
            placed[t] = kps[t] @ R.T.astype(np.float32) + tvec.reshape(3).astype(np.float32)
            ok[t] = True
    smooth = placed.copy()
    if ok.any():
        root, _ = rts_smooth_3d(placed[:, 0], ok, placed[:, 0], 1 / fps, sigma_along=0.15, sigma_perp=0.02,
                                process_accel_std=accel_std)
        smooth = placed + (root - placed[:, 0])[:, None, :]
    return placed, smooth, ok


# ------------------------------------------------------------------ rendering

def draw_video_panel(frame, run, t, K, dist_cv, det, title, size):
    """One stacked panel: the 2D as fed to MotionBERT + the placed H36M derived from it."""
    img = frame.copy()
    H, W = img.shape[:2]
    fs = max(0.5, W / 1920)
    raw, fed, kept, filled = run['raw'][t], run['fed'][t], run['kept'][t], run['filled'][t]
    for a, b in H36M_LIMBS:
        if kept[a] and kept[b]:
            cv2.line(img, tuple(fed[a, :2].astype(int)), tuple(fed[b, :2].astype(int)), _COL_2D, 2, cv2.LINE_AA)
    for j in range(17):
        if kept[j]:
            cv2.circle(img, tuple(fed[j, :2].astype(int)), 4, _COL_2D, -1, cv2.LINE_AA)
        else:
            if raw[j, 2] > 0:                  # detected, dropped by the threshold
                cv2.circle(img, tuple(raw[j, :2].astype(int)), 7, _COL_DROPPED, 2, cv2.LINE_AA)
            if filled[j]:                      # what MotionBERT was given instead
                cv2.circle(img, tuple(fed[j, :2].astype(int)), 5, _COL_2D, 1, cv2.LINE_AA)
    P = run['placed'][t]
    col = hex_bgr(DET_HEX[det])
    placed_ok = bool(np.isfinite(P).all() and (P[:, 2] > _MIN_DEPTH_M).all())
    if placed_ok:
        uv = reproject(P, K, dist_cv)
        for a, b in H36M_LIMBS:
            cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), col, 2, cv2.LINE_AA)
        for p in uv:
            cv2.circle(img, tuple(p.astype(int)), 3, col, -1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'PnP failed this frame', (20, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fs, (0, 0, 255), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * fs, 2)
    cv2.rectangle(img, (10, 10), (30 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(img, title, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.1 * fs, (255, 255, 255), 2, cv2.LINE_AA)
    lines = [(f'{DET_LABEL[det]} 2D fed to MotionBERT: {int(kept.sum())}/17 joints'
              + (f' (+{int(filled.sum())} filled)' if filled.any() else ''), _COL_2D),
             (f'H36M from {DET_LABEL[det]} (placed)', col)]
    if (~kept & (raw[:, 2] > 0)).any():
        lines.append(('dropped by the threshold', _COL_DROPPED))
    for i, (txt, c) in enumerate(lines):
        cv2.putText(img, txt, (20, int(45 * fs + th + 40 * fs * (i + 1))), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * fs, c, 2, cv2.LINE_AA)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def render(out_path, runs, thr, mode, ctx, limits):
    """One video: runs = {'openpose': run, 'yolo': run} for this threshold/encoding."""
    a = ctx['args']
    K, dist_cv, mocap_cam, mocap_joint_ok = ctx['K'], ctx['dist_cv'], ctx['mocap_cam'], ctx['mocap_joint_ok']
    # common timeline: native frames present in both detectors' 2D
    native = np.intersect1d(runs['openpose']['sfi'], runs['yolo']['sfi'])
    row = {d: {int(n): i for i, n in enumerate(runs[d]['sfi'])} for d in DETECTORS}
    cap = cv2.VideoCapture(ctx['video'])
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw = a.panel_width
    ph = int(round(pw * H / W))
    tdw, tdh, gap = pw // 2, 2 * ph + 6, 6
    legend = [('mocap', _MOCAP_HEX, 1.0)]
    for d in DETECTORS:
        m = runs[d]['metrics']
        legend.append((f"H36M from {DET_LABEL[d]}, smooth: MPJPE {m['mpjpe_smooth_mm']:.0f}, PA {m['pa_mpjpe_mm']:.0f} mm",
                       DET_HEX[d], 1.0))
        legend.append((f"H36M from {DET_LABEL[d]}, placed: MPJPE {m['mpjpe_placed_mm']:.0f} mm", DET_HEX[d], 0.35))
    panel = TopDown(tdw, tdh, float(np.arctan(W / (2 * K[0, 0]))), limits[0], limits[1],
                    f'top-down, camera {ctx["cam"]}\nconfidence threshold {thr:g}, missing = {mode}', legend)
    tmp_path = os.path.join(ctx['tmp'], os.path.basename(out_path))
    fps = float(runs['yolo']['fps'])
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (pw + gap + tdw, tdh))
    vsep, hsep = np.zeros((tdh, gap, 3), np.uint8), np.zeros((gap, pw, 3), np.uint8)
    titles = {d: f'{DET_LABEL[d]} 2D -> MotionBERT H36M   (threshold {thr:g}, missing = {mode})' for d in DETECTORS}
    want, fi, n_written = set(int(n) for n in native), 0, 0
    last = int(native[-1]) if len(native) else -1
    while fi <= last:
        if fi not in want:
            if not cap.grab():
                break
            fi += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        panels, skels = [], []
        if fi < len(mocap_cam):
            skels.append((mocap_cam[fi][:, [0, 2]], mocap_joint_ok & np.isfinite(mocap_cam[fi]).all(-1),
                          hex_bgr(_MOCAP_HEX), 2))
        for d in DETECTORS:
            t = row[d][fi]
            panels.append(draw_video_panel(frame, runs[d], t, K, dist_cv, d, titles[d], (pw, ph)))
            valid = np.isfinite(runs[d]['placed'][t]).all(-1)
            skels.append((runs[d]['placed'][t][:, [0, 2]], valid, hex_bgr(DET_HEX[d], fade=0.65), 2))
        for d in DETECTORS:                     # solid smoothed skeletons on top
            t = row[d][fi]
            skels.append((runs[d]['smooth'][t][:, [0, 2]], np.isfinite(runs[d]['smooth'][t]).all(-1),
                          hex_bgr(DET_HEX[d]), 2))
        cv2.putText(panels[1], f'frame {n_written}  t={n_written / fps:.2f}s', (12, ph - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        left = np.concatenate([panels[0], hsep, panels[1]], axis=0)
        writer.write(np.concatenate([left, vsep, panel.render(skels)], axis=1))
        n_written += 1
        fi += 1
    cap.release()
    writer.release()
    publish(tmp_path, out_path)
    print(f'    {n_written} frames -> {out_path}')


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', default='User03')
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--camera', default='06')
    ap.add_argument('--thresholds', default='0,0.2,0.4,0.6,0.8')
    ap.add_argument('--missing', default='zero,fill', help='how dropped joints are encoded: zero, fill, or both')
    ap.add_argument('--betas-gate', type=float, default=0.7,
                    help="step_2b's gate: frames whose mean fed confidence exceeds this give the betas "
                         '(all frames if none do)')
    ap.add_argument('--accel-std', type=float, default=10.0)
    ap.add_argument('--videos', default=None, help="dir holding the trial's {cam}.mp4 if not {TRIAL_DIR}/{user}/{action}")
    ap.add_argument('--openpose-2d', default=None,
                    help='OpenPose {cam}_2d.npz (or its dir); default Analysis/keypoints/openpose/ under the trial')
    ap.add_argument('--yolo-2d', default=None,
                    help='YOLO {cam}_2d.npz (or its dir); default Analysis/keypoints/yolo/ under the trial')
    ap.add_argument('--out-dir', default=None, help='default {RESULTS_DIR}/threshold_sweep/{user}/{action}')
    ap.add_argument('--panel-width', type=int, default=1280, help='width of each video panel; the top-down is half this')
    ap.add_argument('--batch', type=int, default=8, help='MotionBERT windows per forward pass')
    ap.add_argument('--max-frames', type=int, default=0, help='only the first N 2D rows (quick look)')
    args = ap.parse_args()
    user, action, cam = args.user, args.action, args.camera
    thresholds = [float(x) for x in args.thresholds.split(',')]
    modes = [m for m in args.missing.split(',') if m in ('zero', 'fill')]
    if not modes:
        raise SystemExit('--missing must be zero, fill or zero,fill')

    base = os.path.join(TRIAL_DIR, user, action)
    kp_dirs = [os.path.join(r, user, action, 'Analysis', 'keypoints') for r in (TRIAL_DIR, RESULTS_DIR)]

    def first(paths, what):
        p = next((p for p in paths if os.path.exists(p)), None)
        if p is None:
            raise SystemExit(f'no {what}: looked for ' + ' and '.join(paths))
        return p

    ctx = dict(args=args, cam=cam)
    ctx['video'] = first([os.path.join(os.path.abspath(args.videos) if args.videos else base, f'{cam}.mp4')],
                         "video (point --videos at the dir holding the trial's mp4s)")
    mocap_path = first([_mocap_path(user, action)], 'mocap (run step_0_load_mocap.py)')
    calib_path = first([os.path.join(r, user, f'{cam}.mp4-mocAligned.calib') for r in (TRIAL_DIR, RESULTS_DIR)], 'calib')
    meta_path = first([os.path.join(r, user, 'user_meta.json') for r in (TRIAL_DIR, RESULTS_DIR)], 'user_meta.json')
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, 'threshold_sweep', user, action)
    check_out_dir(out_dir)
    with open(meta_path) as f:
        stature = float(json.load(f)['stature_m'])
    w, h, K, L_ext, dist = load_calib(calib_path)
    ctx['K'], ctx['dist_cv'] = K, dist.reshape(1, 5)
    mocap = np.load(mocap_path)
    ctx['mocap_cam'] = (mocap['kps3d'] @ L_ext[:3, :3].T + L_ext[:3, 3]) / 1000.0        # native frames, metres
    ctx['mocap_joint_ok'] = mocap['valid_joint_mask'].astype(bool)
    print(f'trial {user}/{action} camera {cam}, stature {stature:.2f} m\nvideo {ctx["video"]}\n'
          f'thresholds {thresholds}, missing = {modes}\nvideos -> {out_dir}\n')

    # ---- 2D ----
    twod = {}
    for det in DETECTORS:
        print(f'{DET_LABEL[det]} 2D:')
        p = find_2d(det, args.openpose_2d if det == 'openpose' else args.yolo_2d, cam, kp_dirs, user, action)
        d = np.load(p)
        n = args.max_frames or len(d['h36m_2d'])
        twod[det] = dict(h2d=d['h36m_2d'][:n].astype(np.float32), sfi=d['source_frame_idx'][:n].astype(int),
                         fps=float(d['fps']))
        c = twod[det]['h2d'][:, :, 2]
        print(f'  {p}\n  {n} frames; joints detected {100 * (c > 0).mean():.0f}%, confidence of those: '
              f'median {np.median(c[c > 0]):.2f}, 10th pct {np.percentile(c[c > 0], 10):.2f}')

    # ---- MotionBERT -> skeleton -> placement, per detector / threshold / encoding ----
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[init] MotionBERT-Mesh on {device} ...', flush=True)
    model = load_motionbert(device)
    cache, runs, rows = {}, {}, []
    for thr in thresholds:
        for mode in modes:
            for det in DETECTORS:
                src = twod[det]
                fed, kept, filled = apply_threshold(src['h2d'], thr, mode)
                key = (det, hashlib.sha1(fed.tobytes()).hexdigest())
                if key not in cache:             # e.g. YOLO rarely changes between low thresholds
                    t0 = time.time()
                    if kept.sum() < 4:
                        print(f'  {det} thr {thr:g} {mode}: fewer than 4 joints survive -- skipped')
                        cache[key] = None
                    else:
                        rot, betas_t = motionbert_pass(model, fed, device, args.batch)
                        gate = fed[:, :, 2].mean(axis=1) > args.betas_gate
                        n_gate = int(gate.sum())
                        betas = np.median(betas_t[gate] if n_gate else betas_t, axis=0)
                        kps, mesh_h, scale = skeleton_from(model, rot, betas, stature, device)
                        placed, smooth, ok = place(kps, fed, K, ctx['dist_cv'], src['fps'], args.accel_std)
                        cache[key] = dict(placed=placed, smooth=smooth, ok=ok, betas=betas, n_gate=n_gate,
                                          mesh_h=mesh_h, scale=scale, secs=time.time() - t0)
                res = cache[key]
                if res is None:
                    continue
                gt = np.full_like(res['placed'], np.nan)
                inr = src['sfi'] < len(ctx['mocap_cam'])
                gt[inr] = ctx['mocap_cam'][src['sfi'][inr]]
                mask = np.broadcast_to(ctx['mocap_joint_ok'], gt.shape[:2])
                m = dict(mpjpe_placed_mm=1000 * np.nanmean(mpjpe(res['placed'], gt, mask)[0]),
                         mpjpe_smooth_mm=1000 * np.nanmean(mpjpe(res['smooth'], gt, mask)[0]),
                         pa_mpjpe_mm=1000 * np.nanmean(pa_mpjpe(res['placed'], gt, mask)[0]))
                runs[(thr, mode, det)] = dict(res, raw=src['h2d'], fed=fed, kept=kept, filled=filled,
                                              sfi=src['sfi'], fps=src['fps'], metrics=m)
                rows.append(dict(detector=det, threshold=thr, missing=mode,
                                 joints_kept_pct=round(100 * kept.mean(), 1), betas_frames=res['n_gate'],
                                 beta0=round(float(res['betas'][0]), 3), mesh_h_m=round(res['mesh_h'], 3),
                                 scale=round(res['scale'], 3), pnp_ok=int(res['ok'].sum()), n_frames=len(kept),
                                 **{k: round(float(v), 1) for k, v in m.items()}))
                r = rows[-1]
                print(f"  {DET_LABEL[det]:9s} thr {thr:<4g} {mode:4s}  kept {r['joints_kept_pct']:5.1f}%  "
                      f"betas from {(str(r['betas_frames']) + ' fr') if r['betas_frames'] else 'ALL fr (gate passed 0)'}  PnP {r['pnp_ok']}/{r['n_frames']}  "
                      f"MPJPE placed {r['mpjpe_placed_mm']:.0f} smooth {r['mpjpe_smooth_mm']:.0f}  "
                      f"PA {r['pa_mpjpe_mm']:.0f} mm", flush=True)

    # ---- fixed top-down limits across every video, so they can be compared ----
    xs, zs = [ctx['mocap_cam'][..., 0].ravel()], [ctx['mocap_cam'][..., 2].ravel()]
    for r in runs.values():
        xs.append(r['smooth'][..., 0].ravel()); zs.append(r['smooth'][..., 2].ravel())
    xs, zs = np.concatenate(xs), np.concatenate(zs)
    xs, zs = xs[np.isfinite(xs)], zs[np.isfinite(zs)]
    xh = max(float(np.percentile(np.abs(xs), 99)) * 1.25, 0.5)
    limits = ((-xh, xh), (-0.3, max(float(np.percentile(zs, 99)) * 1.15, 1.0)))

    # ---- render (on local disk, then copied out) ----
    print()
    ctx['tmp'] = tempfile.mkdtemp(prefix='threshold_sweep_')
    for thr in thresholds:
        for mode in modes:
            pair = {d: runs.get((thr, mode, d)) for d in DETECTORS}
            if any(v is None for v in pair.values()):
                print(f'  thr {thr:g} {mode}: a detector has no result, not rendered')
                continue
            print(f'  threshold {thr:g}, missing = {mode}')
            render(os.path.join(out_dir, f'{cam}_thr{thr:.2f}_{mode}.mp4'), pair, thr, mode, ctx, limits)

    shutil.rmtree(ctx['tmp'], ignore_errors=True)
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f'{cam}_summary.csv')
        with open(p, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        print(f'\nsummary -> {p}\nvideos  -> {out_dir}')


if __name__ == '__main__':
    main()
