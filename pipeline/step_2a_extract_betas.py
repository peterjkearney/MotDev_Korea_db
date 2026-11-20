#!/usr/bin/env python3
"""step_2a_extract_betas.py -- MotionBERT pass 1: per-frame SMPL rotations and betas.

Runs MotionBERT-Mesh (whole-clip crop_scale, per-frame windowed inference,
--clip-len frames centred on each frame, default 16) on one detector's 2D and
keeps just the rotations (theta) and the per-frame betas; step_2b picks the
body shape from them and step_3 builds the skeleton with it.

--clip-len: the checkpoint (MB_ft_pw3d) was fine-tuned on 16-frame clips (3DPW,
30 fps; H36M at 10 fps), the backbone pretrained on 243.  The window length is
stored in the output; a camera done with a different length is refused unless
--force (steps skip by file existence, so mixing would otherwise be silent --
use a fresh OUT_DIR per length, see datasets/korea/clone_out_dir.py).
Default 16 since 2026-11-13: on the 161 Korea dev reps raw MotionBERT placed error was
100.1 mm at 16 vs 102.4 at 81 (PA 41.4 vs 42.5); 16 at stride 3 (10 fps) gave 103.2.
Outputs made before then are 81 (files without the key); resume such a tree with --clip-len 81.
--stride: frames between window samples (default 1).  Korea video is 30 fps, so
--clip-len 16 --stride 3 feeds 16 frames at 10 fps (1.5 s), H36M's training rate;
stored as 'stride' (files without it = 1) and guarded like --clip-len.

    reads   {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz
    writes  {OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz

--detector picks whose 2D goes in (openpose, the default, or yolo); every
later step takes the same argument and writes to a matching subfolder.  With
no --user / --action it does every camera that has 2D; cameras already done
are skipped (--force redoes them).  Needs a GPU in practice.

Missing joints (confidence 0) are filled from the nearest seen frame before
MotionBERT sees them -- see utils/missing_joints.py; --missing zero keeps
them at (0,0,0) as the pipeline once did.

    python3 step_2a_extract_betas.py                                   # everything, OpenPose
    python3 step_2a_extract_betas.py --detector yolo --user User03 --action P03_CMJM_01
    python3 step_2a_extract_betas.py --dry-run
"""

import argparse
import os
import sys

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rot

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
from config import DETECTORS, twod_path as _twod_path, betas_path as _betas_path, require_out_dir, OUT_DIR as _OUT_DIR, find_cameras
from config import MB_DIR as _MB_DIR      # resolved in config.py; ../MotionBERT no longer holds here

from config import require_motionbert
require_motionbert()
for _d in (_MB_DIR, _SCRIPT_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
from lib.utils.utils_data import crop_scale, flip_data
from lib.utils.utils_mesh import flip_thetas
from lib.model.model_mesh import MeshRegressor
from utils.missing_joints import fill_missing

_MB_MESH_CFG  = os.path.join(_MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml')
_MB_MESH_CKPT = os.path.join(_MB_DIR, 'checkpoint', 'mesh',
                               'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin')
# MB_MESH_CKPT=<path> swaps in another checkpoint of the same model (e.g. training/finetune.py's best.bin)
_MB_MESH_CKPT = os.environ.get('MB_MESH_CKPT') or _MB_MESH_CKPT

_CLIP_LEN = 16      # default window; --clip-len overrides (was 81 before 2026-11-13)


def stored_clip_len(path):
    """Window length a {cam}_betas.npz was made with (files from before --clip-len existed were 81)."""
    z = np.load(path)
    return int(z['clip_len']) if 'clip_len' in z.files else 81


def stored_stride(path):
    """Frame step a {cam}_betas.npz was made with (files from before --stride existed were 1)."""
    z = np.load(path)
    return int(z['stride']) if 'stride' in z.files else 1

def run_backbone(model, batch, clip_len, device):
    if device == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            feat = model.backbone.get_representation(batch)
    else:
        feat = model.backbone.get_representation(batch)
    feat = feat.reshape(batch.shape[0], clip_len, model.feat_J, -1)
    return feat[:, clip_len // 2 : clip_len // 2 + 1, :, :]


def stored_flip(path):
    """--flip the betas file was made with (files from before the option: False)."""
    with np.load(path) as z:
        return bool(z['flip']) if 'flip' in z.files else False


def mean_rotation(R):
    """(K,24,3,3) -> (24,3,3): the chordal mean -- the average matrix projected back onto the rotations (SVD)."""
    U, _, Vt = np.linalg.svd(R.mean(0))
    D = np.ones((R.shape[1], 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[:, None, :]) @ Vt


def _reflect_index(i, n_frames):
    """
    Map a possibly out-of-range window index to [0, n_frames-1] by
    reflection (mirroring at the boundaries) rather than clamping/repeating
    the edge frame. For clips shorter than CLIP_LEN (routine for these
    short hop clips — see extract_hop_features.py's companion discussion),
    clamping means the window near either edge is dominated by dozens of
    copies of a single frozen frame, a badly out-of-distribution input for
    a transformer trained on continuously-varying motion. Reflection gives
    it continuously-varying (if mirrored) motion instead.
    """
    if n_frames == 1:
        return 0
    period = 2 * (n_frames - 1)
    i = i % period
    if i >= n_frames:
        i = period - i
    return i



def load_model(device):
    print("[init] MotionBERT-Mesh ...")
    mb_args = get_config(_MB_MESH_CFG)
    mb_args.data_root = os.path.join(_MB_DIR, 'data', 'mesh')
    backbone = load_backbone(mb_args)
    mesh_model = MeshRegressor(mb_args, backbone=backbone,
                                dim_rep=mb_args.dim_rep,
                                hidden_dim=mb_args.hidden_dim,
                                dropout_ratio=mb_args.dropout)
    print(f'MotionBERT-Mesh checkpoint: {_MB_MESH_CKPT}', flush=True)
    ckpt = torch.load(_MB_MESH_CKPT, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    mesh_model.load_state_dict(sd, strict=True)
    mesh_model.eval()
    if device == 'cuda':
        mesh_model = mesh_model.cuda()
    print(f"       mesh on {device}")
    return mesh_model


def extract(user, action, camera, mesh_model, device, args):
    """One camera's 2D -> {cam}_betas.npz.  Returns a summary line."""
    h36m_2d = np.load(_twod_path(user, action, args.detector, camera))['h36m_2d']
    n_frames = h36m_2d.shape[0]
    n_missing = int((h36m_2d[:, :, 2] <= 0).sum())
    if args.missing == 'fill' and n_missing:
        h36m_2d = fill_missing(h36m_2d)

    motion_norm = crop_scale(h36m_2d.astype(np.float32), scale_range=[1, 1])
    clip_len = args.clip_len
    n_views = 2 if args.flip else 1      # --flip: the window and its mirror image in one batch
    batch = torch.zeros(n_views, clip_len, 17, 3, dtype=torch.float32, device=device)
    half = clip_len // 2       # even lengths: half frames before, half-1 after (16 -> 8 back, 7 forward)
    stride = args.stride       # frames between window samples; the centre sample is always frame_n

    all_rotmats = np.zeros((n_frames, 24, 3, 3), dtype=np.float32)
    all_betas = np.zeros((n_frames,10),dtype=np.float32)

    for frame_n in range(n_frames):
        indices = [_reflect_index(frame_n + (j - half) * stride, n_frames) for j in range(clip_len)] # array of indices for this window (accounting for relfecting at boundaries)
        win_np = motion_norm[indices] # pulling batch of numpy 2D keypoints for frames in window

        with torch.no_grad():
            win_t = torch.from_numpy(win_np[None])
            if args.flip:
                win_t = torch.cat([win_t, flip_data(win_t)])
            batch.copy_(win_t)
            feat_center = run_backbone(mesh_model, batch, clip_len, device) # Runs the DSTformer backbone over the full window
                        # (at 81, every frame's representation is informed by 40 frames of context on each side), then slices out just the
                        # single feature vector for the center frame (clip_len//2, i.e. frame_n itself).
            smpl_out = mesh_model.head(feat_center.float()) # Feeds that one frame's feature vector through the SMPL regression head, producing a dict with the mesh model's outputs for this single frame: 3D keypoints and SMPL pose parameters.

        theta = smpl_out[0]['theta'].reshape(n_views, -1).float().cpu().numpy()
        pose_aa = theta[:, :72].reshape(n_views, 24, 3)  # pose in axis-angle format
        if args.flip:
            # MotionBERT's own test-time flip (train_mesh.py evaluate): the mirror image's pose mirrored back
            # (flip_thetas), then the two averaged -- as rotations here, not as axis-angle numbers
            pose_aa[1] = flip_thetas(pose_aa[1:2])[0]
            R = _Rot.from_rotvec(pose_aa.reshape(-1, 3)).as_matrix().reshape(n_views, 24, 3, 3)
            all_rotmats[frame_n] = mean_rotation(R).astype(np.float32)
        else:
            all_rotmats[frame_n] = _Rot.from_rotvec(pose_aa[0]).as_matrix().astype(np.float32)
        all_betas[frame_n] = theta[:, 72:].mean(0)

    out_path = _betas_path(user, action, args.detector, camera)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, rotmats=all_rotmats, betas=all_betas, avg_betas=np.mean(all_betas, axis=0),
             detector=np.array(args.detector), missing=np.array(args.missing), n_missing_joints=n_missing,
             clip_len=clip_len, stride=stride, flip=bool(args.flip))
    return (f'{n_frames} frames' +(f', {n_missing} missing joints ({100 * n_missing / (n_frames * 17):.1f}%) '
                                    f'{"filled" if args.missing == "fill" else "left at zero"}' if n_missing else ''))


def main():
    ap = argparse.ArgumentParser(description='MotionBERT pass 1 (rotations + betas). With no --user/--action, '
                                             'every camera with 2D from --detector; cameras already done are skipped.')
    ap.add_argument('--detector', choices=DETECTORS, default='openpose', help="whose 2D goes in (default openpose)")
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help='default: every action (of --user, or of every user)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with 2D')
    ap.add_argument('--missing', choices=['fill', 'zero'], default='fill',
                    help='joints the detector did not return: filled from the nearest seen frame (default), '
                         'or left at (0,0,0) as before -- which crop_scale puts in the corner of the crop '
                         '(see utils/missing_joints.py)')
    ap.add_argument('--clip-len', type=int, default=_CLIP_LEN,
                    help=f'frames per MotionBERT window, centred on each frame (default {_CLIP_LEN}; '
                         'the mesh checkpoint was fine-tuned on 16, the backbone pretrained on 243)')
    ap.add_argument('--stride', type=int, default=1,
                    help='frames between window samples (default 1); 3 on 30 fps video = 10 fps, H36M\'s rate')
    ap.add_argument('--flip', action='store_true',
                    help="average each window with its mirror image, as MotionBERT's own mesh evaluation does "
                         "(train_mesh.py, flip: True); twice the MotionBERT time")
    ap.add_argument('--force', action='store_true', help='redo cameras whose output already exists')
    ap.add_argument('--dry-run', action='store_true', help='list what would be run, load nothing')
    args = ap.parse_args()
    if not 1 <= args.clip_len <= 243:
        raise SystemExit(f'--clip-len {args.clip_len}: the backbone takes 1-243 frames (maxlen)')
    if args.stride < 1:
        raise SystemExit(f'--stride {args.stride}: must be >= 1')

    require_out_dir()
    jobs = find_cameras(_twod_path, args.detector, args.user, args.action, set(args.cameras.split(',')) if args.cameras else None)
    if not jobs:
        raise SystemExit(f'no {args.detector} {{cam}}_2d.npz under {_OUT_DIR} for user={args.user or "*"} '
                         f'action={args.action or "*"} -- run step_1_extract_2d.py / step_1_openpose_2d.py first')
    done = [j for j in jobs if os.path.exists(_betas_path(j[0], j[1], args.detector, j[2]))]
    if not args.force:
        want = (args.clip_len, args.stride, bool(args.flip))
        other = [(j, n) for j in done if (n := (stored_clip_len(p := _betas_path(j[0], j[1], args.detector, j[2])),
                                                stored_stride(p), stored_flip(p))) != want]
        if other:
            raise SystemExit(f'{len(other)} camera(s) already done with a different --clip-len/--stride/--flip, e.g. '
                             f'{"/".join(other[0][0])} at clip_len {other[0][1][0]} stride {other[0][1][1]} '
                             f'flip {other[0][1][2]} -- use a fresh OUT_DIR '
                             f'(datasets/korea/clone_out_dir.py) or --force')
    todo = jobs if args.force else [j for j in jobs if j not in done]
    print(f'{args.detector}: {len(jobs)} camera(s) with 2D under {_OUT_DIR}: {len(todo)} to run, '
          f'{len(jobs) - len(todo)} already done; clip_len {args.clip_len}, stride {args.stride}'
          + (', flip-averaged' if args.flip else ''))
    if args.dry_run:
        for user, action in sorted({(u, a) for u, a, _ in todo}):
            print(f'  {user}/{action}: {",".join(c for u, a, c in todo if (u, a) == (user, action))}')
        return
    if not todo:
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh_model = load_model(device)
    n_done, failed = 0, []
    for i, (user, action, cam) in enumerate(todo, 1):
        try:
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: {extract(user, action, cam, mesh_model, device, args)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad camera must not stop the batch
            failed.append((user, action, cam))
            print(f'[{i}/{len(todo)}] {user}/{action} {cam}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\ndone {n_done}, already done {len(jobs) - len(todo)}, failed {len(failed)}')


if __name__ == '__main__':
    main()
