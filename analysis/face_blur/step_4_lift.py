#!/usr/bin/env python3
"""step_4_lift.py -- lift every condition's 2D to 3D, one camera at a time, and
triangulate across all of them.

Lifters (the ones Python_jetson/compare_lifters compares, re-wired to take any
detector's 2D instead of running YOLO themselves):
  mb_pose         MotionBERT-Pose lite (H36M global), normalised units
  mb_mesh         MotionBERT-Mesh -> SMPL -> J_regressor_h36m, mm.  The head is
                  given one frame at a time, as step_2a does: it pools body
                  shape over every frame it is given
  videopose       VideoPose3D, 27-frame TCN (mmpose), centred window
  simplebaseline  SimpleBaseline3D, single frame (mmpose) -- no temporal
                  context, so the most exposed to frame-to-frame jitter
  mp_world        MediaPipe's own world landmarks (mediapipe detector only)

MotionBERT runs over 81-frame windows (step_2a's length, reflected at the
ends) but keeps the central 27 frames of each instead of one, so every frame
still has >= 27 frames of context either side at a twelfth of the cost.

Input 2D: COCO -> H36M with the pipeline's convention (common.coco_to_h36m);
frames with no detection filled from the nearest detected frame at conf 0.1
(utils/missing_joints, as step_2a does) -- step_5 scores only frames the
detector actually found.

lift/{cond}/{det}/{cam}.npz  one (T,17,3) array per lifter, + `detected` (T,)
tri/{cond}/{det}.npz         X (T,17,3) lab mm, n_views (T,17), reproj px (T,17)

    python step_4_lift.py [--conds reenc haar deface oracle] [--dets ...] [--lifters ...]
"""
import argparse
import os
import sys

import numpy as np
import torch

from common import (Trial, det_path, lift_path, tri_path, coco_to_h36m, CAMS, CONDITIONS, DETECTORS,
                    PYTHON_JETSON)
from detectors import MP_TO_COCO
from utils.missing_joints import fill_missing
from utils.triangulate import triangulate_robust

MB_DIR = os.path.join(PYTHON_JETSON, 'MotionBERT')
MMPOSE_CKPTS = os.path.join(PYTHON_JETSON, 'checkpoints', 'mmpose')
LIFTERS = ('mb_pose', 'mb_mesh', 'videopose', 'simplebaseline')
# Checked on P03_CMJM_01 cam 00's OpenPose 2D at 25 fps (PA-MPJPE to mocap): mb_mesh 57 mm, mb_pose 78 mm,
# but videopose 171 mm and simplebaseline 200 mm -- the mmpose pair's output is not a body (hip width ~6 cm),
# so the 2D handed to them is not normalised the way they were trained, and videopose took 15 min a camera.
# Left out by default until that is fixed.
DEFAULT_LIFTERS = ('mb_pose', 'mb_mesh')
CLIP, KEEP = 81, 27
TRI_CONF = 0.5


def _reflect(i, n):
    """step_2a's _reflect_index: mirror out-of-range window indices at the ends."""
    if n == 1:
        return 0
    period = 2 * (n - 1)
    i = i % period
    return period - i if i >= n else i


def _windows(n):
    """(window indices (CLIP,), kept slice in the window, kept frames) covering 0..n-1."""
    lead = (CLIP - KEEP) // 2
    for start in range(0, n, KEEP):
        idx = [_reflect(start - lead + j, n) for j in range(CLIP)]
        k = min(KEEP, n - start)
        yield idx, slice(lead, lead + k), np.arange(start, start + k)


class MotionBert:
    def __init__(self):
        if MB_DIR not in sys.path:
            sys.path.insert(0, MB_DIR)
        from lib.utils.tools import get_config
        from lib.utils.learning import load_backbone
        from lib.model.model_mesh import MeshRegressor
        cfg = get_config(os.path.join(MB_DIR, 'configs', 'pose3d', 'MB_ft_h36m_global_lite.yaml'))
        cfg.flip = False
        self.pose = load_backbone(cfg)
        ck = torch.load(os.path.join(MB_DIR, 'checkpoint', 'pose3d', 'FT_MB_lite_MB_ft_h36m_global_lite',
                                     'best_epoch.bin'), map_location='cpu')
        self.pose.load_state_dict({k.replace('module.', ''): v for k, v in ck['model_pos'].items()})
        self.pose.eval()
        cfg = get_config(os.path.join(MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml'))
        cfg.data_root = os.path.join(MB_DIR, 'data', 'mesh')
        self.mesh = MeshRegressor(cfg, backbone=load_backbone(cfg), dim_rep=cfg.dim_rep,
                                  hidden_dim=cfg.hidden_dim, dropout_ratio=cfg.dropout)
        ck = torch.load(os.path.join(MB_DIR, 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin'),
                        map_location='cpu')
        self.mesh.load_state_dict({k.replace('module.', ''): v for k, v in ck['model'].items()})
        self.mesh.eval()

    @torch.no_grad()
    def __call__(self, h2d, which):
        from lib.utils.utils_data import crop_scale
        x = crop_scale(h2d.astype(np.float32), scale_range=[1, 1]).astype(np.float32)
        out = np.zeros((len(x), 17, 3), np.float32)
        for idx, keep, frames in _windows(len(x)):
            batch = torch.from_numpy(x[idx][None])
            if which == 'mb_pose':
                out[frames] = self.pose(batch)[0, keep].numpy()
            else:
                feat = self.mesh.backbone.get_representation(batch).reshape(1, CLIP, 17, -1)[:, keep]
                feat = feat.reshape(-1, 1, 17, feat.shape[-1])        # one frame per sample: per-frame shape
                out[frames] = self.mesh.head(feat)[0]['kp_3d'].reshape(-1, 17, 3).numpy()
        return out


class MmposeLifter:
    CFG = {'videopose': 'video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m',
           'simplebaseline': 'image-pose-lift_tcn_8xb64-200e_h36m'}
    CKPT = {'videopose': 'videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth',
            'simplebaseline': 'simple3Dbaseline_h36m-f0ad73a4_20210419.pth'}

    def __init__(self, name):
        from mmpose.apis import init_model
        self.model = init_model(os.path.join(MMPOSE_CKPTS, self.CFG[name] + '.py'),
                                os.path.join(MMPOSE_CKPTS, self.CKPT[name]), device='cpu')
        self.model.double()     # mmpose's 2D normalisation hands the TCN float64 inputs; it is tiny, so match them
        ds = self.model.cfg.test_dataloader.dataset
        self.seq_len, self.causal = ds.get('seq_len', 1), ds.get('causal', False)

    def __call__(self, h2d):
        """As mmpose's demo/body3d_pose_lifter_demo.py: each frame's pose moved into
        H36M's average box (norm_pose_2d) and normalised to H36M's 1000x1002 images."""
        from mmengine.structures import InstanceData
        from mmpose.apis import extract_pose_sequence, inference_pose_lifter_model
        from mmpose.structures import PoseDataSample
        seq = []
        for f in range(len(h2d)):
            kp = h2d[f, :, :2]
            s = PoseDataSample()
            s.pred_instances = InstanceData(
                keypoints=kp[None].astype(np.float32), keypoint_scores=h2d[f, :, 2][None],
                bboxes=np.array([[*kp.min(0), *kp.max(0)]], np.float32))
            s.gt_instances = InstanceData()
            s.track_id = 0
            seq.append([s])
        out = np.zeros((len(h2d), 17, 3), np.float32)
        for f in range(len(h2d)):
            win = extract_pose_sequence(seq, frame_idx=f, causal=self.causal, seq_len=self.seq_len, step=1)
            r = inference_pose_lifter_model(self.model, win, with_track_id=False,
                                            image_size=(1000, 1002), norm_pose_2d=True)
            k = np.asarray(r[0].pred_instances.keypoints)
            out[f] = k.reshape(-1, 17, 3)[-1] if k.ndim > 2 else k
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--action', default='P03_CMJM_01')
    ap.add_argument('--conds', nargs='+', default=['reenc', 'haar', 'deface', 'oracle'], choices=CONDITIONS)
    ap.add_argument('--dets', nargs='+', default=list(DETECTORS), choices=DETECTORS + ('openpose',),
                    help='openpose: from step_2b_openpose (run on Colab)')
    ap.add_argument('--lifters', nargs='+', default=list(DEFAULT_LIFTERS), choices=LIFTERS)
    ap.add_argument('--cams', nargs='+', default=CAMS)
    ap.add_argument('--thresh-px', type=float, default=20.0)
    ap.add_argument('--threads', type=int, default=None, help='torch threads (when running one process per detector)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    trial = Trial(args.action)
    models = {}
    for cond in args.conds:
        for d in args.dets:
            cams = [c for c in args.cams if os.path.exists(det_path(args.action, cond, d, c))]
            h2d = {c: coco_to_h36m(np.load(det_path(args.action, cond, d, c))['coco']) for c in cams}

            for c in cams:
                p = lift_path(args.action, cond, d, c)
                if os.path.exists(p) and not args.force:
                    continue
                det = np.load(det_path(args.action, cond, d, c))
                detected = np.isfinite(det['match_px'])
                out = {'detected': detected}
                if detected.sum() >= 2:
                    x = fill_missing(h2d[c])
                    for lf in args.lifters:
                        if lf.startswith('mb_'):
                            models.setdefault('mb', MotionBert())
                            out[lf] = models['mb'](x, lf)
                        else:
                            if lf not in models:
                                models[lf] = MmposeLifter(lf)
                            out[lf] = models[lf](x)
                    if d == 'mediapipe':
                        out['mp_world'] = coco_to_h36m(det['world'][:, MP_TO_COCO])
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, **out)
                print(f'{cond}/{d}/{c}: lifted ({detected.mean():.0%} of frames detected)', flush=True)

            p = tri_path(args.action, cond, d)
            if cams != list(args.cams):          # a triangulation from fewer cameras is not comparable
                print(f'{cond}/{d}: 2D missing for {sorted(set(args.cams) - set(cams))} -- not triangulated')
                continue
            if os.path.exists(p) and not args.force:
                continue
            xy = np.stack([h2d[c][..., :2] for c in args.cams], axis=1)                  # (T,C,17,2)
            vis = np.stack([h2d[c][..., 2] >= TRI_CONF for c in args.cams], axis=1)
            X, used, err = triangulate_robust(xy, vis, [trial.cams[c].as_tuple() for c in args.cams],
                                              thresh_px=args.thresh_px)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, X=X, n_views=used.sum(1), reproj=np.nanmean(err, axis=1), cams=np.array(args.cams))
            print(f'{cond}/{d}: triangulated, {np.isfinite(X[..., 0]).mean():.0%} of frame-joints', flush=True)


if __name__ == '__main__':
    main()
