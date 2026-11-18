"""korea_data.py -- build_dataset.py's sequences as MotionBERT training clips.

Each item is one clip of --clip-len frames cut from a (rep, camera) sequence every --stride frames (the last clip
ends on the last frame).  Input: the step_2a-prepared 2D (clip_len, 17, 3).  Target dict: the pose (axis-angle,
root in this camera), the child's shape and keypoint offsets, and the masks the loss uses -- frame_ok, obs_swing /
obs_full (global segment orientation observed) and kp_mask (BODY_25 keypoints the pseudo-GT fit had data for).

Sequences whose fit projects badly into their camera (> --max-proj-px, a slightly wrong camera) are left out, and so
are clips with too few usable frames.  flip=True mirrors half the clips as MotionBERT's own training does: the 2D
(flip_data), the pose (flip_thetas), the left/right masks, and the lateral part of the central offsets.
"""
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kid_body import BODY25_PAIRS, OFFSET_GROUPS, SMPL24_PAIRS, flip_perm  # noqa: E402

P24 = flip_perm(24, SMPL24_PAIRS)
P25 = flip_perm(25, BODY25_PAIRS)
H36M_LR = ([4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16])
THETA_PAIRS = SMPL24_PAIRS


def flip_2d(x):
    """MotionBERT's flip_data for one clip (T,17,3), x in crop-normalised [-1, 1]."""
    y = x.copy()
    y[..., 0] *= -1
    y[:, H36M_LR[0] + H36M_LR[1]] = y[:, H36M_LR[1] + H36M_LR[0]]
    return y


def flip_pose(p):
    """MotionBERT's flip_thetas: (T,24,3) axis-angle mirrored in the camera's x."""
    q = p.copy()
    q[..., 1] *= -1
    q[..., 2] *= -1
    return q[:, P24]


class KoreaClips(Dataset):
    def __init__(self, npz, clip_len=16, stride=8, flip=False, max_proj_px=15.0, min_ok_frac=0.5, seqs=None):
        d = np.load(npz)
        self.clip_len, self.flip = clip_len, flip
        self.x2d = d['x2d']
        self.pose = d['pose']
        self.obs_swing, self.obs_full = d['obs_swing'], d['obs_full']
        self.ok = d['frame_ok']
        self.kp_mask = np.isfinite(d['x3d_body25']).all(-1)
        self.loo3d, self.loo_ok = d['loo3d'], d['loo_ok']
        self.seq_start, self.seq_len = d['seq_start'], d['seq_len']
        self.seq_rep, self.seq_cam, self.seq_subject = d['seq_rep'], d['seq_cam'], d['seq_subject']
        self.shape = d['seq_shape'].astype(np.float32)
        self.offsets = (d['seq_offsets_rest_mm'] / 1000.0).astype(np.float32)      # groups as OFFSET_GROUPS
        keep = ~(d['seq_proj_px'] > max_proj_px)                                     # NaN (no check) kept
        if seqs is not None:
            keep &= np.isin(np.arange(len(keep)), seqs)
        self.seqs = np.where(keep)[0]
        self.n_dropped_proj = int((d['seq_proj_px'] > max_proj_px).sum())
        self.clips = []
        for s in self.seqs:
            a, L = int(self.seq_start[s]), int(self.seq_len[s])
            starts = list(range(0, max(L - clip_len, 0) + 1, stride))
            if L > clip_len and starts[-1] != L - clip_len:
                starts.append(L - clip_len)
            for o in starts:
                if self.ok[a + o:a + o + clip_len].mean() >= min_ok_frac:
                    self.clips.append((s, a + o))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        s, a = self.clips[i]
        idx = np.arange(a, a + self.clip_len)
        L = int(self.seq_start[s] + self.seq_len[s])
        idx = np.minimum(idx, L - 1)                       # sequences shorter than a clip: hold the last frame
        x = self.x2d[idx]
        pose = self.pose[idx]
        sw, fu, km = self.obs_swing[idx], self.obs_full[idx], self.kp_mask[idx]
        off = self.offsets[s].copy()
        if self.flip and random.random() < 0.5:
            x, pose = flip_2d(x), flip_pose(pose)
            sw, fu, km = sw[:, P24], fu[:, P24], km[:, P25]
            for g in ('neck', 'midhip'):                   # central offsets: their sideways part mirrors
                off[OFFSET_GROUPS.index(g), 0] *= -1
        gt = dict(pose=torch.from_numpy(pose.astype(np.float32)), shape=torch.from_numpy(self.shape[s]),
                  offsets=torch.from_numpy(off), ok=torch.from_numpy(self.ok[idx]),
                  obs_swing=torch.from_numpy(sw), obs_full=torch.from_numpy(fu), kp_mask=torch.from_numpy(km),
                  loo3d=torch.from_numpy(np.nan_to_num(self.loo3d[idx])), loo_ok=torch.from_numpy(self.loo_ok[idx]),
                  seq=torch.tensor(int(s)))
        return torch.from_numpy(x.astype(np.float32)), gt
