"""kid_body.py -- the pseudo-GT's keypoint model as a batched torch module, for the fine-tuning loss.

pseudo_gt.py fits poses with smpl_op_fit.SMPLFit: SMPL (+ AGORA kid template as an 11th shape direction) skinned at
the vertices the keypoints need, BODY_25 keypoints = SMPL-24 joints (knees, ankles, elbows, wrists), surface vertices
(face, toes, heels) and SMPL-24 joints + per-child offsets carried by a bone (hips, shoulders, neck, mid-hip).
KidBody is that model with a different shape and offsets per sample, so a batch of clips from many children poses
each child's own body.  It also returns the global (camera-frame) rotation of every segment and each observed
segment's rest axis (pseudo_gt.OBSERVE): what the orientation loss compares.

    body = KidBody()
    kp, G, axes = body(R, shape, offsets)     # R (M,24,3,3) local rotations, shape (M,11), offsets (M,4,3) metres
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'analysis', 'joint_calib'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
import smpl_op_fit as F  # noqa: E402
from utils.openpose import BODY25_NAMES  # noqa: E402
from utils.smpl_lite import kid_shapedir  # noqa: E402

B25 = {n: i for i, n in enumerate(BODY25_NAMES)}
S24 = F.S24
OFFSET_GROUPS = ('hip', 'shoulder', 'neck', 'midhip')      # order of the offsets array (pseudo_gt's fit_v3 groups)
# each observed segment's rest axis: towards this child joint, or (Pelvis) right hip -> left hip, (Head) -> the nose
AXIS = {'Pelvis': ('R_Hip', 'L_Hip'), 'Head': ('Head', 'nose'),
        **{f'{s}_{a}': (f'{s}_{a}', f'{s}_{b}') for s in 'LR' for a, b in
           (('Hip', 'Knee'), ('Knee', 'Ankle'), ('Ankle', 'Foot'), ('Collar', 'Shoulder'), ('Shoulder', 'Elbow'),
            ('Elbow', 'Wrist'))}}
# left/right partners, for flipping masks with the image
SMPL24_PAIRS = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
BODY25_PAIRS = tuple((B25[f'R{n}'], B25[f'L{n}']) for n in ('Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle',
                                                          'Eye', 'Ear', 'BigToe', 'SmallToe', 'Heel'))


def flip_perm(n, pairs):
    p = np.arange(n)
    for a, b in pairs:
        p[a], p[b] = b, a
    return p


class KidBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = F._load_pkl()
        Jr = m['J_regressor']
        Jr = Jr.toarray() if hasattr(Jr, 'toarray') else np.asarray(Jr)
        surf = list(F.SURFACE.values())
        idx = np.unique(np.r_[np.where(np.abs(Jr).sum(0) > 0)[0], surf])
        v_t = np.asarray(m['v_template'], np.float64)
        S = np.concatenate([np.asarray(m['shapedirs'], np.float64)[:, :, :10], kid_shapedir(v_t)[:, :, None]], 2)
        f32 = dict(dtype=torch.float32)
        self.register_buffer('v_t', torch.tensor(v_t[idx], **f32))
        self.register_buffer('S', torch.tensor(S[idx], **f32))
        self.register_buffer('posedirs', torch.tensor(np.asarray(m['posedirs'])[idx].reshape(len(idx) * 3, 207).T, **f32))
        self.register_buffer('W', torch.tensor(np.asarray(m['weights'])[idx], **f32))
        self.register_buffer('Jr', torch.tensor(Jr[:, idx], **f32))
        self.parents = np.asarray(m['kintree_table'])[0].astype(np.int64)
        self.surf_idx = {n: int(np.searchsorted(idx, v)) for n, v in F.SURFACE.items()}
        self.n = len(idx)
        self.free = {n: (S24[base], S24[bone], OFFSET_GROUPS.index(g), side) for n, (base, bone, g, side) in F.FREE.items()}
        self.axis_joints = [S24[j] for j in AXIS]

    def global_rotations(self, R):
        """R (M,24,3,3) local -> (M,24,3,3) global (the root's frame = the camera's for MotionBERT's poses)."""
        G = [R[:, 0]]
        for k in range(1, 24):
            G.append(G[self.parents[k]] @ R[:, k])
        return torch.stack(G, 1)

    def forward(self, R, shape, offsets):
        """-> keypoints (M,25,3) BODY_25 order (root at the origin's SMPL frame), G (M,24,3,3), axes (M,24,3)."""
        M = R.shape[0]
        v_sh = self.v_t[None] + torch.einsum('nci,mi->mnc', self.S, shape)              # (M,n,3)
        J_rest = torch.einsum('jn,mnc->mjc', self.Jr, v_sh)                               # (M,24,3)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        v_posed = v_sh + ((R[:, 1:] - eye).reshape(M, 207) @ self.posedirs).reshape(M, self.n, 3)
        rel = torch.cat([J_rest[:, :1], J_rest[:, 1:] - J_rest[:, self.parents[1:]]], 1)
        Gt = [None] * 24
        for k in range(24):
            Tk = torch.zeros(M, 4, 4, dtype=R.dtype, device=R.device)
            Tk[:, :3, :3], Tk[:, :3, 3], Tk[:, 3, 3] = R[:, k], rel[:, k], 1.0
            Gt[k] = Tk if k == 0 else Gt[self.parents[k]] @ Tk
        Gt = torch.stack(Gt, 1)                                                           # (M,24,4,4)
        A = Gt.clone()
        A[:, :, :3, 3] = Gt[:, :, :3, 3] - (Gt[:, :, :3, :3] @ J_rest[..., None])[..., 0]
        Tv = torch.einsum('nk,mkij->mnij', self.W, A)
        v = (Tv[:, :, :3, :3] @ v_posed[..., None])[..., 0] + Tv[:, :, :3, 3]
        J24 = torch.einsum('jn,mnc->mjc', self.Jr, v)
        kp = torch.zeros(M, 25, 3, dtype=R.dtype, device=R.device)
        for n, s in F.ANCHOR_JOINTS.items():
            kp[:, B25[n]] = J24[:, S24[s]]
        for n, i in self.surf_idx.items():
            kp[:, B25[n]] = v[:, i]
        mirror = torch.tensor([-1.0, 1.0, 1.0], dtype=R.dtype, device=R.device)
        for n, (base, bone, g, side) in self.free.items():
            d = offsets[:, g] * (mirror if side == 'R' else 1.0)
            kp[:, B25[n]] = J24[:, base] + (Gt[:, bone, :3, :3] @ d[..., None])[..., 0]
        # rest axes of the observed segments
        nose = v_sh[:, self.surf_idx['Nose']]
        axes = torch.zeros(M, 24, 3, dtype=R.dtype, device=R.device)
        for j, (a, b) in AXIS.items():
            pa = J_rest[:, S24[a]]
            pb = nose if b == 'nose' else J_rest[:, S24[b]]
            axes[:, S24[j]] = torch.nn.functional.normalize(pb - pa, dim=-1)
        return kp, Gt[:, :, :3, :3], axes
