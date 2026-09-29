"""smpl_lite.py -- the parts of SMPL the child pipeline needs, without MotionBERT's model class.

Two views of the same model:

  * linear (numpy): rest-pose vertices and H36M joints as a linear function of the shape
    vector, for fitting shape to bone lengths (fit_betas) and the T-pose height step_3
    scales by (mesh_height_m);
  * MiniSMPL (torch): full linear blend skinning, evaluated only at the vertices the joint
    regressors read, so a whole trial's forward kinematics is milliseconds.  Reproduces
    step_3's kps_H36M_scaled to <0.1 mm at zero pose offsets.

Shape vectors are 10 betas, or 11 with AGORA's kid template as the last direction: weight 0 is
the adult template, 1 the infant-proportioned kid template (SMIL-like, ~0.54 m), and the 10
betas act on top.  The template file (AGORA's smpl_kid_template.npy, a licensed asset kept out
of git) is found at SMPL_KID_TEMPLATE, the repo's models/ folder, or next to SMPL_NEUTRAL.pkl
in MotionBERT's mesh folder.  Built as smplx builds it: the mean-centred kid template minus
the adult template.

The H36M regressor is the one MotionBERT's mesh checkpoint carries (utils_smpl.SMPL registers
it as a buffer, so the checkpoint's copy overrides data/mesh/J_regressor_h36m_correct.npy,
whose leg rows are swapped relative to it).
"""
import os
import pickle
import warnings

import numpy as np
import scipy.sparse as sp

from config import MB_DIR, MODELS_DIR

MB_MESH = os.path.join(MB_DIR, 'data', 'mesh')
MB_CKPT = os.path.join(MB_DIR, 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin')
# AGORA's kid template is a licensed model asset, kept out of git (*.npy is ignored) like the SMPL pickle and
# the MotionBERT weights.  Looked for at SMPL_KID_TEMPLATE, then MODELS_DIR (the repo's models/ folder, local),
# then next to SMPL_NEUTRAL.pkl in MotionBERT's mesh folder (on Drive for Colab: copy it there once).
KID_TEMPLATE_CANDIDATES = [os.environ.get('SMPL_KID_TEMPLATE'),
                           os.path.join(MODELS_DIR, 'smpl_kid_template.npy'),
                           os.path.join(MB_MESH, 'smpl_kid_template.npy')]


def kid_template_path():
    for c in KID_TEMPLATE_CANDIDATES:
        if c and os.path.isfile(c):
            return c
    raise SystemExit('smpl_kid_template.npy (AGORA kid template) not found -- looked at:\n  '
                     + '\n  '.join(c for c in KID_TEMPLATE_CANDIDATES if c)
                     + f'\nCopy it next to SMPL_NEUTRAL.pkl ({MB_MESH}) or set SMPL_KID_TEMPLATE to its path.')


KID_TEMPLATE = next((c for c in KID_TEMPLATE_CANDIDATES if c and os.path.isfile(c)), KID_TEMPLATE_CANDIDATES[1])

H36M = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
J = {n: i for i, n in enumerate(H36M)}
SMPL24 = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
          'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
          'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand']

# Segments a shape is fitted to.  Spine / Nose / Head have no OpenPose joint, so the torso is
# Hip->Thorax directly; the two widths are the most shape-informative segments available.
SEGMENTS = [('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
            ('Hip', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
            ('Hip', 'Thorax'),
            ('Thorax', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
            ('Thorax', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
            ('RHip', 'LHip'), ('RShoulder', 'LShoulder')]


def _load_pkl():
    p = os.path.join(MB_MESH, 'SMPL_NEUTRAL.pkl')
    if not os.path.isfile(p):
        raise SystemExit(f'{p} not found -- set MB_DIR to the MotionBERT folder')
    with open(p, 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pickle.load(f, encoding='latin1')


def load_h36m_regressor():
    """(17,6890) H36M regressor as step_3 uses it (the mesh checkpoint's copy)."""
    if os.path.exists(MB_CKPT):
        import torch
        sd = torch.load(MB_CKPT, map_location='cpu')['model']
        for k, v in sd.items():
            if k.endswith('head.smpl.J_regressor_h36m'):
                return v.numpy().astype(np.float64)
    warnings.warn('mesh checkpoint not found; using data/mesh/J_regressor_h36m_correct.npy (legs may be swapped)')
    return np.load(os.path.join(MB_MESH, 'J_regressor_h36m_correct.npy')).astype(np.float64)


def kid_shapedir(v_template):
    k = np.load(kid_template_path()).astype(np.float64)
    return (k - k.mean(0)) - v_template


# ----------------------------------------------------------------------------- linear model (numpy)
def load_linear(n_betas=10, kid=False):
    """v_t (6890,3), S (6890,3,n), J0 (17,3), A (17,3,n): rest-pose vertices / H36M joints linear in shape."""
    m = _load_pkl()
    v_t = np.asarray(m['v_template'], dtype=np.float64)
    S = np.asarray(m['shapedirs'], dtype=np.float64)[:, :, :n_betas]
    if kid:
        S = np.concatenate([S, kid_shapedir(v_t)[:, :, None]], axis=2)
    Jr = load_h36m_regressor()
    return dict(v_t=v_t, S=S, J0=Jr @ v_t, A=np.einsum('jv,vck->jck', Jr, S), kid=kid, n_betas=n_betas)


def mesh_height_m(lin, shape):
    """Crown-to-sole height of the T-pose mesh (y-up), as step_3 measures it."""
    v = lin['v_t'] + lin['S'] @ np.asarray(shape, dtype=np.float64)
    return float(v[:, 1].max() - v[:, 1].min())


def seg_lengths_mm(lin, shape, segs=SEGMENTS):
    j = 1000.0 * (lin['J0'] + lin['A'] @ np.asarray(shape, dtype=np.float64))
    return np.array([np.linalg.norm(j[J[a]] - j[J[b]]) for a, b in segs])


def data_seg_lengths(kps3d_mm, segs=SEGMENTS):
    """Median, IQR and count over frames of each segment's length (frames where both joints exist)."""
    med, iqr, n = [], [], []
    for a, b in segs:
        d = np.linalg.norm(kps3d_mm[:, J[a]] - kps3d_mm[:, J[b]], axis=-1)
        d = d[np.isfinite(d)]
        med.append(np.median(d) if d.size else np.nan)
        iqr.append((np.percentile(d, 75) - np.percentile(d, 25)) if d.size else np.nan)
        n.append(d.size)
    return np.array(med), np.array(iqr), np.array(n)


def fit_betas(lin, target_mm, segs=SEGMENTS, lam=1.0, sigma_mm=10.0):
    """Damped least squares: (fitted - target)/sigma per segment plus sqrt(lam)*beta as a ridge
    toward the mean shape.  With a kid model the last parameter is the kid weight, bounded [0,1],
    unregularised, started at 0.5.  Returns (shape, rms residual mm)."""
    from scipy.optimize import least_squares
    ok = np.isfinite(target_mm)
    segs_ok = [s for s, k in zip(segs, ok) if k]
    tgt = np.asarray(target_mm)[ok]
    n = lin['A'].shape[-1]
    n_reg = n - 1 if lin['kid'] else n

    def resid(x):
        return np.concatenate([(seg_lengths_mm(lin, x, segs_ok) - tgt) / sigma_mm, np.sqrt(lam) * x[:n_reg]])

    if lin['kid']:
        x0 = np.zeros(n)
        x0[-1] = 0.5
        lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
        lo[-1], hi[-1] = 0.0, 1.0
        sol = least_squares(resid, x0, bounds=(lo, hi), method='trf')
    else:
        sol = least_squares(resid, np.zeros(n), method='lm')
    rms = float(np.sqrt(np.mean((seg_lengths_mm(lin, sol.x, segs_ok) - tgt) ** 2)))
    return sol.x, rms


# ----------------------------------------------------------------------------- skinned model (torch)
class MiniSMPL:
    """SMPL linear blend skinning at the vertices J_regressor / J_regressor_h36m read.
    shape: 10 betas, or 11 with the kid weight last."""

    def __init__(self, shape, device='cpu'):
        import torch
        m = _load_pkl()
        Jr = m['J_regressor']
        Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr)
        Jh = load_h36m_regressor()
        idx = np.where((np.abs(Jr).sum(0) > 0) | (np.abs(Jh).sum(0) > 0))[0]
        v_full = np.asarray(m['v_template'], dtype=np.float64)
        shape = np.asarray(shape, dtype=np.float64)
        S = np.asarray(m['shapedirs'])[idx][:, :, :10]
        if shape.size > 10:
            S = np.concatenate([S, kid_shapedir(v_full)[idx][:, :, None]], axis=2)
        v_shaped = v_full[idx] + S @ shape[:S.shape[-1]]
        f64 = dict(dtype=torch.float64, device=device)
        self.device = device
        self.v_shaped = torch.tensor(v_shaped, **f64)
        self.posedirs = torch.tensor(np.asarray(m['posedirs'])[idx].reshape(len(idx) * 3, 207).T, **f64)
        self.W = torch.tensor(np.asarray(m['weights'])[idx], **f64)
        self.Jr = torch.tensor(Jr[:, idx], **f64)
        self.Jh = torch.tensor(Jh[:, idx], **f64)
        self.parents = np.asarray(m['kintree_table'])[0].astype(np.int64)
        self.parents[0] = -1
        self.J_rest = self.Jr @ self.v_shaped
        self.n = len(idx)

    def forward(self, rotmats):
        """rotmats (T,24,3,3) -> SMPL-24 joints (T,24,3), H36M joints (T,17,3); metres, SMPL's frame."""
        import torch
        T = rotmats.shape[0]
        eye = torch.eye(3, dtype=rotmats.dtype, device=rotmats.device)
        v_posed = self.v_shaped + ((rotmats[:, 1:] - eye).reshape(T, 207) @ self.posedirs).reshape(T, self.n, 3)
        G = [None] * 24
        rel = self.J_rest.clone()
        rel[1:] = self.J_rest[1:] - self.J_rest[self.parents[1:]]
        for k in range(24):
            Tk = torch.zeros(T, 4, 4, dtype=rotmats.dtype, device=rotmats.device)
            Tk[:, :3, :3] = rotmats[:, k]
            Tk[:, :3, 3] = rel[k]
            Tk[:, 3, 3] = 1.0
            G[k] = Tk if k == 0 else G[self.parents[k]] @ Tk
        G = torch.stack(G, 1)
        Jh4 = torch.cat([self.J_rest, torch.zeros(24, 1, dtype=rotmats.dtype, device=rotmats.device)], 1)
        A = G.clone()
        A[:, :, :3, 3] = G[:, :, :3, 3] - (G[:, :, :3, :] @ Jh4[None, :, :, None])[..., 0]
        Tv = torch.einsum('nk,tkij->tnij', self.W, A)
        v = (Tv[:, :, :3, :3] @ v_posed[..., None])[..., 0] + Tv[:, :, :3, 3]
        return torch.einsum('jn,tnc->tjc', self.Jr, v), torch.einsum('jn,tnc->tjc', self.Jh, v)


def rodrigues(aa):
    """(...,3) axis-angle -> (...,3,3), differentiable at zero."""
    import torch
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-12)
    k = aa / theta
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def project(X_cam, K, dist):
    """OpenCV pinhole + radial/tangential distortion, torch.  X_cam (...,3) m -> (...,2) px."""
    import torch
    x, y = X_cam[..., 0] / X_cam[..., 2], X_cam[..., 1] / X_cam[..., 2]
    k1, k2, p1, p2, k3 = [float(d) for d in np.asarray(dist).ravel()[:5]]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], -1)


def kabsch(P, Q):
    """Rigid R, t with Q = R P + t over rows; P, Q (n,3)."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - R @ mp
