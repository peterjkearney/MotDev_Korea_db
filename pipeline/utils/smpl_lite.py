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

# The hip and shoulder widths measure the joint DEFINITION as much as the body: on BioCV P03 the triangulated
# OpenPose hips are 49 mm wider than mocap's hip joint centres, shoulders 58 mm, and SMPL's regressed H36M hips
# wider than both (294 mm on the mean adult).  They are also the only terms that constrain girth, so fitting them
# sets the torso's girth from the definition gap.  The width-free set leaves girth to the shape prior.
WIDTH_SEGMENTS = [('Hip', 'RHip'), ('Hip', 'LHip'), ('RHip', 'LHip'),
                  ('Thorax', 'LShoulder'), ('Thorax', 'RShoulder'), ('RShoulder', 'LShoulder')]
SEGMENTS_NO_WIDTH = [s for s in SEGMENTS if s not in WIDTH_SEGMENTS]


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


# H36M-17 joints from SMPL's own 24-joint regressor (the SMPLify-X convention for OpenPose's body joints):
# limb joints at SMPL's joint centres, Hip / Thorax / Spine built as the targets build them (midpoints),
# Nose / Head kept from the H36M regressor (SMPL-24 has no counterpart; both are out of the 2D term and the
# scoring).  The H36M regressor's hips sit 80 mm lateral and 123 mm above SMPL's on the mean adult (40 / 53 mm
# at kid weight 0.75) -- up on the pelvis, so its thigh shortens ~30% as the hip flexes to 120 deg, where
# SMPL-24's shortens 2-4%.
SMPL24_FOR_H36M = {'RHip': 'R_Hip', 'RKnee': 'R_Knee', 'RAnkle': 'R_Ankle', 'LHip': 'L_Hip', 'LKnee': 'L_Knee',
                   'LAnkle': 'L_Ankle', 'LShoulder': 'L_Shoulder', 'LElbow': 'L_Elbow', 'LWrist': 'L_Wrist',
                   'RShoulder': 'R_Shoulder', 'RElbow': 'R_Elbow', 'RWrist': 'R_Wrist'}
JOINTS = ('h36m', 'smpl24')


def joint_regressor(joints='h36m'):
    """(17,6890) vertex regressor for the H36M-17 layout: 'h36m' the mesh checkpoint's H36M regressor,
    'smpl24' the layout rebuilt from SMPL's J_regressor (SMPL24_FOR_H36M)."""
    Jh = load_h36m_regressor()
    if joints == 'h36m':
        return Jh
    if joints != 'smpl24':
        raise ValueError(f'unknown joints {joints!r} (one of {JOINTS})')
    Jr = _load_pkl()['J_regressor']
    Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr, dtype=np.float64)
    out = Jh.copy()                                                  # Nose, Head stay H36M's
    for h, s in SMPL24_FOR_H36M.items():
        out[J[h]] = Jr[SMPL24.index(s)]
    out[J['Hip']] = 0.5 * (out[J['RHip']] + out[J['LHip']])
    out[J['Thorax']] = 0.5 * (out[J['RShoulder']] + out[J['LShoulder']])
    out[J['Spine']] = 0.5 * (out[J['Hip']] + out[J['Thorax']])
    return out


def kid_shapedir(v_template):
    k = np.load(kid_template_path()).astype(np.float64)
    return (k - k.mean(0)) - v_template


# ----------------------------------------------------------------------------- linear model (numpy)
def load_linear(n_betas=10, kid=False, joints='h36m'):
    """v_t (6890,3), S (6890,3,n), J0 (17,3), A (17,3,n): rest-pose vertices / H36M-17 joints linear in shape
    (joints: see joint_regressor)."""
    m = _load_pkl()
    v_t = np.asarray(m['v_template'], dtype=np.float64)
    S = np.asarray(m['shapedirs'], dtype=np.float64)[:, :, :n_betas]
    if kid:
        S = np.concatenate([S, kid_shapedir(v_t)[:, :, None]], axis=2)
    Jr = joint_regressor(joints)
    return dict(v_t=v_t, S=S, J0=Jr @ v_t, A=np.einsum('jv,vck->jck', Jr, S), kid=kid, n_betas=n_betas, joints=joints)


def midpoint_centres(lin):
    """Copy of a linear model whose Hip and Thorax are the midpoints of its hips / shoulders, as the
    OpenPose and mocap targets build them.  The regressor's own Thorax sits up at the neck base (36 mm above
    the shoulder midpoint on the mean adult, 15 mm at kid weight 0.75), so Hip->Thorax would compare
    different points.  For fitting only: the skinned model (MiniSMPL) keeps the regressor's joints."""
    lin = dict(lin, J0=lin['J0'].copy(), A=lin['A'].copy())
    for c, (a, b) in (('Hip', ('RHip', 'LHip')), ('Thorax', ('RShoulder', 'LShoulder'))):
        lin['J0'][J[c]] = 0.5 * (lin['J0'][J[a]] + lin['J0'][J[b]])
        lin['A'][J[c]] = 0.5 * (lin['A'][J[a]] + lin['A'][J[b]])
    return lin


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


# ----------------------------------------------------------------------------- girth
# Trunk girths at fixed fractions of the way from the Hip to the Thorax joint (rest pose, y up).
GIRTH_LEVELS = {'hip': 0.0, 'waist': 0.5, 'chest': 0.8}


def girth_rings(lin, shape_ref, slab_m=0.008):
    """{name: vertex indices in ring order} for each GIRTH_LEVELS slice of the reference body: the convex
    outline (x-z) of the vertices within slab_m of the level, trunk only (T-pose arms and the thighs' far
    sides cut by a half-width from the shoulders).  Chosen once, so girths_mm is smooth in the shape."""
    from scipy.spatial import ConvexHull
    v = lin['v_t'] + lin['S'] @ np.asarray(shape_ref, dtype=np.float64)
    j = lin['J0'] + lin['A'] @ np.asarray(shape_ref, dtype=np.float64)
    yh, yt, xc = j[J['Hip'], 1], j[J['Thorax'], 1], j[J['Hip'], 0]
    half = 0.8 * abs(j[J['RShoulder'], 0] - j[J['LShoulder'], 0])
    rings = {}
    for name, f in GIRTH_LEVELS.items():
        idx = np.where((np.abs(v[:, 1] - (yh + f * (yt - yh))) < slab_m) & (np.abs(v[:, 0] - xc) < half))[0]
        rings[name] = idx[ConvexHull(v[idx][:, [0, 2]]).vertices]
    return rings


def girths_mm(lin, shape, rings):
    """{name: perimeter (mm) of that ring in the x-z plane} for this shape."""
    shape = np.asarray(shape, dtype=np.float64)
    out = {}
    for name, idx in rings.items():
        p = (lin['v_t'][idx] + lin['S'][idx] @ shape)[:, [0, 2]]
        out[name] = 1000.0 * float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())
    return out


def fit_betas(lin, target_mm, segs=SEGMENTS, lam=1.0, sigma_mm=10.0, lam_rest=None, girth=None, sigma_girth_mm=10.0):
    """Damped least squares: (fitted - target)/sigma per segment plus sqrt(lam)*beta as a ridge
    toward the mean shape (lam_rest, when given, replaces lam for betas 1-9: beta 0 is mostly size,
    the rest proportions and girth).  With a kid model the last parameter is the kid weight, bounded
    [0,1], unregularised, started at 0.5.

    girth (girth_rings output): a girth prior -- each ring's perimeter is held to that of the same body
    with betas 1-9 at zero (size and kid weight still scale girth), (difference)/sigma_girth_mm per ring.
    Bone lengths carry no girth information, so this keeps girth at the mean for the size while betas
    1-9 stay free (under a light lam_rest) to fit proportions.  Returns (shape, rms residual mm)."""
    from scipy.optimize import least_squares
    ok = np.isfinite(target_mm)
    segs_ok = [s for s, k in zip(segs, ok) if k]
    tgt = np.asarray(target_mm)[ok]
    n = lin['A'].shape[-1]
    n_reg = n - 1 if lin['kid'] else n
    w = np.full(n_reg, np.sqrt(lam))
    if lam_rest is not None:
        w[1:] = np.sqrt(lam_rest)

    def girth_resid(x):
        x0 = x.copy()
        x0[1:n_reg] = 0.0
        g, g0 = girths_mm(lin, x, girth), girths_mm(lin, x0, girth)
        return np.array([(g[k] - g0[k]) / sigma_girth_mm for k in girth])

    def resid(x):
        r = [(seg_lengths_mm(lin, x, segs_ok) - tgt) / sigma_mm, w * x[:n_reg]]
        if girth is not None:
            r.append(girth_resid(x))
        return np.concatenate(r)

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
    shape: 10 betas, or 11 with the kid weight last.  full=True skins all 6890 vertices
    (for rendering: vertices() and faces).  joints: which regressor the H36M-17 output uses (joint_regressor)."""

    def __init__(self, shape, device='cpu', full=False, joints='h36m'):
        import torch
        m = _load_pkl()
        Jr = m['J_regressor']
        Jr = Jr.toarray() if sp.issparse(Jr) else np.asarray(Jr)
        Jh = joint_regressor(joints)
        idx = (np.arange(Jr.shape[1]) if full else
               np.where((np.abs(Jr).sum(0) > 0) | (np.abs(Jh).sum(0) > 0))[0])
        self.faces = np.asarray(m['f'], dtype=np.int64) if full else None
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

    def forward(self, rotmats, moments=None):
        """rotmats (T,24,3,3) -> SMPL-24 joints (T,24,3), H36M joints (T,17,3); metres, SMPL's frame.
        moments (P,24,4) (see body_points) adds a third output: those P skinned points (T,P,3)."""
        import torch
        A = self.bone_transforms(rotmats)
        v = self.vertices(rotmats, A)
        J24, Jh = torch.einsum('jn,tnc->tjc', self.Jr, v), torch.einsum('jn,tnc->tjc', self.Jh, v)
        if moments is None:
            return J24, Jh
        return J24, Jh, torch.einsum('pkj,tkij->tpi', moments, A[:, :, :3, :])

    def vertices(self, rotmats, A=None):
        """rotmats (T,24,3,3) -> skinned vertices (T,n,3), metres, SMPL's frame (all 6890 when full)."""
        import torch
        T = rotmats.shape[0]
        eye = torch.eye(3, dtype=rotmats.dtype, device=rotmats.device)
        v_posed = self.v_shaped + ((rotmats[:, 1:] - eye).reshape(T, 207) @ self.posedirs).reshape(T, self.n, 3)
        A = self.bone_transforms(rotmats) if A is None else A
        Tv = torch.einsum('nk,tkij->tnij', self.W, A)
        return (Tv[:, :, :3, :3] @ v_posed[..., None])[..., 0] + Tv[:, :, :3, 3]

    def bone_transforms(self, rotmats):
        """rotmats (T,24,3,3) -> per-bone skinning transforms A (T,24,4,4): rest-pose point -> posed."""
        import torch
        T = rotmats.shape[0]
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
        return A


# OpenPose BODY_25 foot keypoints as SMPL vertices (smplx's vertex_ids, SMPL/SMPL-H topology)
OPENPOSE_FOOT_VERTICES = {'LBigToe': 3216, 'LSmallToe': 3226, 'LHeel': 3387,
                          'RBigToe': 6617, 'RSmallToe': 6624, 'RHeel': 6787}


def ankle_height_m(shape, joints='h36m', ref='sole'):
    """Rest-pose height of the H36M-17 ankle joints, metres, mean of left and right: where a grounded ankle
    sits above the floor for this body.  ref 'sole': above the lowest vertex (a barefoot floor).  ref
    'foot_kp': above the mean height of the mesh's OpenPose toe/heel keypoint vertices -- the floor as
    Korea's build_child_gt fits it (a plane through triangulated BODY_25 toes and heels, which sit ~10 mm
    above the sole on a child), and unchanged by shoes, which raise the keypoints with the ankle."""
    import torch
    smpl = MiniSMPL(shape, full=True, joints=joints)
    with torch.no_grad():
        v = smpl.vertices(torch.eye(3, dtype=torch.float64).expand(1, 24, 3, 3))[0]
        Jh = smpl.Jh @ v
    y = v[:, 1]                                                  # SMPL rest frame is y-up
    if ref == 'sole':
        base = y.min()
    elif ref == 'foot_kp':
        base = y[list(OPENPOSE_FOOT_VERTICES.values())].mean()
    else:
        raise ValueError(f'unknown ref {ref!r}')
    return float(np.mean([Jh[H36M.index(j), 1] - base for j in ('RAnkle', 'LAnkle')]))


SOLE = ['RHeel', 'RToe', 'RMedial', 'RLateral', 'LHeel', 'LToe', 'LMedial', 'LLateral']


def body_points(shape, n_cells=150, sole_mm=15.0):
    """Skinning moments (9,24,4) for MiniSMPL.forward(..., moments=): row 0 the whole-body centre of
    mass, rows 1-8 the sole extremes named in SOLE (heel, toe, medial, lateral of each foot).

    COM: the closed rest-pose mesh is filled with a grid (n_cells across its largest extent; rays
    along the depth axis, inside = between alternate surface crossings), each interior point gets
    the inverse-distance skin weights of its 8 nearest vertices, and the moments are the mean of
    w_k [x; 1] over the points -- so the posed COM is exactly the mean of the LBS-skinned interior,
    uniform density, mass conserved however the limbs overlap.  Pose blend shapes are ignored.
    Sole: vertices within sole_mm of the foot's lowest point (rest frame y-up, z forward, +x = left)."""
    from scipy.spatial import cKDTree
    m = _load_pkl()
    f = np.asarray(m['f'], dtype=np.int64)
    W = np.asarray(m['weights'], dtype=np.float64)
    v_full = np.asarray(m['v_template'], dtype=np.float64)
    shape = np.asarray(shape, dtype=np.float64)
    S = np.asarray(m['shapedirs'])[:, :, :10]
    if shape.size > 10:
        S = np.concatenate([S, kid_shapedir(v_full)[:, :, None]], axis=2)
    v = v_full + S @ shape[:S.shape[-1]]

    lo, hi = v.min(0), v.max(0)
    h = (hi - lo).max() / n_cells
    gx = np.arange(lo[0] + h / 2, hi[0], h)
    gy = np.arange(lo[1] + h / 2, hi[1], h)
    hits = {}
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    for i in range(len(f)):                                     # cast +z rays through every column a triangle covers
        P = np.stack([a[i], b[i], c[i]])
        ix = np.where((gx >= P[:, 0].min()) & (gx <= P[:, 0].max()))[0]
        iy = np.where((gy >= P[:, 1].min()) & (gy <= P[:, 1].max()))[0]
        if not len(ix) or not len(iy):
            continue
        X, Y = np.meshgrid(gx[ix], gy[iy], indexing='ij')
        d = (P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1])
        if abs(d) < 1e-15:
            continue
        l1 = ((X - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (Y - P[0, 1])) / d
        l2 = ((P[1, 0] - P[0, 0]) * (Y - P[0, 1]) - (X - P[0, 0]) * (P[1, 1] - P[0, 1])) / d
        inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        z = P[0, 2] + l1 * (P[1, 2] - P[0, 2]) + l2 * (P[2, 2] - P[0, 2])
        for (jx, jy), zz in zip(np.argwhere(inside), z[inside]):
            hits.setdefault((ix[jx], iy[jy]), []).append(zz)
    pts = []
    for (jx, jy), zs in hits.items():
        zs = np.sort(zs)
        if len(zs) % 2:                                          # grazed an edge: column ambiguous, skip
            continue
        for z0, z1 in zip(zs[0::2], zs[1::2]):
            zc = np.arange(np.ceil((z0 - lo[2]) / h - 0.5) + 0.5, (z1 - lo[2]) / h, 1.0) * h + lo[2]
            pts.extend((gx[jx], gy[jy], zz) for zz in zc if z0 <= zz <= z1)
    pts = np.asarray(pts)
    dist, nn = cKDTree(v).query(pts, k=8)
    iw = 1.0 / np.maximum(dist, 1e-6)
    w_pts = (W[nn] * iw[..., None]).sum(1) / iw.sum(1, keepdims=True)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    rows = [np.einsum('pk,pj->kj', w_pts, ph) / len(pts)]

    part = W.argmax(1)
    for side, (ja, jf), sgn in (('R', (SMPL24.index('R_Ankle'), SMPL24.index('R_Foot')), -1.0),
                                ('L', (SMPL24.index('L_Ankle'), SMPL24.index('L_Foot')), 1.0)):
        foot = np.where(np.isin(part, (ja, jf)))[0]
        sole = foot[v[foot, 1] < v[foot, 1].min() + sole_mm / 1000.0 * (hi[1] - lo[1]) / 1.7]
        pick = [sole[v[sole, 2].argmin()], sole[v[sole, 2].argmax()],
                sole[(-sgn * v[sole, 0]).argmax()], sole[(sgn * v[sole, 0]).argmax()]]   # heel toe medial lateral
        rows += [W[k][:, None] * np.r_[v[k], 1.0][None, :] for k in pick]
    return np.stack(rows)


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
