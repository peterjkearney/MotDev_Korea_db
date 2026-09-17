#!/usr/bin/env python3
"""
step_0_load_mocap_gt.py — convert a BioCV trial's markers.c3d into H36M-17-ordered
ground-truth joint positions, for comparing against MotionBERT-Mesh output
(extract_3d_keypoints.py's kps3d).

Marker -> H36M mapping (decided against markers.c3d's Visual3D-computed
joint centres, not raw skin landmarks):

    Hip (root) <- HIP_MIDPOINT      Spine    <- T10 + geometric offset (see below)
    RHip       <- RIGHT_HIP         Thorax   <- midpoint(LShoulder, RShoulder) (see below)
    RKnee      <- RIGHT_KNEE        Nose     <- (unavailable, no head/face markers)
    RAnkle     <- RAJC              Head     <- (unavailable, no head/face markers)
    LHip       <- LEFT_HIP          LShoulder<- ACROM_L + geometric offset (see below)
    LKnee      <- LEFT_KNEE         LElbow   <- LEFT_ELBOW
    LAnkle     <- LAJC              LWrist   <- LEFT_WRIST
                                     RShoulder<- ACROM_R + geometric offset (see below)
                                     RElbow   <- RIGHT_ELBOW
                                     RWrist   <- RIGHT_WRIST

RAJC/LAJC (Ankle Joint Centre) chosen over RIGHT_ANKLE/LEFT_ANKLE as the
more refined estimate. RKnee/LKnee (RIGHT_KNEE/LEFT_KNEE) verified directly
against MotionBERT-style keypoints: both conventions place the knee
internally (joint-centre style), not on a skin marker, so no correction is
needed there. Nose and Head have no marker-set coverage at all (this rig
tracks limbs/trunk, not the head) — left as NaN and flagged in
valid_joint_mask, not silently zero-filled, so aggregate error metrics can
exclude them explicitly.

Shoulder joint centre — NOT markers.c3d's own LEFT_SHO/RIGHT_SHO. Verified
directly against ACROM_L/ACROM_R: LEFT_ELBOW/LEFT_WRIST etc. are exact
midpoints of their lateral/medial marker pairs (genuine joint-centre
estimates), but LEFT_SHO/RIGHT_SHO turned out to be ACROM offset by a flat,
constant 25.00mm every single frame — not derived from this subject's
anatomy at all (almost certainly just a marker-radius/standoff allowance),
and empirically not close to where a vision keypoint detector's "shoulder"
lands (see the YOLO-vs-mocap diagnostic video, camera 07).

Replaced with a simple geometric approximation instead: offset medially
(toward the body midline, along the ACROM_L-ACROM_R axis) and inferiorly
(down the trunk, along the C7-to-T10 axis -- not C7-to-HIP_MIDPOINT: that
was the first version, but HIP_MIDPOINT is occluded for exactly the static
calibration window this whole exercise cares about, per the earlier
occlusion investigation, which silently zeroed out every shoulder bone's
availability there. T10 is available throughout the static window and
still gives a reasonable trunk-down direction) from the raw ACROM marker,
each offset scaled as a fraction of the subject's own biacromial (shoulder)
width. THE FRACTION BELOW (_SHOULDER_OFFSET_FRAC) IS A ROUGH, UN-CITED
APPROXIMATION, not a published regression equation -- the general
anatomical fact (true GH joint centre sits medial+inferior to the skin-
surface acromion, roughly proportional to shoulder size) is well
established, but this specific magnitude is a starting guess, adjustable,
not verified against literature or this population.

Thorax joint centre -- NOT markers.c3d's own SHO_MIDPOINT (that's the raw
average of the two skin-surface ACROM markers). Redefined as the midpoint
of the two CORRECTED LShoulder/RShoulder points above, so Thorax stays
geometrically consistent with the same shoulder joint-centre estimate used
everywhere else, rather than mixing a skin-marker-based Thorax with a
joint-centre-based shoulder.

Spine joint centre -- NOT markers.c3d's own T10 directly. T10 is a
posterior skin marker (over the spinous process); MotionBERT/H36M's
"Spine" keypoint is an internal skeletal concept sitting closer to the
vertebral column than to the skin surface, i.e. offset slightly ANTERIOR
from T10, toward the front of the torso. Approximated here as T10 moved a
fraction of the way toward XIP_PROC (the xiphoid process marker, anterior
chest, roughly the same vertebral level) -- this fraction should be small
relative to the shoulder one, since a skin marker directly over the spine
is already close to the true vertebral location, unlike ACROM which sits
well lateral of the true GH joint. THE FRACTION BELOW (_SPINE_OFFSET_FRAC)
IS A ROUGH, UN-CITED APPROXIMATION, same caveat as the shoulder offset.

Per-marker occlusion (C3D residual < 0) is propagated as NaN rather than
trusting whatever stale/garbage position a dropped-tracking frame holds.

Output units are native mocap millimetres — NOT converted to the "metres"
label MotionBERT's kps3d uses, since that label was already established to
be non-metric (see extract_features.py's NOTE ON UNITS). Reconciling scale
between this ground truth and MotionBERT's camera-space output is left to
the evaluation step (Procrustes/p_mpjpe), not baked in here.

Run on host (NOT the motor-dev container — this only needs numpy + ezc3d,
no torch/YOLO):
    export LD_LIBRARY_PATH=/ssd/miniconda3/lib/python3.13/site-packages/ezc3d:$LD_LIBRARY_PATH
    python3 step_0_load_mocap.py --user User28 --action P28_CMJM_01
"""

import os
import argparse

import numpy as np
import ezc3d

from utils.frame_decimation import nearest_frame_indices

from config import TRIAL_DIR as _TRIAL_DIR, mocap_path, require_out_dir

# Same order as extract_3d_keypoints.py's H36M_JOINT_NAMES — keep these two
# files' joint index convention identical so downstream code never needs to
# re-map between them.
H36M_JOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Nose', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist',
]

# H36M joint name -> markers.c3d marker name, or None where no marker exists,
# or 'COMPUTED' for the shoulder/thorax/spine geometric approximations (see
# below and the module docstring).
_JOINT_TO_MARKER = {
    'Hip':       'HIP_MIDPOINT',
    'RHip':      'RIGHT_HIP',
    'RKnee':     'RIGHT_KNEE',
    'RAnkle':    'RAJC',
    'LHip':      'LEFT_HIP',
    'LKnee':     'LEFT_KNEE',
    'LAnkle':    'LAJC',
    'Spine':     'COMPUTED',
    'Thorax':    'COMPUTED',
    'Nose':      None,
    'Head':      None,
    'LShoulder': 'COMPUTED',
    'LElbow':    'LEFT_ELBOW',
    'LWrist':    'LEFT_WRIST',
    'RShoulder': 'COMPUTED',
    'RElbow':    'RIGHT_ELBOW',
    'RWrist':    'RIGHT_WRIST',
}

# fraction of biacromial (shoulder) width used for each shoulder offset
# direction -- rough approximation, not a cited regression, see module
# docstring
_SHOULDER_OFFSET_FRAC = 0.10

# fraction of the T10-to-XIP_PROC (posterior-to-anterior torso) distance
# used to move T10 toward the true vertebral column -- rough approximation,
# not a cited regression, see module docstring. Deliberately smaller than
# _SHOULDER_OFFSET_FRAC: T10 already sits over the spine, unlike ACROM.
_SPINE_OFFSET_FRAC = 0.15


def _compute_shoulder_joint_centres(pts, name_to_idx):
    """
    Returns (left_xyz, right_xyz, occluded): (T,3), (T,3), (T,) bool -- ACROM
    offset medially + inferiorly, scaled by this subject's own biacromial
    width each frame. See module docstring for the approximation this makes.
    """
    def marker(name):
        xyz = pts[:3, name_to_idx[name], :].T
        occ = (pts[3, name_to_idx[name], :] < 0) | np.all(xyz == 0, axis=1)
        return xyz, occ

    acrom_l, occ_al = marker('ACROM_L')
    acrom_r, occ_ar = marker('ACROM_R')
    c7, occ_c7 = marker('C7')
    t10, occ_t10 = marker('T10')

    biacromial = acrom_r - acrom_l
    width = np.linalg.norm(biacromial, axis=1, keepdims=True)
    medial_unit = np.divide(biacromial, width, out=np.zeros_like(biacromial), where=width > 0)

    trunk = t10 - c7   # down the spine, not down to the (often-occluded) hip
    trunk_len = np.linalg.norm(trunk, axis=1, keepdims=True)
    inferior_unit = np.divide(trunk, trunk_len, out=np.zeros_like(trunk), where=trunk_len > 0)

    offset_mag = _SHOULDER_OFFSET_FRAC * width   # (T,1), medial and inferior share this scale
    left = acrom_l + offset_mag * medial_unit + offset_mag * inferior_unit
    right = acrom_r - offset_mag * medial_unit + offset_mag * inferior_unit

    occluded = occ_al | occ_ar | occ_c7 | occ_t10 | (width[:, 0] == 0) | (trunk_len[:, 0] == 0)
    return left, right, occluded


def _compute_spine_point(pts, name_to_idx):
    """
    Returns (xyz, occluded): (T,3), (T,) bool -- T10 offset anteriorly (toward
    XIP_PROC) to approximate MotionBERT/H36M's internal "Spine" keypoint,
    which sits closer to the vertebral column than the posterior skin
    surface. See module docstring for the approximation this makes.
    """
    def marker(name):
        xyz = pts[:3, name_to_idx[name], :].T
        occ = (pts[3, name_to_idx[name], :] < 0) | np.all(xyz == 0, axis=1)
        return xyz, occ

    t10, occ_t10 = marker('T10')
    xip, occ_xip = marker('XIP_PROC')

    anterior = xip - t10   # posterior skin -> anterior skin, roughly through-torso
    depth = np.linalg.norm(anterior, axis=1, keepdims=True)
    anterior_unit = np.divide(anterior, depth, out=np.zeros_like(anterior), where=depth > 0)

    spine = t10 + (_SPINE_OFFSET_FRAC * depth) * anterior_unit

    occluded = occ_t10 | occ_xip | (depth[:, 0] == 0)
    return spine, occluded


def load_mocap_h36m(c3d_path, target_fps=None):
    """
    Returns:
        kps3d_mm          : (T, 17, 3) float32, mm, NaN where unmapped/occluded
        valid_joint_mask   : (17,) bool — False for Nose/Head (unmapped)
        fps                : float, output frame rate (mocap native, or
                              target_fps if given and lower)
        joint_names        : H36M_JOINT_NAMES
        source_frame_idx   : (T,) int — indices into the native 200Hz stream;
                              identity (0..n-1) if target_fps not given. This
                              is the same nearest_frame_indices() used by
                              extract_2d_keypoints.py, applied with the same
                              (n_frames, src_fps, target_fps) so both streams
                              land on identical frame indices when capped to
                              the same n_frames (see that script's docstring).
    """
    c = ezc3d.c3d(c3d_path)
    labels = c['parameters']['POINT']['LABELS']['value']
    name_to_idx = {name: i for i, name in enumerate(labels)}

    pts = c['data']['points']          # (4, n_markers, T) = X,Y,Z,residual
    fps = float(c['parameters']['POINT']['RATE']['value'][0])
    n_frames = pts.shape[2]

    kps3d = np.full((n_frames, 17, 3), np.nan, dtype=np.float32)
    valid_joint_mask = np.zeros(17, dtype=bool)

    missing_markers = []

    # Pre-compute every 'COMPUTED' joint up front (rather than lazily inside
    # the main loop below), since Thorax depends on the already-corrected
    # LShoulder/RShoulder and H36M_JOINT_NAMES lists Thorax BEFORE the
    # shoulders -- a single-pass lazy computation would see Thorax first.
    computed = {}
    if any(m == 'COMPUTED' for m in _JOINT_TO_MARKER.values()):
        required = ('ACROM_L', 'ACROM_R', 'C7', 'T10', 'XIP_PROC')
        req_missing = [r for r in required if r not in name_to_idx]
        if req_missing:
            missing_markers.extend(req_missing)
        else:
            shoulder_l, shoulder_r, shoulder_occ = _compute_shoulder_joint_centres(pts, name_to_idx)
            thorax = (shoulder_l + shoulder_r) / 2.0
            spine, spine_occ = _compute_spine_point(pts, name_to_idx)
            computed = {
                'LShoulder': (shoulder_l, shoulder_occ),
                'RShoulder': (shoulder_r, shoulder_occ),
                'Thorax':    (thorax, shoulder_occ),
                'Spine':     (spine, spine_occ),
            }

    for j, joint_name in enumerate(H36M_JOINT_NAMES):
        marker_name = _JOINT_TO_MARKER[joint_name]
        if marker_name is None:
            continue

        if marker_name == 'COMPUTED':
            if joint_name not in computed:
                continue   # required markers missing, already recorded above
            xyz, occ = computed[joint_name]
            xyz = xyz.copy()
            xyz[occ] = np.nan
            kps3d[:, j, :] = xyz
            valid_joint_mask[j] = True
            continue

        if marker_name not in name_to_idx:
            missing_markers.append(marker_name)
            continue
        m = name_to_idx[marker_name]
        xyz = pts[:3, m, :].T                    # (T, 3)
        residual = pts[3, m, :]                   # (T,)
        # This export doesn't use the standard "residual < 0 means occluded"
        # convention -- residual sits at a constant 1.0 for every sample,
        # valid or not. Occluded/dropped-tracking frames are instead
        # zero-filled (verified directly: e.g. RIGHT_KNEE reads exactly
        # [0,0,0] with residual==1.0 on frames where the marker was lost).
        # Treat both signals as occlusion so we don't miss either convention.
        occluded = (residual < 0) | np.all(xyz == 0, axis=1)
        xyz = xyz.copy()
        xyz[occluded] = np.nan
        kps3d[:, j, :] = xyz
        valid_joint_mask[j] = True

    if missing_markers:
        raise RuntimeError(
            f"{c3d_path}: expected markers not found in this file: {missing_markers} "
            f"— marker naming may differ for this participant/trial."
        )

    n_occluded_frames = np.isnan(kps3d[:, valid_joint_mask, :]).any(axis=(1, 2)).sum()
    if n_occluded_frames:
        print(f"  [warn] {n_occluded_frames}/{n_frames} frames have at least one "
              f"occluded mapped marker (propagated as NaN, not interpolated)")

    if target_fps is not None and target_fps < fps:
        source_frame_idx = nearest_frame_indices(n_frames, fps, target_fps)
        kps3d = kps3d[source_frame_idx]           # selection, not interpolation --
        out_fps = min(target_fps, fps)             # NaN occlusion stays exact
        print(f"  [fps] {n_frames} -> {len(source_frame_idx)} frames "
              f"({fps:.1f}Hz -> {out_fps:.1f}Hz)")
    else:
        source_frame_idx = np.arange(n_frames)
        out_fps = fps

    return kps3d, valid_joint_mask, out_fps, H36M_JOINT_NAMES, source_frame_idx


def convert_trial(user, action):
    """One trial's markers.c3d -> {OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz."""
    c3d_path = os.path.join(_TRIAL_DIR, user, action, 'markers.c3d')
    output_path = mocap_path(user, action)
    # No decimation: kept at native (200Hz) resolution, one row per native
    # frame. Video and mocap share the identical native capture rate
    # (confirmed: both 200Hz for every trial), so a camera's own native
    # frame index already means the same real instant in mocap's stream
    # too -- step_4b_mocap_comparison.py indexes this directly with that
    # shared native index rather than needing a second, independent
    # decimation to line up against.
    kps3d, valid_joint_mask, fps, joint_names, source_frame_idx = load_mocap_h36m(c3d_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path,
             kps3d=kps3d,
             source_frame_idx=source_frame_idx,
             units=np.array('mm'),
             valid_joint_mask=valid_joint_mask,
             fps=fps,
             joint_names=np.array(joint_names))
    return (f"{kps3d.shape[0]} frames @ {fps:g} Hz, {valid_joint_mask.sum()}/17 joints "
            f"(missing: {[n for n, v in zip(joint_names, valid_joint_mask) if not v]})")


def find_trials(user=None, action=None):
    """(user, action) for every folder under TRIAL_DIR holding a markers.c3d, narrowed by
    --user and/or --action when given."""
    users = [user] if user else sorted(d for d in os.listdir(_TRIAL_DIR)
                                       if os.path.isdir(os.path.join(_TRIAL_DIR, d)))
    trials = []
    for u in users:
        udir = os.path.join(_TRIAL_DIR, u)
        if not os.path.isdir(udir):
            continue
        for a in ([action] if action else sorted(os.listdir(udir))):
            if os.path.exists(os.path.join(udir, a, 'markers.c3d')):
                trials.append((u, a))
    return trials


def main():
    ap = argparse.ArgumentParser(description='mocap -> H36M joints. With no --user/--action, every trial '
                                             'under BIOCV_ROOT; trials already converted are skipped.')
    ap.add_argument('--user', default=None, help='default: every user')
    ap.add_argument('--action', default=None, help="default: every action (of --user, or of every user)")
    ap.add_argument('--force', action='store_true', help='redo trials whose output already exists')
    args = ap.parse_args()

    require_out_dir()
    trials = find_trials(args.user, args.action)
    if not trials:
        raise SystemExit(f'no markers.c3d found under {_TRIAL_DIR} for user={args.user or "*"} action={args.action or "*"}')
    print(f'{len(trials)} trial(s) under {_TRIAL_DIR}')
    n_done = n_skip = 0
    failed = []
    for user, action in trials:
        if os.path.exists(mocap_path(user, action)) and not args.force:
            n_skip += 1
            continue
        try:
            print(f'{user}/{action}: {convert_trial(user, action)}', flush=True)
            n_done += 1
        except Exception as e:                      # one bad c3d must not stop the other 400
            failed.append((user, action, f'{type(e).__name__}: {e}'))
            print(f'{user}/{action}: FAILED -- {type(e).__name__}: {e}', flush=True)
    print(f'\nconverted {n_done}, already done {n_skip}, failed {len(failed)}')
    for user, action, err in failed:
        print(f'  {user}/{action}: {err[:200]}')
    if failed and len(trials) == 1:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
