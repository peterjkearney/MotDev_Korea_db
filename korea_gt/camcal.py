"""Camera self-calibration from synchronised multi-view 2D pose tracks.

The scene is a moving person observed by three fixed cameras.  We solve for
per-camera focal length and radial distortion, the relative pose of cameras 2
and 3, and the 3D position of every observed joint, by bundle adjustment.

Two things make an otherwise weakly-constrained problem tractable:

  * skeletal constraints -- every bone must keep the same length in every
    frame, and left/right bones share a length.  This is what breaks the
    focal-length / subject-distance degeneracy.
  * a good initialisation from the roughly-known rig layout.

During optimisation the world frame IS camera 1's frame (R = I, t = 0); that
fixes 6 of the 7 gauge degrees of freedom.  The 7th (scale) is fixed by a soft
prior on the camera 1 -- camera 2 baseline.  A gravity-aligned frame, and hence
camera heights and tilts, is recovered afterwards from foot contacts.

Joint convention is OpenPose BODY_25.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

# --------------------------------------------------------------------------
# BODY_25 skeleton
# --------------------------------------------------------------------------

BODY25_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]

FOOT_JOINTS = [19, 20, 21, 22, 23, 24]   # toes and heels -- these touch the floor
ANKLES = [11, 14]

# (joint_a, joint_b, shared_length_key).  Left/right pairs share a key, so the
# symmetry constraint costs nothing extra.  Face and foot bones are excluded:
# they are short, noisy, and contribute little.
BONES = [
    (1, 8, "torso"),
    (1, 2, "clavicle"), (1, 5, "clavicle"),
    (2, 3, "upperarm"), (5, 6, "upperarm"),
    (3, 4, "forearm"), (6, 7, "forearm"),
    (8, 9, "hip_offset"), (8, 12, "hip_offset"),
    (9, 10, "thigh"), (12, 13, "thigh"),
    (10, 11, "shank"), (13, 14, "shank"),
]

BONE_KEYS = sorted({k for _, _, k in BONES})

# Segment lengths as a fraction of standing stature.  Children are not scaled
# adults: at 3-4.5 years the trunk is relatively longer and the legs
# relatively shorter, so an adult table applied to this cohort biases the
# recovered scale -- and with it every distance and height -- by tens of
# percent.  Values are approximate (Snyder et al. 1977; Jensen 1986); replace
# them with your own measurements if you have them.
#
# Note these are OpenPose keypoint separations, not anatomical segment
# lengths: "Neck" is the shoulder midpoint and "MidHip" the hip midpoint, so
# `torso` is shoulder-line to hip-line, and `thigh` is hip keypoint to knee
# keypoint.  They are consistent estimators, not clinical measurements.
STATURE_FRACTION = {
    "child_3_5": {
        "torso": 0.320, "clavicle": 0.105, "upperarm": 0.165,
        "forearm": 0.145, "hip_offset": 0.060, "thigh": 0.205,
        "shank": 0.195,
    },
    "adult": {
        "torso": 0.294, "clavicle": 0.106, "upperarm": 0.176,
        "forearm": 0.153, "hip_offset": 0.059, "thigh": 0.247,
        "shank": 0.247,
    },
}

# Median stature: ~1.02 m at 3.75 years (WHO growth standards, both sexes).
DEFAULT_STATURE = {"child_3_5": 1.02, "adult": 1.70}


def bone_lengths_for(stature=None, group="child_3_5"):
    """Expected keypoint separations (m) for a subject of this stature."""
    if group not in STATURE_FRACTION:
        raise ValueError(f"unknown group {group!r}; "
                         f"choose from {sorted(STATURE_FRACTION)}")
    if stature is None:
        stature = DEFAULT_STATURE[group]
    return {k: v * stature for k, v in STATURE_FRACTION[group].items()}


# This dataset is children aged 3 to 4.5.
BONE_INIT = bone_lengths_for(group="child_3_5")


# --------------------------------------------------------------------------
# Lens arithmetic: turning a camera spec into a focal length in pixels
# --------------------------------------------------------------------------

def focal_px_from_hfov(hfov_deg, width_px=1920):
    """Pixel focal length from horizontal field of view."""
    return 0.5 * width_px / np.tan(0.5 * np.radians(hfov_deg))


def hfov_from_focal_px(f_px, width_px=1920):
    return np.degrees(2 * np.arctan(0.5 * width_px / np.asarray(f_px, float)))


def focal_px_from_mm(f_mm, sensor_width_mm, width_px=1920):
    """Pixel focal length from a physical focal length and sensor width.

    `sensor_width_mm` must be the width ACTUALLY READ OUT in the recording
    mode, not the full sensor: most cameras crop or scale in video, and many
    apply a further crop for stabilisation.  Getting this wrong is the usual
    reason a spec-sheet focal length disagrees with the fitted one.
    """
    return np.asarray(f_mm, float) / sensor_width_mm * width_px


def focal_35mm_equiv(f_px, width_px=1920):
    """35mm-equivalent focal length -- the number to compare against a spec."""
    return 36.0 * np.asarray(f_px, float) / width_px


# Known cameras.  `sensor_width_mm` is the width read out in the RECORDING
# mode, and `video_crop` any further crop applied in movie mode (including
# electronic stabilisation) -- both raise the effective pixel focal length, so
# a wrong value here shifts the whole bound.
CAMERA_PRESETS = {
    # Sony DSC-RX100 (Mark I): 1"-type sensor, Zeiss 10.4-37.1 mm zoom,
    # quoted as 28-100 mm equivalent.  Later RX100 marks differ (III-V are
    # 8.8-25.7 mm / 24-70 eq; VI-VII are 24-200 eq), so check the mark.
    "sony-rx100": dict(sensor_width_mm=13.2, f_mm=(10.4, 37.1), video_crop=1.0),
}


def focal_bounds_for_camera(name, width_px=1920, video_crop=None):
    """Physically possible pixel focal range (lo, hi) for a known camera.

    Pass the result as `BAOptions.f_bounds`.  For a zoom lens this is a range,
    not a discrete set -- but the lower bound alone is worth a lot, because it
    excludes solutions where the lens would have to be wider than it can be.
    """
    if name not in CAMERA_PRESETS:
        raise ValueError(f"unknown camera {name!r}; "
                         f"known: {sorted(CAMERA_PRESETS)}")
    spec = CAMERA_PRESETS[name]
    crop = spec["video_crop"] if video_crop is None else video_crop
    w_mm = spec["sensor_width_mm"] / crop
    return tuple(focal_px_from_mm(f, w_mm, width_px) for f in spec["f_mm"])


# --------------------------------------------------------------------------
# Rig layout
# --------------------------------------------------------------------------

def layout_from_polar(dist_deg, target_dist=3.0):
    """Plan-view camera centres from (distance-to-target, bearing) pairs.

    The bearing is the angle subtended AT THE TARGET between camera 1's line of
    sight and this camera's, positive to the right.  Camera 1 sits at the plan
    origin looking down +y; the target is at (0, target_dist).

    Returns an (n, 2) array of (x, y) in metres.
    """
    target = np.array([0.0, target_dist])
    out = []
    for dist, deg in dist_deg:
        th = np.radians(deg)
        out.append(target + dist * np.array([np.sin(th), -np.cos(th)]))
    return np.array(out)


# The user's rig: cam1 3 m from the target, cam2 45 deg right at 1.8 m,
# cam3 45 deg left at 2.5 m.
DEFAULT_LAYOUT = layout_from_polar([(3.0, 0.0), (1.8, 45.0), (2.5, -45.0)])


def look_at(centre, target, up=(0.0, 0.0, 1.0)):
    """Extrinsics (R, t) for a camera at `centre` aimed at `target`.

    Input coordinates are in a gravity-aligned rig frame (x right, y forward,
    z up).  Camera convention is x right, y down, z forward, so that
    X_cam = R @ X_rig + t.
    """
    centre = np.asarray(centre, float)
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ centre


def init_cameras(layout=None, heights=(1.2, 1.2, 1.2), target_dist=3.0,
                 target_height=1.0, image_size=(1920, 1080), hfov_deg=50.0):
    """Build a starting CameraSet from the roughly-known rig geometry.

    Cameras are placed at `layout` (plan view) at the given `heights` and aimed
    at a point `target_height` above the floor at the target.  The result is
    then rotated/translated so camera 1 is the identity, which is the frame the
    bundle adjustment works in.
    """
    layout = DEFAULT_LAYOUT if layout is None else np.asarray(layout, float)
    w, h = image_size
    f = 0.5 * w / np.tan(0.5 * np.radians(hfov_deg))

    target = np.array([0.0, target_dist, target_height])
    Rs, ts = [], []
    for (px, py), cz in zip(layout, heights):
        R, t = look_at((px, py, cz), target)
        Rs.append(R)
        ts.append(t)

    # Re-express relative to camera 1.  With X_rig = R_1^T (X_1 - t_1),
    #   X_c = (R_c R_1^T) X_1 + (t_c - (R_c R_1^T) t_1)
    # so the translation subtracts the NEW rotation applied to t_1 -- not
    # R_c R_1^T R_1^T t_1, which is what reusing the loop variable would give.
    R1, t1 = Rs[0], ts[0]
    Rnew = [R @ R1.T for R in Rs]
    ts = [t - Rn @ t1 for Rn, t in zip(Rnew, ts)]
    Rs = Rnew
    # (Rs[0] is now I and ts[0] is now 0 up to float error -- make it exact.)
    Rs[0] = np.eye(3)
    ts[0] = np.zeros(3)

    n = len(layout)
    return CameraSet(
        f=np.full(n, f),
        k1=np.zeros(n),
        k2=np.zeros(n),
        rvec=np.stack([Rotation.from_matrix(R).as_rotvec() for R in Rs]),
        t=np.stack(ts),
        cx=np.full(n, 0.5 * w),
        cy=np.full(n, 0.5 * h),
        image_size=image_size,
    )


def estimate_focal_from_scale(obs, target_dists, subject_height=1.02,
                              top=(0, 1), bottom=ANKLES, span_frac=0.86):
    """Data-driven focal initialisation from apparent subject size.

    A guessed field of view is a poor starting point, and the initial
    triangulation inherits its error.  But you already know roughly how far
    each camera is from the target, and roughly how tall a person is, so

        f = (pixel span) * (distance to subject) / (metric span)

    `target_dists` is the known camera-to-target distance per camera -- for
    this rig, (3.0, 1.8, 2.5).  The span used is neck/nose to ankle, which is
    about `span_frac` of stature.
    """
    C = obs.xy.shape[1]
    out = np.zeros(C)
    for c in range(C):
        px = []
        for f in range(obs.xy.shape[0]):
            t = [j for j in top if obs.vis[f, c, j]]
            b = [j for j in bottom if obs.vis[f, c, j]]
            if t and b:
                px.append(obs.xy[f, c, b, 1].mean() - obs.xy[f, c, t, 1].mean())
        if not px:
            raise RuntimeError(f"camera {c}: no frame with both head and ankles")
        out[c] = np.median(px) * target_dists[c] / (subject_height * span_frac)
    return out


@dataclass
class CameraSet:
    f: np.ndarray            # (n,) focal length in pixels
    k1: np.ndarray           # (n,) radial distortion
    k2: np.ndarray           # (n,)
    rvec: np.ndarray         # (n, 3) world -> camera rotation vectors
    t: np.ndarray            # (n, 3) world -> camera translation
    cx: np.ndarray           # (n,) principal point
    cy: np.ndarray
    image_size: tuple = (1920, 1080)

    @property
    def n(self):
        return len(self.f)

    @property
    def R(self):
        return Rotation.from_rotvec(self.rvec).as_matrix()

    @property
    def centres(self):
        """Camera centres in world (= camera 1) coordinates."""
        return np.einsum("nji,nj->ni", self.R, -self.t)

    @property
    def axes(self):
        """Optical axis directions in world coordinates."""
        return self.R[:, 2, :]

    def hfov_deg(self):
        return np.degrees(2 * np.arctan(0.5 * self.image_size[0] / self.f))

    def project(self, X, cam):
        """Project world points X (m, 3) through camera `cam`."""
        R = Rotation.from_rotvec(self.rvec[cam]).as_matrix()
        Xc = X @ R.T + self.t[cam]
        z = np.maximum(Xc[:, 2], 1e-4)
        u, v = Xc[:, 0] / z, Xc[:, 1] / z
        r2 = u * u + v * v
        d = 1 + self.k1[cam] * r2 + self.k2[cam] * r2 * r2
        return np.stack([self.f[cam] * d * u + self.cx[cam],
                         self.f[cam] * d * v + self.cy[cam]], axis=1)

    def P(self, cam):
        """3x4 pinhole projection matrix (ignores distortion)."""
        K = np.array([[self.f[cam], 0, self.cx[cam]],
                      [0, self.f[cam], self.cy[cam]],
                      [0, 0, 1.0]])
        R = Rotation.from_rotvec(self.rvec[cam]).as_matrix()
        return K @ np.hstack([R, self.t[cam][:, None]])


# --------------------------------------------------------------------------
# Data loading and frame selection
# --------------------------------------------------------------------------

class InsufficientData(RuntimeError):
    """Raised when a session has too little usable data to calibrate."""

@dataclass
class Observations:
    """A selected set of frames, flattened to per-joint observations.

    xy   : (F, C, J, 2) pixel coordinates
    vis  : (F, C, J)    bool, True where the joint was detected in that view
    conf : (F, C, J)
    meta : list of (subject, stub, frame_index), length F
    subject_of_frame : (F,) index into `subjects`
    """
    xy: np.ndarray
    vis: np.ndarray
    conf: np.ndarray
    meta: list = field(default_factory=list)
    subjects: list = field(default_factory=list)
    subject_of_frame: np.ndarray = None


def load_candidates(datapath, subjects=None, actions=None, max_files=None,
                    conf_thresh=0.5, min_joints=None, min_views=2, min_corr=4,
                    stride=5, verbose=True):
    """Scan aligned .npz files and keep frames that carry usable correspondences.

    A joint counts as seen only if its confidence clears `conf_thresh` AND its
    coordinates are not exactly (0, 0) -- the pose estimator writes (0, 0) with
    confidence 0 for a missing joint, so a threshold alone is not enough.

    A joint seen in at least `min_views` cameras is a correspondence, and a
    frame is kept if it has at least `min_corr` of them.  Nothing in the
    calibration needs a frame to be complete: the bundle adjustment works per
    observation, and the bone term per bone.  What conditions it is the total
    number of correspondences and their spread, so reliability is judged per
    joint (confidence threshold, robust loss), not per frame.

    `min_joints` restores the old rule -- at least that many joints seen in
    ALL views -- which throws away good correspondences wholesale: with
    cameras at +-45 degrees one ear is always hidden from some camera, and
    heels/toes are often faint, so clearly-seen bodies failed it.
    """
    files = sorted(f for f in os.listdir(datapath)
                   if f.endswith(".npz") and not f.startswith("."))
    if subjects is not None:
        subjects = set(subjects)
        files = [f for f in files if f.split("_")[0] in subjects]
    if actions is not None:
        actions = {str(a) for a in actions}
        files = [f for f in files if f.split("_")[2] in actions]
    if max_files is not None:
        files = files[:max_files]

    xy_out, vis_out, conf_out, meta = [], [], [], []
    for i, fname in enumerate(files):
        if verbose and i % 25 == 0:
            print(f"\rscanning {i}/{len(files)}", end="")
        d = np.load(os.path.join(datapath, fname))
        xy = np.stack([d["xy1"], d["xy2"], d["xy3"]], axis=1)        # (T,C,J,2)
        conf = np.stack([d["score1"], d["score2"], d["score3"]], 1)  # (T,C,J)
        vis = (conf > conf_thresh) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))

        if min_joints is not None:                                   # legacy rule
            good = vis.all(axis=1).sum(axis=1) >= min_joints
        else:
            n_corr = (vis.sum(axis=1) >= min_views).sum(axis=1)      # (T,)
            good = n_corr >= min_corr
        keep = np.where(good)[0][::stride]
        if len(keep) == 0:
            continue
        stub = fname[:-4]
        subject = stub.split("_")[0]
        xy_out.append(xy[keep])
        vis_out.append(vis[keep])
        conf_out.append(conf[keep])
        meta.extend((subject, stub, int(t)) for t in keep)

    if verbose:
        print(f"\rscanned {len(files)} files -> {len(meta)} candidate frames")
    if not meta:
        raise RuntimeError("no candidate frames survived filtering")

    subs = sorted({m[0] for m in meta})
    sub_idx = {s: i for i, s in enumerate(subs)}
    return Observations(
        xy=np.concatenate(xy_out).astype(float),
        vis=np.concatenate(vis_out),
        conf=np.concatenate(conf_out).astype(float),
        meta=meta,
        subjects=subs,
        subject_of_frame=np.array([sub_idx[m[0]] for m in meta]),
    )


_UPPER = [0, 1, 2, 5, 15, 16, 17, 18]            # head and shoulders
_LOWER = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]  # knees, ankles, feet


def select_spread(obs, n_frames=400, pos_bins=5, scale_bins=4, seed=0,
                  min_frames=80, target_corr=None, min_corr_total=None,
                  min_views=2):
    """Thin candidate frames down to a set that covers each image well.

    Focal length is constrained by seeing the subject at many image positions
    and many apparent sizes, so we bucket frames by where the subject sits in
    each view plus how large they appear, and sample evenly across buckets
    rather than taking a contiguous or random slice.

    Apparent size is the vertical extent in camera 1, taken only when both
    upper and lower body are visible there; a partial body would read as
    small, so those frames get their own bucket instead.

    Budget.  By default, `n_frames` frames.  With `target_corr`, frames are
    added until they hold that many correspondences (joints seen in >=
    `min_views` views), with `n_frames` as a cap -- frames now range from a
    handful of joints to 25, so a frame count is the wrong unit.  In that mode
    the richest frame of each bucket is taken first, rather than a random one.

    Raises InsufficientData below `min_frames` frames or, if given,
    `min_corr_total` correspondences.
    """
    rng = np.random.default_rng(seed)
    F, C, J, _ = obs.xy.shape
    m = obs.vis[..., None]
    cen = (obs.xy * m).sum(axis=2) / np.maximum(m.sum(axis=2), 1)     # (F,C,2)
    n_corr = (obs.vis.sum(axis=1) >= min_views).sum(axis=1)           # (F,)
    n_all = obs.vis.all(axis=1).sum(axis=1)                           # seen by every camera

    v1 = obs.vis[:, 0]
    whole = v1[:, _UPPER].any(axis=1) & v1[:, _LOWER].any(axis=1)
    ys = np.where(v1, obs.xy[:, 0, :, 1], np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        height = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)        # (F,)
    logh = np.log(np.where(whole & (height > 0), height, np.nan))

    def binned(v, lo, hi, nb):
        return np.clip(((v - lo) / max(hi - lo, 1e-9) * nb).astype(int), 0, nb - 1)

    w, h = obs.xy[..., 0].max(), obs.xy[..., 1].max()
    if np.isfinite(logh).any():
        lo, hi = np.nanmin(logh), np.nanmax(logh)
        scale_key = np.where(np.isfinite(logh),
                             binned(np.nan_to_num(logh, nan=lo), lo, hi, scale_bins), -1)
    else:
        scale_key = np.full(F, -1)
    key = np.stack(
        [binned(cen[:, c, 0], 0, w, pos_bins) for c in range(C)] +
        [binned(cen[:, c, 1], 0, h, pos_bins) for c in range(C)] +
        [scale_key],
        axis=1,
    )

    buckets = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)
    for v in buckets.values():
        rng.shuffle(v)
        if target_corr is not None:
            # Richest first -- and "rich" means seen by EVERY camera before
            # seen by some: a joint in all three views constrains all three
            # cameras at once, which is what made complete frames worth more.
            v.sort(key=lambda i: (-n_all[i], -n_corr[i]))   # stable: ties stay shuffled

    # Round-robin across buckets so no region of the image dominates.
    chosen, order, depth, total = [], list(buckets.values()), 0, 0

    def full():
        if len(chosen) >= n_frames:
            return True
        return target_corr is not None and total >= target_corr

    while not full():
        added = False
        for v in order:
            if depth < len(v):
                chosen.append(v[depth])
                total += int(n_corr[v[depth]])
                added = True
                if full():
                    break
        if not added:
            break
        depth += 1

    idx = np.sort(np.array(chosen, dtype=int))
    total = int(n_corr[idx].sum()) if len(idx) else 0
    short_frames = len(idx) < min_frames
    short_corr = min_corr_total is not None and total < min_corr_total
    if short_frames or short_corr:
        need = [f">= {min_frames} frames"]
        if min_corr_total is not None:
            need.append(f">= {min_corr_total} correspondences")
        raise InsufficientData(
            f"only {len(idx)} frames / {total} correspondences available "
            f"(need {' and '.join(need)}). This session cannot be calibrated: "
            f"too few joints are seen by two or more cameras at once. Forcing "
            f"it produces a confident-looking but meaningless fit.")
    return Observations(
        xy=obs.xy[idx], vis=obs.vis[idx], conf=obs.conf[idx],
        meta=[obs.meta[i] for i in idx],
        subjects=obs.subjects,
        subject_of_frame=obs.subject_of_frame[idx],
    )


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------

def undistort(cams, xy, cam, iters=8):
    """Pixel coords -> ideal normalised coords, inverting the radial model."""
    u = (xy[..., 0] - cams.cx[cam]) / cams.f[cam]
    v = (xy[..., 1] - cams.cy[cam]) / cams.f[cam]
    ud, vd = u.copy(), v.copy()
    for _ in range(iters):
        r2 = u * u + v * v
        d = 1 + cams.k1[cam] * r2 + cams.k2[cam] * r2 * r2
        u, v = ud / d, vd / d
    return np.stack([u, v], axis=-1)


def triangulate(cams, xy, vis):
    """Multi-view DLT.  xy (F,C,J,2), vis (F,C,J) -> X (F,J,3), ok (F,J).

    Points are undistorted first -- triangulating distorted pixels through a
    pinhole matrix costs several centimetres of 3D error on a wide lens.
    Invisible views contribute zero rows, which leaves the null space
    untouched, so the whole thing batches into one SVD call.
    """
    F, C, J, _ = xy.shape
    N = F * J
    A = np.zeros((N, 2 * C, 4))
    for c in range(C):
        n = undistort(cams, xy[:, c], c).reshape(N, 2)
        P = np.hstack([cams.R[c], cams.t[c][:, None]])       # normalised coords
        m = vis[:, c, :].reshape(N)[:, None]
        A[:, 2 * c] = (n[:, 0:1] * P[2] - P[0]) * m
        A[:, 2 * c + 1] = (n[:, 1:2] * P[2] - P[1]) * m

    n_views = vis.transpose(0, 2, 1).reshape(N, C).sum(axis=1)
    usable = n_views >= 2
    X = np.full((N, 3), np.nan)
    if usable.any():
        _, _, Vt = np.linalg.svd(A[usable])
        hom = Vt[:, -1, :]
        w = hom[:, 3]
        good = np.abs(w) > 1e-12
        Xu = np.full((usable.sum(), 3), np.nan)
        Xu[good] = hom[good, :3] / w[good, None]
        X[usable] = Xu

    X = X.reshape(F, J, 3)
    ok = usable.reshape(F, J) & np.isfinite(X).all(axis=2)

    # Reject points that land behind a camera which claims to see them.
    Xf = np.where(ok[..., None], X, 0.0)
    for c in range(C):
        z = Xf @ cams.R[c][2] + cams.t[c][2]
        ok &= ~(vis[:, c, :] & (z <= 0.1))
    return X, ok


# --------------------------------------------------------------------------
# Bundle adjustment
# --------------------------------------------------------------------------

@dataclass
class BAProblem:
    """Flattened problem: point list, observation list, bone list."""
    n_points: int
    pt_of: np.ndarray          # (F, J) -> point index, -1 if unused
    obs_cam: np.ndarray        # (M,) camera index per observation
    obs_pt: np.ndarray         # (M,) point index per observation
    obs_xy: np.ndarray         # (M, 2)
    obs_w: np.ndarray          # (M,) weight
    bone_a: np.ndarray         # (B,) point index
    bone_b: np.ndarray         # (B,)
    bone_len: np.ndarray       # (B,) index into bone-length params
    n_bone_params: int
    subjects: list
    X0: np.ndarray             # (n_points, 3)
    bone_prior: np.ndarray = None   # (n_bone_params,) anthropometric prior
    scale_baseline: float = None    # if set, |t_1| is pinned to this (metres)
    f_shared: bool = False          # if set, all cameras share camera 1's focal
    use_bone_prior: bool = False    # tether bone lengths to an assumed table


def build_problem(cams, obs, X, ok, sigma_px=3.0, weight_by_conf=True,
                  bone_target=None, use_bone_prior=False):
    F, C, J, _ = obs.xy.shape
    pt_of = np.full((F, J), -1)
    good = ok.copy()
    pt_of[good] = np.arange(good.sum())
    n_points = int(good.sum())

    fi, ci, ji = np.where(obs.vis & good[:, None, :])
    obs_pt = pt_of[fi, ji]
    w = np.full(len(fi), 1.0 / sigma_px)
    if weight_by_conf:
        w = w * np.sqrt(obs.conf[fi, ci, ji])

    key_idx = {k: i for i, k in enumerate(BONE_KEYS)}
    n_sub = len(obs.subjects)
    ba, bb, bl = [], [], []
    for f in range(F):
        s = obs.subject_of_frame[f]
        for ja, jb, key in BONES:
            if ja >= J or jb >= J:
                continue
            if pt_of[f, ja] >= 0 and pt_of[f, jb] >= 0:
                ba.append(pt_of[f, ja])
                bb.append(pt_of[f, jb])
                bl.append(s * len(BONE_KEYS) + key_idx[key])

    prob = BAProblem(
        n_points=n_points,
        pt_of=pt_of,
        obs_cam=ci, obs_pt=obs_pt, obs_xy=obs.xy[fi, ci, ji], obs_w=w,
        bone_a=np.array(ba, int), bone_b=np.array(bb, int),
        bone_len=np.array(bl, int),
        n_bone_params=n_sub * len(BONE_KEYS),
        subjects=obs.subjects,
        X0=X[good],
        bone_prior=np.tile([(bone_target or BONE_INIT)[k]
                            for k in BONE_KEYS], n_sub),
        use_bone_prior=use_bone_prior,
    )
    # Initialise each bone length from the triangulated geometry rather than
    # from a table, so that with the prior switched off no assumed
    # anthropometry enters the solve at any point.
    ba, bb, bl = prob.bone_a, prob.bone_b, prob.bone_len
    if len(ba):
        dl = np.linalg.norm(prob.X0[ba] - prob.X0[bb], axis=1)
        init = prob.bone_prior.copy()
        for i in range(prob.n_bone_params):
            m = bl == i
            if m.any():
                init[i] = float(np.median(dl[m]))
        prob.bone_init = init
    else:
        prob.bone_init = prob.bone_prior.copy()
    return prob


# Parameter layout: [f(C), k1(C), k2(C), rvec+t for cams 1..C-1, bones, points]
def _offsets(C, prob):
    o = {}
    o["f"] = 0
    o["k1"] = C
    o["k2"] = 2 * C
    o["rt"] = 3 * C                        # 6 * (C-1) values, cameras 1..C-1
    o["bone"] = 3 * C + 6 * (C - 1)
    o["pts"] = o["bone"] + prob.n_bone_params
    o["n"] = o["pts"] + 3 * prob.n_points
    return o


def pack(cams, prob, bone_lengths):
    C = cams.n
    o = _offsets(C, prob)
    p = np.zeros(o["n"])
    p[o["f"]:o["f"] + C] = cams.f
    p[o["k1"]:o["k1"] + C] = cams.k1
    p[o["k2"]:o["k2"] + C] = cams.k2
    for c in range(1, C):
        b = o["rt"] + 6 * (c - 1)
        p[b:b + 3] = cams.rvec[c]
        p[b + 3:b + 6] = cams.t[c]
    p[o["bone"]:o["pts"]] = bone_lengths
    p[o["pts"]:] = prob.X0.ravel()
    return p


def unpack(p, cams, prob):
    C = cams.n
    o = _offsets(C, prob)
    out = CameraSet(
        f=p[o["f"]:o["f"] + C].copy(),
        k1=p[o["k1"]:o["k1"] + C].copy(),
        k2=p[o["k2"]:o["k2"] + C].copy(),
        rvec=np.zeros((C, 3)), t=np.zeros((C, 3)),
        cx=cams.cx.copy(), cy=cams.cy.copy(), image_size=cams.image_size,
    )
    for c in range(1, C):
        b = o["rt"] + 6 * (c - 1)
        out.rvec[c] = p[b:b + 3]
        out.t[c] = p[b + 3:b + 6]

    # Hard scale gauge.  A global similarity leaves every reprojection exactly
    # unchanged, so scale is an exact null space of the data term and no soft
    # prior can hold it -- a handful of prior residuals lose to thousands of
    # reprojection residuals, and the whole reconstruction quietly shrinks.
    # Pinning |t_2| removes the direction instead of penalising it; the radial
    # part of t_2 then has zero gradient and simply never moves.
    if prob is not None and prob.f_shared:
        # Same model, same recording mode => one focal length, not three.
        # Tying them here (rather than adding a constraint) leaves the other
        # two parameters inert, exactly as with the scale gauge above.
        out.f[:] = out.f[0]
    if prob is not None and prob.scale_baseline is not None:
        nrm = np.linalg.norm(out.t[1])
        if nrm > 1e-9:
            out.t[1] = out.t[1] * (prob.scale_baseline / nrm)
    return out, p[o["bone"]:o["pts"]], p[o["pts"]:].reshape(-1, 3)


# Stage schedule for `solve`: (loss, f_scale, fit_distortion).  Start
# forgiving and pinhole-only, then free the distortion while still forgiving,
# and only then tighten to a hard robust loss that rejects outliers.
DEFAULT_STAGES = (
    ("soft_l1", 8.0, False),
    ("soft_l1", 4.0, True),
    ("huber", 3.0, True),
)


@dataclass
class BAOptions:
    sigma_bone: float = 0.010      # m -- how tightly bones must hold length
    # A loose tether to adult anthropometry.  It is NOT the scale anchor --
    # scale is pinned hard in unpack() -- so keep it wide and treat the
    # recovered bone lengths as a validation of the assumed baseline instead.
    sigma_bone_prior: float = 0.15   # m
    # Off by default: with scale pinned to a measured camera-subject distance,
    # limb lengths are an OUTPUT of the fit and make a far better check than a
    # constraint.  Bone-length constancy and left/right symmetry still apply --
    # both are pure shape constraints and carry no assumed sizes.
    use_bone_prior: bool = False
    sigma_scale: float = 0.25      # m -- vestigial; |t_2| is pinned exactly
    sigma_pos: float = 0.50        # m -- weak prior keeping cams near the layout
    sigma_k: float = 0.05          # weak prior pulling distortion toward zero
    fit_k1: bool = True
    fit_k2: bool = False
    fit_focal: bool = True
    # Hard, physically motivated bounds.  Without them the distortion terms
    # absorb arbitrary model error and drag the focal length with them.
    k1_bounds: tuple = (-0.6, 0.3)
    k2_bounds: tuple = (-0.3, 0.3)
    f_ratio_bounds: tuple = (0.5, 2.0)   # multiples of the initial focal
    # Absolute focal bounds in pixels, (lo, hi), scalar or per-camera. Takes
    # precedence over f_ratio_bounds. Use when you know the lens: see
    # focal_px_from_hfov / focal_px_from_mm.
    f_bounds: tuple = None
    # Tie all cameras to a single focal length. Only correct if they really
    # are the same model in the same recording mode -- test it, do not assume.
    f_shared: bool = False
    # Fix focals outright at cams0.f (fit_focal=False) -- by far the strongest
    # option if you can identify the camera and mode.
    loss: str = "huber"            # robust loss for reprojection residuals
    f_scale: float = 2.0           # robust-loss knee, in units of sigma
    max_nfev: int = 200
    verbose: int = 2


def make_residual(cams0, prob, opts, baseline_nominal, centre_prior):
    C = cams0.n
    o = _offsets(C, prob)
    cx, cy = cams0.cx, cams0.cy
    per_cam = [np.where(prob.obs_cam == c)[0] for c in range(C)]

    def fun(p):
        cams, L, X = unpack(p, cams0, prob)
        Rs = cams.R
        rep = np.zeros((len(prob.obs_pt), 2))
        for c in range(C):
            m = per_cam[c]
            if len(m) == 0:
                continue
            Xc = X[prob.obs_pt[m]] @ Rs[c].T + cams.t[c]
            z = np.maximum(Xc[:, 2], 1e-3)
            u, v = Xc[:, 0] / z, Xc[:, 1] / z
            r2 = u * u + v * v
            d = 1 + cams.k1[c] * r2 + cams.k2[c] * r2 * r2
            rep[m, 0] = cams.f[c] * d * u + cx[c] - prob.obs_xy[m, 0]
            rep[m, 1] = cams.f[c] * d * v + cy[c] - prob.obs_xy[m, 1]
        rep *= prob.obs_w[:, None]

        dl = np.linalg.norm(X[prob.bone_a] - X[prob.bone_b], axis=1)
        bone = (dl - L[prob.bone_len]) / opts.sigma_bone
        bone_pri = ((L - prob.bone_prior) / opts.sigma_bone_prior
                    if prob.use_bone_prior else np.empty(0))

        centres = cams.centres
        # |t_2| is pinned in unpack(), so this is ~0 by construction; it is kept
        # only so the residual vector keeps a fixed shape.
        scale = np.array([(np.linalg.norm(cams.t[1]) - baseline_nominal)
                          / opts.sigma_scale])
        pos = ((centres[1:] - centre_prior[1:]) / opts.sigma_pos).ravel()
        kpri = np.concatenate([cams.k1 / opts.sigma_k,
                               cams.k2 / opts.sigma_k])
        return np.concatenate([rep.ravel(), bone, bone_pri, scale, pos, kpri])

    return fun, o


def make_sparsity(cams0, prob, opts):
    C = cams0.n
    o = _offsets(C, prob)
    M, B = len(prob.obs_pt), len(prob.bone_a)
    n_bp = prob.n_bone_params if prob.use_bone_prior else 0
    n_res = 2 * M + B + n_bp + 1 + 3 * (C - 1) + 2 * C
    S = lil_matrix((n_res, o["n"]), dtype=int)

    rows = np.arange(M)
    for k in (0, 1):
        r = 2 * rows + k
        if prob.f_shared:
            S[r[:, None], np.arange(o["f"], o["f"] + C)[None, :]] = 1
        else:
            S[r, o["f"] + prob.obs_cam] = 1
        S[r, o["k1"] + prob.obs_cam] = 1
        S[r, o["k2"] + prob.obs_cam] = 1
        for d in range(3):
            S[r, o["pts"] + 3 * prob.obs_pt + d] = 1
        moving = prob.obs_cam > 0
        rm, cm = r[moving], prob.obs_cam[moving]
        for d in range(6):
            S[rm, o["rt"] + 6 * (cm - 1) + d] = 1

    br = 2 * M + np.arange(B)
    for d in range(3):
        S[br, o["pts"] + 3 * prob.bone_a + d] = 1
        S[br, o["pts"] + 3 * prob.bone_b + d] = 1
    S[br, o["bone"] + prob.bone_len] = 1

    # bone-length priors: each depends on exactly one bone-length parameter
    bp = 2 * M + B
    for i in range(n_bp):
        S[bp + i, o["bone"] + i] = 1

    gr = bp + n_bp
    S[gr, o["rt"]:o["bone"]] = 1                      # scale gauge
    for c in range(1, C):
        for d in range(3):
            S[gr + 1 + 3 * (c - 1) + d, o["rt"] + 6 * (c - 1):o["rt"] + 6 * c] = 1
    kr = gr + 1 + 3 * (C - 1)
    for c in range(C):
        S[kr + c, o["k1"] + c] = 1
        S[kr + C + c, o["k2"] + c] = 1
    return S.tocsr(), o


def bundle_adjust(cams0, obs, X, ok, opts=None, baseline_nominal=None,
                  centre_prior=None, sigma_px=3.0):
    """Run the bundle adjustment.  Returns (cams, bone_lengths, X, result, prob)."""
    opts = opts or BAOptions()
    prob = build_problem(cams0, obs, X, ok, sigma_px=sigma_px,
                         use_bone_prior=opts.use_bone_prior)

    if centre_prior is None:
        centre_prior = cams0.centres
    if baseline_nominal is None:
        baseline_nominal = float(np.linalg.norm(centre_prior[1]))
    prob.scale_baseline = baseline_nominal
    prob.f_shared = opts.f_shared

    p0 = pack(cams0, prob, prob.bone_init)

    fun, o = make_residual(cams0, prob, opts, baseline_nominal, centre_prior)
    S, _ = make_sparsity(cams0, prob, opts)

    lo = np.full_like(p0, -np.inf)
    hi = np.full_like(p0, np.inf)
    C = cams0.n
    if not opts.fit_focal:
        lo[o["f"]:o["f"] + C] = p0[o["f"]:o["f"] + C] - 1e-9
        hi[o["f"]:o["f"] + C] = p0[o["f"]:o["f"] + C] + 1e-9
    else:
        if opts.f_bounds is not None:
            lo[o["f"]:o["f"] + C] = np.broadcast_to(opts.f_bounds[0], C)
            hi[o["f"]:o["f"] + C] = np.broadcast_to(opts.f_bounds[1], C)
        else:
            # Keep the focal within a sane multiple of its (data-driven) init.
            lo[o["f"]:o["f"] + C] = opts.f_ratio_bounds[0] * p0[o["f"]:o["f"] + C]
            hi[o["f"]:o["f"] + C] = opts.f_ratio_bounds[1] * p0[o["f"]:o["f"] + C]
    for name, on, bnd in (("k1", opts.fit_k1, opts.k1_bounds),
                          ("k2", opts.fit_k2, opts.k2_bounds)):
        if on:
            lo[o[name]:o[name] + C] = bnd[0]
            hi[o[name]:o[name] + C] = bnd[1]
        else:
            lo[o[name]:o[name] + C] = -1e-9
            hi[o[name]:o[name] + C] = 1e-9
    lo[o["bone"]:o["pts"]] = 0.01

    # The data-driven focal init can land outside physically possible bounds
    # (e.g. below a lens's wide end); least_squares rejects that outright, so
    # clip the start into the feasible box rather than failing.
    outside = (p0 < lo) | (p0 > hi)
    if outside.any():
        p0 = np.clip(p0, lo, hi)

    res = least_squares(
        fun, p0, jac_sparsity=S, bounds=(lo, hi), method="trf",
        loss=opts.loss, f_scale=opts.f_scale, tr_solver="lsmr",
        x_scale="jac", max_nfev=opts.max_nfev, verbose=opts.verbose,
    )
    cams, L, Xopt = unpack(res.x, cams0, prob)
    return cams, L, Xopt, res, prob


def solve(cams0, obs, opts=None, baseline_nominal=None, centre_prior=None,
          sigma_px=3.0, stages=DEFAULT_STAGES, verbose=True):
    """Staged solve: re-triangulate between passes and tighten the loss.

    A single hard-robust pass from a guessed focal length gets stuck -- the
    initial 3D points carry the focal error, the reprojection residuals are
    large everywhere, and a Huber loss then flattens the gradient that would
    have fixed it.  So we start forgiving (soft_l1, wide knee, no distortion),
    re-triangulate with the improved cameras, and only then switch to a hard
    Huber that can actually reject pose-estimator outliers.

    Jumping straight to a hard Huber is the failure mode to avoid: with every
    residual already several sigma out, the loss saturates everywhere and the
    optimiser stops on ftol having barely moved -- distortion in particular
    never leaves zero.  Three stages fix it.

    Each stage is (loss, f_scale, fit_distortion).
    """
    opts = opts or BAOptions()
    if centre_prior is None:
        centre_prior = cams0.centres
    if baseline_nominal is None:
        baseline_nominal = float(np.linalg.norm(centre_prior[1]))

    cams = cams0
    L = X = prob = res = None
    for i, (loss, f_scale, fit_dist) in enumerate(stages):
        X0, ok0 = triangulate(cams, obs.xy, obs.vis)
        if verbose:
            print(f"[stage {i+1}/{len(stages)}] loss={loss} f_scale={f_scale} "
                  f"distortion={'on' if fit_dist else 'off'} "
                  f"points={int(ok0.sum())}")
        st = BAOptions(**{**opts.__dict__})
        st.f_scale = f_scale
        st.fit_k1 = opts.fit_k1 and fit_dist
        st.fit_k2 = opts.fit_k2 and fit_dist
        st.loss = loss
        cams, L, X, res, prob = bundle_adjust(
            cams, obs, X0, ok0, opts=st, baseline_nominal=baseline_nominal,
            centre_prior=centre_prior, sigma_px=sigma_px)
        if verbose:
            e = reprojection_errors(cams, prob, X)
            print("   " + "  ".join(f"cam{c+1} rms {s['rms']:.2f}px" for c, s in e.items())
                  + f"   f = {np.round(cams.f, 0)}")
    return cams, L, X, res, prob


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def reprojection_errors(cams, prob, X):
    """Per-camera reprojection error in pixels (unweighted)."""
    out = {}
    for c in range(cams.n):
        m = prob.obs_cam == c
        if not m.any():
            continue
        pred = cams.project(X[prob.obs_pt[m]], c)
        e = np.linalg.norm(pred - prob.obs_xy[m], axis=1)
        out[c] = dict(n=int(m.sum()), rms=float(np.sqrt((e ** 2).mean())),
                      median=float(np.median(e)), p95=float(np.percentile(e, 95)))
    return out


def bone_stats(prob, X, L):
    """Realised bone length vs. the fitted constant, per shared length key."""
    dl = np.linalg.norm(X[prob.bone_a] - X[prob.bone_b], axis=1)
    out = {}
    nk = len(BONE_KEYS)
    for i, key in enumerate(BONE_KEYS):
        m = (prob.bone_len % nk) == i
        if not m.any():
            continue
        out[key] = dict(n=int(m.sum()), mean=float(dl[m].mean()),
                        std=float(dl[m].std()),
                        cv_pct=float(100 * dl[m].std() / max(dl[m].mean(), 1e-9)))
    return out


# --------------------------------------------------------------------------
# Gravity-aligned frame from foot contacts
# --------------------------------------------------------------------------

def fit_floor(X, prob, obs, joints=FOOT_JOINTS, thresh=0.03, iters=2000,
              seed=0, up_hint=None):
    """RANSAC a floor plane through the reconstructed foot keypoints.

    Returns (normal, offset, inlier_mask, points) with the normal oriented so
    that the torso lies on the positive side.
    """
    rng = np.random.default_rng(seed)
    idx = prob.pt_of[:, joints].ravel()
    idx = idx[idx >= 0]
    P = X[idx]
    if len(P) < 3:
        raise RuntimeError("not enough reconstructed foot points")

    best_n, best_d, best_in = None, None, np.zeros(len(P), bool)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        inl = np.abs(P @ n + d) < thresh
        if inl.sum() > best_in.sum():
            best_n, best_d, best_in = n, d, inl

    # Refit on inliers.
    Q = P[best_in]
    c = Q.mean(axis=0)
    _, _, Vt = np.linalg.svd(Q - c)
    n = Vt[-1]
    d = -n @ c

    # Orient "up": the torso must be on the positive side.
    torso = prob.pt_of[:, [1, 8]].ravel()
    torso = torso[torso >= 0]
    if (X[torso] @ n + d).mean() < 0:
        n, d = -n, -d
    if up_hint is not None and n @ np.asarray(up_hint) < 0:
        n, d = -n, -d
    return n, d, best_in, P


def gravity_frame(cams, normal, offset):
    """Express the rig in a floor frame: origin under camera 1, +z up.

    +y is camera 1's optical axis projected onto the floor, +x completes a
    right-handed set.  Returns a dict with per-camera position, height above
    the floor, downward tilt and roll, in degrees and metres.
    """
    up = np.asarray(normal, float)
    up /= np.linalg.norm(up)

    axis1 = cams.axes[0]
    fwd = axis1 - (axis1 @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)

    Rw = np.stack([right, fwd, up])          # world -> floor frame rotation
    c1 = cams.centres[0]
    origin = c1 - (c1 @ up + offset) * up    # foot of camera 1 on the floor

    centres = (cams.centres - origin) @ Rw.T
    axes = cams.axes @ Rw.T
    ups = (-cams.R[:, 1, :]) @ Rw.T          # camera "up" (image -y) in floor frame

    tilt = -np.degrees(np.arcsin(np.clip(axes[:, 2], -1, 1)))     # + = pointing down
    yaw = np.degrees(np.arctan2(axes[:, 0], axes[:, 1]))          # + = toward +x
    roll = np.degrees(np.arctan2(
        np.einsum("ij,ij->i", ups, np.cross(np.broadcast_to([0, 0, 1.0], axes.shape), axes)),
        ups[:, 2]))

    return dict(
        R_world_to_floor=Rw, origin=origin, up=up,
        position=centres, height=centres[:, 2],
        tilt_deg=tilt, yaw_deg=yaw, roll_deg=roll,
        plan=centres[:, :2],
    )


def summarise(cams, prob, X, L, floor=None, layout_prior=None):
    """Print the numbers you actually need to judge whether the fit is sane."""
    print("focal length / field of view")
    for c in range(cams.n):
        print(f"  cam{c+1}: f = {cams.f[c]:8.1f} px   hfov = {cams.hfov_deg()[c]:5.1f} deg"
              f"   ~{focal_35mm_equiv(cams.f[c], cams.image_size[0]):4.1f} mm equiv"
              f"   k1 = {cams.k1[c]:+.4f}  k2 = {cams.k2[c]:+.4f}")

    print("\nreprojection error (px)")
    for c, s in reprojection_errors(cams, prob, X).items():
        print(f"  cam{c+1}: rms {s['rms']:6.2f}  median {s['median']:6.2f}"
              f"  p95 {s['p95']:7.2f}   n = {s['n']}")

    print("\nbone lengths (m) -- cv is the diagnostic; want < ~2%")
    for k, s in bone_stats(prob, X, L).items():
        print(f"  {k:11s} {s['mean']:.3f} +- {s['std']:.3f}   cv {s['cv_pct']:5.2f}%   n = {s['n']}")

    print("\ncamera centres in camera-1 frame (m)")
    for c in range(cams.n):
        print(f"  cam{c+1}: {np.round(cams.centres[c], 3)}")
    b = np.linalg.norm(cams.centres[1:] - cams.centres[0], axis=1)
    print(f"  baselines from cam1: {np.round(b, 3)}")

    if floor is not None:
        print("\ngravity-aligned frame (origin on the floor below camera 1)")
        for c in range(cams.n):
            print(f"  cam{c+1}: plan {np.round(floor['plan'][c], 3)}  "
                  f"height {floor['height'][c]:5.3f} m  "
                  f"tilt {floor['tilt_deg'][c]:+6.2f} deg  "
                  f"yaw {floor['yaw_deg'][c]:+7.2f} deg  "
                  f"roll {floor['roll_deg'][c]:+6.2f} deg")
        if layout_prior is not None:
            err = np.linalg.norm(floor["plan"] - np.asarray(layout_prior), axis=1)
            print(f"  plan-view deviation from your layout: {np.round(err, 3)} m")


# --------------------------------------------------------------------------
# Data-driven initialisation via the essential matrix
# --------------------------------------------------------------------------

def init_from_essential(obs, f_est, baseline, image_size=(1920, 1080),
                        ransac_px=3.0, verbose=True):
    """Recover the rig geometry from the correspondences themselves.

    Seeding the bundle adjustment from an assumed layout bakes in whichever way
    round you believe the cameras were placed; if that reading is wrong the
    optimiser descends into a mirrored or otherwise wrong basin and never comes
    back.  This instead reads the geometry off the data:

      1. essential matrix between cameras 1 and 2, with the estimated focals,
         then recoverPose -- which resolves the four-fold sign ambiguity by
         cheirality, so it cannot hand back a mirrored rig;
      2. scale the (unit) translation to the known camera 1 - camera 2
         baseline, which makes the reconstruction metric;
      3. triangulate with that metric pair and solve PnP for camera 3.

    Only ONE external number is used -- the 1-2 baseline -- which is exactly
    the single scale degree of freedom images cannot supply.
    """
    import cv2

    C = obs.xy.shape[1]
    w, h = image_size
    cx, cy = 0.5 * w, 0.5 * h
    K = [np.array([[f_est[c], 0, cx], [0, f_est[c], cy], [0, 0, 1.0]])
         for c in range(C)]

    def norm_pts(c, p):
        return np.stack([(p[:, 0] - cx) / f_est[c], (p[:, 1] - cy) / f_est[c]], 1)

    # --- pair 1-2 ---
    m = obs.vis[:, 0, :] & obs.vis[:, 1, :]
    p1, p2 = obs.xy[:, 0][m], obs.xy[:, 1][m]
    n1, n2 = norm_pts(0, p1), norm_pts(1, p2)
    E, inl = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999,
                                  ransac_px / np.mean(f_est[:2]))
    inl = inl.ravel().astype(bool)
    ninl, R2, t2, _ = cv2.recoverPose(E, n1[inl], n2[inl], np.eye(3))
    t2 = t2.ravel()
    t2 = t2 / np.linalg.norm(t2) * baseline
    if verbose:
        print(f"  E(1,2): {inl.sum()}/{len(n1)} inliers, cheirality {ninl}")

    cams = CameraSet(
        f=np.array(f_est, float), k1=np.zeros(C), k2=np.zeros(C),
        rvec=np.zeros((C, 3)), t=np.zeros((C, 3)),
        cx=np.full(C, cx), cy=np.full(C, cy), image_size=image_size,
    )
    cams.rvec[1] = Rotation.from_matrix(R2).as_rotvec()
    cams.t[1] = t2

    # --- metric triangulation on the 1-2 pair, then PnP for the rest ---
    vis12 = obs.vis.copy()
    vis12[:, 2, :] = False
    X, ok = triangulate(cams, obs.xy, vis12)

    for c in range(2, C):
        sel = ok & obs.vis[:, c, :]
        obj = X[sel].astype(np.float64)
        img = obs.xy[:, c][sel].astype(np.float64)
        if len(obj) < 20:
            raise RuntimeError(f"camera {c+1}: too few points for PnP")
        good, rvec, tvec, pnp_in = cv2.solvePnPRansac(
            obj, img, K[c], None, reprojectionError=8.0, iterationsCount=5000,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not good:
            raise RuntimeError(f"camera {c+1}: PnP failed")
        rvec, tvec = cv2.solvePnPRefineLM(
            obj[pnp_in.ravel()], img[pnp_in.ravel()], K[c], None, rvec, tvec)
        cams.rvec[c] = rvec.ravel()
        cams.t[c] = tvec.ravel()
        if verbose:
            print(f"  PnP(cam{c+1}): {len(pnp_in)}/{len(obj)} inliers")

    if verbose:
        print("  centres from data:\n   ",
              np.array2string(cams.centres, precision=3).replace("\n", "\n    "))
    return cams


def rescale_to_anthropometry(cams, X, L, prob, keys=("thigh", "shank", "torso"),
                             target=None, verbose=True):
    """Rescale the whole solution so limb lengths match adult anthropometry.

    Images fix everything except scale, so the reconstruction is only as metric
    as the one number fed in -- here the assumed camera 1 - camera 2 baseline.
    A rough guess at that baseline is a much weaker anchor than the fact that
    the subjects are adult humans, so solve for the scale that best matches
    known limb lengths and report the baseline that implies.  If the two
    disagree, the anthropometry is the one to trust.

    Returns (cams, X, L, s) with everything scaled by s.
    """
    target = BONE_INIT if target is None else target
    nk = len(BONE_KEYS)
    dl = np.linalg.norm(X[prob.bone_a] - X[prob.bone_b], axis=1)
    num = den = 0.0
    rows = []
    for key in keys:
        i = BONE_KEYS.index(key)
        m = (prob.bone_len % nk) == i
        if not m.any():
            continue
        obs_len, want = float(dl[m].mean()), target[key]
        rows.append((key, obs_len, want, want / obs_len))
        num += obs_len * want
        den += obs_len * obs_len
    if den == 0:
        raise RuntimeError("no usable bones for rescaling")
    s = num / den

    if verbose:
        print(f"  scale factor from anthropometry: {s:.3f}")
        for key, obs_len, want, r in rows:
            print(f"    {key:8s} {obs_len:.3f} m -> {obs_len*s:.3f} m "
                  f"(expected {want:.2f}, individual ratio {r:.2f})")

    out = CameraSet(f=cams.f.copy(), k1=cams.k1.copy(), k2=cams.k2.copy(),
                    rvec=cams.rvec.copy(), t=cams.t * s,
                    cx=cams.cx.copy(), cy=cams.cy.copy(),
                    image_size=cams.image_size)
    return out, X * s, L * s, s


def subject_geometry(X, prob, floor, cams):
    """Where the subject actually stood, and how far each camera really was.

    The distances you believed (3 m / 1.8 m / 2.5 m) are checkable outputs, not
    inputs -- this is the comparison that tells you whether the rig was where
    you thought.
    """
    idx = prob.pt_of[:, [1, 8]].ravel()      # neck and mid-hip: the trunk
    idx = idx[idx >= 0]
    P = X[idx]
    Rw, org = floor["R_world_to_floor"], floor["origin"]
    Pf = (P - org) @ Rw.T
    centre = Pf.mean(axis=0)
    d = np.linalg.norm(floor["position"][:, None, :] - Pf[None, :, :], axis=2)
    return dict(centroid=centre,
                spread=Pf.std(axis=0),
                dist_mean=d.mean(axis=1),
                dist_median=np.median(d, axis=1))


def rescale_to_camera_distance(cams, X, L, prob, normal, offset, distance,
                               cam=0, mode="ground", verbose=True):
    """Pin metric scale to a measured camera-to-subject distance.

    Preferable to `rescale_to_anthropometry` whenever you actually measured
    something on site: it replaces an assumed body size (which you cannot
    check) with a tape measure (which you can).  Limb lengths then become an
    OUTPUT of the fit, and comparing them against expected values for the
    cohort is a real, independent validation rather than a tautology.

    Because a global similarity is an exact null direction of the reprojection
    cost, rescaling after the solve is mathematically identical to having
    constrained it during -- not an approximation.

    `mode` is "ground" for a tape measure along the floor (from the point below
    the camera to the target) or "slant" for the straight-line distance from
    the camera itself.  The two differ by well under a percent for this rig.

    Returns (cams, X, L, floor, s).
    """
    floor = gravity_frame(cams, normal, offset)
    sg = subject_geometry(X, prob, floor, cams)
    if mode == "ground":
        current = float(np.linalg.norm(floor["plan"][cam] - sg["centroid"][:2]))
    elif mode == "slant":
        current = float(np.linalg.norm(floor["position"][cam] - sg["centroid"]))
    else:
        raise ValueError("mode must be 'ground' or 'slant'")
    if current <= 0:
        raise RuntimeError("degenerate camera-subject distance")
    s = distance / current

    if verbose:
        print(f"  scale from camera {cam+1} -> subject = {distance:.2f} m "
              f"({mode}): factor {s:.3f}")

    out = CameraSet(f=cams.f.copy(), k1=cams.k1.copy(), k2=cams.k2.copy(),
                    rvec=cams.rvec.copy(), t=cams.t * s,
                    cx=cams.cx.copy(), cy=cams.cy.copy(),
                    image_size=cams.image_size)
    return out, X * s, L * s, gravity_frame(out, normal, offset * s), s


def implied_stature(prob, X, L, group="child_3_5"):
    """Back out the subject's height from the fitted limb lengths.

    With scale pinned to a measured distance rather than to assumed body size,
    this is a genuine prediction -- if it lands in the right range for the
    cohort, the whole reconstruction is corroborated by something that was
    never fed in.
    """
    frac = STATURE_FRACTION[group]
    nk = len(BONE_KEYS)
    dl = np.linalg.norm(X[prob.bone_a] - X[prob.bone_b], axis=1)
    est = {}
    for key in BONE_KEYS:
        i = BONE_KEYS.index(key)
        m = (prob.bone_len % nk) == i
        if m.any():
            est[key] = float(dl[m].mean()) / frac[key]
    vals = [est[k] for k in ("thigh", "shank", "torso") if k in est]
    return dict(per_bone=est, estimate=float(np.mean(vals)) if vals else np.nan)


def scale_from_distances(cams, X, prob, floor, measured, mode="ground",
                         verbose=True):
    """Least-squares scale from several measured camera-to-subject distances.

    You measured three distances but only need one to fix scale, so the other
    two are a free consistency check.  The reconstruction already fixes the
    RATIOS between them, so if the measurements disagree about those ratios no
    single scale can satisfy all three -- and the residuals say which
    measurement is the odd one out.

    Returns (s, implied_distances, residuals).
    """
    sg = subject_geometry(X, prob, floor, cams)
    if mode == "ground":
        cur = np.linalg.norm(floor["plan"] - sg["centroid"][:2], axis=1)
    else:
        cur = np.linalg.norm(floor["position"] - sg["centroid"], axis=1)
    measured = np.asarray(measured, float)
    m = np.isfinite(measured)
    s = float((cur[m] @ measured[m]) / (cur[m] @ cur[m]))
    implied = cur * s
    resid = implied - measured
    if verbose:
        print(f"  best single scale for all measurements: x{s:.3f}")
        for c in range(len(cur)):
            note = "" if not m[c] else f"  measured {measured[c]:.2f}  resid {resid[c]:+.2f}"
            print(f"    cam{c+1}: implied {implied[c]:.2f} m{note}")
    return s, implied, resid


def anchor_spread(X, prob, floor, cam=0):
    """How much does the subject move relative to the anchor distance?

    Pinning scale to "camera 1 is 3 m from the target" is only as well defined
    as the target is.  The subject moves, so report the distribution -- if the
    spread is a few percent the anchor is sound; if it is tens of percent, the
    scale inherits that.
    """
    idx = prob.pt_of[:, [1, 8]]
    Rw, org = floor["R_world_to_floor"], floor["origin"]
    d = []
    for f in range(idx.shape[0]):
        ii = idx[f][idx[f] >= 0]
        if len(ii):
            p = ((X[ii] - org) @ Rw.T).mean(axis=0)
            d.append(float(np.linalg.norm(p[:2] - floor["plan"][cam])))
    d = np.array(d)
    return dict(mean=d.mean(), median=float(np.median(d)), sd=d.std(),
                p10=float(np.percentile(d, 10)), p90=float(np.percentile(d, 90)),
                rel_sd_pct=100 * d.std() / d.mean(), per_frame=d)


PLAUSIBLE_STATURE = {"child_3_5": (0.90, 1.15)}


def check_stature(prob, X, L, group="child_3_5", verbose=True):
    """Is the implied subject height plausible for the cohort?

    With scale anchored on a measured distance, stature is a prediction.  It is
    therefore the test that tells you whether that measured distance actually
    applies to THIS session -- which matters here, because the rig demonstrably
    moved between sessions while the tape measure was taken only once.
    """
    st = implied_stature(prob, X, L, group=group)
    lo, hi = PLAUSIBLE_STATURE[group]
    ok = lo <= st["estimate"] <= hi
    if verbose:
        flag = "plausible" if ok else "IMPLAUSIBLE -- anchor likely wrong here"
        print(f"  implied stature {st['estimate']:.2f} m "
              f"(expected {lo:.2f}-{hi:.2f} for this cohort): {flag}")
        if not ok:
            mid = 0.5 * (lo + hi)
            print(f"    to reach {mid:.2f} m the anchor distance would have to be "
                  f"x{mid/st['estimate']:.2f} what you assumed")
    return dict(**st, plausible=ok)


def axis_convergence(cams):
    """The point where the cameras' optical axes come closest to meeting.

    This is the rig's aim point, and it is a much better scale anchor than the
    subject's average position: it depends only on where the cameras POINT, so
    it is completely unaffected by the subject moving, standing off-centre, or
    spending more time on one side of the capture volume.

    Least squares over the three axis lines.  Returns (point, miss) where
    `miss` is each axis's perpendicular distance from that point -- a small
    miss means the axes genuinely converge and the aim point is well defined.
    """
    C, D = cams.centres, cams.axes
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in zip(C, D):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    p = np.linalg.solve(A, b)
    miss = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - c))
                     for c, d in zip(C, D)])
    return p, miss


def rescale_to_convergence(cams, X, L, prob, normal, offset, distance,
                           cam=0, mode="slant", verbose=True):
    """Pin scale to the distance from one camera to the rig's aim point.

    "Camera 1 is pointed at a target 3 m away" describes where the camera was
    AIMED, so measure it there rather than at the subject's time-averaged
    position.  Returns (cams, X, L, floor, s, info).
    """
    p, miss = axis_convergence(cams)
    floor = gravity_frame(cams, normal, offset)
    Rw, org = floor["R_world_to_floor"], floor["origin"]
    p_f = (p - org) @ Rw.T

    if mode == "ground":
        current = float(np.linalg.norm(p_f[:2] - floor["plan"][cam]))
    elif mode == "slant":
        current = float(np.linalg.norm(p_f - floor["position"][cam]))
    else:
        raise ValueError("mode must be 'ground' or 'slant'")
    s = distance / current

    if verbose:
        print(f"  aim point at {np.round(p_f, 3)} in the floor frame")
        print(f"  axes miss it by {np.round(miss * s, 3)} m "
              f"({np.round(100 * miss / current, 1)}% of the anchor distance)")
        print(f"  scale: camera {cam+1} -> aim point = {distance:.2f} m "
              f"({mode}), factor {s:.3f}")

    out = CameraSet(f=cams.f.copy(), k1=cams.k1.copy(), k2=cams.k2.copy(),
                    rvec=cams.rvec.copy(), t=cams.t * s,
                    cx=cams.cx.copy(), cy=cams.cy.copy(),
                    image_size=cams.image_size)
    info = dict(aim_point=p_f * s, miss=miss * s,
                miss_rel_pct=100 * miss / current)
    return out, X * s, L * s, gravity_frame(out, normal, offset * s), s, info


def init_focal_from_spec(bounds, n_cams=3, where="wide"):
    """Starting focal length taken from the lens spec, not from body size.

    `estimate_focal_from_scale` needs an assumed subject height, which is an
    anthropometric input -- unwanted when scale is anchored on a measured
    distance and limb lengths are meant to be an output.  Once the camera is
    known the spec supplies a starting point directly, with no such assumption.

    `where` is "wide" (the wide end -- a good guess for a fixed rig framing a
    play area), "mid" (geometric mean of the range), or a float in [0, 1]
    interpolating between the two ends.
    """
    lo, hi = bounds
    if where == "wide":
        f = lo
    elif where == "mid":
        f = float(np.sqrt(lo * hi))
    else:
        f = lo + float(where) * (hi - lo)
    return np.full(n_cams, f)


# --------------------------------------------------------------------------
# Triangulating whole sequences with fixed, calibrated cameras
# --------------------------------------------------------------------------

def reprojection_per_view(cams, X, xy, vis):
    """Pixel error of every observed view against 3D points.

    X (F,J,3), xy (F,C,J,2), vis (F,C,J) -> err (F,C,J), NaN where the view
    did not observe the joint or the 3D point is invalid.
    """
    F, C, J, _ = xy.shape
    err = np.full((F, C, J), np.nan)
    Xf = X.reshape(-1, 3)
    good = np.isfinite(Xf).all(axis=1)
    for c in range(C):
        pred = np.full((F * J, 2), np.nan)
        if good.any():
            pred[good] = cams.project(Xf[good], c)
        e = np.linalg.norm(pred.reshape(F, J, 2) - xy[:, c], axis=-1)
        err[:, c] = np.where(vis[:, c], e, np.nan)
    return err


def _max_over_views(err):
    """Max over the view axis, ignoring NaN; +inf where no view observed."""
    m = np.where(np.isnan(err), -np.inf, err).max(axis=1)
    m[np.isneginf(m)] = np.inf
    return m


def triangulate_robust(cams, xy, vis, thresh_px=15.0):
    """Triangulate every joint, dropping one view when it disagrees.

    With three views of a joint there is one view of redundancy, so a single
    bad detection -- most often a left/right swap in one camera -- shows up as
    a large reprojection error.  Where the all-views solution misses the
    threshold, re-triangulate from each pair and keep the best pair if it
    passes.  Two-view joints get no second chance (a pair cannot outvote
    itself) but are held to the same threshold.

    Returns
      X     (F,J,3)  NaN where invalid
      ok    (F,J)
      used  (F,C,J)  which views the final point was triangulated from
      err   (F,C,J)  reprojection error in each used view, NaN elsewhere
    """
    F, C, J, _ = xy.shape
    X, ok = triangulate(cams, xy, vis)
    err = reprojection_per_view(cams, X, xy, vis)
    worst = _max_over_views(err)

    best_X, best_worst = X.copy(), worst.copy()
    best_used, best_err = vis.copy(), err.copy()

    retry = (vis.sum(axis=1) >= 3) & ~(ok & (worst <= thresh_px))
    if retry.any():
        for drop in range(C):
            v = vis & retry[:, None, :]
            v[:, drop, :] = False
            Xp, okp = triangulate(cams, xy, v)
            ep = reprojection_per_view(cams, Xp, xy, v)
            wp = _max_over_views(ep)
            better = retry & okp & (wp < best_worst)
            best_X[better] = Xp[better]
            best_worst[better] = wp[better]
            best_used = np.where(better[:, None, :], v, best_used)
            best_err = np.where(better[:, None, :], ep, best_err)

    ok = (np.isfinite(best_X).all(axis=-1) & (best_worst <= thresh_px)
          & (best_used.sum(axis=1) >= 2))
    best_X[~ok] = np.nan
    best_used &= ok[:, None, :]
    best_err = np.where(best_used, best_err, np.nan)
    return best_X, ok, best_used, best_err
