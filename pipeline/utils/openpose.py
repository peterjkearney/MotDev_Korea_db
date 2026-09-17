"""openpose.py -- OpenPose BODY_25 handling shared by both work streams.

BODY_25 -> H36M-17 conventions here are the SAME as step_1_extract_2d.py's
coco2h36m (pelvis = mid-hip, thorax = mid-shoulder, spine = mid(pelvis,
thorax), head = mid-ear), so a MotionBERT run on OpenPose input is directly
comparable to one on YOLO input.  Korea_db/build_child_gt.py carries the same
two conversions; keep them in step if either changes.

The one deliberate difference from coco2h36m: OpenPose writes a missing joint
as (0, 0) with confidence 0, which YOLO never does.  Averaging that into a
midpoint would put the pelvis halfway to the image corner at about half the
present part's confidence -- enough to pass step_4's PnP gate of 0.4.  So a
composite joint with ANY missing part is itself written as missing (0, 0, 0).
MotionBERT's crop_scale only ignores those for its bounding box -- it still
normalises them into the corner of the crop -- so step_2a fills them before
lifting (utils/missing_joints.py).
"""
import glob
import json
import os
import re

import numpy as np

BODY25_NAMES = [
    'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow',
    'LWrist', 'MidHip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel',
    'RBigToe', 'RSmallToe', 'RHeel',
]

H36M_NAMES = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
              'Spine', 'Thorax', 'Nose', 'Head',
              'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

# H36M joint -> the BODY_25 joints it is built from (mean of them)
H36M_FROM_BODY25 = {0: [9, 12], 1: [9], 2: [10], 3: [11], 4: [12], 5: [13], 6: [14],
                    8: [2, 5], 9: [0], 10: [17, 18],
                    11: [5], 12: [6], 13: [7], 14: [2], 15: [3], 16: [4]}
# spine (7) = mid(pelvis, thorax), done after the table above

# H36M limb joints with a direct BODY_25 counterpart -- use these for any
# per-joint or per-bone comparison; the rest are synthesised midpoints
H36M_CORE = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]

# (a, b, key): bones for length/proportion checks, left/right sharing a key
BODY25_BONES = [
    (1, 8, 'torso'),
    (1, 2, 'clavicle'), (1, 5, 'clavicle'),
    (2, 3, 'upperarm'), (5, 6, 'upperarm'),
    (3, 4, 'forearm'), (6, 7, 'forearm'),
    (8, 9, 'hip_offset'), (8, 12, 'hip_offset'),
    (9, 10, 'thigh'), (12, 13, 'thigh'),
    (10, 11, 'shank'), (13, 14, 'shank'),
]


def present_mask(xy, conf):
    """A BODY_25 joint is detected iff conf > 0 and it is not the (0, 0) sentinel."""
    return (conf > 0) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))


def body25_to_h36m_2d(xy, conf):
    """(T,25,2), (T,25) -> (T,17,3) x, y, conf -- the array step_1 saves."""
    T = xy.shape[0]
    present = present_mask(xy, conf)
    out = np.zeros((T, 17, 3), np.float32)
    have = np.zeros((T, 17), bool)
    for k, idx in H36M_FROM_BODY25.items():
        out[:, k, :2] = xy[:, idx].mean(axis=1)
        out[:, k, 2] = conf[:, idx].mean(axis=1)
        have[:, k] = present[:, idx].all(axis=1)
    out[:, 7] = 0.5 * (out[:, 0] + out[:, 8])
    have[:, 7] = have[:, 0] & have[:, 8]
    out[~have] = 0.0
    return out


def body25_to_h36m_3d(P, ok):
    """(N,25,3), (N,25) valid -> (N,17,3) NaN where invalid, (N,17) valid.

    Midpoints commute with rigid transforms, so this can be applied in any
    frame.  A composite joint is valid only if all its parts are.
    """
    N = P.shape[0]
    out = np.full((N, 17, 3), np.nan)
    valid = np.zeros((N, 17), bool)
    for k, idx in H36M_FROM_BODY25.items():
        out[:, k] = P[:, idx].mean(axis=1)
        valid[:, k] = ok[:, idx].all(axis=1)
    out[:, 7] = 0.5 * (out[:, 0] + out[:, 8])
    valid[:, 7] = valid[:, 0] & valid[:, 8]
    out[~valid] = np.nan
    return out, valid


# ---------------------------------------------------------------------------
# Reading OpenPose --write_json output
# ---------------------------------------------------------------------------

_FRAME_RE = re.compile(r'(\d+)_keypoints\.json$')


def list_openpose_jsons(json_dir):
    """{frame number: path} for one video's OpenPose JSON directory.

    OpenPose names files {prefix}_{frame:012d}_keypoints.json, frame from 0.
    """
    out = {}
    for p in glob.glob(os.path.join(json_dir, '*_keypoints.json')):
        m = _FRAME_RE.search(os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def read_openpose_frame(path):
    """All people in one OpenPose JSON: xy (P,25,2), conf (P,25)."""
    with open(path) as f:
        d = json.load(f)
    people = d.get('people', [])
    if not people:
        return np.zeros((0, 25, 2), np.float32), np.zeros((0, 25), np.float32)
    kp = np.array([p['pose_keypoints_2d'] for p in people], np.float32).reshape(-1, 25, 3)
    return kp[:, :, :2], kp[:, :, 2]


# One JSON per video.  OpenPose itself writes one file per frame; hundreds of tiny files per
# camera are slow on Drive, and say nothing about WHICH video frame each one is -- they are
# numbered by the video OpenPose was given, which tools/run_openpose.py decimates first.
# Reading them by the wrong numbering once desynchronised every OpenPose result.  The
# per-video file carries `native_frame_idx`, so there is nothing left to guess:
#
#   {"format": "openpose_video_v1", "model": "BODY_25", "video": "06.mp4",
#    "video_fps": 200.0, "video_n_frames": 1253, "target_fps": 60,
#    "native_frame_idx": [0, 3, 7, ...],               entry i is this frame of the source video
#    "frames": [{"people": [{"pose_keypoints_2d": [x, y, c, ... 75 numbers]}, ...]}, ...]}
OPENPOSE_VIDEO_FORMAT = 'openpose_video_v1'


def consolidate_openpose_dir(json_dir, native_frame_idx, **meta):
    """OpenPose's per-frame JSON dir -> the one-file-per-video document above.

    Frames are taken in the order of OpenPose's own numbering.  `native_frame_idx` may be
    one longer than the JSONs (OpenPose stops at the last decodable frame, which can be one
    short of the video's frame count); any other mismatch is an error."""
    files = list_openpose_jsons(json_dir)
    n = len(files)
    idx = [int(i) for i in native_frame_idx]
    if not (n > 0 and n <= len(idx) <= n + 1):
        raise ValueError(f'{json_dir}: {n} JSONs but {len(idx)} frame indices')
    frames = []
    for k in sorted(files):
        with open(files[k]) as f:
            people = json.load(f).get('people', [])
        frames.append({'people': [{'pose_keypoints_2d': p['pose_keypoints_2d']} for p in people]})
    return dict(format=OPENPOSE_VIDEO_FORMAT, model='BODY_25', **meta,
                native_frame_idx=idx[:n], frames=frames)


def save_openpose_video(path, doc):
    """Written to a temp name and renamed, so an interrupted write never looks complete."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(doc, f, separators=(',', ':'))
    os.replace(tmp, path)


def load_openpose_video(path):
    """(doc, people): the document, and per frame (xy (P,25,2), conf (P,25)) for its P people."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get('format') != OPENPOSE_VIDEO_FORMAT:
        raise ValueError(f'{path}: not an {OPENPOSE_VIDEO_FORMAT} file')
    people = []
    for fr in doc['frames']:
        kp = np.array([p['pose_keypoints_2d'] for p in fr['people']], np.float32).reshape(-1, 25, 3)
        people.append((kp[:, :, :2], kp[:, :, 2]))
    return doc, people


def pick_person(xy_people, conf_people, ref_xy, ref_ok, conf_thresh=0.3,
                penalty_px=960.0, min_joints=4):
    """Index of the OpenPose person best matching a reference 2D skeleton.

    Both sides in H36M-17 order.  Same scoring as step_1's
    find_closest_user_to_mocap, per frame: mean distance over the reference's
    valid core joints, with a missing/low-confidence detection charged
    `penalty_px`.  Returns (index or None, score).
    """
    if len(xy_people) == 0:
        return None, np.inf
    core = np.array(H36M_CORE)
    ok = ref_ok[core]
    if ok.sum() < min_joints:
        return None, np.inf
    best, best_s = None, np.inf
    for i in range(len(xy_people)):
        h = body25_to_h36m_2d(xy_people[i:i + 1], conf_people[i:i + 1])[0]
        det = h[core, 2] > conf_thresh
        d = np.linalg.norm(h[core, :2] - ref_xy[core], axis=1)
        d = np.where(det, np.minimum(d, penalty_px), penalty_px)
        s = float(d[ok].mean())
        if s < best_s:
            best, best_s = i, s
    return best, best_s
