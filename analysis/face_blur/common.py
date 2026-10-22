"""common.py -- paths, the trial, and the helpers every face-blur step shares.

The study: how much does blurring faces before keypoint detection (as the
ethics protocol requires) move 2D keypoints, and how much of that survives
into lifted and triangulated 3D?  One BioCV trial (P03_CMJM_01, 4 of its 9 cameras,
mocap), decimated to 25 fps (see CAMS and TARGET_FPS below).

Conditions (the frames each detector sees):
  raw     the original video's frames, decoded directly
  reenc   raw re-encoded with the same x264 settings as the blurred videos --
          the baseline every blur is compared against, so compression is not
          counted as blur error
  haar    the ethics proposal's method: OpenCV frontal-face Haar cascade
          (Viola & Jones) + cv2.medianBlur over each detected box
  deface  `deface` 1.5 defaults: CenterFace CNN, threshold 0.2, box blur in an
          ellipse at 1.3x the face box
  oracle  deface's blur over the projected head (from triangulated clean face
          keypoints) in every frame -- what perfect face detection would give

Outputs live on the external SSD (the Mac's disk is nearly full); FACEBLUR_OUT
overrides.  Layout under {OUT_ROOT}/{action}/:
  videos/{cond}/{cam}.mp4         the condition's TARGET_FPS video (not raw)
  faces/{method}/{cam}.npz        the anonymiser's blur regions per frame
  2d/{cond}/{det}/{cam}.npz       the subject's COCO-17 keypoints per frame
  head.npz                        triangulated head, for the oracle and recall
  lift/{cond}/{det}/{cam}.npz     every lifter's (T,17,3) output
  tri/{cond}/{det}.npz            all-camera triangulation, H36M-17, mm
  report/                         tables and figures
"""
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(REPO, 'pipeline'))      # utils/

from utils.calibration import load_calib                 # noqa: E402
from utils.frame_decimation import nearest_frame_indices  # noqa: E402
from utils.triangulate import project                     # noqa: E402

TRIAL_ROOT = os.environ.get('FACEBLUR_TRIAL_ROOT', os.path.join(REPO, 'data', 'sampleBiocv'))
OUT_ROOT = os.environ.get('FACEBLUR_OUT', '/Volumes/Sandisk_SSD/MotorDevelopment/FaceBlur')
PYTHON_JETSON = os.path.join(os.path.dirname(REPO), 'Python_jetson')

ALL_CAMS = [f'{c:02d}' for c in range(9)]
# A subset for time: Haar found the face in 00/03/07 but not 05 in a probe, MediaPipe kept to the
# subject in all four (it locked on to someone else in 04/06/08).  FACEBLUR_CAMS overrides.
CAMS = os.environ.get('FACEBLUR_CAMS', '00 03 05 07').split()
# 25 fps for time; the temporal lifters were trained at H36M's 50, so absolute lifting error may rise a
# little, but equally in every condition.  FACEBLUR_FPS overrides.
TARGET_FPS = float(os.environ.get('FACEBLUR_FPS', 25))
CONDITIONS = ('raw', 'reenc', 'haar', 'deface', 'oracle')
BLURS = ('haar', 'deface', 'oracle')
DETECTORS = ('yolo', 'rtmpose', 'mediapipe')
BASELINE = 'reenc'

COCO_NAMES = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder',
              'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
              'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
FACE = [0, 1, 2, 3, 4]
H36M_NAMES = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax',
              'Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
# H36M limb joint -> COCO index, for matching a detection to the projected mocap
H36M_TO_COCO_LIMBS = {1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15,
                      11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}


def out_dir(action, *parts):
    return os.path.join(OUT_ROOT, action, *parts)


def video_path(action, cam, cond):
    if cond == 'raw':
        return os.path.join(TRIAL_ROOT, action, f'{cam}.mp4')
    return out_dir(action, 'videos', cond, f'{cam}.mp4')


def faces_path(action, method, cam):
    return out_dir(action, 'faces', method, f'{cam}.npz')


def det_path(action, cond, det, cam):
    return out_dir(action, '2d', cond, det, f'{cam}.npz')


def head_path(action):
    return out_dir(action, 'head.npz')


def lift_path(action, cond, det, cam):
    return out_dir(action, 'lift', cond, det, f'{cam}.npz')


def tri_path(action, cond, det):
    return out_dir(action, 'tri', cond, f'{det}.npz')


class Camera:
    def __init__(self, path):
        self.w, self.h, self.K, L, self.dist = load_calib(path)
        self.R, self.t = L[:3, :3], L[:3, 3]
        self.centre = -self.R.T @ self.t                    # lab mm

    def project(self, X):
        """(..., 3) lab mm -> (..., 2) px, NaN where missing or behind the camera."""
        X = np.asarray(X, float)
        flat = X.reshape(-1, 3)
        uv = project(flat, self.K, self.dist, self.R, self.t)
        z = flat @ self.R.T[:, 2] + self.t[2]
        uv[~(z > 0)] = np.nan
        return uv.reshape(*X.shape[:-1], 2)

    def depth(self, X):
        X = np.asarray(X, float)
        return X @ self.R.T[:, 2] + self.t[2]

    def as_tuple(self):
        return self.K, self.dist, self.R, self.t


class Trial:
    """One BioCV trial at TARGET_FPS: the kept native frames, mocap, cameras."""

    def __init__(self, action, root=TRIAL_ROOT):
        self.action = action
        self.dir = os.path.join(root, action)
        m = np.load(os.path.join(self.dir, 'Analysis', 'H36M', 'mocap_h36m.npz'))
        # mocap is 1247 frames, the videos ~1253: cap at the mocap so frame N is the same instant everywhere
        self.frame_idx = nearest_frame_indices(len(m['kps3d']), float(m['fps']), TARGET_FPS)
        self.mocap = m['kps3d'][self.frame_idx].astype(float)        # (T,17,3) lab mm, NaN missing
        self.mocap_mask = np.asarray(m['valid_joint_mask'], bool)     # Nose/Head have no markers
        self.cams = {c: Camera(self._calib(c)) for c in CAMS}
        self.T = len(self.frame_idx)

    def _calib(self, cam):
        for d in (self.dir, os.path.dirname(self.dir)):
            p = os.path.join(d, f'{cam}.mp4-mocAligned.calib')
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f'{cam}.mp4-mocAligned.calib near {self.dir}')

    def mocap2d(self, cam):
        return self.cams[cam].project(self.mocap)                    # (T,17,2)


def iter_frames(trial, cam, cond):
    """BGR frames of one camera at TARGET_FPS, one per trial.frame_idx entry.
    raw decodes the original and keeps the trial's frames; every other
    condition's video was written at TARGET_FPS, so all its frames are kept."""
    cap = cv2.VideoCapture(video_path(trial.action, cam, cond))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {video_path(trial.action, cam, cond)}')
    try:
        if cond == 'raw':
            keep = set(trial.frame_idx.tolist())
            for i in range(int(trial.frame_idx[-1]) + 1):
                if i in keep:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame
                elif not cap.grab():
                    break
        else:
            for _ in range(trial.T):
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
    finally:
        cap.release()


def pick_subject(cands, mocap2d_f, max_px=150.0, conf_thr=0.3, min_joints=4):
    """Choose the detection (N,17,3 COCO) closest to the projected mocap of one
    frame ((17,2) H36M).  Returns (index or -1, median px distance)."""
    best, best_d = -1, np.inf
    for i, kp in enumerate(cands):
        d = []
        for h, c in H36M_TO_COCO_LIMBS.items():
            if kp[c, 2] >= conf_thr and np.isfinite(mocap2d_f[h]).all():
                d.append(np.linalg.norm(kp[c, :2] - mocap2d_f[h]))
        if len(d) >= min_joints and np.median(d) < best_d:
            best, best_d = i, float(np.median(d))
    if best_d > max_px:
        return -1, best_d
    return best, best_d


def coco_to_h36m(coco):
    """(T,17,3) COCO x, y, conf -> (T,17,3) H36M, the pipeline's step_1 convention
    (Head = mid-ear, Nose = nose, composites average their parts' confidence)."""
    c = np.asarray(coco, np.float32)
    h = np.zeros_like(c)
    mid = lambda a, b: (c[:, a] + c[:, b]) * 0.5          # noqa: E731
    h[:, 0] = mid(11, 12)
    h[:, 1], h[:, 2], h[:, 3] = c[:, 12], c[:, 14], c[:, 16]
    h[:, 4], h[:, 5], h[:, 6] = c[:, 11], c[:, 13], c[:, 15]
    h[:, 8] = mid(5, 6)
    h[:, 7] = (h[:, 0] + h[:, 8]) * 0.5
    h[:, 9] = c[:, 0]
    h[:, 10] = mid(3, 4)
    h[:, 11], h[:, 12], h[:, 13] = c[:, 5], c[:, 7], c[:, 9]
    h[:, 14], h[:, 15], h[:, 16] = c[:, 6], c[:, 8], c[:, 10]
    return h


def procrustes_fit(P, G):
    """Similarity (s, R, t) taking P (n,3) onto G (n,3), reflection-free --
    utils/metrics._procrustes's maths, returned so it can move other joints too."""
    mp, mg = P.mean(0), G.mean(0)
    P0, G0 = P - mp, G - mg
    U, S, Vt = np.linalg.svd(P0.T @ G0)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (P0 ** 2).sum()
    return s, R, mg - s * mp @ R


def write_video(path, fps, w, h, crf=18):
    """An ffmpeg x264 pipe taking BGR frames on .stdin -- the same settings for
    every condition, so re-encoding is identical across them."""
    import subprocess
    import imageio_ffmpeg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)
