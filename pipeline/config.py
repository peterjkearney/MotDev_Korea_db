"""Shared paths for the PnP_depth pipeline.

Every step imports TRIAL_DIR from here instead of hardcoding the dataset root,
so moving the data (local disk vs Drive vs another machine) is one env var and
never an edit to a tracked file.

Colab:  %env BIOCV_ROOT=/content/data/BioCV
"""
import os

# Dataset root: {TRIAL_DIR}/{user}/{action}/...
# On Colab keep this on local disk -- step_1 reads video, which is painfully
# slow over the Drive FUSE mount.
TRIAL_DIR = os.environ.get('BIOCV_ROOT', '/content/data/BioCV')

# Where finished results are kept. Deliberately NOT inside TRIAL_DIR: local
# disk dies with the runtime, and keeping results out of the input tree means
# a re-run can never half-overwrite a good previous result.
RESULTS_DIR = os.environ.get(
    'BIOCV_RESULTS', '/content/drive/MyDrive/MotorDevelopment/results')

# Where every step's OUTPUT is kept: the original BioCV folder on Drive, under
# {OUT_DIR}/{user}/{action}/Analysis/.  Inputs (videos, c3d, calibs) are read
# from TRIAL_DIR, the fast local copy unzipped on the VM; nothing is written
# there, so nothing needs syncing and nothing dies with the runtime.
OUT_DIR = os.environ.get('BIOCV_OUT', '/content/drive/MyDrive/MotorDevelopment/Data/BioCV')


def analysis_dir(user, action):
    return os.path.join(OUT_DIR, user, action, 'Analysis')


# One function per output: the only place a file's location is decided, used by
# the step that writes it and by every step that reads it.
def mocap_path(user, action):
    """step_0: mocap as H36M-17 joints, native rate."""
    return os.path.join(analysis_dir(user, action), 'H36M', 'mocap_h36m.npz')


def twod_path(user, action, detector, cam):
    """2D keypoints as H36M-17, one folder per detector so neither overwrites the other:
    'yolo' (step_1_extract_2d) or 'openpose' (step_1_openpose_2d)."""
    return os.path.join(analysis_dir(user, action), 'keypoints', detector, f'{cam}_2d.npz')


def tri_target_path(user, action):
    """step_1b: triangulated-OpenPose ground truth, mocap_h36m.npz's layout (leave-one-out per camera)."""
    return os.path.join(analysis_dir(user, action), 'H36M', 'openpose_tri_h36m.npz')


def tri_vs_mocap_path(user, action):
    """step_1b: how far the triangulated target sits from mocap, per joint and camera."""
    return os.path.join(analysis_dir(user, action), 'H36M', 'tri_vs_mocap.npz')


def openpose_json_path(user, action, cam):
    """tools/run_openpose.py: OpenPose's BODY_25 output for one video -- every frame and every
    person in ONE json, with the native frame number of each entry (utils/openpose.py)."""
    return os.path.join(analysis_dir(user, action), 'keypoints', 'openpose', f'{cam}_openpose.json')


DETECTORS = ('openpose', 'yolo')


def betas_path(user, action, detector, cam):
    """step_2a: MotionBERT pass 1 on one detector's 2D -- per-frame SMPL rotations and betas."""
    return os.path.join(analysis_dir(user, action), 'mesh', detector, f'{cam}_betas.npz')


def require_out_dir():
    """A path under /content/drive with Drive not mounted is silently created on the
    VM's own disk and lost with the runtime -- stop instead."""
    if os.path.abspath(OUT_DIR).startswith('/content/drive') and not os.path.ismount('/content/drive'):
        raise SystemExit(f"OUT_DIR {OUT_DIR} is on Google Drive but Drive is not mounted -- run "
                         f"drive.mount('/content/drive'), or set BIOCV_OUT")
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'OUT_DIR {OUT_DIR} does not exist -- set BIOCV_OUT to the BioCV folder on Drive')


# This is a working copy of Python_jetson/PnP_depth_clean, moved into Korea_db
# so it can be amended for the child study without touching the original.
# MotionBERT and the model weights stay where they were; the original
# resolved them as ../MotionBERT relative to the script, which no longer
# holds here, so point at them explicitly (override with env vars).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find(name, env):
    """First existing candidate: env var, the original's ../<name> next to
    this folder (the Colab layout), or the Mac layout under Python_jetson."""
    cands = [os.environ.get(env)] + [
        os.path.abspath(os.path.join(_SCRIPT_DIR, *rel, name))
        for rel in (('..',), ('..', '..', 'Python_jetson'), ('..', '..'))]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[1]      # ../<name>: what the error message should point at


MB_DIR     = _find('MotionBERT', 'MB_DIR')
MODELS_DIR = _find('models', 'MODELS_DIR')
