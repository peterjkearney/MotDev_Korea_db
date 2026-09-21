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


def final_betas_path(user, action, detector, cam):
    """step_2b: the one body shape (10 betas) this camera's skeleton is built with."""
    return os.path.join(analysis_dir(user, action), 'mesh', detector, f'{cam}_final_betas.npz')


def mesh_pose_path(user, action, detector, cam):
    """step_3: H36M-17 and SMPL-24 joints per frame from SMPL (fixed betas + pass-1 rotations),
    scaled to the subject's stature, in the mesh's own camera-facing frame (not yet placed)."""
    return os.path.join(analysis_dir(user, action), 'mesh', detector, f'{cam}_mesh_pose.npz')


def pnp_path(user, action, detector, cam):
    """step_4: the skeleton placed in this camera's frame by PnP (per frame) + the smoothed copy."""
    return os.path.join(analysis_dir(user, action), 'PnP', detector, f'{cam}_pnp.npz')


def features_path(user, action, detector, cam):
    """step_6: joint angles, heights and velocities in the lab frame, per frame."""
    return os.path.join(analysis_dir(user, action), 'features', detector, f'{cam}_features.npz')


def metrics_path(user, action, detector, gt='mocap'):
    """step_8: per-camera error metrics against mocap ('mocap') or the triangulated target ('triangulated')."""
    return os.path.join(analysis_dir(user, action), 'diagnostics', detector,
                        'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz')


def spider_path(user, action, detector, gt='mocap'):
    """step_8: the error-by-camera chart that goes with metrics_path."""
    return os.path.join(analysis_dir(user, action), 'diagnostics', detector,
                        f'spider_error_{action}{"" if gt == "mocap" else "_tri"}.png')


def depth_video_path(user, action, cam):
    """step_5: the video with both detectors' smoothed skeletons + the top-down comparison against
    mocap and the triangulated target."""
    return os.path.join(analysis_dir(user, action), 'diagnostics', f'{cam}_pnp_depth_vs_mocap.mp4')


def openpose_check_path(user, action, cam):
    """tools/render_openpose_check.py: the video with OpenPose's people, confidences and the projected mocap."""
    return os.path.join(analysis_dir(user, action), 'diagnostics', 'openpose', f'{cam}_openpose_check.mp4')


def calib_path(user, cam):
    """{cam}.mp4-mocAligned.calib: with the local copy of the data, else on Drive."""
    for root in (TRIAL_DIR, OUT_DIR):
        p = os.path.join(root, user, f'{cam}.mp4-mocAligned.calib')
        if os.path.exists(p):
            return p
    return os.path.join(TRIAL_DIR, user, f'{cam}.mp4-mocAligned.calib')


def stature_path(user):
    """user_meta.json ({"stature_m": ...}): with the local copy of the data, else on Drive."""
    for root in (TRIAL_DIR, OUT_DIR):
        p = os.path.join(root, user, 'user_meta.json')
        if os.path.exists(p):
            return p
    return os.path.join(TRIAL_DIR, user, 'user_meta.json')


def find_cameras(path_fn, detector, user=None, action=None, cameras=None):
    """[(user, action, cam)] for every file `path_fn(user, action, detector, cam)` under OUT_DIR,
    narrowed by --user / --action / --cameras when given.  How the batch steps discover work."""
    import glob
    out = []
    for p in sorted(glob.glob(path_fn(user or '*', action or '*', detector, '*'))):
        suffix = os.path.basename(path_fn('u', 'a', detector, ''))          # e.g. '_2d.npz'
        cam = os.path.basename(p)[:-len(suffix)]
        if '_' in cam:        # '*_betas.npz' also matches '00_final_betas.npz'; camera names have no '_'
            continue
        a = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p))))   # .../{user}/{action}
        if cameras is None or cam in cameras:
            out.append((os.path.basename(os.path.dirname(a)), os.path.basename(a), cam))
    return out


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


def require_motionbert():
    """Stop with directions if MotionBERT is not where MB_DIR says, rather than dying on
    `No module named 'lib'` when its package is imported."""
    need = [os.path.join(MB_DIR, 'lib'),
            os.path.join(MB_DIR, 'configs', 'mesh', 'MB_ft_pw3d.yaml'),
            os.path.join(MB_DIR, 'checkpoint', 'mesh', 'FT_MB_release_MB_ft_pw3d', 'best_epoch.bin'),
            os.path.join(MB_DIR, 'data', 'mesh')]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        env = os.environ.get('MB_DIR')
        why = ('MB_DIR is not set in this session (a %env setting is lost when the runtime restarts), so the '
               'default location next to the repo was tried' if not env else
               f'MB_DIR is set to {env!r}, which is not a folder, so the default location was tried instead'
               if not os.path.isdir(env) else f'MB_DIR = {env!r}')
        raise SystemExit('MotionBERT not found at ' + MB_DIR + ' -- missing:\n  ' + '\n  '.join(missing)
                         + f'\n{why}.\nSet it to the MotionBERT folder '
                           '(Colab: %env MB_DIR=/content/drive/MyDrive/.../MotionBERT)')
