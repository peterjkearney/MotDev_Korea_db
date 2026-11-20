#!/usr/bin/env python3
"""bp_extract.py -- MediaPipe BlazePose 3D on exactly the frames our pipeline scores, cached per camera.

For each trial's chosen cameras ({BIOCV_OUT}/bp_cameras.json, datasets/biocv/bp_views.py) BlazePose runs over
the source frames in the 30 fps OpenPose 2d.npz (source_frame_idx: native 200 Hz video frames), in video mode
(tracking + landmark smoothing, the way it is used live): heavy model, model_complexity=2, as
blazepose_vs_ours.py and the face-blur study ran it.

--crop ('subject given'): each frame is first cropped to the OpenPose subject our side used (keypoint box + 25%
per side, smoothed), so BlazePose gets the same identity information; cached as {cam}_mp_crop.npz.

Writes {BIOCV_OUT}/{user}/{action}/Analysis/blazepose/{cam}_mp.npz:
    world        (T,17,3) H36M-17 mm from the 33 world landmarks (hip-centred, camera-like axes: x right, y down,
                 z away); NaN where BlazePose found nobody
    world33      (T,33,4) raw world landmarks (x, y, z m, visibility)
    image33      (T,33,4) normalised image landmarks (x, y, z, visibility)
    source_frame_idx, fps, mediapipe version, model_complexity

--api tasks (default): PoseLandmarker heavy (pose_landmarker_heavy.task, downloaded to models/ on first use) --
mediapipe 0.10.30+ (Colab) has only this API.  --api solutions: the legacy API (mediapipe <= 0.10.21, the Mac's
motEnv), which the 2026-10-27 P03 numbers used.

    python3 analysis/blazepose/bp_extract.py --trials User04/P04_CMJM_01 [--cameras 00,05] [--force]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(REPO, 'pipeline'))
from config import OUT_DIR, TRIAL_DIR, analysis_dir, twod_path  # noqa: E402

MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]   # detectors.py's map
COMPLEXITY = 2


def coco_to_h36m(c):
    """(T,17,k) COCO -> (T,17,k) H36M, the pipeline's step_1 convention (Head = mid-ear) -- face_blur/common.py's."""
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


def mp_path(user, action, cam):
    return os.path.join(analysis_dir(user, action), 'blazepose', f'{cam}_mp.npz')


def find_video(user, action, cam):
    for root in (TRIAL_DIR, OUT_DIR):
        p = os.path.join(root, user, action, f'{cam}.mp4')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{cam}.mp4 under {TRIAL_DIR}/{user}/{action} or {OUT_DIR}/{user}/{action}')


TASK_URL = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/'
            'pose_landmarker_heavy.task')
TASK_MODEL = os.path.join(REPO, 'models', 'pose_landmarker_heavy.task')       # models/ is git-ignored


def _frames(video, frames):
    """Yield (row, native frame number, RGB image) for the wanted native frames, in order."""
    want = {int(f): i for i, f in enumerate(frames)}
    cap = cv2.VideoCapture(video)
    for f in range(int(frames[-1]) + 1):
        if f not in want:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        yield want[f], f, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap.release()


def subject_boxes(d2, pad=0.25, smooth=5):
    """(T,4) x0, y0, x1, y1 pixel crop per row around the OpenPose subject step_1_openpose_2d selected (the identity
    our side is given): keypoint extent grown by `pad` of its size on every side, gaps interpolated, ends held,
    moving average over `smooth` rows so the crop does not jump."""
    b25 = d2['body25_2d'].astype(float)
    T = len(b25)
    box = np.full((T, 4), np.nan)
    for i in range(T):
        k = b25[i][b25[i, :, 2] > 0.1, :2]
        if len(k) >= 4:
            box[i] = [k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()]
    ok = np.isfinite(box[:, 0])
    if not ok.any():
        return None
    t = np.arange(T)
    box = np.stack([np.interp(t, t[ok], box[ok, j]) for j in range(4)], 1)
    if smooth > 1:
        kern = np.ones(smooth) / smooth
        box = np.stack([np.convolve(np.pad(box[:, j], smooth // 2, mode='edge'), kern, 'valid') for j in range(4)], 1)
    c, h = (box[:, :2] + box[:, 2:]) / 2, (box[:, 2:] - box[:, :2]) / 2 * (1 + 2 * pad)
    return np.concatenate([c - h, c + h], 1)


def _crop(rgb, box):
    """Crop to box (clipped to the image) -> (crop, x0, y0)."""
    H, W = rgb.shape[:2]
    x0, y0 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x1, y1 = int(min(W, np.ceil(box[2]))), int(min(H, np.ceil(box[3])))
    return np.ascontiguousarray(rgb[y0:y1, x0:x1]), x0, y0


def run_blazepose(video, frames, api, video_fps, boxes=None):
    """BlazePose heavy, video mode, over the given native frames -> (world33, image33, version string).

    api 'tasks': PoseLandmarker (mediapipe >= 0.10; the only API from 0.10.30 on), timestamps = native frame / video
    fps.  api 'solutions': the legacy mp.solutions.pose (<= 0.10.21), what the 2026-10-27 P03 numbers used.
    boxes (T,4): crop every frame to its box first; image landmarks are mapped back to the full frame (the world
    landmarks are hip-centred, so a crop does not move them)."""
    import mediapipe as mp
    T = len(frames)
    w33 = np.full((T, 33, 4), np.nan, np.float32)
    i33 = np.full((T, 33, 4), np.nan, np.float32)
    off = np.zeros((T, 4))                          # per row: x0, y0, crop w, crop h -- then image size

    def prep(i, rgb):
        H, W = rgb.shape[:2]
        if boxes is None:
            off[i] = (0, 0, W, H)
            return rgb, W, H
        c, x0, y0 = _crop(rgb, boxes[i])
        off[i] = (x0, y0, c.shape[1], c.shape[0])
        return c, W, H

    def full_frame(i, lms, W, H):
        a = np.array(lms, np.float32)
        a[:, 0] = (a[:, 0] * off[i, 2] + off[i, 0]) / W
        a[:, 1] = (a[:, 1] * off[i, 3] + off[i, 1]) / H
        return a

    if api == 'solutions':
        if not hasattr(mp, 'solutions'):
            raise SystemExit(f'mediapipe {mp.__version__} has no legacy solutions API -- use --api tasks')
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=COMPLEXITY, smooth_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
        for i, _, rgb in _frames(video, frames):
            img, W, H = prep(i, rgb)
            r = pose.process(img)
            if r.pose_world_landmarks is not None:
                w33[i] = [(p.x, p.y, p.z, p.visibility) for p in r.pose_world_landmarks.landmark]
                i33[i] = full_frame(i, [(p.x, p.y, p.z, p.visibility) for p in r.pose_landmarks.landmark], W, H)
        pose.close()
        return w33, i33, f'{mp.__version__} solutions'
    from mediapipe.tasks.python import BaseOptions, vision
    if not os.path.exists(TASK_MODEL):
        import urllib.request
        os.makedirs(os.path.dirname(TASK_MODEL), exist_ok=True)
        print(f'downloading {TASK_URL}', flush=True)
        urllib.request.urlretrieve(TASK_URL, TASK_MODEL + '.part')
        os.replace(TASK_MODEL + '.part', TASK_MODEL)
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=TASK_MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1,
                                        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                                        min_tracking_confidence=0.5)
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        for i, f, rgb in _frames(video, frames):
            img, W, H = prep(i, rgb)
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)),
                                    int(round(f * 1000.0 / video_fps)))
            if r.pose_world_landmarks:
                vis = lambda p: p.visibility if p.visibility is not None else np.nan      # noqa: E731
                w33[i] = [(p.x, p.y, p.z, vis(p)) for p in r.pose_world_landmarks[0]]
                i33[i] = full_frame(i, [(p.x, p.y, p.z, vis(p)) for p in r.pose_landmarks[0]], W, H)
    return w33, i33, f'{mp.__version__} tasks'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', required=True, help='comma-separated user/action')
    ap.add_argument('--cameras', default=None, help='override bp_cameras.json (comma-separated)')
    ap.add_argument('--api', choices=['tasks', 'solutions'], default='tasks')
    ap.add_argument('--crop', action='store_true',
                    help="crop each frame around our OpenPose subject first ('subject given' condition); "
                         "cached as {cam}_mp_crop.npz unless --suffix says otherwise")
    ap.add_argument('--suffix', default=None, help="cache name suffix, e.g. '_sol' to keep a second API's output")
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.suffix is None:
        args.suffix = '_crop' if args.crop else ''
    sel = json.load(open(os.path.join(OUT_DIR, 'bp_cameras.json'))) if not args.cameras else {}
    for t in args.trials.split(','):
        u, a = t.split('/')
        if args.cameras:
            cams = args.cameras.split(',')
        else:
            r = sel.get(u, {}).get(a)
            if not r:
                print(f'{u}/{a}: not in bp_cameras.json, skipped', flush=True)
                continue
            cams = [r[k] for k in ('front', 'side', 'oblique') if k in r]
        for c in cams:
            out = mp_path(u, a, c).replace('_mp.npz', f'_mp{args.suffix}.npz')
            if os.path.exists(out) and not args.force:
                continue
            d = np.load(twod_path(u, a, 'openpose', c))
            sfi, fps = d['source_frame_idx'].astype(int), float(d['fps'])
            t0 = time.time()
            boxes = subject_boxes(d) if args.crop else None
            if args.crop and boxes is None:
                print(f'{u}/{a} cam {c}: no OpenPose subject to crop to, skipped', flush=True)
                continue
            w33, i33, ver = run_blazepose(find_video(u, a, c), sfi, args.api, float(d['video_fps']), boxes)
            world = coco_to_h36m(w33[:, MP_TO_COCO, :3]) * 1000.0
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out + '.part.npz', world=world.astype(np.float32), world33=w33, image33=i33,
                     source_frame_idx=sfi, fps=fps, mediapipe=np.array(ver), model_complexity=COMPLEXITY,
                     crop=bool(args.crop), **({'crop_boxes': boxes.astype(np.float32)} if args.crop else {}))
            os.replace(out + '.part.npz', out)
            det = np.isfinite(world[:, 0, 0]).mean()
            print(f'{u}/{a} cam {c}: {len(sfi)} frames, detected {det:.0%}, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
