#!/usr/bin/env python3
"""bp_extract.py -- MediaPipe BlazePose 3D on exactly the frames our pipeline scores, cached per camera.

For each trial's chosen cameras ({BIOCV_OUT}/bp_cameras.json, datasets/biocv/bp_views.py) BlazePose runs over
the source frames in the 30 fps OpenPose 2d.npz (source_frame_idx: native 200 Hz video frames), in video mode
(tracking + landmark smoothing, the way it is used live): heavy model, model_complexity=2, as
blazepose_vs_ours.py and the face-blur study ran it.

Writes {BIOCV_OUT}/{user}/{action}/Analysis/blazepose/{cam}_mp.npz:
    world        (T,17,3) H36M-17 mm from the 33 world landmarks (hip-centred, camera-like axes: x right, y down,
                 z away); NaN where BlazePose found nobody
    world33      (T,33,4) raw world landmarks (x, y, z m, visibility)
    image33      (T,33,4) normalised image landmarks (x, y, z, visibility)
    source_frame_idx, fps, mediapipe version, model_complexity

Needs the legacy solutions API: pip install mediapipe==0.10.21 (the version the P03 numbers came from).

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


def run_blazepose(video, frames):
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise SystemExit(f'mediapipe {mp.__version__} has no legacy solutions API -- pip install mediapipe==0.10.21')
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=COMPLEXITY, smooth_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
    T = len(frames)
    w33 = np.full((T, 33, 4), np.nan, np.float32)
    i33 = np.full((T, 33, 4), np.nan, np.float32)
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
        r = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if r.pose_world_landmarks is not None:
            w33[want[f]] = [(p.x, p.y, p.z, p.visibility) for p in r.pose_world_landmarks.landmark]
            i33[want[f]] = [(p.x, p.y, p.z, p.visibility) for p in r.pose_landmarks.landmark]
    cap.release()
    pose.close()
    return w33, i33, mp.__version__


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', required=True, help='comma-separated user/action')
    ap.add_argument('--cameras', default=None, help='override bp_cameras.json (comma-separated)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
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
            out = mp_path(u, a, c)
            if os.path.exists(out) and not args.force:
                continue
            d = np.load(twod_path(u, a, 'openpose', c))
            sfi, fps = d['source_frame_idx'].astype(int), float(d['fps'])
            t0 = time.time()
            w33, i33, ver = run_blazepose(find_video(u, a, c), sfi)
            world = coco_to_h36m(w33[:, MP_TO_COCO, :3]) * 1000.0
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out + '.part.npz', world=world.astype(np.float32), world33=w33, image33=i33,
                     source_frame_idx=sfi, fps=fps, mediapipe=np.array(ver), model_complexity=COMPLEXITY)
            os.replace(out + '.part.npz', out)
            det = np.isfinite(world[:, 0, 0]).mean()
            print(f'{u}/{a} cam {c}: {len(sfi)} frames, detected {det:.0%}, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
