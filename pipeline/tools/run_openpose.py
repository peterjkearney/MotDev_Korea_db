#!/usr/bin/env python3
"""Run OpenPose over every trial and camera, writing ONE json per video -- the file
step_1_openpose_2d.py reads.  The OpenPose analogue of step_1_extract_2d.py's YOLO pass.

For each {user}/{action}/{cam}.mp4 under TRIAL_DIR (the local copy of BioCV):
  1. decimate the 200 Hz video to --target-fps frames (the same frames step_1 uses --
     3.3x fewer for OpenPose),
  2. run the OpenPose binary on it with BODY_25 and --write_json into a temp dir,
  3. check one JSON came out per frame, and fold them into
        {OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_openpose.json
     with the native frame number of every entry (utils/openpose.py has the format).

OUT_DIR is the BioCV folder on Drive, so nothing needs syncing: a camera is done when its
json exists, and re-running the same command after a disconnect resumes where it stopped,
losing at most the camera in flight.

Per-frame JSON dirs from before this layout ({root}/{user}/{action}/openpose/{cam}/, as
kept in the old results folder) need not be recomputed: --import-from {root} folds any
complete one into the new file instead of running the binary.

    python3 tools/run_openpose.py --openpose-bin /content/openpose/build/examples/openpose/openpose.bin \\
                                  --model-folder /content/openpose/models/
    python3 tools/run_openpose.py ... --users P08 --actions P08_CMJM_01,P08_WALK_01
    python3 tools/run_openpose.py --import-from /content/drive/MyDrive/MotorDevelopment/results
    python3 tools/run_openpose.py ... --dry-run

Throughput: measured on a Colab T4 with the cuDNN-free build, ~5 fps --
about 77 s per camera, ~12 minutes per 9-camera BioCV trial.  Use
--users/--actions to pick a subset; the adult control does not need every
trial.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
sys.path.insert(0, _PIPE)
sys.path.insert(0, _HERE)
from config import TRIAL_DIR, OUT_DIR, openpose_json_path, require_out_dir
from run_batch import discover
from utils.frame_decimation import nearest_frame_indices
from utils.openpose import list_openpose_jsons, consolidate_openpose_dir, save_openpose_video


def video_cameras(user, action):
    d = os.path.join(TRIAL_DIR, user, action)
    return sorted(os.path.splitext(os.path.basename(v))[0]
                  for v in glob.glob(os.path.join(d, '0*.mp4')))


def decimate(src, dst, target_fps):
    """Returns (frames written, native indices of those frames, video fps, video n)."""
    cap = cv2.VideoCapture(src)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or n <= 0 or fps <= 0:
        cap.release()
        return 0, [], fps, n          # unreadable video: reported as a failure, not a crash
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keep = set(nearest_frame_indices(n, fps, target_fps).tolist())
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), min(target_fps, fps), (w, h))
    i, kept = 0, []
    while True:
        if i in keep:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            kept.append(i)
        elif not cap.grab():
            break
        i += 1
    cap.release()
    out.release()
    return len(kept), kept, fps, n


def video_info(video):
    cap = cv2.VideoCapture(video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def import_old_dir(old_dir, video, target_fps):
    """A per-frame JSON dir from the old layout -> the one-file document, or None if it is
    not a complete run.  Which native frame each JSON is comes from its frames.json; older
    runs have none, and then a JSON count well below the video's can only be a decimated run
    (the same tick list run_openpose used), anything else one JSON per native frame."""
    n_json = len(list_openpose_jsons(old_dir))
    if n_json == 0 or not os.path.exists(os.path.join(old_dir, 'done.json')):
        return None
    rec = os.path.join(old_dir, 'frames.json')
    if os.path.exists(rec):
        with open(rec) as f:
            r = json.load(f)
        idx, vfps, vn, tfps = r['native_frame_idx'], r['video_fps'], r['video_n_frames'], r.get('target_fps', target_fps)
    else:
        vn, vfps = video_info(video)
        if vn <= 0 or vfps <= 0:
            return None
        tfps = target_fps
        idx = nearest_frame_indices(vn, vfps, tfps).tolist() if n_json < 0.8 * vn else list(range(vn))
        idx = idx[:n_json + 1]
    return consolidate_openpose_dir(old_dir, idx, video=os.path.basename(video), video_fps=float(vfps),
                                    video_n_frames=int(vn), target_fps=float(tfps))


def run_one(user, action, cam, video, args, work):
    out = openpose_json_path(user, action, cam)
    if os.path.exists(out) and not args.force:
        return 'skip', 0, 0.0
    t0 = time.time()
    if args.import_from:
        doc = import_old_dir(os.path.join(args.import_from, user, action, 'openpose', cam), video, args.target_fps)
        if doc is not None:
            save_openpose_video(out, doc)
            return 'imported', len(doc['frames']), time.time() - t0
    if not args.openpose_bin:
        return 'fail nothing to import and no --openpose-bin', 0, time.time() - t0
    json_dir = os.path.join(work, 'json')
    shutil.rmtree(json_dir, ignore_errors=True)
    os.makedirs(json_dir)
    dec = os.path.join(work, 'dec.mp4')
    written, kept, vfps, vn = decimate(video, dec, args.target_fps)
    if written == 0:
        return f'fail cannot read {video}', 0, time.time() - t0
    cmd = [args.openpose_bin, '--video', dec, '--write_json', json_dir,
           '--model_pose', 'BODY_25', '--display', '0', '--render_pose', '0',
           '--number_people_max', str(args.max_people)]
    if args.model_folder:
        cmd += ['--model_folder', args.model_folder]
    if args.net_resolution:
        cmd += ['--net_resolution', args.net_resolution]
    log_path = os.path.join(work, 'openpose.log')
    with open(log_path, 'w') as log:
        log.write(' '.join(cmd) + '\n\n')
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=args.cwd or None)
    got = len(list_openpose_jsons(json_dir))
    if p.returncode != 0 or got < written:
        with open(log_path) as f:
            tail = ' | '.join(f.read().strip().splitlines()[-3:])
        return f'fail rc={p.returncode} {got}/{written} JSONs: {tail[:300]}', got, time.time() - t0
    # `kept` is the one record of which native frame each JSON is; it goes into the file, so
    # the 2D rows, the video and the mocap can never drift apart.
    doc = consolidate_openpose_dir(json_dir, kept, video=os.path.basename(video), video_fps=float(vfps),
                                   video_n_frames=int(vn), target_fps=float(args.target_fps),
                                   cmd=' '.join(cmd))
    save_openpose_video(out, doc)           # written last, atomically: its presence marks the camera done
    return 'ok', got, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openpose-bin', default=None, help='the OpenPose binary (not needed for a pure --import-from)')
    ap.add_argument('--model-folder', default=None, help="OpenPose's models/ dir")
    ap.add_argument('--cwd', default=None, help='run the binary from here (some builds need their root)')
    ap.add_argument('--users')
    ap.add_argument('--actions')
    ap.add_argument('--cameras', default=None, help='comma-separated subset, default all')
    ap.add_argument('--target-fps', type=float, default=60)
    ap.add_argument('--max-people', type=int, default=3)
    ap.add_argument('--net-resolution', default=None, help='e.g. -1x368 (default) or -1x256 for speed')
    ap.add_argument('--import-from', default=None,
                    help='root of old per-frame JSON dirs ({root}/{user}/{action}/openpose/{cam}/); complete ones '
                         'are folded into the new file instead of re-running OpenPose')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not args.openpose_bin and not args.import_from:
        raise SystemExit('give --openpose-bin to run OpenPose, and/or --import-from to reuse old JSON dirs')
    require_out_dir()

    trials, unmatched = discover(
        [u for u in args.users.split(',') if u] if args.users else None,
        [a for a in args.actions.split(',') if a] if args.actions else None)
    if unmatched:
        print(f'WARNING: no match for {",".join(unmatched)}')
    want = set(args.cameras.split(',')) if args.cameras else None

    jobs = [(user, action, cam, os.path.join(TRIAL_DIR, user, action, f'{cam}.mp4'))
            for user, action in trials for cam in video_cameras(user, action) if not want or cam in want]
    print(f'videos from {TRIAL_DIR}\njsons to    {OUT_DIR}\n{len(trials)} trial(s), {len(jobs)} camera video(s)')
    if args.dry_run:
        for user, action, cam, video in jobs:
            old = args.import_from and os.path.exists(os.path.join(args.import_from, user, action, 'openpose', cam, 'done.json'))
            state = ('done' if os.path.exists(openpose_json_path(user, action, cam)) and not args.force
                     else 'import' if old else 'run')
            print(f'  {user}/{action}/{cam}  {state}')
        return

    work = tempfile.mkdtemp(prefix='openpose_')
    counts = {}
    try:
        for i, (user, action, cam, video) in enumerate(jobs, 1):
            status, n, secs = run_one(user, action, cam, video, args, work)
            counts[status.split()[0]] = counts.get(status.split()[0], 0) + 1
            if status != 'skip':
                print(f'[{i}/{len(jobs)}] {user}/{action}/{cam}: {status} {n} frames ({secs:.0f}s)', flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('\n' + ', '.join(f'{k} {v}' for k, v in counts.items()))


if __name__ == '__main__':
    main()
