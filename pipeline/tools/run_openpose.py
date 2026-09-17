#!/usr/bin/env python3
"""Run OpenPose over every trial and camera, writing the JSON dirs
step_1_openpose_2d.py reads.  The OpenPose analogue of tools/run_batch.py.

For each {user}/{action}/{cam}.mp4 under TRIAL_DIR:
  1. decimate the 200 Hz video to --target-fps frames (tools/decimate_video.py
     logic, the same frames step_1 would use -- 3.3x fewer for OpenPose),
  2. run the OpenPose binary on it with BODY_25 and --write_json,
  3. check one JSON came out per frame,
into {user}/{action}/openpose/{cam}/.

Surviving a dead runtime.  TRIAL_DIR is the Colab VM's local disk, which is
wiped when the runtime ends -- so, like run_batch.py does for Analysis/, each
camera's JSON dir is copied to RESULTS_DIR (on Drive) the moment it finishes,
and at the start of a run any camera already complete on Drive is copied back
and skipped.  Re-running the same command after a disconnect therefore
resumes where it stopped, losing at most the camera in flight.  Run it before
run_batch.py in every session, even when all cameras are done: that is what
restores the JSONs to local disk for step_1op.  (--no-sync disables this.)

Then run run_batch.py with --json-frames decimated (its default).

    python3 tools/run_openpose.py --openpose-bin /content/openpose/build/examples/openpose/openpose.bin \\
                                  --model-folder /content/openpose/models/
    python3 tools/run_openpose.py ... --users P08 --actions P08_CMJM_01,P08_WALK_01
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
from config import TRIAL_DIR, RESULTS_DIR
from run_batch import discover
from utils.frame_decimation import nearest_frame_indices
from utils.openpose import list_openpose_jsons


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


def is_complete(json_dir):
    """Done iff a previous run left its marker AND the JSONs it counted are
    still there.  (Predicting the count from the video does not work:
    CAP_PROP_FRAME_COUNT can overstate the decodable frames by one.)"""
    marker = os.path.join(json_dir, 'done.json')
    if not os.path.exists(marker):
        return False
    with open(marker) as f:
        n = json.load(f).get('n_frames', -1)
    return len(list_openpose_jsons(json_dir)) >= n > 0


def drive_dir(user, action, cam, args):
    return None if args.no_sync else os.path.join(args.sync_dir, user, action, 'openpose', cam)


def _copy_dir(src, dst):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_one(video, json_dir, ddir, args, work):
    if is_complete(json_dir) and not args.force:
        return 'skip', len(list_openpose_jsons(json_dir)), 0.0
    if ddir and is_complete(ddir) and not args.force:
        t0 = time.time()
        _copy_dir(ddir, json_dir)
        return 'restored from drive', len(list_openpose_jsons(json_dir)), time.time() - t0
    t0 = time.time()
    if os.path.isdir(json_dir):
        shutil.rmtree(json_dir)
    os.makedirs(json_dir)
    dec = os.path.join(work, 'dec.mp4')
    written, kept, vfps, vn = decimate(video, dec, args.target_fps)
    if written == 0:
        return f'fail cannot read {video}', 0, time.time() - t0
    # The one record of which native frame each JSON is.  step_1_openpose_2d.py
    # reads this rather than re-deriving the frame list from the mocap, so the
    # 2D rows, the video and the mocap can never drift apart.
    with open(os.path.join(json_dir, 'frames.json'), 'w') as f:
        json.dump({'native_frame_idx': kept, 'video_fps': vfps, 'video_n_frames': vn,
                   'target_fps': args.target_fps}, f)
    cmd = [args.openpose_bin, '--video', dec, '--write_json', json_dir,
           '--model_pose', 'BODY_25', '--display', '0', '--render_pose', '0',
           '--number_people_max', str(args.max_people)]
    if args.model_folder:
        cmd += ['--model_folder', args.model_folder]
    if args.net_resolution:
        cmd += ['--net_resolution', args.net_resolution]
    with open(os.path.join(json_dir, 'openpose.log'), 'w') as log:
        log.write(' '.join(cmd) + '\n\n')
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=args.cwd or None)
    got = len(list_openpose_jsons(json_dir))
    if p.returncode != 0 or got < written:
        return f'fail rc={p.returncode} {got}/{written} JSONs', got, time.time() - t0
    with open(os.path.join(json_dir, 'done.json'), 'w') as f:
        json.dump({'n_frames': got, 'target_fps': args.target_fps, 'cmd': cmd}, f)
    if ddir:
        _copy_dir(json_dir, ddir)          # ~1 MB per camera; survives the runtime
    return 'ok', got, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--openpose-bin', required=True)
    ap.add_argument('--model-folder', default=None, help="OpenPose's models/ dir")
    ap.add_argument('--cwd', default=None, help='run the binary from here (some builds need their root)')
    ap.add_argument('--users')
    ap.add_argument('--actions')
    ap.add_argument('--cameras', default=None, help='comma-separated subset, default all')
    ap.add_argument('--target-fps', type=float, default=60)
    ap.add_argument('--max-people', type=int, default=3)
    ap.add_argument('--net-resolution', default=None, help='e.g. -1x368 (default) or -1x256 for speed')
    ap.add_argument('--sync-dir', default=RESULTS_DIR,
                    help='where finished JSON dirs are kept between sessions (default RESULTS_DIR)')
    ap.add_argument('--no-sync', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if not args.no_sync and not os.path.isdir(args.sync_dir):
        raise SystemExit(f'--sync-dir {args.sync_dir} does not exist -- mount Drive, '
                         f'set BIOCV_RESULTS, or pass --no-sync')

    trials, unmatched = discover(
        [u for u in args.users.split(',') if u] if args.users else None,
        [a for a in args.actions.split(',') if a] if args.actions else None)
    if unmatched:
        print(f'WARNING: no match for {",".join(unmatched)}')
    want = set(args.cameras.split(',')) if args.cameras else None

    jobs = []
    for user, action in trials:
        for cam in video_cameras(user, action):
            if want and cam not in want:
                continue
            video = os.path.join(TRIAL_DIR, user, action, f'{cam}.mp4')
            jdir = os.path.join(TRIAL_DIR, user, action, 'openpose', cam)
            jobs.append((user, action, cam, video, jdir))
    print(f'TRIAL_DIR {TRIAL_DIR}\nsync     {"off" if args.no_sync else args.sync_dir}\n'
          f'{len(trials)} trial(s), {len(jobs)} camera video(s)')
    if args.dry_run:
        for user, action, cam, video, jdir in jobs:
            ddir = drive_dir(user, action, cam, args)
            state = ('done' if is_complete(jdir) else
                     'done on drive' if ddir and is_complete(ddir) else 'todo')
            print(f'  {user}/{action}/{cam}  {state}')
        return

    work = tempfile.mkdtemp(prefix='openpose_')
    n_ok = n_skip = n_fail = n_rest = 0
    try:
        for i, (user, action, cam, video, jdir) in enumerate(jobs, 1):
            status, n, secs = run_one(video, jdir, drive_dir(user, action, cam, args), args, work)
            n_ok += status == 'ok'
            n_skip += status == 'skip'
            n_rest += status.startswith('restored')
            n_fail += status.startswith('fail')
            print(f'[{i}/{len(jobs)}] {user}/{action}/{cam}: {status} '
                  f'{n} frames ({secs:.0f}s)', flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f'\nok {n_ok}, restored {n_rest}, skipped {n_skip}, failed {n_fail}')


if __name__ == '__main__':
    main()
