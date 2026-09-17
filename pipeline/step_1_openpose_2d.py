#!/usr/bin/env python3
"""step_1_openpose_2d.py -- OpenPose BODY_25 -> H36M-17 2D, in place of step_1.

Stream 1 (adults, BioCV).  Drop-in replacement for step_1_extract_2d.py that
takes OpenPose's --write_json output instead of running YOLO, so the adult
result is measured with the same detector the child test uses.  Writes the
same {cam}_2d.npz step_1 writes (h36m_2d, source_frame_idx, video, fps) plus
body25_2d, which step_1b_triangulate_2d.py needs.

Expected layout (one JSON per frame, OpenPose's own naming):
    {TRIAL_DIR}/{user}/{action}/openpose/{cam}/{anything}_{frame:012d}_keypoints.json

Which native frame each JSON is -- the thing that keeps the 2D, the video and
the mocap on one clock -- comes from the VIDEO, never from the mocap:
    decimated  (tools/run_openpose.py) JSON i is native frame
               frames.json['native_frame_idx'][i]; if that record is missing
               (older runs) the same list is recomputed from the video with
               the same function run_openpose used, which reproduces it
               exactly.  Deriving it from the mocap instead -- as this step
               once did -- desynchronised trials whose mocap and video differ
               in length or whose video fps metadata is off.
    original   OpenPose was run on the native video: JSON frame n is native
               frame n, and the 60 fps ticks are taken from the video's own
               length and fps.
The mocap is then sampled at those same native frame numbers (video and mocap
are frame-aligned at the native rate, per the mocAligned calibration); rows
past the end of the mocap are missing, and a duration mismatch is reported.

Person selection.  OpenPose has no track ids, so the subject is chosen PER
FRAME: the person whose H36M limb joints lie closest to the projected mocap
(same scoring as step_1's find_closest_user_to_mocap, without the temporal
bookkeeping a tracker needs).  Frames with no acceptable match are written as
missing (all zeros), and `detected` records which frames matched.

    python3 step_1_openpose_2d.py --user P08 --action P08_CMJM_01
"""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

from config import TRIAL_DIR as _TRIAL_DIR, mocap_path as _mocap_path
from utils.calibration import load_calib, reproject
from utils.frame_decimation import nearest_frame_indices
from utils.openpose import (H36M_NAMES, body25_to_h36m_2d, list_openpose_jsons,
                            load_openpose_video, pick_person)


def json_layout(jdir, n_json, video_n, mode):
    """'decimated' or 'original': which frames the JSONs are numbered by.

    tools/run_openpose.py decimates the video first and leaves frames.json
    next to its JSONs, so that file settles it.  Without it, a JSON count far
    below the video's frame count can only be a decimated run.  Reading
    decimated JSONs as 'original' was the bug that left the 2D empty past the
    first third of the clip and time-stretched what it did find, so the
    mode is only taken on trust when it is asked for explicitly.
    """
    if mode != 'auto':
        return mode
    if os.path.exists(os.path.join(jdir, 'frames.json')):
        return 'decimated'
    if video_n and n_json < 0.8 * video_n:
        return 'decimated'
    return 'original'


def video_frame_list(jdir, video_path, mode, target_fps):
    """(native frame index per JSON row, video fps, video n) from the video."""
    rec = os.path.join(jdir, 'frames.json')
    if os.path.exists(rec):
        with open(rec) as f:
            r = json.load(f)
        idx = np.asarray(r['native_frame_idx'], int)
        if mode == 'original':
            idx = np.arange(int(r['video_n_frames']))
            idx = nearest_frame_indices(len(idx), float(r['video_fps']), target_fps)
        return idx, float(r['video_fps']), int(r['video_n_frames'])
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if n > 0 and fps > 0:
            idx = nearest_frame_indices(n, fps, target_fps)
            if mode == 'decimated':
                # run_openpose stops at the last decodable frame, which can be one
                # short of CAP_PROP_FRAME_COUNT: trust the JSON count
                idx = idx[:len(list_openpose_jsons(jdir))]
            return idx, float(fps), n
    return None, None, None


def project_mocap(calib_path, kps3d_mm):
    """Mocap lab-mm -> this camera's pixels; NaN where behind the camera."""
    w, h, K, L, dist = load_calib(calib_path)
    R, t = L[:3, :3], L[:3, 3]
    cam = kps3d_mm @ R.T + t
    T, J = cam.shape[:2]
    uv = reproject(cam.reshape(-1, 3), K, dist.reshape(1, 5)).reshape(T, J, 2)
    uv[cam[..., 2] <= 0] = np.nan
    return uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cameras', default=None, help='default: every camera with a JSON dir')
    ap.add_argument('--json-root', default=None,
                    help='dir holding {cam}/ JSON dirs; default {trial}/openpose')
    ap.add_argument('--json-frames', choices=['auto', 'original', 'decimated'], default='auto',
                    help='what the JSONs are numbered by: the decimated 60 fps video (run_openpose.py) '
                         'or the native one; auto reads it off the JSON dir (default)')
    ap.add_argument('--target-fps', type=float, default=60)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--max-match-px', type=float, default=150.0,
                    help='reject the best person if its mean joint distance to mocap exceeds this')
    ap.add_argument('--no-activate', action='store_true',
                    help='only write Analysis/keypoints/openpose/{cam}_2d.npz; leave the {cam}_2d.npz the '
                         'later steps read alone')
    args = ap.parse_args()

    trial = os.path.join(_TRIAL_DIR, args.user, args.action)
    out_dir = os.path.join(trial, 'Analysis', 'keypoints')
    os.makedirs(out_dir, exist_ok=True)
    json_root = args.json_root or os.path.join(trial, 'openpose')

    mocap_path = _mocap_path(args.user, args.action)
    if not os.path.exists(mocap_path):
        raise SystemExit(f'no {mocap_path} -- run step_0_load_mocap.py first')
    mocap = np.load(mocap_path)
    kps_native = mocap['kps3d']
    native_fps = float(mocap['fps'])
    valid_joint = mocap['valid_joint_mask']

    cams = (args.cameras.split(',') if args.cameras else
            sorted(os.path.basename(d) for d in glob.glob(os.path.join(json_root, '*'))
                   if os.path.isdir(d)))
    if not cams:
        raise SystemExit(f'no camera JSON dirs under {json_root}')

    for cam in cams:
        jdir = os.path.join(json_root, cam)
        n_json = len(list_openpose_jsons(jdir))
        if n_json == 0:
            print(f'=== camera {cam}: no JSONs in {jdir}, skipping ===')
            continue

        # native frame number of every JSON, from the video
        video_path = os.path.join(trial, f'{cam}.mp4')
        vn_hint = None
        if os.path.exists(os.path.join(jdir, 'frames.json')):
            with open(os.path.join(jdir, 'frames.json')) as f:
                vn_hint = int(json.load(f)['video_n_frames'])
        elif os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            vn_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        mode = json_layout(jdir, n_json, vn_hint, args.json_frames)
        source_frame_idx, vfps, vn = video_frame_list(jdir, video_path, mode, args.target_fps)
        if source_frame_idx is None:
            print(f'=== camera {cam}: no frames.json and no video to derive the frame list from; '
                  f'skipping (falling back to the mocap would desynchronise it) ===')
            continue
        T = len(source_frame_idx)
        out_fps = min(args.target_fps, vfps if vfps else native_fps)
        frames = source_frame_idx if mode == 'original' else np.arange(T)
        json_nums = list_openpose_jsons(jdir)
        n_hit = sum(int(n) in json_nums for n in frames)
        if n_hit < 0.9 * T:
            raise SystemExit(
                f'camera {cam}: only {n_hit}/{T} rows have a JSON when read as {mode!r} '
                f'({n_json} JSONs numbered up to {max(json_nums)}, video {vn} frames) -- the JSONs are '
                f'numbered by the other frame list; pass --json-frames '
                f'{"decimated" if mode == "original" else "original"} or leave it on auto')
        people = load_openpose_video(jdir, frames)

        # mocap sampled at the same native frames; beyond its end -> missing
        in_range = source_frame_idx < len(kps_native)
        kps = np.full((T, 17, 3), np.nan)
        kps[in_range] = kps_native[source_frame_idx[in_range]]
        if vn and abs(vn / vfps - len(kps_native) / native_fps) > 0.25:
            print(f'    WARNING: video {vn / vfps:.2f} s vs mocap {len(kps_native) / native_fps:.2f} s '
                  f'-- {int((~in_range).sum())} of {T} rows have no mocap')

        calib = os.path.join(_TRIAL_DIR, args.user, f'{cam}.mp4-mocAligned.calib')
        mocap2d = project_mocap(calib, kps)                     # (T,17,2)

        h36m = np.zeros((T, 17, 3), np.float32)
        body25 = np.zeros((T, 25, 3), np.float32)
        detected = np.zeros(T, bool)
        score = np.full(T, np.nan)
        n_people = np.zeros(T, int)
        for i, (xy_p, conf_p) in enumerate(people):
            n_people[i] = len(xy_p)
            ref_ok = valid_joint & np.isfinite(mocap2d[i]).all(axis=1)
            k, s = pick_person(xy_p, conf_p, mocap2d[i], ref_ok, conf_thresh=args.conf)
            if k is None or s > args.max_match_px:
                continue
            body25[i, :, :2] = xy_p[k]
            body25[i, :, 2] = conf_p[k]
            h36m[i] = body25_to_h36m_2d(xy_p[k:k + 1], conf_p[k:k + 1])[0]
            detected[i] = True
            score[i] = s

        video = f'{cam}.mp4'
        # Each detector keeps its own copy (keypoints/openpose/, keypoints/yolo/), so neither
        # overwrites the other; keypoints/{cam}_2d.npz is whichever one the later steps should use.
        det_dir = os.path.join(out_dir, 'openpose')
        os.makedirs(det_dir, exist_ok=True)
        out = os.path.join(det_dir, f'{cam}_2d.npz')
        np.savez(out, h36m_2d=h36m, body25_2d=body25, source_frame_idx=source_frame_idx,
                 video=video, fps=out_fps, detected=detected, match_px=score,
                 n_people=n_people, joint_names=np.array(H36M_NAMES),
                 detector=np.array('openpose_body25'), json_frames=np.array(mode),
                 video_fps=vfps if vfps else np.nan, video_n_frames=vn if vn else -1,
                 mocap_fps=native_fps, mocap_n_frames=len(kps_native))
        if not args.no_activate:
            shutil.copy2(out, os.path.join(out_dir, f'{cam}_2d.npz'))
        print(f'=== camera {cam}: {n_json} JSONs ({mode}), {T} frames, matched {detected.sum()} '
              f'(median {np.nanmedian(score):.1f} px to mocap; '
              f'{(n_people > 1).mean():.0%} of frames had >1 person) -> {out}')


if __name__ == '__main__':
    main()
