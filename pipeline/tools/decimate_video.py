#!/usr/bin/env python3
"""Write a video containing only the frames step_1 would have used.

BioCV videos are 200 Hz.  OpenPose is slow, and step_1_openpose_2d.py only
takes the frames nearest_frame_indices() picks for --target-fps (60 by
default), so run OpenPose on this decimated copy instead -- 3.3x fewer
frames -- and pass --json-frames decimated.  The frame selection is the same
function step_1 uses, so decimated frame i is exactly native frame
source_frame_idx[i].

    python3 tools/decimate_video.py in.mp4 out.mp4 --target-fps 60
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.frame_decimation import nearest_frame_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--target-fps', type=float, default=60)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f'cannot open {args.src}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    keep = set(nearest_frame_indices(n, src_fps, args.target_fps).tolist())
    out_fps = min(args.target_fps, src_fps)
    writer = cv2.VideoWriter(args.dst, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w, h))
    i = written = 0
    while True:
        if i in keep:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        else:
            if not cap.grab():
                break
        i += 1
    cap.release()
    writer.release()
    print(f'{args.src}: {n} frames @ {src_fps:g} fps -> {written} frames @ {out_fps:g} fps -> {args.dst}')


if __name__ == '__main__':
    main()
