#!/usr/bin/env python3
"""render_openpose_check.py -- look at what OpenPose gave us, against the projected mocap.

One camera of one trial.  Over the source video:
  * every person OpenPose found, as BODY_25 joints and limbs -- the one the matcher picks
    in orange, everyone else in grey -- with the CONFIDENCE written beside each joint
    (red where it is under --conf, the threshold below which a joint does not count);
  * the mocap, projected through this camera's calibration, in green;
  * a status line per frame: people found, which one was picked, its distance to the mocap
    over the joints it has, how many limb joints that was, and ACCEPTED / REJECTED under
    the same rule step_1_openpose_2d.py applies (utils/openpose.pick_person, --max-match-px).
The view is cropped to where the subject goes (from the projected mocap) so the labels can
be read; --no-crop keeps the whole frame.

    reads   {TRIAL_DIR}/{user}/{action}/{cam}.mp4          (or --videos DIR)
            {OUT_DIR}/.../Analysis/keypoints/openpose/{cam}_openpose.json
            {OUT_DIR}/.../Analysis/H36M/mocap_h36m.npz, the camera's calib
    writes  {OUT_DIR}/.../Analysis/diagnostics/openpose/{cam}_openpose_check.mp4
            (rendered on local disk, then copied)

    python3 tools/render_openpose_check.py --user User03 --action P03_CMJM_01 --camera 01
    python3 tools/render_openpose_check.py ... --slow 4          # quarter speed
"""
import argparse
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import (TRIAL_DIR, openpose_json_path, mocap_path, calib_path, openpose_check_path, require_out_dir)
from step_1_openpose_2d import project_mocap
from utils.openpose import BODY25_NAMES, load_openpose_video, pick_person

H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
BODY25_LIMBS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11),
                (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
                (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
BODY_JOINTS = list(range(15)) + [17, 18]           # what the H36M conversion uses; eyes and feet are extra
_PICKED, _OTHER, _MOCAP, _LOW = (0, 140, 255), (160, 160, 160), (80, 220, 80), (60, 60, 255)   # BGR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--camera', required=True)
    ap.add_argument('--videos', default=None, help="dir holding {cam}.mp4 if not {TRIAL_DIR}/{user}/{action}")
    ap.add_argument('--conf', type=float, default=0.3, help='confidence under which a joint does not count (labels go red)')
    ap.add_argument('--max-match-px', type=float, default=80.0)
    ap.add_argument('--all-joints', action='store_true', help='also draw and label the eyes and the six foot joints')
    ap.add_argument('--no-crop', action='store_true')
    ap.add_argument('--height', type=int, default=1080, help='output height in px')
    ap.add_argument('--slow', type=float, default=2.0, help='slow-down factor (2 = half speed)')
    ap.add_argument('--out', default=None, help='default: config.openpose_check_path, on Drive')
    args = ap.parse_args()
    user, action, cam = args.user, args.action, args.camera

    video = os.path.join(args.videos or os.path.join(TRIAL_DIR, user, action), f'{cam}.mp4')
    if not os.path.exists(video):
        raise SystemExit(f"no {video} -- point --videos at the dir holding the trial's mp4s")
    out_path = args.out or openpose_check_path(user, action, cam)
    if not args.out:
        require_out_dir()
    doc, people = load_openpose_video(openpose_json_path(user, action, cam))
    native = np.asarray(doc['native_frame_idx'], int)
    mocap = np.load(mocap_path(user, action))
    kps = np.full((len(native), 17, 3), np.nan)
    ok = native < len(mocap['kps3d'])
    kps[ok] = mocap['kps3d'][native[ok]]
    uv = project_mocap(calib_path(user, cam), kps)                      # (T,17,2), NaN where no marker
    valid_joint = mocap['valid_joint_mask'].astype(bool)

    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fixed crop: where the projected mocap goes over the clip, plus a margin, at the frame's aspect
    x0, y0, x1, y1 = 0, 0, W, H
    fin = uv[np.isfinite(uv).all(-1)]
    if not args.no_crop and len(fin):
        lo, hi = fin.min(0), fin.max(0)
        cx, cy = (lo + hi) / 2
        hh = max((hi[1] - lo[1]) * 0.75, (hi[0] - lo[0]) * 0.75 * H / W, 150)
        hw = hh * W / H
        x0, x1 = int(np.clip(cx - hw, 0, W - 2 * hw)), 0
        y0, y1 = int(np.clip(cy - hh, 0, H - 2 * hh)), 0
        x1, y1 = int(min(W, x0 + 2 * hw)), int(min(H, y0 + 2 * hh))
    sc = args.height / (y1 - y0)
    ow, oh = int(round((x1 - x0) * sc)), args.height
    tf = lambda p: tuple(int(round(v)) for v in ((p[0] - x0) * sc, (p[1] - y0) * sc))

    fps = min(float(doc.get('target_fps', 60)), float(doc['video_fps'])) / max(args.slow, 1e-6)
    tmp = os.path.join(tempfile.gettempdir(), f'{action}_{cam}_openpose_check.mp4')
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh))
    show = list(range(25)) if args.all_joints else BODY_JOINTS
    want = {int(n): i for i, n in enumerate(native)}
    n_acc, fi, last = 0, 0, int(native[-1])
    while fi <= last:
        if fi not in want:
            if not cap.grab():
                break
            fi += 1
            continue
        good, frame = cap.read()
        if not good:
            break
        i = want[fi]
        img = cv2.resize(frame[y0:y1, x0:x1], (ow, oh), interpolation=cv2.INTER_LINEAR)
        xy_p, conf_p = people[i]
        ref_ok = valid_joint & np.isfinite(uv[i]).all(axis=1)
        k, dist, nm = pick_person(xy_p, conf_p, uv[i], ref_ok, conf_thresh=args.conf)
        accepted = k is not None and dist <= args.max_match_px
        n_acc += accepted

        for pi in sorted(range(len(xy_p)), key=lambda q: q == k):       # the picked person is drawn last, on top
            col, th = (_PICKED, 2) if pi == k else (_OTHER, 1)
            seen = conf_p[pi] > 0
            for a, b in BODY25_LIMBS:
                if seen[a] and seen[b] and a in show and b in show:
                    cv2.line(img, tf(xy_p[pi][a]), tf(xy_p[pi][b]), col, th, cv2.LINE_AA)
            for j in show:
                if not seen[j]:
                    continue
                c = float(conf_p[pi][j])
                pt = tf(xy_p[pi][j])
                cv2.circle(img, pt, 4 if pi == k else 3, col, -1, cv2.LINE_AA)
                lab_col = _LOW if c < args.conf else ((255, 255, 255) if pi == k else _OTHER)
                cv2.putText(img, f'{c:.2f}', (pt[0] + 6, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, f'{c:.2f}', (pt[0] + 6, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lab_col, 1, cv2.LINE_AA)
        for a, b in H36M_LIMBS:                                          # projected mocap
            if ref_ok[a] and ref_ok[b]:
                cv2.line(img, tf(uv[i][a]), tf(uv[i][b]), _MOCAP, 2, cv2.LINE_AA)
        for j in np.where(ref_ok)[0]:
            cv2.circle(img, tf(uv[i][j]), 4, _MOCAP, 1, cv2.LINE_AA)

        status = (f'people {len(xy_p)} | ' + ('nobody qualifies' if k is None else
                  f'picked #{k}: {dist:.0f} px over {nm}/12 limb joints') + ' | '
                  + ('ACCEPTED' if accepted else 'REJECTED' if ok[i] else 'no mocap this frame'))
        lines = [(f'{user}/{action} cam {cam}   row {i}  native frame {fi}  t={fi / float(doc["video_fps"]):.2f}s', (255, 255, 255)),
                 (status, (80, 220, 80) if accepted else _LOW),
                 ('orange: picked OpenPose person   grey: other people   green: projected mocap   '
                  f'red label: confidence < {args.conf:g}', (220, 220, 220))]
        cv2.rectangle(img, (0, 0), (ow, 92), (0, 0, 0), -1)
        for r, (txt, colr) in enumerate(lines):
            cv2.putText(img, txt, (10, 26 + 28 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.7 if r < 2 else 0.55, colr, 2 if r < 2 else 1, cv2.LINE_AA)
        writer.write(img)
        fi += 1
    cap.release()
    writer.release()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copy2(tmp, out_path)
    os.remove(tmp)
    print(f'{len(native)} rows, accepted on {n_acc}; crop x {x0}-{x1}, y {y0}-{y1} -> {out_path}')


if __name__ == '__main__':
    main()
