#!/usr/bin/env python3
"""demo_frames.py -- example frames showing what deface does, picked from
recall_survey's samples to span subjects, tasks, cameras, facing angles and
face sizes.

Each panel: the whole frame after deface (its regions outlined), and a zoom on
the subject's face before and after.  The "before" crops show identifiable
faces -- BioCV participants consented to data release, but check before
putting them in a public document.

    python demo_frames.py [--n 10]
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from common import Camera, out_dir, pick_subject, TRIAL_ROOT
import blur
import detectors

ROOTS = ['/Volumes/Expansion/MotorDevelopment/BioCV', TRIAL_ROOT]


def pick(df, n):
    """Greedy: each pick adds the most unseen (subject, task, cam, facing bin, size bin) values."""
    df = df.assign(fbin=pd.cut(df.facing_deg, [0, 30, 60, 90]).astype(str),
                   sbin=pd.cut(df.face_px, [0, 40, 60, 80, 1000]).astype(str))
    keys = ['subject', 'task', 'cam', 'fbin', 'sbin']
    seen = {k: set() for k in keys}
    out = []
    rest = df.sample(frac=1, random_state=0)
    for _ in range(n):
        gain = sum((~rest[k].isin(seen[k])).astype(int) * (3 if k == 'subject' else 1) for k in keys)
        r = rest.loc[gain.idxmax()]
        out.append(r)
        for k in keys:
            seen[k].add(r[k])
        rest = rest.drop(gain.idxmax())
    return pd.DataFrame(out)


def trial_dir(action):
    for root in ROOTS:
        for d in (os.path.join(root, action.split('_')[0], action), os.path.join(root, action)):
            if os.path.exists(os.path.join(d, '00.mp4')):
                return d
    raise FileNotFoundError(action)


def calib(d, cam):
    for p in (os.path.join(d, f'{cam}.mp4-mocAligned.calib'),
              os.path.join(os.path.dirname(d), f'{cam}.mp4-mocAligned.calib')):
        if os.path.exists(p):
            return p


def read_frame(path, f):
    cap = cv2.VideoCapture(path)
    for _ in range(f):
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=10)
    args = ap.parse_args()
    csvs = [out_dir('recall_survey', 'frames.csv'), out_dir('recall_survey_test', 'frames.csv')]
    df = pd.concat([pd.read_csv(p, dtype={'cam': str}) for p in csvs if os.path.exists(p)], ignore_index=True)
    df = df[(df.facing_deg < 90) & df.in_frame & df.deface_covered]
    rows = pick(df, args.n)
    dest = out_dir('demo_frames')
    os.makedirs(dest, exist_ok=True)
    deface = blur.Deface()
    yolo = detectors.Yolo()
    for k, r in enumerate(rows.itertuples(), 1):
        cam = f'{int(r.cam):02d}'
        d = trial_dir(r.action)
        frame = read_frame(os.path.join(d, f'{cam}.mp4'), int(r.frame))
        regs = deface.find(frame)
        after = deface.apply(frame.copy(), regs)
        # the subject's face: the deface region holding the nose of the YOLO person nearest the projected mocap
        cm = Camera(calib(d, cam))
        H, W = frame.shape[:2]
        mocap = np.load(os.path.join(TRIAL_ROOT, r.action, 'Analysis', 'H36M', 'mocap_h36m.npz'))['kps3d']
        people = yolo(frame)
        j, _ = pick_subject(people, cm.project(mocap[int(r.frame)].astype(float)))
        nose = people[j, 0, :2] if j >= 0 else np.array([W / 2, H / 2])
        hit = [i for i in range(len(regs)) if blur.inside(regs[i][None], 'ellipse', nose[None])[0]]
        big = hit[0] if hit else int(np.argmin(np.linalg.norm((regs[:, :2] + regs[:, 2:4]) / 2 - nose, axis=1)))
        x1, y1, x2, y2 = regs[big, :4]
        s = max(x2 - x1, y2 - y1) * 1.6
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bx1, by1 = int(max(0, cx - s)), int(max(0, cy - s))
        bx2, by2 = int(min(W, cx + s)), int(min(H, cy + s))
        crop_b = cv2.resize(frame[by1:by2, bx1:bx2], (400, 400), interpolation=cv2.INTER_NEAREST)
        crop_a = cv2.resize(after[by1:by2, bx1:bx2], (400, 400), interpolation=cv2.INTER_NEAREST)
        full = after.copy()
        for x1_, y1_, x2_, y2_, _ in regs.astype(int):
            cv2.rectangle(full, (x1_, y1_), (x2_, y2_), (0, 0, 255), 3)
        full = cv2.resize(full, (int(W * 800 / H), 800))
        right = np.vstack([crop_b, crop_a])
        panel = np.hstack([full, right])
        label = (f'{r.action}  cam {cam}  frame {int(r.frame)}  |  facing {r.facing_deg:.0f} deg  '
                 f'|  head ~{r.face_px:.0f} px  |  deface regions: {len(regs)}')
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, 'before', (full.shape[1] + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel, 'after', (full.shape[1] + 10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        p = os.path.join(dest, f'{k:02d}_{r.action}_cam{cam}_f{int(r.frame)}.jpg')
        cv2.imwrite(p, panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(p, flush=True)


if __name__ == '__main__':
    main()
