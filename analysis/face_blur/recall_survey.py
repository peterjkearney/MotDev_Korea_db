#!/usr/bin/env python3
"""recall_survey.py -- how often each anonymiser covers the face, over many
subjects, actions and all nine cameras, from a few frames of each trial.

The full study (steps 1-5) measures what blur does to keypoints on one trial;
this only asks "did deface blur the face?", which needs no videos written and
no lifting, so it can sample widely.  (Haar is left out: its recall on the main
trial was already too poor to be a candidate.)  Per trial, --frames frames
spread evenly over the mocap's valid span; per sampled frame and camera:
  deface   blur.Deface regions (deface 1.5 defaults)
  yolo     YOLO26-m keypoints, the subject picked by proximity to the projected mocap
The head is triangulated per frame from the nine cameras' YOLO face points, as
step_3_head does, and a camera's face counts as covered when its projected
nose and eyes lie in a region.  Frames where the face is turned > 90 deg from a
camera are kept but marked, as there is nothing identifying to blur.

Trials are found under --root as any folder holding 0*.mp4 and a mocap
(Analysis/H36M/mocap_h36m.npz there, or under the local sample by action);
calibrations are looked for in the trial folder, then its parent.

    python recall_survey.py --root /Volumes/Expansion/MotorDevelopment/BioCV --frames 12 --workers 4
"""
import argparse
import concurrent.futures as cf
import glob
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import pandas as pd

from common import Camera, pick_subject, out_dir, ALL_CAMS, TRIAL_ROOT, FACE
from utils.triangulate import triangulate_robust

CONF = 0.5
_models = {}


def find_trials(root, actions=None):
    out = []
    if actions:     # named trials: look in {root}/{action} and {root}/{subject}/{action}, not the whole drive
        dirs = [p for a in actions for p in (os.path.join(root, a), os.path.join(root, a.split('_')[0], a))]
    else:
        dirs = sorted(glob.glob(os.path.join(root, '**', ''), recursive=True))
    for d in dirs:
        d = d.rstrip('/')
        name = os.path.basename(d)
        if not glob.glob(os.path.join(d, '0*.mp4')) or (actions and name not in actions):
            continue
        mocap = next((p for p in (os.path.join(d, 'Analysis', 'H36M', 'mocap_h36m.npz'),
                                  os.path.join(TRIAL_ROOT, name, 'Analysis', 'H36M', 'mocap_h36m.npz'))
                      if os.path.exists(p)), None)
        calibs = {}
        for c in ALL_CAMS:
            calibs[c] = next((p for p in (os.path.join(d, f'{c}.mp4-mocAligned.calib'),
                                          os.path.join(os.path.dirname(d), f'{c}.mp4-mocAligned.calib'))
                              if os.path.exists(p)), None)
        if (mocap and all(calibs.values()) and all(os.path.exists(os.path.join(d, f'{c}.mp4')) for c in ALL_CAMS)
                and name not in {t['action'] for t in out}):          # one copy per trial, not render folders
            out.append(dict(action=name, dir=d, mocap=mocap, calibs=calibs))
    return out


def sample_frames(mocap_path, n):
    m = np.load(mocap_path)
    ok = np.where(np.isfinite(m['kps3d']).all(-1).sum(-1) >= 8)[0]
    idx = np.unique(np.linspace(ok[0], ok[-1], n).round().astype(int))
    return idx, m['kps3d'][idx].astype(float)


def _init():
    import blur
    import detectors
    import torch
    torch.set_num_threads(2)
    _models.update(deface=blur.Deface(), yolo=detectors.Yolo())


def run_camera(trial, cam, idx, mocap):
    """One camera of one trial: regions and the subject's COCO keypoints at each sampled frame."""
    camera = Camera(trial['calibs'][cam])
    m2d = camera.project(mocap)
    cap = cv2.VideoCapture(os.path.join(trial['dir'], f'{cam}.mp4'))
    res = dict(deface=[], coco=np.zeros((len(idx), 17, 3), np.float32), secs=[])
    pos = 0
    for i, f in enumerate(idx):
        while pos < f:                       # read forward: ~12 ms a frame, where a seek costs ~2 s
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            res['deface'].append(np.zeros((0, 5)))
            continue
        t = time.time()
        res['deface'].append(_models['deface'].find(frame))
        res['secs'].append(time.time() - t)
        k = _models['yolo'](frame)
        j, _ = pick_subject(k, m2d[i])
        if j >= 0:
            res['coco'][i] = k[j]
        res['size'] = frame.shape[:2]
    cap.release()
    return trial['action'], cam, res


def score(trial, idx, per_cam):
    from blur import inside
    cams = {c: Camera(trial['calibs'][c]) for c in ALL_CAMS}
    xy = np.stack([per_cam[c]['coco'][:, FACE, :2] for c in ALL_CAMS], 1)
    vis = np.stack([per_cam[c]['coco'][:, FACE, 2] >= CONF for c in ALL_CAMS], 1)
    face, _, _ = triangulate_robust(xy, vis, [cams[c].as_tuple() for c in ALL_CAMS], thresh_px=20.0)
    ear_mid = np.nanmean(face[:, 3:5], 1)
    nose = np.where(np.isfinite(face[:, 0]), face[:, 0], np.nanmean(face[:, 1:3], 1))
    centre = (nose + ear_mid) / 2
    fwd = nose - ear_mid
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    rows = []
    for c in ALL_CAMS:
        cm = cams[c]
        to_cam = cm.centre - centre
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.degrees(np.arccos(np.clip((fwd * to_cam).sum(1), -1, 1)))
        pts = cm.project(face[:, :3])
        ctr = cm.project(centre)
        for i, f in enumerate(idx):
            p = pts[i][np.isfinite(pts[i]).all(1)]
            if not np.isfinite(ctr[i]).all() or not np.isfinite(facing[i]):
                continue                                   # head not solved in this frame
            if len(p) == 0:
                p = ctr[i][None]
            row = dict(action=trial['action'], subject=trial['action'].split('_')[0],
                       task=trial['action'].split('_')[1], cam=c, frame=int(f), facing_deg=facing[i],
                       face_px=cm.K[0, 0] * 160.0 / cm.depth(centre[i]),     # ~ 16 cm head height in px
                       in_frame=bool((0 <= ctr[i][0] < per_cam[c]['size'][1]) & (0 <= ctr[i][1] < per_cam[c]['size'][0])))
            regs = np.asarray(per_cam[c]['deface'][i]).reshape(-1, 5)
            row['deface_covered'] = bool(inside(regs, 'ellipse', p).all())
            row['deface_n_regions'] = len(regs)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=TRIAL_ROOT)
    ap.add_argument('--actions', nargs='*', default=None, help='trial folder names; default every trial found')
    ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=out_dir('recall_survey'))
    args = ap.parse_args()

    trials = find_trials(args.root, args.actions)
    if not trials:
        raise SystemExit(f'no trial under {args.root} with 9 videos, 9 calibrations and a mocap')
    print(f'{len(trials)} trials: ' + ' '.join(t['action'] for t in trials), flush=True)
    samples = {t['action']: sample_frames(t['mocap'], args.frames) for t in trials}
    per = {t['action']: {} for t in trials}
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=_init) as pool:
        futs = [pool.submit(run_camera, t, c, *samples[t['action']]) for t in trials for c in ALL_CAMS]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            a, c, res = fut.result()
            per[a][c] = res
            if n % 9 == 0:
                print(f'  {n}/{len(futs)} camera jobs, {time.time() - t0:.0f} s', flush=True)

    rows = []
    for t in trials:
        rows += score(t, samples[t['action']][0], per[t['action']])
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'frames.csv'), index=False, float_format='%.4g')

    vis = df[(df.facing_deg < 90) & df.in_frame]
    vis = vis.assign(facing=pd.cut(vis.facing_deg, [0, 30, 60, 90]),
                     face_size=pd.cut(vis.face_px, [0, 40, 60, 80, 1000]))
    print(f'\n{len(df)} frame-camera samples, {len(vis)} with the face turned towards the camera\n')
    for by in ('task', 'subject', 'cam', 'facing', 'face_size'):
        print(vis.groupby(by, observed=True).agg(n=('deface_covered', 'size'), deface=('deface_covered', 'mean'))
              .round(3).to_string(), '\n')
    print(f'overall: deface {vis.deface_covered.mean():.1%} of {len(vis)}')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
