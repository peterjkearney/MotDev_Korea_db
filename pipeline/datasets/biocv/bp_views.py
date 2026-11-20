#!/usr/bin/env python3
"""bp_views.py -- viewing angle of each camera on the subject, and the front / oblique / side camera per trial
for the BlazePose head-to-head (run_biocv_bp.py's 'cameras' stage).

Viewing angle (deg, 0..180) per frame = the angle, in the floor plane, between the subject's facing direction
(mocap: up x (RHip - LHip)) and the direction from the mocap pelvis to the camera centre.  0 = the camera sees the
front, 90 = a side, 180 = the back.  Per-frame bins (VIEW_BINS) label the frames for the breakdown by view, so a
walk that turns round is labelled frame by frame, not by camera.

Camera choice per trial: among cameras where OpenPose matched the subject on >= --min-coverage of the frames,
the one whose median viewing angle is closest to 0 (front), then 90 (side), then 45 (oblique), each camera used
once.  Writes {OUT_DIR}/bp_cameras.json:

    {user: {action: {"front": cam, "side": cam, "oblique": cam,
                     "cams": {cam: {"coverage": f, "median_angle": deg}}}}}

    python3 datasets/biocv/bp_views.py --trials User04/P04_CMJM_01,User04/P04_WALK_01
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # pipeline/
from config import OUT_DIR, calib_path, mocap_path, twod_path  # noqa: E402
from utils.calibration import load_calib  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])
ROLES = (('front', 0.0), ('side', 90.0), ('oblique', 45.0))        # picking order
VIEW_BINS = (('front', 0, 30), ('oblique', 30, 60), ('side', 60, 120), ('rear', 120, 180.01))


def bp_cameras_path():
    return os.path.join(OUT_DIR, 'bp_cameras.json')


def facing(kps):
    """(T,3) unit facing direction in the floor plane from H36M-17 lab joints (NaN where a hip is missing)."""
    r = kps[:, 1] - kps[:, 4]
    r[:, 2] = 0
    f = np.cross(UP, r)
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def camera_centre(L):
    """Lab-mm camera centre from the lab->camera extrinsic."""
    R, t = L[:3, :3], L[:3, 3]
    return -R.T @ t


def view_angle(kps, L):
    """(T,) viewing angle in degrees (0 front, 90 side, 180 back) of camera L on the subject in kps (lab mm)."""
    f = facing(kps)
    d = camera_centre(L)[None] - kps[:, 0]
    d[:, 2] = 0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((f * d).sum(1), -1, 1)))


def view_bin(angle):
    out = np.full(len(angle), '', dtype=object)
    for name, lo, hi in VIEW_BINS:
        out[(angle >= lo) & (angle < hi)] = name
    return out


def trial_mocap(user, action, sfi):
    m = np.load(mocap_path(user, action))
    k = m['kps3d'].astype(float)
    out = np.full((len(sfi), 17, 3), np.nan)
    ok = sfi < len(k)
    out[ok] = k[sfi[ok]]
    return out


def pick(user, action, cams, min_coverage):
    info = {}
    for c in cams:
        p = twod_path(user, action, 'openpose', c)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        sfi, det = d['source_frame_idx'].astype(int), d['detected'].astype(bool)
        G = trial_mocap(user, action, sfi)
        ang = view_angle(G, load_calib(calib_path(user, c, action))[3])
        ok = det & np.isfinite(ang)
        info[c] = dict(coverage=float(ok.mean()), median_angle=float(np.median(ang[ok])) if ok.any() else None)
    usable = {c: v for c, v in info.items() if v['coverage'] >= min_coverage and v['median_angle'] is not None}
    out = {}
    for role, target in ROLES:
        left = {c: v for c, v in usable.items() if c not in out.values()}
        if left:
            out[role] = min(left, key=lambda c: abs(left[c]['median_angle'] - target))
    out['cams'] = info
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trials', required=True, help='comma-separated user/action')
    ap.add_argument('--cams', default='00,01,02,03,04,05,06,07,08')
    ap.add_argument('--min-coverage', type=float, default=0.6)
    ap.add_argument('--force', action='store_true', help='re-pick trials already in bp_cameras.json')
    args = ap.parse_args()
    path = bp_cameras_path()
    doc = json.load(open(path)) if os.path.exists(path) else {}
    for t in args.trials.split(','):
        u, a = t.split('/')
        if a in doc.get(u, {}) and not args.force:
            continue
        r = pick(u, a, args.cams.split(','), args.min_coverage)
        doc.setdefault(u, {})[a] = r
        roles = ', '.join(f"{k} {r[k]} ({r['cams'][r[k]]['median_angle']:.0f} deg)" for k, _ in ROLES if k in r)
        print(f'{u}/{a}: {roles or "NO usable camera"}', flush=True)
        if any(k not in r for k, _ in ROLES):
            print(f'  WARNING fewer than 3 cameras with coverage >= {args.min_coverage}: '
                  + ', '.join(f"{c} {v['coverage']:.2f}" for c, v in r['cams'].items()))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
