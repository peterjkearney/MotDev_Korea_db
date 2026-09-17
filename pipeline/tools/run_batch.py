#!/usr/bin/env python3
"""Run the full pipeline over every user/action found under TRIAL_DIR.

Discovers users, then the action folders inside them, then for each trial runs
steps 0,1,2a,2b,3,4,6,8 (the renders, 5 and 7, are left out by default), copies
Analysis/ to RESULTS_DIR, and appends to a results table -- rewritten after
every trial, so a Colab runtime dying loses at most the trial in flight.

    python3 tools/run_batch.py                        # everything
    python3 tools/run_batch.py --dry-run              # show the plan, run nothing
    python3 tools/run_batch.py --users User03,User04
    python3 tools/run_batch.py --force                # redo trials already done

Resume is keyed on RESULTS_DIR (which is on Drive and survives the runtime),
not on the local working tree, so re-running after a disconnect picks up where
it left off.

Outputs, all under RESULTS_DIR:
    results_table.csv   one row per (user, action, camera) -- betas, per-joint
                        and mean error vs mocap, camera angle
    run_log.csv         one row per (user, action, step) -- ok/fail/skip, secs
    {user}/{action}/Analysis/...   the synced per-trial outputs
"""
import argparse
import csv
import datetime
import glob
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
sys.path.insert(0, _PIPE)
from config import TRIAL_DIR, RESULTS_DIR, mocap_path, twod_path
from sync_results import sync_trial

import numpy as np

# name -> (script, takes --cameras, outputs relative to Analysis/ given cameras)
# Two 2D sources and two ground truths, chosen with --steps and --gt:
#   YOLO adults   : step_0,step_1,step_2a,step_2b,step_3,step_4,step_6,step_8   (--gt mocap)
#   OpenPose adults: step_0,step_1op,step_1t,step_2a,step_2b,step_3,step_4,step_6,step_8
#                   (--gt mocap or --gt triangulated; step_8 runs once per --gt)
#   Korea children : step_2a,step_2b,step_3,step_4,step_6,step_8  (--gt triangulated),
#                   after step_1_korea_2d.py has laid the trials out
STEPS = [
    ('step_0',   'step_0_load_mocap.py',       False, lambda c: ['keypoints/mocap_h36m.npz']),
    ('step_1',   'step_1_extract_2d.py',       False, lambda c: [f'keypoints/{x}_2d.npz' for x in c]),
    ('step_1op', 'step_1_openpose_2d.py',      True,  lambda c: [f'keypoints/{x}_2d.npz' for x in c]),
    ('step_1t',  'step_1b_triangulate_2d.py',  True,  lambda c: ['keypoints/openpose_tri_h36m.npz']),
    ('step_2a',  'step_2a_extract_betas.py',   True,  lambda c: [f'keypoints/mesh/{x}_betas.npz' for x in c]),
    ('step_2b',  'step_2b_finalise_betas.py',  True,  lambda c: [f'keypoints/mesh/{x}_final_betas.npz' for x in c]),
    ('step_3',   'step_3_extract_3d.py',       True,  lambda c: [f'keypoints/mesh/{x}_mesh_pose.npz' for x in c]),
    ('step_4',   'step_4_PnP.py',              True,  lambda c: [f'keypoints/PnP/{x}_pnp.npz' for x in c]),
    ('step_6',   'step_6_extract_features.py', True,  lambda c: [f'features/{x}_features.npz' for x in c]),
    ('step_8',   'step_8_spider_error.py',     True,  lambda c: ['diagnostics/error_metrics.npz']),
]
STEP_NAMES = [s[0] for s in STEPS]
DEFAULT_STEPS = 'step_0,step_1,step_2a,step_2b,step_3,step_4,step_6,step_8'


def metrics_file(gt):
    return 'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz'

# step_8 records per-joint error four ways; the table carries one of them as
# columns and the npz keeps all four.
VARIANTS = ('smooth_all', 'smooth_conf', 'placed_all', 'placed_conf')


# ---------------------------------------------------------------- discovery

def _subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def cameras_for(user, action):
    """Camera ids actually present.

    Read off disk rather than assumed to be 00..08, so trials missing a camera
    do not have every later step try to load a file that was never made.
    From the videos where there are any (BioCV); otherwise from the 2D files
    already laid out (Korea, which has no footage) or the OpenPose JSON dirs.
    """
    d = os.path.join(TRIAL_DIR, user, action)
    vids = glob.glob(os.path.join(d, '0*.mp4'))
    if vids:
        return sorted(os.path.splitext(os.path.basename(v))[0] for v in vids)
    twod = glob.glob(os.path.join(d, 'Analysis', 'keypoints', '*_2d.npz'))
    if twod:
        return sorted(os.path.basename(p)[:-7] for p in twod)
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, 'openpose', '*'))
                  if os.path.isdir(p))


def is_action_dir(user, action):
    d = os.path.join(TRIAL_DIR, user, action)
    return (bool(glob.glob(os.path.join(d, '0*.mp4')))
            or os.path.exists(os.path.join(d, 'markers.c3d'))
            or bool(glob.glob(os.path.join(d, 'Analysis', 'keypoints', '*_2d.npz'))))


def discover(users_filter, actions_filter):
    """(trials, unmatched) -- folder names are matched case-insensitively, so
    --users user06 finds User06. Names that matched nothing come back in
    `unmatched` rather than silently narrowing the run to zero trials."""
    users_lc = {u.lower() for u in users_filter} if users_filter else None
    actions_lc = {a.lower() for a in actions_filter} if actions_filter else None
    seen_users, seen_actions, trials = set(), set(), []

    for user in _subdirs(TRIAL_DIR):
        if users_lc is not None and user.lower() not in users_lc:
            continue
        seen_users.add(user.lower())
        for action in _subdirs(os.path.join(TRIAL_DIR, user)):
            if not is_action_dir(user, action):
                continue
            if actions_lc is not None and action.lower() not in actions_lc:
                continue
            seen_actions.add(action.lower())
            trials.append((user, action))

    unmatched = []
    if users_lc:
        unmatched += [u for u in users_filter if u.lower() not in seen_users]
    if actions_lc:
        unmatched += [a for a in actions_filter if a.lower() not in seen_actions]
    return trials, unmatched


def blockers(user, action, cameras, steps):
    """Reasons this trial cannot run at all -- checked up front so a bad trial
    fails in a second rather than after step_1 has chewed through nine videos."""
    names = {s[0] for s in steps}
    out = []
    if not cameras:
        out.append('no cameras found (videos, 2D files or OpenPose dirs)')
    if 'step_0' in names and not os.path.exists(os.path.join(TRIAL_DIR, user, action, 'markers.c3d')):
        out.append('no markers.c3d (step_0)')
    if not os.path.exists(os.path.join(TRIAL_DIR, user, 'user_meta.json')):
        out.append('no user_meta.json (step_3 needs stature_m)')
    missing = [c for c in cameras
               if not os.path.exists(os.path.join(TRIAL_DIR, user, f'{c}.mp4-mocAligned.calib'))]
    if missing:
        out.append(f'no calib for camera(s) {",".join(missing)}')
    return out


# ------------------------------------------------------------------ running

def analysis_dir(user, action):
    return os.path.join(TRIAL_DIR, user, action, 'Analysis')


def outputs_present(user, action, step, cameras):
    name, _, _, outs = step
    # steps already moved to OUT_DIR (their paths come from config, not from `outs`)
    if name == 'step_0':
        return os.path.exists(mocap_path(user, action))
    if name in ('step_1', 'step_1op'):
        det = 'yolo' if name == 'step_1' else 'openpose'
        return all(os.path.exists(twod_path(user, action, det, c)) for c in cameras)
    base = analysis_dir(user, action)
    return all(os.path.exists(os.path.join(base, r)) for r in outs(cameras))


def is_done(user, action, gt='mocap'):
    """Completion is judged on Drive, not local disk -- local disk is wiped
    when the runtime dies, Drive is what actually persists."""
    return os.path.exists(os.path.join(RESULTS_DIR, user, action, 'Analysis',
                                       'diagnostics', metrics_file(gt)))


def run_step(user, action, step, cameras, args):
    name, script, takes_cameras, _ = step
    cmd = [sys.executable, os.path.join(_PIPE, script),
           '--user', user, '--action', action]
    if takes_cameras:
        cmd += ['--cameras', ','.join(cameras)]
    if name in ('step_6', 'step_8'):
        cmd += ['--conf-thresh', str(args.conf_thresh)]
    if name == 'step_8':
        cmd += ['--gt', args.gt]
    if name == 'step_4':
        cmd += ['--process-accel-std', str(args.accel_std)]

    log_dir = os.path.join(analysis_dir(user, action), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{name}.log')

    t0 = time.time()
    with open(log_path, 'w') as log:
        log.write(' '.join(cmd) + '\n\n')
        log.flush()
        try:
            # Each step is a subprocess, not an import: the torch/CUDA state a
            # step builds is released when it exits, and a segfault or OOM in
            # one step cannot take the whole batch down with it.
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               cwd=_PIPE, env=os.environ.copy(),
                               timeout=args.step_timeout or None)
            rc, err = p.returncode, ''
        except subprocess.TimeoutExpired:
            rc, err = -1, f'timeout after {args.step_timeout}s'
    return rc, time.time() - t0, err, log_path


# ------------------------------------------------------------------- tables

def _atomic_write_csv(path, fieldnames, rows):
    """Write via a temp file + rename so a runtime death mid-write cannot leave
    a half-written table on Drive."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def metric_rows(user, action, variant, gt='mocap'):
    """One row per camera for this trial, from step_8's npz plus step_2b's betas."""
    m_path = os.path.join(analysis_dir(user, action), 'diagnostics', metrics_file(gt))
    if not os.path.exists(m_path):
        return []
    m = np.load(m_path, allow_pickle=True)

    names = [str(x) for x in m['joint_names']]
    per_joint = m[f'perjoint_{variant}']
    cameras = [str(c) for c in m['cameras']]
    stature = float(m['stature_mm']) if 'stature_mm' in m.files else np.nan
    bone_names = [str(b) for b in m['bone_names']] if 'bone_names' in m.files else []

    rows = []
    for i, cam in enumerate(cameras):
        # betas are per camera now (step_2b no longer averages across views)
        betas, beta_frames = np.full(10, np.nan), ''
        b_path = os.path.join(analysis_dir(user, action), 'keypoints', 'mesh', f'{cam}_final_betas.npz')
        if os.path.exists(b_path):
            bz = np.load(b_path, allow_pickle=True)
            b = bz['betas'].ravel()
            betas[:len(b)] = b[:10]
            if 'n_frames_used' in bz.files:
                beta_frames = int(bz['n_frames_used'])
        row = {
            'user': user, 'action': action, 'camera': cam, 'gt': gt,
            'n_cameras': len(cameras),
            'angle_deg': round(float(m['angles_deg'][i]), 2),
            'n_frames': int(m['n_frames'][i]),
            'conf_thresh': float(m['conf_thresh']),
            'stature_mm': '' if np.isnan(stature) else round(stature, 1),
            'beta_n_frames': beta_frames,
            'joint_variant': variant,
        }
        for v in VARIANTS:
            e = float(m[f'err_{v}'][i])
            row[f'err_{v}_mm'] = round(e, 3)
            row[f'err_{v}_pct_stature'] = '' if np.isnan(stature) else round(100 * e / stature, 3)
        if 'pa_mpjpe' in m.files:
            row['pa_mpjpe_mm'] = round(float(m['pa_mpjpe'][i]), 3)
            row['n_mpjpe_mm'] = round(float(m['n_mpjpe'][i]), 3)
        for k in range(10):
            row[f'beta_{k:02d}'] = round(float(betas[k]), 5)
        for j, nm in enumerate(names):
            val = float(per_joint[i, j])
            row[f'errj_{nm}_mm'] = '' if np.isnan(val) else round(val, 3)
        for j, bn in enumerate(bone_names):
            val = float(m['bone_ratio'][i, j])
            row[f'bone_{bn}_ratio'] = '' if np.isnan(val) else round(val, 4)
        rows.append(row)
    return rows


def results_fieldnames(names, bone_names=()):
    f = ['user', 'action', 'camera', 'gt', 'n_cameras', 'beta_n_frames', 'angle_deg',
         'n_frames', 'conf_thresh', 'stature_mm', 'joint_variant']
    f += [f'err_{v}_mm' for v in VARIANTS]
    f += [f'err_{v}_pct_stature' for v in VARIANTS]
    f += ['pa_mpjpe_mm', 'n_mpjpe_mm']
    f += [f'beta_{k:02d}' for k in range(10)]
    f += [f'errj_{n}_mm' for n in names]
    f += [f'bone_{b}_ratio' for b in bone_names]
    return f


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', help='comma-separated subset, default all found')
    ap.add_argument('--actions', help='comma-separated subset, default all found')
    ap.add_argument('--steps', default=DEFAULT_STEPS,
                    help=f'default: {DEFAULT_STEPS}; pick from {",".join(STEP_NAMES)}')
    ap.add_argument('--gt', default='mocap', choices=['mocap', 'triangulated'],
                    help='ground truth for step_8 and for the results table')
    ap.add_argument('--joint-variant', default='smooth_all', choices=VARIANTS,
                    help='which error series fills the per-joint columns')
    ap.add_argument('--conf-thresh', type=float, default=0.3)
    ap.add_argument('--accel-std', type=float, default=10.0)
    ap.add_argument('--step-timeout', type=int, default=0, help='seconds, 0 = none')
    ap.add_argument('--force', action='store_true', help='redo trials already in RESULTS_DIR')
    ap.add_argument('--rerun-steps', action='store_true',
                    help='rerun steps whose outputs already exist locally')
    ap.add_argument('--include-videos', action='store_true', help='sync renders too')
    ap.add_argument('--stop-on-fail', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    steps = [s for s in STEPS if s[0] in args.steps.split(',')]
    if not steps:
        ap.error(f'--steps matched nothing; pick from {",".join(STEP_NAMES)}')

    trials, unmatched = discover(
        [u for u in args.users.split(',') if u] if args.users else None,
        [a for a in args.actions.split(',') if a] if args.actions else None)

    print(f'TRIAL_DIR   {TRIAL_DIR}')
    print(f'RESULTS_DIR {RESULTS_DIR}')
    print(f'steps       {",".join(s[0] for s in steps)}')
    print(f'found       {len(trials)} trial(s)\n')

    if unmatched:
        print(f'WARNING: no match for {",".join(unmatched)}')
        print(f'         users present: {",".join(_subdirs(TRIAL_DIR)) or "(none)"}\n')

    if not trials:
        # Say which of the two it is: an empty/wrong root, or a filter that
        # excluded everything. They need opposite fixes.
        present = _subdirs(TRIAL_DIR)
        if not present:
            print(f'nothing to do -- no user folders under {TRIAL_DIR}')
            print('  check BIOCV_ROOT, or that the copy to local disk finished')
        else:
            print('nothing to do -- filters excluded every trial')
            print(f'  users present: {",".join(present)}')
        return

    plan = []
    for user, action in trials:
        cams = cameras_for(user, action)
        why = blockers(user, action, cams, steps)
        if is_done(user, action, args.gt) and not args.force:
            state = 'done'
        elif why:
            state = 'blocked: ' + '; '.join(why)
        else:
            state = 'run'
        plan.append((user, action, cams, state))
        print(f'  {user}/{action:<22} {len(cams)} cam  {state}')

    if args.dry_run:
        print('\ndry run -- nothing executed')
        return

    results_path = os.path.join(RESULTS_DIR, 'results_table.csv' if args.gt == 'mocap'
                                else 'results_table_tri.csv')
    log_path = os.path.join(RESULTS_DIR, 'run_log.csv')
    results = _read_csv(results_path)
    run_log = _read_csv(log_path)
    log_fields = ['timestamp', 'user', 'action', 'step', 'status', 'seconds', 'detail']

    todo = [p for p in plan if p[3] == 'run']
    print(f'\nrunning {len(todo)} trial(s)\n' + '=' * 60)

    for n, (user, action, cams, _) in enumerate(todo, 1):
        print(f'\n[{n}/{len(todo)}] {user}/{action}  cameras {",".join(cams)}')
        failed = None
        for step in steps:
            name = step[0]
            stamp = datetime.datetime.now().isoformat(timespec='seconds')
            if not args.rerun_steps and outputs_present(user, action, step, cams):
                print(f'    {name:<8} skip (outputs present)')
                run_log.append(dict(timestamp=stamp, user=user, action=action,
                                    step=name, status='skip', seconds=0, detail=''))
                continue
            rc, secs, err, lp = run_step(user, action, step, cams, args)
            status = 'ok' if rc == 0 else 'fail'
            print(f'    {name:<8} {status} ({secs:.0f}s)' +
                  ('' if rc == 0 else f'  rc={rc} {err} -- see {lp}'))
            run_log.append(dict(timestamp=stamp, user=user, action=action, step=name,
                                status=status, seconds=round(secs, 1),
                                detail=err or (f'rc={rc}' if rc else '')))
            if rc != 0:
                failed = name
                break

        # Sync and write tables even on failure: the logs are the only record of
        # why it broke, and local disk does not survive the runtime.
        copied, _ = sync_trial(user, action, args.include_videos)

        rows = metric_rows(user, action, args.joint_variant, args.gt)
        if rows:
            results = [r for r in results
                       if not (r.get('user') == user and r.get('action') == action)]
            results += rows
            names = [k[5:-3] for k in rows[0] if k.startswith('errj_')]
            bones = [k[5:-6] for k in rows[0] if k.startswith('bone_') and k.endswith('_ratio')]
            _atomic_write_csv(results_path, results_fieldnames(names, bones), results)
            print(f'    table    +{len(rows)} row(s) -> {results_path}')
        elif not failed:
            print('    table    no error_metrics.npz -- no rows added')

        _atomic_write_csv(log_path, log_fields, run_log)

        if failed and args.stop_on_fail:
            print(f'\nstopping: {user}/{action} failed at {failed}')
            break

    ok = sum(1 for r in run_log if r.get('status') == 'ok')
    bad = [r for r in run_log if r.get('status') == 'fail']
    print('\n' + '=' * 60)
    print(f'steps ok {ok}, failed {len(bad)}')
    for r in bad:
        print(f"  FAIL {r['user']}/{r['action']} {r['step']}")
    print(f'results  {results_path}')
    print(f'log      {log_path}')


if __name__ == '__main__':
    main()
