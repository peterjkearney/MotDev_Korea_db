#!/usr/bin/env python3
"""run_biocv_bp.py -- our best single-camera pipeline on BioCV adults, for the head-to-head against MediaPipe
BlazePose (analysis/blazepose/bp_extract.py, bp_compare.py).

Ours = OpenPose 2D at 30 fps -> fine-tuned MotionBERT-Mesh (MB_MESH_CKPT, clip 16, no flip) -> C19 with the
subject's step_2c shape (fitted to multi-camera OpenPose bone lengths: the stand-in for limb lengths known from
calibration poses) and the mocap floor.  Each camera is its own single-camera run.

Everything is written to a FRESH tree (BIOCV_OUT): the existing BioCV tree holds 60 fps 2D and P03's older runs,
which 30 fps outputs must not overwrite.  Inputs -- videos, calibrations, user_meta.json, OpenPose JSONs and
mocap_h36m.npz -- are read from the existing tree (BIOCV_ROOT); 'setup' copies the JSONs and the mocap across.

Stages (each step skips what is done, so re-running resumes):
    setup     copy {cam}_openpose.json + mocap_h36m.npz per trial; user_meta.json where the source has none
    step_1    OpenPose JSON -> 2D at 30 fps, all 9 cameras        (datasets/biocv/step_1_openpose_2d.py)
    step_1b   triangulated OpenPose target, all 9 cameras          (step_2c's bone lengths)
    cameras   front / side / oblique camera per trial              (datasets/biocv/bp_views.py -> bp_cameras.json)
    step_2a..step_4   MotionBERT (GPU) on the 3 chosen cameras
    step_2c   shapes per subject
    step_4b   C19 on the 3 chosen cameras
    step_8    our own mocap scoring of C19 (cross-check for bp_compare)
    blazepose analysis/blazepose/bp_extract.py on the same frames, whole frame AND cropped to our subject (Tasks API; pip install mediapipe==0.10.35 on Colab)
    compare   analysis/blazepose/bp_compare.py for both -> bp_compare/ (as is) and bp_compare_crop/ (subject given)

Colab:
    %env BIOCV_ROOT=/content/drive/MyDrive/MotorDevelopment/Data/BioCV
    %env BIOCV_OUT=/content/drive/MyDrive/MotorDevelopment/Data/BioCV_bp30
    %env MB_DIR=/content/drive/MyDrive/.../MotionBERT
    %env MB_MESH_CKPT=/content/drive/MyDrive/MotorDevelopment/Data/finetune_runs/run1/best.bin
    !python3 pipeline/datasets/biocv/run_biocv_bp.py                      # test subjects + P03
    !python3 pipeline/datasets/biocv/run_biocv_bp.py --group p03 --until cameras
    !python3 pipeline/datasets/biocv/run_biocv_bp.py --from step_4b
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(os.path.dirname(_HERE))
_REPO = os.path.dirname(_PIPE)
sys.path.insert(0, _PIPE)
from config import OUT_DIR, TRIAL_DIR, mocap_path, openpose_json_path, require_out_dir  # noqa: E402

# the agreed test set: 10 adults, CMJ + walk for all, hop + run where they exist (2026-11-20 Drive listing)
_HOP_RUN = ('04', '06', '08', '09', '10', '13', '28')
TEST = {f'User{n}': [f'P{n}_CMJM_01', f'P{n}_WALK_01'] + ([f'P{n}_HOP_01', f'P{n}_RUN_01'] if n in _HOP_RUN else [])
        for n in ('04', '06', '08', '09', '10', '13', '16', '17', '26', '28')}
P03 = {'User03': ['P03_CMJM_01', 'P03_HOP_01', 'P03_RUN_01', 'P03_WALK_01']}     # tuned on: reported separately
# participantData.csv (BioCV release); used only where the source tree has no user_meta.json
STATURE_M = {'User03': 1.89, 'User04': 1.896, 'User06': 1.68, 'User08': 1.82, 'User09': 1.79, 'User10': 1.658,
             'User13': 1.76, 'User16': 1.82, 'User17': 1.59, 'User26': 1.58, 'User28': 1.56}
CAMS = [f'{i:02d}' for i in range(9)]
FPS = 30
CONFIG = 'C19'
STAGES = ['setup', 'step_1', 'step_1b', 'cameras', 'step_2a', 'step_2b', 'step_3', 'step_4', 'step_2c',
          'step_4b', 'step_8', 'blazepose', 'compare']


def run(script, *cli):
    cmd = [sys.executable, os.path.join(_REPO, script), *cli]
    print(f'\n$ {script} ' + ' '.join(cli), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=_PIPE)
    print(f'  [{"ok" if r.returncode == 0 else "FAILED rc=" + str(r.returncode)}, {(time.time() - t0) / 60:.1f} min]', flush=True)
    return r.returncode == 0


def setup(trials):
    """Copy the per-trial inputs into OUT_DIR; returns the trials that have every camera's JSON."""
    if os.path.abspath(TRIAL_DIR) == os.path.abspath(OUT_DIR):
        raise SystemExit('BIOCV_OUT is BIOCV_ROOT -- point BIOCV_OUT at a fresh folder (30 fps outputs must not '
                         'overwrite the 60 fps tree)')
    ready = []
    for u, a in trials:
        src = os.path.join(TRIAL_DIR, u, a, 'Analysis')
        missing = []
        for c in CAMS:
            dst = openpose_json_path(u, a, c)
            if os.path.exists(dst):
                continue
            cand = [os.path.join(src, 'keypoints', 'openpose', f'{c}_openpose.json'),
                    os.path.join(src, 'openpose', f'{c}_openpose.json')]          # older layout
            s = next((p for p in cand if os.path.exists(p)), None)
            if s is None:
                missing.append(c)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(s, dst + '.part')
            os.replace(dst + '.part', dst)
        mdst, msrc = mocap_path(u, a), os.path.join(src, 'H36M', 'mocap_h36m.npz')
        if not os.path.exists(mdst):
            if os.path.exists(msrc):
                os.makedirs(os.path.dirname(mdst), exist_ok=True)
                shutil.copy2(msrc, mdst)
            else:
                missing.append('mocap_h36m')
        calibs = [c for c in CAMS if not any(os.path.exists(os.path.join(r, u, f'{c}.mp4-mocAligned.calib'))
                                             for r in (TRIAL_DIR, OUT_DIR))]
        if calibs:
            missing.append('calib ' + ','.join(calibs))
        print(f'{u}/{a}: ' + ('ready' if not missing else 'MISSING ' + ' '.join(missing)), flush=True)
        if not missing:
            ready.append((u, a))
    for u in sorted({u for u, _ in trials}):
        meta_src, meta_dst = os.path.join(TRIAL_DIR, u, 'user_meta.json'), os.path.join(OUT_DIR, u, 'user_meta.json')
        if os.path.exists(meta_src):
            got = json.load(open(meta_src)).get('stature_m')
            if got is not None and abs(got - STATURE_M[u]) > 0.005:
                print(f'WARNING {u}: user_meta.json says {got} m, participantData.csv {STATURE_M[u]} m', flush=True)
        elif not os.path.exists(meta_dst):
            os.makedirs(os.path.dirname(meta_dst), exist_ok=True)
            with open(meta_dst, 'w') as f:
                json.dump({'stature_m': STATURE_M[u], 'source': 'participantData.csv'}, f)
            print(f'{u}: wrote user_meta.json ({STATURE_M[u]} m)', flush=True)
    return ready


def chosen_cameras(u, a):
    p = os.path.join(OUT_DIR, 'bp_cameras.json')
    doc = json.load(open(p)) if os.path.exists(p) else {}
    r = doc.get(u, {}).get(a)
    return [r[k] for k in ('front', 'side', 'oblique') if k in r] if r else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', choices=['test', 'p03', 'all'], default='all')
    ap.add_argument('--users', default=None, help='comma-separated, narrows the group (e.g. User04,User06)')
    ap.add_argument('--actions', default=None, help='comma-separated, narrows the group')
    ap.add_argument('--from', dest='start', default='setup', choices=STAGES)
    ap.add_argument('--until', default='compare', choices=STAGES)
    ap.add_argument('--device', default='auto', help='step_4b: cpu | cuda | auto')
    ap.add_argument('--stop-on-fail', action='store_true')
    args = ap.parse_args()
    require_out_dir()

    groups = {'test': TEST, 'p03': P03, 'all': {**P03, **TEST}}[args.group]
    users = set(args.users.split(',')) if args.users else None
    actions = set(args.actions.split(',')) if args.actions else None
    trials = [(u, a) for u, acts in groups.items() for a in acts
              if (users is None or u in users) and (actions is None or a in actions)]
    i0, i1 = STAGES.index(args.start), STAGES.index(args.until)
    stages = STAGES[i0:i1 + 1]
    ckpt = os.environ.get('MB_MESH_CKPT')
    print(f'BIOCV_ROOT {TRIAL_DIR}\nBIOCV_OUT  {OUT_DIR}\nMB_MESH_CKPT {ckpt}\nstages {", ".join(stages)}\n'
          f'{len(trials)} trials of {len({u for u, _ in trials})} subjects', flush=True)
    if any(s in stages for s in ('step_2a', 'step_3')) and not ckpt:
        raise SystemExit('MB_MESH_CKPT is not set -- the test uses the fine-tuned run1 best.bin')

    if 'setup' in stages:
        ready = setup(trials)
        if len(ready) < len(trials):
            print(f'\n{len(trials) - len(ready)} trial(s) not ready (see MISSING above); they are skipped by later stages')
    sel = lambda u, a: ['--user', u, '--action', a]            # noqa: E731
    ok = True
    for st in stages:
        if st == 'setup':
            continue
        if st == 'step_1':
            ok = all([run('pipeline/datasets/biocv/step_1_openpose_2d.py', *sel(u, a), '--target-fps', str(FPS))
                      for u, a in trials if os.path.exists(openpose_json_path(u, a, '00'))])
        elif st == 'step_1b':
            ok = all([run('pipeline/datasets/biocv/step_1b_triangulate_2d.py', *sel(u, a)) for u, a in trials])
        elif st == 'cameras':
            ok = run('pipeline/datasets/biocv/bp_views.py', '--trials', ','.join(f'{u}/{a}' for u, a in trials))
        elif st in ('step_2a', 'step_2b', 'step_3', 'step_4', 'step_4b', 'step_8'):
            script = {'step_2a': 'step_2a_extract_betas.py', 'step_2b': 'step_2b_finalise_betas.py',
                      'step_3': 'step_3_extract_3d.py', 'step_4': 'step_4_PnP.py', 'step_4b': 'step_4b_refine.py',
                      'step_8': 'step_8_spider_error.py'}[st]
            # the Korea recipe's flags (run_korea.py); step_2a: clip 16 (its default), stride 1 on 30 fps 2D, no flip
            extra = {'step_4': ['--exclude-joints', 'Hip,RHip,LHip'],
                     'step_4b': ['--configs', CONFIG, '--device', args.device],
                     'step_8': ['--gt', 'mocap', '--config', CONFIG]}.get(st, [])
            ok = True
            for u, a in trials:
                cams = chosen_cameras(u, a)
                if not cams:
                    print(f'{u}/{a}: no chosen cameras (run the cameras stage)', flush=True)
                    ok = False
                    continue
                ok = run(f'pipeline/{script}', *sel(u, a), '--cameras', ','.join(cams), *extra) and ok
        elif st == 'step_2c':
            ok = all([run('pipeline/step_2c_fit_shape.py', '--user', u) for u in sorted({u for u, _ in trials})])
            ok = run('analysis/blazepose/bp_compare.py', '--shapes-only') and ok
        elif st == 'blazepose':
            tl = ','.join(f'{u}/{a}' for u, a in trials)
            ok = run('analysis/blazepose/bp_extract.py', '--trials', tl)                  # as is
            ok = run('analysis/blazepose/bp_extract.py', '--trials', tl, '--crop') and ok   # subject given
        elif st == 'compare':
            ok = run('analysis/blazepose/bp_compare.py', '--config', CONFIG)
            ok = run('analysis/blazepose/bp_compare.py', '--config', CONFIG, '--bp-suffix', '_crop') and ok
        if not ok and args.stop_on_fail:
            print(f'stopping at {st}')
            break


if __name__ == '__main__':
    main()
