#!/usr/bin/env python3
"""sample_hip_ablation.py -- one Korea rep through the pipeline, PnP with and without the hips.

The hips are the one place the pipeline's "same body point" assumption fails:
MotionBERT's H36M hips (regressed from the SMPL mesh) are wide and sit at
femoral-head height, OpenPose's are narrower and lower.  step_4 feeds both to
solvePnP as if they matched.  This runs a single rep twice -- PnP on every
confident joint (the pipeline as it was) and PnP on the limb joints only
(--exclude-joints Hip,RHip,LHip) -- and renders step_5's animation for each,
so the alignment can be compared frame for frame.  step_8 scores both against
the triangulated target too.

Input, Korea (children): one rep file (--rep).  Either
  * the raw aligned file (xy1..3, score1..3), e.g. data/sampleKorea/B023_GMS_2_2.npz.
    Its calibrated twin is looked up under --gt3d (build_child_gt.py's output,
    which holds the whole session's camera calibration); if that is not
    reachable the rep is calibrated from its own frames, with relaxed gates,
    which is a weaker calibration -- the log says which was used;
  * or a build_child_gt.py rep file directly.

Input, BioCV (adults): one trial folder (--trial), e.g. data/sampleBiocv/P03_CMJM_01,
holding {cam}.mp4 and markers.c3d, with the subject's {cam}.mp4-mocAligned.calib and
user_meta.json in the folder above it, and OpenPose's {cam}_openpose.json under
Analysis/openpose/ or Analysis/keypoints/openpose/.  The subject is the action's
prefix (P03).  step_0 converts the mocap, step_1_openpose_2d the JSON, step_1b
triangulates the target; step_5 then draws over the footage, and step_8 scores
against mocap as well as the triangulated target.

Everything is written under --out (default: next to the rep, in
<rep folder>/out/<stub>/):
    GT3D/<subject>/<stub>.npz              the calibrated rep the pipeline is laid out from
    pnp_all_joints/                        an OUT_DIR (config.OUT_DIR layout) for the baseline
    pnp_no_hips/                           the same, PnP without Hip, RHip, LHip
    videos/<stub>_cam<c>_<variant>.mp4     step_5's animation for each camera and variant
    comparison.txt                         step_8's numbers, side by side

MotionBERT runs on the CPU here (a couple of minutes for a 100-frame rep, ~3 min per
camera for a 375-frame BioCV trial).  Run with the motEnv python:
    ~/anaconda3/envs/motEnv/bin/python sample_hip_ablation.py --rep data/sampleKorea/B023_GMS_2_2.npz
    ~/anaconda3/envs/motEnv/bin/python sample_hip_ablation.py --trial data/sampleBiocv/P03_CMJM_01 --cameras 00,04,07
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, 'pipeline')
sys.path.insert(0, os.path.join(PIPE, 'utils'))
DEFAULT_GT3D = '/Volumes/Expansion/MotorDevelopment/Korea/B/GT3D'
HIPS = 'Hip,RHip,LHip'
VARIANTS = {                     # name: (step_4 --exclude-joints, label written into the video)
    'pnp_all_joints': ('', 'PnP on all joints'),
    'pnp_no_hips': (HIPS, 'PnP without Hip, RHip, LHip'),
}


def log(msg):
    print(f'\n### {msg}', flush=True)


TRIAL_ROOT = None      # BioCV mode: the TRIAL_DIR (videos, mocap, calibs) every step is given


def run_step(script, out_dir, *cli):
    """One pipeline step as a subprocess, its OUT_DIR pointed at out_dir."""
    env = dict(os.environ, BIOCV_OUT=out_dir)
    if TRIAL_ROOT:
        env['BIOCV_ROOT'] = TRIAL_ROOT
    cmd = [sys.executable, os.path.join(PIPE, script), *cli]
    print('$ BIOCV_OUT=' + out_dir + ' ' + ' '.join(os.path.basename(c) if c == cmd[1] else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=PIPE, env=env, check=True)


# ---------------------------------------------------------------------------
# 1. the calibrated rep (build_child_gt.py's format)
# ---------------------------------------------------------------------------

def is_gt3d(path):
    with np.load(path) as d:
        return 'h36m_2d' in d.files and 'K_1080' in d.files


def self_calibrate(raw_rep, gt3d_out, fps):
    """No calibrated twin: run build_child_gt.process_session on the rep's own folder, gates
    relaxed so ~100 frames can calibrate.  Returns the rep file it wrote."""
    sys.path.insert(0, HERE)
    import build_child_gt as bcg
    stub = os.path.basename(raw_rep)[:-4]
    subject = stub.split('_')[0]
    bcg.GATES.update(min_calib_frames_strict=20, min_calib_frames=20, min_calib_corr=400,
                     min_upright_frames=5)
    args = SimpleNamespace(out=gt3d_out, data=os.path.dirname(os.path.abspath(raw_rep)),
                           json_dir=os.path.join(gt3d_out, '_no_json'), video_dir=os.path.join(gt3d_out, '_no_video'),
                           scale='stature', fps=fps, calib_stride=1, calib_max_frames=800, calib_corr=6000,
                           save_all=True)
    print(f'calibrating from {args.data} (every {subject}_*.npz there), relaxed gates', flush=True)
    summ = bcg.process_session(subject, args, video_index={})
    print(f"session status {summ['status']}, usable {summ['usable']}" + (f", reasons: {summ['reasons']}" if summ['reasons'] else ''))
    if summ['status'] != 'ok':
        raise SystemExit('calibration from this rep alone failed -- copy the GT3D rep file (build_child_gt.py output) '
                         'next to it, or point --gt3d at the GT3D root')
    meta = os.path.join(gt3d_out, subject, 'user_meta.json')
    if not os.path.exists(meta):
        print(f'WARNING stature could not be measured from this rep; using the cohort median {bcg.COHORT_STATURE} m')
        with open(meta, 'w') as f:
            json.dump({'stature_m': bcg.COHORT_STATURE, 'note': 'cohort median: not measurable from this rep alone'}, f, indent=1)
    rep = os.path.join(gt3d_out, subject, stub + '.npz')
    if not os.path.exists(rep):
        raise SystemExit(f'{rep} was not written')
    return rep


def stage_gt3d(rep, gt3d_root, gt3d_out, fps):
    """-> the calibrated rep file under gt3d_out/<subject>/, plus session_summary.json and user_meta.json."""
    stub = os.path.basename(rep)[:-4]
    subject = stub.split('_')[0]
    sdir = os.path.join(gt3d_out, subject)
    os.makedirs(sdir, exist_ok=True)
    dst = os.path.join(sdir, stub + '.npz')

    if is_gt3d(rep):
        src, how = rep, 'the given rep file is already calibrated (build_child_gt.py format)'
    else:
        twin = os.path.join(gt3d_root, subject, stub + '.npz')
        if os.path.exists(twin):
            src, how = twin, f'session calibration from {twin}'
        else:
            print(f'no calibrated twin at {twin}')
            src = self_calibrate(rep, gt3d_out, fps)
            how = 'calibrated from the rep itself (relaxed gates -- weaker than a session calibration)'
    print(f'calibrated rep: {how}')
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
    # session_summary.json (step_1_korea_2d needs one) and user_meta.json (stature, for step_3)
    src_dir = os.path.dirname(os.path.abspath(src))
    for name in ('session_summary.json', 'user_meta.json'):
        p, q = os.path.join(src_dir, name), os.path.join(sdir, name)
        if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(q):
            shutil.copy2(p, q)
    if not os.path.exists(os.path.join(sdir, 'session_summary.json')):
        with open(os.path.join(sdir, 'session_summary.json'), 'w') as f:
            json.dump({'subject': subject, 'status': 'ok', 'usable': True, 'note': 'written by sample_hip_ablation.py'}, f)
    if not os.path.exists(os.path.join(sdir, 'user_meta.json')):
        with np.load(dst) as d:
            st = float(d['stature_m'])
        with open(os.path.join(sdir, 'user_meta.json'), 'w') as f:
            json.dump({'stature_m': round(st, 4), 'note': 'from the rep file (stature_m)'}, f, indent=1)
    with np.load(dst) as d:
        print(f"  {stub}: {d['h36m_2d'].shape[1]} frames, cameras {list(d['cameras'])}, stature {float(d['stature_m']):.3f} m, "
              f"usable {bool(d['usable'])}, usable frames (all-camera target) {int(d['frame_usable'].sum())}")
    return subject, stub


def stage_biocv(trial_dir, out, base):
    """A BioCV trial folder -> (user, action).  Builds out/trial/{user}/{action}/ as the pipeline's
    TRIAL_DIR (symlinks to the videos, mocap, calibs and stature) and drops the OpenPose JSONs
    into the base OUT_DIR where step_1_openpose_2d looks."""
    global TRIAL_ROOT
    trial_dir = os.path.abspath(trial_dir)
    action = os.path.basename(trial_dir)
    user = action.split('_')[0]
    subj_dir = os.path.dirname(trial_dir)
    TRIAL_ROOT = os.path.join(out, 'trial')
    tdir = os.path.join(TRIAL_ROOT, user, action)
    os.makedirs(tdir, exist_ok=True)

    def link(src, dst):
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    for name in os.listdir(subj_dir):                       # calibs and user_meta.json, next to the trial
        if name.endswith('.calib') or name == 'user_meta.json':
            link(os.path.join(subj_dir, name), os.path.join(TRIAL_ROOT, user, name))
    n_vid = 0
    for name in os.listdir(trial_dir):
        if name.endswith('.mp4') or name == 'markers.c3d':
            link(os.path.join(trial_dir, name), os.path.join(tdir, name))
            n_vid += name.endswith('.mp4')
    need = [os.path.join(TRIAL_ROOT, user, 'user_meta.json'), os.path.join(tdir, 'markers.c3d')]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise SystemExit('BioCV trial is missing: ' + ', '.join(missing))
    jsons = sorted(glob.glob(os.path.join(trial_dir, 'Analysis', 'openpose', '*_openpose.json'))
                   + glob.glob(os.path.join(trial_dir, 'Analysis', 'keypoints', 'openpose', '*_openpose.json')))
    if not jsons:
        raise SystemExit(f'no {{cam}}_openpose.json under {trial_dir}/Analysis/(keypoints/)openpose/')
    kdir = os.path.join(base, user, action, 'Analysis', 'keypoints', 'openpose')
    os.makedirs(kdir, exist_ok=True)
    for j in jsons:
        shutil.copy2(j, os.path.join(kdir, os.path.basename(j)))
    with open(need[0]) as f:
        st = json.load(f)['stature_m']
    print(f'BioCV trial {user}/{action}: {n_vid} videos, {len(jsons)} OpenPose JSONs, stature {st} m; TRIAL_DIR {TRIAL_ROOT}')
    return user, action


# ---------------------------------------------------------------------------
# 2. the pipeline, twice
# ---------------------------------------------------------------------------

def copy_layout(src_out, dst_out):
    """The laid-out trial and MotionBERT results, without PnP and everything after it."""
    if os.path.isdir(dst_out):
        shutil.rmtree(dst_out)
    shutil.copytree(src_out, dst_out, ignore=shutil.ignore_patterns('PnP', 'diagnostics', 'features'))


def scores(out_dir, subject, stub, gt='triangulated'):
    p = os.path.join(out_dir, subject, stub, 'Analysis', 'diagnostics', 'openpose',
                     'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz')
    if not os.path.exists(p):
        return None
    m = np.load(p)
    return dict(cameras=[str(c) for c in m['cameras']], placed=m['err_placed_all'], smooth=m['err_smooth_all'],
                pa=m['pa_mpjpe'], n=m['n_mpjpe'], perjoint=m['perjoint_smooth_all'], names=[str(n) for n in m['joint_names']],
                bias=m['bias_xyz'])


def comparison_text(res, subject, stub, gt='triangulated'):
    what = 'mocap' if gt == 'mocap' else 'the all-camera triangulated OpenPose target'
    lines = [f'{subject}/{stub}: error against {what} (mm), per camera', '']
    names = [k for k in VARIANTS if res.get(k)]
    if len(names) < 2:
        return '\n'.join(lines + ['(step_8 did not score both variants)'])
    a, b = res[names[0]], res[names[1]]
    lines.append(f'{"camera":>8} {"":>14} {names[0]:>16} {names[1]:>16} {"change":>10}')
    for i, cam in enumerate(a['cameras']):
        j = b['cameras'].index(cam) if cam in b['cameras'] else None
        if j is None:
            continue
        for key, label in (('placed', 'MPJPE placed'), ('smooth', 'MPJPE smooth'), ('n', 'N-MPJPE'), ('pa', 'PA-MPJPE')):
            x, y = float(a[key][i]), float(b[key][j])
            lines.append(f'{cam:>8} {label:>14} {x:16.1f} {y:16.1f} {y - x:+10.1f}')
        lines.append('')
    lines.append('per-joint error of the smoothed skeleton, mean over cameras (mm):')
    lines.append(f'{"joint":>10} {names[0]:>16} {names[1]:>16} {"change":>10}')
    pa_, pb_ = np.nanmean(a['perjoint'], axis=0), np.nanmean(b['perjoint'], axis=0)
    for k, nm in enumerate(a['names']):
        if np.isfinite(pa_[k]) or np.isfinite(pb_[k]):
            lines.append(f'{nm:>10} {pa_[k]:16.1f} {pb_[k]:16.1f} {pb_[k] - pa_[k]:+10.1f}')
    lines += ['', 'signed bias, prediction minus target, camera frame (mm; y is image-down, so a negative y is ABOVE the target):']
    lines.append(f'{"joint":>10} {names[0] + " x y z":>28} {names[1] + " x y z":>28}')
    ba, bb = np.nanmean(a['bias'], axis=0), np.nanmean(b['bias'], axis=0)
    for k, nm in enumerate(a['names']):
        if np.isfinite(ba[k]).all() or np.isfinite(bb[k]).all():
            lines.append(f'{nm:>10} ' + ' '.join(f'{v:8.0f}' for v in ba[k]) + '   ' + ' '.join(f'{v:8.0f}' for v in bb[k]))
    lines += ['', 'MPJPE is the mean joint distance with no alignment; N-MPJPE after the best scale; PA-MPJPE after Procrustes.',
              'Hip, RHip and LHip are still SCORED in every variant (only the PnP correspondences change), so their',
              'error reflects the H36M/' + ('mocap' if gt == 'mocap' else 'OpenPose') + ' definition offset in both columns.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rep', default=None,
                    help='Korea: the rep, raw aligned (xy1..3, score1..3) or a build_child_gt.py rep file '
                         '(default data/sampleKorea/B023_GMS_2_2.npz unless --trial is given)')
    ap.add_argument('--trial', default=None, help='BioCV: the trial folder, e.g. data/sampleBiocv/P03_CMJM_01')
    ap.add_argument('--gt3d', default=DEFAULT_GT3D, help="build_child_gt.py's output root, for the rep's calibrated twin")
    ap.add_argument('--out', default=None, help='output root (default <rep folder>/out/<stub>)')
    ap.add_argument('--cameras', default=None, help='comma-separated subset (Korea 1,2,3; BioCV 00..08); default all')
    ap.add_argument('--fps', type=float, default=30.0, help='only used when calibrating from the rep itself')
    ap.add_argument('--tri', choices=['all', 'loo'], default='all', help='which triangulated target step_5 draws / step_8 scores')
    ap.add_argument('--force', action='store_true', help='redo everything, including MotionBERT')
    args = ap.parse_args()

    if args.trial and args.rep:
        raise SystemExit('give --rep (Korea) or --trial (BioCV), not both')
    biocv = args.trial is not None
    src = os.path.abspath(args.trial if biocv else (args.rep or os.path.join(HERE, 'data', 'sampleKorea', 'B023_GMS_2_2.npz')))
    if not os.path.exists(src):
        raise SystemExit(f'{src} not found')
    stub = os.path.basename(src) if biocv else os.path.basename(src)[:-4]
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(src), 'out', stub))
    if args.force and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    cams = ['--cameras', args.cameras] if args.cameras else []
    base = os.path.join(out, 'pnp_all_joints')
    os.makedirs(base, exist_ok=True)

    if biocv:
        log('1. BioCV trial: videos, mocap, calibs, OpenPose JSONs')
        subject, stub = stage_biocv(src, out, base)
        log('2. mocap -> H36M (step_0), OpenPose JSON -> H36M 2D (step_1_openpose_2d), triangulated target (step_1b)')
        run_step('step_0_load_mocap.py', base, '--user', subject, '--action', stub)
        run_step('step_1_openpose_2d.py', base, '--user', subject, '--action', stub, *cams)
        run_step('step_1b_triangulate_2d.py', base, '--user', subject, '--action', stub, *cams, '--verbose')
    else:
        log('1. calibrated rep')
        subject, stub = stage_gt3d(src, args.gt3d, os.path.join(out, 'GT3D'), args.fps)
        log('2. lay the rep out as a pipeline trial (step_1_korea_2d)')
        run_step('step_1_korea_2d.py', base, '--gt3d', os.path.join(out, 'GT3D'), '--subjects', subject, '--include-unusable')

    log('3. MotionBERT: betas (step_2a), one body shape (step_2b), skeleton (step_3) -- once, shared by both variants')
    run_step('step_2a_extract_betas.py', base, '--user', subject, '--action', stub, *cams)
    run_step('step_2b_finalise_betas.py', base, '--user', subject, '--action', stub, *cams)
    run_step('step_3_extract_3d.py', base, '--user', subject, '--action', stub, *cams)

    res, res_mocap = {}, {}
    for name, (exclude, label) in VARIANTS.items():
        vdir = os.path.join(out, name)
        if vdir != base:
            copy_layout(base, vdir)
        log(f'4. PnP -- {label} ({name})')
        run_step('step_4_PnP.py', vdir, '--user', subject, '--action', stub, *cams, '--force',
                 *(['--exclude-joints', exclude] if exclude else []))
        log(f'5. animation -- {label}')
        run_step('step_5_mocap_comparison.py', vdir, '--user', subject, '--action', stub, *cams, '--force',
                 '--tri', args.tri, '--label', label, *(['--draw-2d', '--draw-target'] if biocv else []))
        log(f'8. score against the triangulated target -- {label}')
        run_step('step_8_spider_error.py', vdir, '--user', subject, '--action', stub, *cams, '--force',
                 '--gt', 'triangulated', '--tri', args.tri, '--verbose')
        res[name] = scores(vdir, subject, stub)
        if biocv:
            log(f'8. score against mocap -- {label}')
            run_step('step_8_spider_error.py', vdir, '--user', subject, '--action', stub, *cams, '--force',
                     '--gt', 'mocap', '--verbose')
            res_mocap[name] = scores(vdir, subject, stub, 'mocap')

    log('collecting')
    vid_dir = os.path.join(out, 'videos')
    os.makedirs(vid_dir, exist_ok=True)
    for name in VARIANTS:
        for p in sorted(glob.glob(os.path.join(out, name, subject, stub, 'Analysis', 'diagnostics', '*_pnp_depth_vs_mocap.mp4'))):
            cam = os.path.basename(p).split('_')[0]
            q = os.path.join(vid_dir, f'{stub}_cam{cam}_{name}.mp4')
            shutil.copy2(p, q)
            print(f'  {q}')
    text = comparison_text(res, subject, stub)
    if res_mocap:
        text = comparison_text(res_mocap, subject, stub, 'mocap') + '\n\n' + '=' * 100 + '\n\n' + text
    with open(os.path.join(out, 'comparison.txt'), 'w') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nvideos in {vid_dir}, numbers in {os.path.join(out, "comparison.txt")}')


if __name__ == '__main__':
    main()
