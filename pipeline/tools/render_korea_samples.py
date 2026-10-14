#!/usr/bin/env python3
"""render_korea_samples.py -- render_mesh.py videos for every Korea sample rep, end to end.

For each rep folder under --root (data/sampleKorea/out/{user}_{action}/), using the inputs the old
sample experiments left in betas_ablation/motionbert/:

    1. builds a clean OUT_DIR at {rep}/step4b/ (2D keypoints, MotionBERT pass-1 rotations and
       final betas, the triangulated target -- copied once)
    2. step_2c_fit_shape.py   the subject's adult / kid shapes (from this rep alone; see --out-dir)
    3. step_4b_refine.py      the --configs, every camera
    4. render_mesh.py         one video per camera, against the leave-one-out target and/or the
                              all-camera target (*_gtall.mp4), re-encoded to H.264 when ffmpeg is there

--out-dir instead works on an OUT_DIR laid out by step_1_korea_2d (e.g. data/sampleKorea/out_v4, built
from GT3D_v4) with MotionBERT already run (step_2a/2b): nothing is copied, step_2c fits each subject
from every rep laid out there, and the reps are the trials with pass-1 rotations.

Steps 2-3 skip what already exists unless --force.  A camera step_4b could not place (e.g. no
final betas for the MotionBERT shape) is reported and skipped.  Videos land in
{OUT_DIR}/{user}/{action}/Analysis/diagnostics/openpose/.

    python3 tools/render_korea_samples.py                                  # every rep, cameras 1-3
    python3 tools/render_korea_samples.py --reps B023_GMS_2_2,B011_GMS_2_3 --gt loo
    python3 tools/render_korea_samples.py --configs C0,C5,C8 --smooth C0 --cams 1,2
    python3 tools/render_korea_samples.py --reps B010_GMS_1_1,B011_GMS_2_3 --configs C8,C10 --smooth '' --com
    python3 tools/render_korea_samples.py --out-dir ../data/sampleKorea/out_v4 --reps B011_GMS_2_3
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(_HERE)
DEFAULT_ROOT = os.path.join(os.path.dirname(PIPELINE), 'data', 'sampleKorea', 'out')


def run(cmd, env, quiet=False):
    p = subprocess.run(cmd, cwd=PIPELINE, env=env, capture_output=True, text=True)
    if p.returncode != 0 or not quiet:
        out = (p.stdout + p.stderr).strip()
        if out:
            print('    ' + out.replace('\n', '\n    '))
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of {user}_{action} rep folders')
    ap.add_argument('--out-dir', default=None,
                    help='a step_1_korea_2d OUT_DIR with MotionBERT run (instead of --root sample folders)')
    ap.add_argument('--reps', default=None, help='comma-separated rep folders (default: every one with inputs)')
    ap.add_argument('--configs', default='C0,C5,C8')
    ap.add_argument('--smooth', default='C0', help="configurations drawn RTS-smoothed ('' for none)")
    ap.add_argument('--cams', default='1,2,3')
    ap.add_argument('--gt', choices=('loo', 'all', 'both'), default='both')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--force', action='store_true', help='redo shapes and step_4b outputs that exist')
    ap.add_argument('--com', action='store_true', help="render_mesh's centre-of-mass / base-of-support overlay")
    args = ap.parse_args()

    py = sys.executable
    ffmpeg = shutil.which('ffmpeg')
    if args.out_dir:
        args.out_dir = os.path.abspath(args.out_dir)
        found = glob.glob(os.path.join(args.out_dir, '*', '*', 'Analysis', 'mesh', 'openpose', '*_betas.npz'))
        default = sorted({p.split(os.sep)[-5] for p in found})
    else:
        default = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                         for p in glob.glob(os.path.join(args.root, '*', 'betas_ablation', 'motionbert')))
    reps = args.reps.split(',') if args.reps else default
    gts = {'loo': [False], 'all': [True], 'both': [False, True]}[args.gt]
    made, skipped = [], []
    for rep in reps:
        user, action = rep.split('_', 1)[0], rep
        if args.out_dir:                                         # already laid out: calibs and stature live here too
            out = src = args.out_dir
            ana = os.path.join(out, user, action, 'Analysis')
            if not os.path.isdir(os.path.join(ana, 'mesh', 'openpose')):
                print(f'{rep}: no MotionBERT output under {out} (run step_2a/2b), skipped')
                continue
            print(f'{rep}')
        else:
            src = os.path.join(args.root, rep, 'betas_ablation', 'motionbert')
            out = os.path.join(args.root, rep, 'step4b')
            ana = os.path.join(out, user, action, 'Analysis')
            if not os.path.isdir(os.path.join(src, user, action, 'Analysis')):
                print(f'{rep}: no betas_ablation/motionbert inputs, skipped')
                continue
            print(f'{rep}')
            for sub in ('keypoints', 'mesh', 'H36M'):
                if not os.path.isdir(os.path.join(ana, sub)):
                    shutil.copytree(os.path.join(src, user, action, 'Analysis', sub), os.path.join(ana, sub))
        env = dict(os.environ, BIOCV_OUT=out, BIOCV_ROOT=src)
        force = ['--force'] if args.force else []
        print('  step_2c: shapes')
        run([py, 'step_2c_fit_shape.py', '--user', user] + force, env, quiet=True)
        print(f'  step_4b: {args.configs}')
        run([py, 'step_4b_refine.py', '--user', user, '--action', action, '--configs', args.configs,
             '--device', args.device] + force, env, quiet=True)
        for cam in args.cams.split(','):
            missing = [c for c in args.configs.split(',')
                       if not os.path.exists(os.path.join(ana, 'PnP', 'openpose', c, f'{cam}_pnp.npz'))]
            if missing:
                print(f'  cam {cam}: no step_4b output for {",".join(missing)}, skipped')
                skipped.append(f'{rep} cam {cam}')
                continue
            for all_cams in gts:
                cmd = [py, os.path.join('tools', 'render_mesh.py'), '--user', user, '--action', action, '--cam', cam,
                       '--gt', 'tri', '--configs', args.configs, '--smooth', args.smooth]
                if all_cams:
                    cmd.append('--all-cameras')
                if args.com:
                    cmd.append('--com')
                tag = 'all-camera' if all_cams else 'leave-one-out'
                print(f'  cam {cam}: rendering vs {tag} target')
                p = subprocess.run(cmd, cwd=PIPELINE, env=env, capture_output=True, text=True)
                if p.returncode != 0:
                    print('    FAILED\n    ' + (p.stdout + p.stderr).strip().replace('\n', '\n    '))
                    skipped.append(f'{rep} cam {cam} ({tag})')
                    continue
                for line in p.stdout.splitlines():
                    if 'MPJPE' in line or 'COM' in line:
                        print('    ' + line)
                video = [l.split('wrote ', 1)[1] for l in p.stdout.splitlines() if l.startswith('wrote ')][0]
                if ffmpeg:                                           # mp4v -> H.264, playable everywhere
                    tmp = video[:-4] + '_h264.mp4'
                    if subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', video, '-c:v', 'libx264', '-pix_fmt',
                                       'yuv420p', '-crf', '20', tmp]).returncode == 0:
                        os.replace(tmp, video)
                made.append(video)
    print(f'\n{len(made)} video(s) written' + (f'; skipped: {", ".join(skipped)}' if skipped else ''))
    for v in made:
        print(f'  {v}')


if __name__ == '__main__':
    main()
