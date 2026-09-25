#!/usr/bin/env python3
"""sample_variant_matrix.py -- one BioCV trial through the pipeline (YOLO 2D), then the shape x placement
matrix scored against mocap:

    shape      : MotionBERT betas (step_2b)          | betas fitted to the mocap bone lengths (sample_fit_betas_bones)
    placement  : PnP (step_4, the pipeline as is)
                 refined pose, free rotation, no foot terms      (sample_refine_2d)
                 refined pose, free rotation, feet pinned + floor
                 MotionBERT pose kept, rotation frozen to stance, translation with feet pinned + floor

Stages the trial folder the way sample_hip_ablation does (symlinks under <out>/trial), runs
step_0 (mocap -> H36M), step_1_extract_2d (YOLO), step_2a/2b/3/4 for --detector yolo, scores with
step_8 (--gt mocap), builds the fitted-betas copy and re-runs steps 3/4/8 on it, then runs the three
refinement variants on both shapes.  Everything lands under --out; nothing under pipeline/ changes.

    ~/anaconda3/envs/motEnv/bin/python sample_variant_matrix.py --trial data/sampleBiocv/P03_RUN_01 --cameras 00,05,07
    ~/anaconda3/envs/motEnv/bin/python sample_variant_matrix.py --trial ... --skip-pipeline     # matrix only, trial already through step_4
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
sys.path.insert(0, HERE)
from sample_fit_betas_bones import load_smpl                      # noqa: E402
from sample_betas_ablation import write_fitted_betas, copy_layout  # noqa: E402

PY = sys.executable
REFINE = os.path.join(HERE, 'sample_refine_2d.py')
COMMON = ['--exclude-2d', 'Hip,RHip,LHip', '--huber', '20']
PLACEMENTS = {                       # name: extra sample_refine_2d flags (None = step_4 PnP as is)
    'pnp': None,
    'refine_nofoot': COMMON,
    'refine_foot': COMMON + ['--w-foot', '3000', '--w-floor', '3000'],
    'fixrot_feet': COMMON + ['--w-foot', '3000', '--w-floor', '3000', '--fix-rotation', 'stance', '--no-body'],
}
LABELS = {
    ('motionbert', 'pnp'): 'MotionBERT betas + PnP (original)',
    ('motionbert', 'refine_nofoot'): 'MotionBERT betas + refined pose (no fixed feet)',
    ('motionbert', 'refine_foot'): 'MotionBERT betas + refined pose + fixed feet',
    ('motionbert', 'fixrot_feet'): 'MotionBERT betas + fixed feet + PnP depth (no rotation)',
    ('bones_mocap', 'pnp'): 'Fitted betas + PnP (MotionBERT rotations unchanged)',
    ('bones_mocap', 'refine_nofoot'): 'Fitted betas + refined pose (no fixed feet)',
    ('bones_mocap', 'refine_foot'): 'Fitted betas + refined pose + fixed feet',
    ('bones_mocap', 'fixrot_feet'): 'Fitted betas + fixed feet + PnP depth (no rotation)',
}


def log(msg):
    print(f'\n### {msg}', flush=True)


def run(cmd, out_dir, trial_root, cwd=PIPE, quiet=False):
    env = dict(os.environ, BIOCV_OUT=out_dir, BIOCV_ROOT=trial_root)
    print('$ ' + ' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL if quiet else None)


def stage(trial_dir, out):
    """<out>/trial/{user}/{action}/ with symlinks to videos + markers.c3d; calibs + user_meta.json at <out>/trial/{user}/."""
    trial_dir = os.path.abspath(trial_dir)
    action = os.path.basename(trial_dir)
    user = action.split('_')[0]
    subj_dir = os.path.dirname(trial_dir)
    trial_root = os.path.join(out, 'trial')
    tdir = os.path.join(trial_root, user, action)
    os.makedirs(tdir, exist_ok=True)

    def link(src, dst):
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    for name in os.listdir(subj_dir):
        if name.endswith('.calib') or name == 'user_meta.json':
            link(os.path.join(subj_dir, name), os.path.join(trial_root, user, name))
    for name in os.listdir(trial_dir):
        if (name.endswith('.mp4') and '_' not in name) or name == 'markers.c3d':
            link(os.path.join(trial_dir, name), os.path.join(tdir, name))
    for need in (os.path.join(trial_root, user, 'user_meta.json'), os.path.join(tdir, 'markers.c3d')):
        if not os.path.exists(need):
            raise SystemExit(f'missing {need}')
    with open(os.path.join(trial_root, user, 'user_meta.json')) as f:
        st = json.load(f)['stature_m']
    print(f'{user}/{action}: stature {st} m, TRIAL_DIR {trial_root}')
    return user, action, trial_root


def metrics(vdir, user, action, det):
    p = os.path.join(vdir, user, action, 'Analysis', 'diagnostics', det, 'error_metrics.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def comparison(out, user, action, det, cams):
    rows = []
    for shape in ('motionbert', 'bones_mocap'):
        for pl in PLACEMENTS:
            vdir = os.path.join(out, shape if pl == 'pnp' else f'{shape}__{pl}')
            m = metrics(vdir, user, action, det)
            rows.append((LABELS[(shape, pl)], m))
    lines = [f'{user}/{action}, {det.upper()} 2D, error against mocap (mm); cameras {", ".join(cams)}', '']
    for key, label in (('err_placed_all', 'MPJPE placed'), ('n_mpjpe', 'N-MPJPE (best scale)'), ('pa_mpjpe', 'PA-MPJPE (pose only)')):
        lines.append(f'{label}')
        lines.append(f'{"variant":58s}' + ''.join(f'{"cam " + c:>10s}' for c in cams) + f'{"mean":>10s}')
        for name, m in rows:
            if m is None:
                lines.append(f'{name:58s}' + '   (not scored)')
                continue
            cm = [str(c) for c in m['cameras']]
            vals = [float(m[key][cm.index(c)]) if c in cm else np.nan for c in cams]
            lines.append(f'{name:58s}' + ''.join(f'{v:10.1f}' for v in vals) + f'{np.nanmean(vals):10.1f}')
        lines.append('')
    ref = rows[0][1]
    if ref is not None:
        names = [str(n) for n in ref['joint_names']]
        lines.append('per-joint MPJPE placed, mean over cameras (mm)')
        lines.append(f'{"joint":>10s}' + ''.join(f'{i + 1:>8d}' for i in range(len(rows))))
        for k, nm in enumerate(names):
            vals = [np.nanmean(m['perjoint_placed_all'][:, k]) if m is not None else np.nan for _, m in rows]
            if np.isfinite(vals[0]):
                lines.append(f'{nm:>10s}' + ''.join(f'{v:8.0f}' for v in vals))
        lines.append('columns: ' + '; '.join(f'{i + 1} = {n}' for i, (n, _) in enumerate(rows)))
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trial', required=True, help='BioCV trial folder, e.g. data/sampleBiocv/P03_RUN_01')
    ap.add_argument('--out', default=None, help='output root (default <trial>/../out/<action>)')
    ap.add_argument('--cameras', default='00,05,07')
    ap.add_argument('--detector', default='yolo')
    ap.add_argument('--skip-pipeline', action='store_true', help='steps 0-4 already done under <out>/motionbert')
    ap.add_argument('--skip-refine', action='store_true', help='only (re)build the comparison table')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--sigma-mm', type=float, default=10.0)
    args = ap.parse_args()

    trial = os.path.abspath(args.trial)
    out = os.path.abspath(args.out or os.path.join(os.path.dirname(trial), 'out', os.path.basename(trial)))
    os.makedirs(out, exist_ok=True)
    cams = args.cameras.split(',')
    det = args.detector
    user, action, trial_root = stage(trial, out)
    base = os.path.join(out, 'motionbert')
    os.makedirs(base, exist_ok=True)
    cam_cli = ['--cameras', ','.join(cams)]
    ua = ['--user', user, '--action', action]

    if not args.skip_pipeline:
        log('step_0: mocap -> H36M')
        run([PY, 'step_0_load_mocap.py', *ua], base, trial_root)
        log(f'step_1: {det} 2D, cameras {cams} in parallel')
        if det == 'yolo':
            procs = [subprocess.Popen([PY, 'step_1_extract_2d.py', *ua, '--pattern', f'{c}.mp4', '--force'], cwd=PIPE,
                                      env=dict(os.environ, BIOCV_OUT=base, BIOCV_ROOT=trial_root),
                                      stdout=open(os.path.join(out, f'step1_{c}.log'), 'w'), stderr=subprocess.STDOUT) for c in cams]
            for p in procs:
                if p.wait() != 0:
                    raise SystemExit('step_1 failed; see step1_*.log under ' + out)
        else:
            run([PY, 'step_1_openpose_2d.py', *ua, *cam_cli, '--force'], base, trial_root)
        log('step_2a: MotionBERT rotations + betas (cameras in parallel)')
        procs = [subprocess.Popen([PY, 'step_2a_extract_betas.py', '--detector', det, *ua, '--cameras', c, '--force'], cwd=PIPE,
                                  env=dict(os.environ, BIOCV_OUT=base, BIOCV_ROOT=trial_root),
                                  stdout=open(os.path.join(out, f'step2a_{c}.log'), 'w'), stderr=subprocess.STDOUT) for c in cams]
        for p in procs:
            if p.wait() != 0:
                raise SystemExit('step_2a failed; see step2a_*.log under ' + out)
        for step in ('step_2b_finalise_betas.py', 'step_3_extract_3d.py', 'step_4_PnP.py'):
            log(step)
            run([PY, step, '--detector', det, *ua, *cam_cli, '--force'], base, trial_root)
        log('step_8: score the PnP baseline')
        run([PY, 'step_8_spider_error.py', '--detector', det, *ua, *cam_cli, '--force', '--gt', 'mocap'], base, trial_root)

    fitted = os.path.join(out, 'bones_mocap')
    if not args.skip_refine:
        log('fitted betas: copy the trial, fit betas to the mocap bone lengths, re-run steps 3/4/8')
        copy_layout(base, fitted)
        smpl = load_smpl(10)
        write_fitted_betas(fitted, user, action, det, cams, 'mocap_h36m.npz', smpl,
                           SimpleNamespace(lam=args.lam, sigma_mm=args.sigma_mm))
        for step in ('step_3_extract_3d.py', 'step_4_PnP.py'):
            run([PY, step, '--detector', det, *ua, *cam_cli, '--force'], fitted, trial_root)
        run([PY, 'step_8_spider_error.py', '--detector', det, *ua, *cam_cli, '--force', '--gt', 'mocap'], fitted, trial_root)

        for shape, src in (('motionbert', base), ('bones_mocap', fitted)):
            for pl, flags in PLACEMENTS.items():
                if flags is None:
                    continue
                log(f'{LABELS[(shape, pl)]}')
                run([PY, REFINE, '--src', src, '--out', os.path.join(out, f'{shape}__{pl}'), '--trial-root', trial_root,
                     '--detector', det, *cam_cli, *flags], src, trial_root, cwd=HERE, quiet=True)

    log('comparison')
    text = comparison(out, user, action, det, cams)
    with open(os.path.join(out, 'matrix_comparison.txt'), 'w') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nnumbers in {os.path.join(out, "matrix_comparison.txt")}')


if __name__ == '__main__':
    main()
