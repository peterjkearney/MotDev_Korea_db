#!/usr/bin/env python3
"""sample_betas_ablation.py -- the pipeline from step_3 on, with MotionBERT's betas and with
betas fitted to the bone lengths of a 3D skeleton (sample_fit_betas_bones.py).

Betas enter the pipeline in one place: step_3 reads {cam}_final_betas.npz to build the
SMPL skeleton, and steps 4 and 8 never look at them.  So a laid-out trial that is already
through step_2b is copied once per variant, the betas files are swapped, and steps 3, 4
and 8 are re-run on each copy.  Steps 0-2b are shared; nothing under pipeline/ changes.

Variants:
    motionbert    step_2b's betas as they are (the pipeline as it stands)
    bones_tri     one shape per subject, fitted to the all-camera triangulated OpenPose skeleton
                  (H36M/openpose_tri_h36m.npz) -- uses no mocap, so it is deployable
    bones_kid     the same fit with AGORA's kid template (models/smpl_kid_template.npy) as an
                  11th shape direction, weight in [0,1]: adult SMPL cannot reach a child's leg
                  proportions, the kid blend can.  step_3 takes 10 betas only, so this variant
                  builds its skeleton here (build_mesh_fitted) and continues with steps 4 and 8
Bone-fitted bodies are metric (fitted to segment lengths in the target's units) and are not
rescaled to the stature unless --rescale-fitted; MotionBERT's are rescaled by step_3 as always.
    bones_mocap   the same, fitted to the mocap skeleton (H36M/mocap_h36m.npz) -- an oracle
                  upper bound on what better shape can buy

step_3 rescales MotionBERT's mesh to the measured stature; the fitted bodies carry their own
metric size.  The log shows each variant's T-pose height and the scale applied.

Input: --base, an OUT_DIR (config.OUT_DIR layout) holding one trial through step_2b, e.g.
sample_hip_ablation.py's pnp_all_joints/.  Calibs and user_meta.json are found through
BIOCV_ROOT (--trial-root, default <base>/../trial as the hip ablation lays it out).
Output, under --out (default <base>/../betas_ablation/):
    <variant>/                 an OUT_DIR per variant (steps 3, 4, 8 re-run inside)
    comparison.txt             step_8's numbers side by side, against mocap and the triangulated target

    ~/anaconda3/envs/motEnv/bin/python sample_betas_ablation.py
    ~/anaconda3/envs/motEnv/bin/python sample_betas_ablation.py --exclude-joints Hip,RHip,LHip   # PnP without hips
    ~/anaconda3/envs/motEnv/bin/python sample_betas_ablation.py --n-betas 3                     # size + 2 proportions only
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, 'pipeline')
sys.path.insert(0, HERE)
from sample_fit_betas_bones import (SEGMENTS, data_seg_lengths, fit_betas, load_smpl,   # noqa: E402
                                    mesh_height_m, seg_lengths_mm)

DEFAULT_BASE = os.path.join(HERE, 'data', 'sampleBiocv', 'out', 'P03_CMJM_01', 'pnp_all_joints')
VARIANTS = {                      # name: (skeleton file the betas are fitted to (None = MotionBERT's), kid blend?)
    'motionbert': (None, False),
    'bones_tri': ('openpose_tri_h36m.npz', False),
    'bones_kid': ('openpose_tri_h36m.npz', True),     # + AGORA's kid template as an 11th shape direction
    'bones_mocap': ('mocap_h36m.npz', False),
}


def log(msg):
    print(f'\n### {msg}', flush=True)


def run_step(script, out_dir, trial_root, *cli):
    env = dict(os.environ, BIOCV_OUT=out_dir, BIOCV_ROOT=trial_root)
    cmd = [sys.executable, os.path.join(PIPE, script), *cli]
    print(f'$ BIOCV_OUT={out_dir} {script} ' + ' '.join(cli), flush=True)
    subprocess.run(cmd, cwd=PIPE, env=env, check=True)


def copy_layout(src_out, dst_out):
    """The laid-out trial and MotionBERT pass-1 results, without step_3's mesh and everything after."""
    if os.path.isdir(dst_out):
        shutil.rmtree(dst_out)
    shutil.copytree(src_out, dst_out, ignore=shutil.ignore_patterns('PnP', 'diagnostics', 'features',
                                                                     '*_mesh_pose.npz'))


def find_trial(base):
    hits = sorted(glob.glob(os.path.join(base, '*', '*', 'Analysis')))
    if len(hits) != 1:
        raise SystemExit(f'expected exactly one {{user}}/{{action}}/Analysis under {base}, found {len(hits)}')
    action = os.path.basename(os.path.dirname(hits[0]))
    user = os.path.basename(os.path.dirname(os.path.dirname(hits[0])))
    return user, action


def write_fitted_betas(vdir, user, action, detector, cams, skel_file, smpl, args):
    """Fit one shape to the skeleton's bone lengths and write it as every camera's final betas.
    With a kid model the file holds 11 values, the last the kid weight (step_3 cannot read that:
    build_mesh_fitted replaces it)."""
    skel = os.path.join(vdir, user, action, 'Analysis', 'H36M', skel_file)
    kps = np.load(skel, allow_pickle=True)['kps3d'].astype(np.float64)
    tgt, _, _ = data_seg_lengths(kps, SEGMENTS)
    beta, _ = fit_betas(smpl, tgt, SEGMENTS, lam=args.lam, sigma_mm=args.sigma_mm)
    rms = float(np.sqrt(np.nanmean((seg_lengths_mm(smpl, beta, SEGMENTS) - tgt) ** 2)))
    kid = smpl.get('kid', False)
    full = np.zeros(11 if kid else 10, dtype=np.float32)
    if kid:
        full[:len(beta) - 1] = beta[:-1]
        full[-1] = beta[-1]
    else:
        full[:len(beta)] = beta
    print(f'  betas from {skel_file}: ' + ' '.join(f'{b:+.3f}' for b in full[:10])
          + (f'   kid weight {full[-1]:.3f}' if kid else '')
          + f'   T-pose height {mesh_height_m(smpl, beta):.3f} m, segment RMS {rms:.1f} mm')
    for cam in cams:
        p = os.path.join(vdir, user, action, 'Analysis', 'mesh', detector, f'{cam}_final_betas.npz')
        old = dict(np.load(p, allow_pickle=True))
        old['motionbert_betas'] = old['betas']
        old['betas'] = full
        old['betas_source'] = np.array(f'bones:{skel_file}')
        old['segment_rms_mm'] = np.array(rms)
        np.savez(p, **old)
    return full, rms


def stature_m(trial_root, vdir, user):
    import json
    for root in (trial_root, vdir):
        p = os.path.join(root, user, 'user_meta.json')
        if os.path.exists(p):
            with open(p) as f:
                return float(json.load(f)['stature_m'])
    raise SystemExit(f'no {user}/user_meta.json under {trial_root} or {vdir}')


def build_mesh_fitted(vdir, user, action, detector, cams, betas, smpl_lin, stature, rescale):
    """step_3 for a bone-fitted shape (10 betas, or 11 with the kid blend): the SMPL skeleton from
    pass-1 rotations, written in step_3's {cam}_mesh_pose.npz format.  Uses sample_refine_2d's
    MiniSMPL (verified to reproduce step_3 to <0.1 mm on adult shapes).

    Scale: the betas were fitted to segment lengths in the target's own units, so the skeleton is
    already metric and is NOT rescaled (scale 1).  step_3's rescale to the stature exists for
    MotionBERT's betas, which carry no size; applied to a fitted body it inflates every bone by
    stature / mesh-height (the kid body's crown-to-sole comes out 4% under the measured stature),
    and PnP then mis-places it.  --rescale-fitted restores step_3's behaviour."""
    import torch
    from sample_refine_2d import MiniSMPL
    A = os.path.join(vdir, user, action, 'Analysis')
    mesh_height = mesh_height_m(smpl_lin, betas.astype(np.float64))
    scale = stature / mesh_height if rescale else 1.0
    smpl = MiniSMPL(betas, torch.device('cpu'))
    for cam in cams:
        rot = np.load(os.path.join(A, 'mesh', detector, f'{cam}_betas.npz'))['rotmats'].astype(np.float64)
        with torch.no_grad():
            J24, Jh = smpl.forward(torch.tensor(rot))
        out = os.path.join(A, 'mesh', detector, f'{cam}_mesh_pose.npz')
        np.savez(out, kps_H36M_scaled=(Jh.numpy() * scale).astype(np.float32),
                 kps_SMPL24_scaled=(J24.numpy() * scale).astype(np.float32),
                 rotmats=rot.astype(np.float32), scale_factor=scale, mesh_height_m=mesh_height,
                 stature_m=stature, betas=betas.astype(np.float32), detector=np.array(detector),
                 built_by=np.array('sample_betas_ablation.build_mesh_fitted (bone-fitted shape, metric, '
                                   + ('rescaled to stature' if rescale else 'not rescaled') + ')'))
        print(f'  {cam}: {rot.shape[0]} frames, mesh T-pose {mesh_height:.3f} m vs subject {stature:.2f} m -> scale x{scale:.3f}')


def scores(vdir, user, action, detector, gt):
    p = os.path.join(vdir, user, action, 'Analysis', 'diagnostics', detector,
                     'error_metrics.npz' if gt == 'mocap' else 'error_metrics_tri.npz')
    if not os.path.exists(p):
        return None
    m = np.load(p, allow_pickle=True)
    return dict(cameras=[str(c) for c in m['cameras']], placed=m['err_placed_all'], smooth=m['err_smooth_all'],
                pa=m['pa_mpjpe'], n=m['n_mpjpe'], perjoint=m['perjoint_smooth_all'], perjoint_pa=m['perjoint_pa'],
                names=[str(n) for n in m['joint_names']], bias=m['bias_xyz'],
                bone_names=[str(b) for b in m['bone_names']], bone_ratio=m['bone_ratio'])


def comparison_text(res, user, action, gt, scale):
    what = 'mocap' if gt == 'mocap' else 'the all-camera triangulated OpenPose target'
    names = [k for k in VARIANTS if res.get(k)]
    lines = [f'{user}/{action}: error against {what} (mm), per camera', '']
    if len(names) < 2:
        return '\n'.join(lines + ['(step_8 did not score at least two variants)'])
    base = res[names[0]]
    hdr = f'{"camera":>8} {"":>14}' + ''.join(f'{n:>14}' for n in names) + ''.join(f'{"d " + n:>14}' for n in names[1:])
    lines.append(hdr)
    for i, cam in enumerate(base['cameras']):
        for key, label in (('placed', 'MPJPE placed'), ('smooth', 'MPJPE smooth'), ('n', 'N-MPJPE'), ('pa', 'PA-MPJPE')):
            vals = []
            for n in names:
                r = res[n]
                j = r['cameras'].index(cam) if cam in r['cameras'] else None
                vals.append(float(r[key][j]) if j is not None else np.nan)
            lines.append(f'{cam:>8} {label:>14}' + ''.join(f'{v:14.1f}' for v in vals)
                         + ''.join(f'{v - vals[0]:+14.1f}' for v in vals[1:]))
        lines.append('')
    lines.append(f'{"mean":>8} {"":>14}' + ''.join(f'{n:>14}' for n in names) + ''.join(f'{"d " + n:>14}' for n in names[1:]))
    for key, label in (('placed', 'MPJPE placed'), ('smooth', 'MPJPE smooth'), ('n', 'N-MPJPE'), ('pa', 'PA-MPJPE')):
        vals = [float(np.nanmean(res[n][key])) for n in names]
        lines.append(f'{"":>8} {label:>14}' + ''.join(f'{v:14.1f}' for v in vals)
                     + ''.join(f'{v - vals[0]:+14.1f}' for v in vals[1:]))
    lines += ['', 'per-joint error of the smoothed skeleton, mean over cameras (mm):']
    lines.append(f'{"joint":>10}' + ''.join(f'{n:>14}' for n in names) + ''.join(f'{"d " + n:>14}' for n in names[1:]))
    pj = {n: np.nanmean(res[n]['perjoint'], axis=0) for n in names}
    for k, nm in enumerate(base['names']):
        if any(np.isfinite(pj[n][k]) for n in names):
            vals = [pj[n][k] for n in names]
            lines.append(f'{nm:>10}' + ''.join(f'{v:14.1f}' for v in vals) + ''.join(f'{v - vals[0]:+14.1f}' for v in vals[1:]))
    lines += ['', 'per-joint error after Procrustes (PA), mean over cameras (mm) -- shape and pose only, no placement:']
    lines.append(f'{"joint":>10}' + ''.join(f'{n:>14}' for n in names) + ''.join(f'{"d " + n:>14}' for n in names[1:]))
    pj = {n: np.nanmean(res[n]['perjoint_pa'], axis=0) for n in names}
    for k, nm in enumerate(base['names']):
        if any(np.isfinite(pj[n][k]) for n in names):
            vals = [pj[n][k] for n in names]
            lines.append(f'{nm:>10}' + ''.join(f'{v:14.1f}' for v in vals) + ''.join(f'{v - vals[0]:+14.1f}' for v in vals[1:]))
    lines += ['', 'bone length, prediction / target, mean over cameras (1.00 = same length as the target):']
    lines.append(f'{"bone":>16}' + ''.join(f'{n:>14}' for n in names))
    br = {n: np.nanmean(res[n]['bone_ratio'], axis=0) for n in names}
    for k, nm in enumerate(base['bone_names']):
        lines.append(f'{nm:>16}' + ''.join(f'{br[n][k]:14.3f}' for n in names))
    lines += ['', 'step_3 scale applied to reach the measured stature (T-pose mesh height -> stature):']
    for n in names:
        if n in scale:
            lines.append(f'{n:>14}: mesh {scale[n][0]:.3f} m  x{scale[n][1]:.3f}')
    lines += ['', 'MPJPE is the mean joint distance with no alignment; N-MPJPE after the best scale; PA-MPJPE after Procrustes.',
              'All variants are scaled to the same stature by step_3, so differences are in proportions, not size.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=DEFAULT_BASE, help='OUT_DIR with one trial through step_2b')
    ap.add_argument('--trial-root', default=None, help='BIOCV_ROOT with {user}/*.calib and user_meta.json (default <base>/../trial)')
    ap.add_argument('--out', default=None, help='output root (default <base>/../betas_ablation)')
    ap.add_argument('--detector', default='openpose')
    ap.add_argument('--cameras', default=None, help='comma-separated subset; default every camera with final betas')
    ap.add_argument('--variants', default=','.join(VARIANTS), help='subset of ' + ','.join(VARIANTS))
    ap.add_argument('--n-betas', type=int, default=10)
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--sigma-mm', type=float, default=10.0)
    ap.add_argument('--exclude-joints', default='', help="passed to step_4 (e.g. Hip,RHip,LHip)")
    ap.add_argument('--tri', choices=['all', 'loo'], default='all', help='triangulated target step_8 scores')
    ap.add_argument('--rescale-fitted', action='store_true',
                    help="rescale bone-fitted bodies to the stature as step_3 does (default: keep them metric, scale 1)")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    trial_root = os.path.abspath(args.trial_root or os.path.join(base, '..', 'trial'))
    out = os.path.abspath(args.out or os.path.join(base, '..', 'betas_ablation'))
    user, action = find_trial(base)
    mesh_dir = os.path.join(base, user, action, 'Analysis', 'mesh', args.detector)
    cams = (args.cameras.split(',') if args.cameras else
            sorted(os.path.basename(p).split('_')[0] for p in glob.glob(os.path.join(mesh_dir, '*_final_betas.npz'))))
    if not cams:
        raise SystemExit(f'no {{cam}}_final_betas.npz under {mesh_dir} -- run the trial through step_2b first')
    if not os.path.isdir(os.path.join(trial_root, user)):
        raise SystemExit(f'{trial_root}/{user} not found -- give --trial-root (calibs + user_meta.json)')
    variants = [v for v in VARIANTS if v in args.variants.split(',')]
    print(f'{user}/{action}, cameras {cams}, variants {variants}\n  base {base}\n  out  {out}')
    os.makedirs(out, exist_ok=True)
    smpl = load_smpl(args.n_betas)
    smpl_kid = load_smpl(args.n_betas, kid=True) if any(VARIANTS[v][1] for v in variants) else None
    cam_cli = ['--cameras', ','.join(cams)]

    res = {'mocap': {}, 'triangulated': {}}
    scale = {}
    for name in variants:
        vdir = os.path.join(out, name)
        skel_file, kid = VARIANTS[name]
        log(f'{name}: copy the trial through step_2b')
        copy_layout(base, vdir)
        full = None
        if skel_file:
            full, _ = write_fitted_betas(vdir, user, action, args.detector, cams, skel_file,
                                         smpl_kid if kid else smpl, args)
        log(f'{name}: step_3 (SMPL skeleton), step_4 (PnP), step_8 (score)')
        if skel_file:
            build_mesh_fitted(vdir, user, action, args.detector, cams, full, smpl_kid if kid else smpl,
                              stature_m(trial_root, vdir, user), args.rescale_fitted)
        else:
            run_step('step_3_extract_3d.py', vdir, trial_root, '--user', user, '--action', action, *cam_cli, '--force')
        run_step('step_4_PnP.py', vdir, trial_root, '--user', user, '--action', action, *cam_cli, '--force',
                 *(['--exclude-joints', args.exclude_joints] if args.exclude_joints else []))
        for gt, fn in (('mocap', 'mocap_h36m.npz'), ('triangulated', 'openpose_tri_h36m.npz')):
            if not os.path.exists(os.path.join(vdir, user, action, 'Analysis', 'H36M', fn)):
                print(f'  no {fn}: not scored against {gt}')       # Korea has no mocap
                continue
            run_step('step_8_spider_error.py', vdir, trial_root, '--user', user, '--action', action, *cam_cli,
                     '--force', '--gt', gt, '--tri', args.tri, '--verbose')
            res[gt][name] = scores(vdir, user, action, args.detector, gt)
        mp = np.load(os.path.join(vdir, user, action, 'Analysis', 'mesh', args.detector, f'{cams[0]}_mesh_pose.npz'))
        scale[name] = (float(mp['mesh_height_m']), float(mp['scale_factor']))

    log('comparison')
    text = (comparison_text(res['mocap'], user, action, 'mocap', scale) + '\n\n' + '=' * 110 + '\n\n'
            + comparison_text(res['triangulated'], user, action, 'triangulated', scale))
    if args.exclude_joints:
        text = f'PnP without {args.exclude_joints}\n\n' + text
    with open(os.path.join(out, 'comparison.txt'), 'w') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nnumbers in {os.path.join(out, "comparison.txt")}')


if __name__ == '__main__':
    main()
