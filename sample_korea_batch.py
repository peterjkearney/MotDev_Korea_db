#!/usr/bin/env python3
"""sample_korea_batch.py -- every Korea rep in a folder through the child recipe, one table at the end.

Per rep (data/sampleKorea/B0xx_GMS_a_b.npz, raw aligned or build_child_gt.py format):
  1. sample_hip_ablation.py       calibrated twin from --gt3d (else self-calibration), layout, MotionBERT,
                                  PnP with / without hips, step_5 videos, step_8       (skipped if done)
  2. sample_betas_ablation.py     shapes: motionbert | bones_tri (adult SMPL) | bones_kid (AGORA kid blend),
                                  PnP without hips, step_8
  3. sample_refine_2d.py          three refinements, ray_const rotation, feet pinned, floor hinge, 30 fps thresholds:
       motionbert   hips out of the 2D term                                   (the adult-body recipe)
       bones_tri    hips/spine/nose/head out, floor-contact
       bones_kid    spine/nose/head out, floor-contact                        (the child recipe)
Everything lands under <folder>/out/<stub>/ as the single-rep tools lay it out; this adds
<folder>/out/batch_summary.txt and batch_summary.csv: per rep and variant, MPJPE placed / N-MPJPE /
PA-MPJPE (mean over cameras and per camera), plus the fitted kid weight and the number of scored joints.

    ~/anaconda3/envs/motEnv/bin/python sample_korea_batch.py
    ~/anaconda3/envs/motEnv/bin/python sample_korea_batch.py --reps B010_GMS_1_1,B023_GMS_2_2
    ~/anaconda3/envs/motEnv/bin/python sample_korea_batch.py --summary-only
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GT3D = '/Volumes/Expansion/MotorDevelopment/Korea/B/GT3D'
DEFAULT_ALIGNED = '/Volumes/Expansion/MotorDevelopment/Korea/B/Aligned'   # every rep of every subject
KOREA_FPS = ['--min-contact-frames', '3', '--foot-speed-px', '3', '--foot-lowest-margin-px', '30']
FLOORC = ['--w-floor-contact', '3000', '--floor-contact-m', '0.025']
BASE = ['--huber', '20', '--w-foot', '3000', '--w-floor', '3000', '--fix-rotation', 'ray_const'] + KOREA_FPS
REFINES = {                     # name: (shape variant, extra flags)
    'motionbert': ('motionbert', ['--exclude-2d', 'Hip,RHip,LHip']),
    'bones_tri': ('bones_tri', ['--exclude-2d', 'Hip,RHip,LHip,Spine,Nose,Head'] + FLOORC),
    'bones_kid': ('bones_kid', ['--exclude-2d', 'Spine,Nose,Head'] + FLOORC),
}
COLS = ['placed', 'n', 'pa']


def run(cmd, log_path):
    with open(log_path, 'w') as f:
        f.write('$ ' + ' '.join(cmd) + '\n\n')
        f.flush()
        r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f'{os.path.basename(cmd[1])} failed (exit {r.returncode}), see {log_path}')


def metrics(vdir, subject, stub):
    p = os.path.join(vdir, subject, stub, 'Analysis', 'diagnostics', 'openpose', 'error_metrics_tri.npz')
    if not os.path.exists(p):
        return None
    m = np.load(p, allow_pickle=True)
    return dict(cameras=[str(c) for c in m['cameras']], placed=m['err_placed_all'], n=m['n_mpjpe'], pa=m['pa_mpjpe'])


def process(rep, out_root, gt3d, calib_data, force):
    stub = os.path.basename(rep)[:-4]
    subject = stub.split('_')[0]
    out = os.path.join(out_root, stub)
    py = sys.executable
    t0 = time.time()
    base = os.path.join(out, 'pnp_no_hips')
    os.makedirs(out, exist_ok=True)
    if force or not os.path.exists(os.path.join(base, subject, stub, 'Analysis', 'mesh', 'openpose', '1_final_betas.npz')):
        print(f'  [{stub}] 1. layout + MotionBERT + PnP (sample_hip_ablation)', flush=True)
        run([py, os.path.join(HERE, 'sample_hip_ablation.py'), '--rep', rep, '--gt3d', gt3d, '--out', out]
            + (['--calib-data', calib_data] if calib_data and os.path.isdir(calib_data) else [])
            + (['--force'] if force else []), os.path.join(out, 'batch_1_hip_ablation.log'))
    else:
        print(f'  [{stub}] 1. layout exists, skipped', flush=True)
    print(f'  [{stub}] 2. shapes (sample_betas_ablation)', flush=True)
    run([py, os.path.join(HERE, 'sample_betas_ablation.py'), '--base', base, '--trial-root', base,
         '--out', os.path.join(out, 'betas_ablation'), '--variants', 'motionbert,bones_tri,bones_kid',
         '--exclude-joints', 'Hip,RHip,LHip'], os.path.join(out, 'batch_2_betas_ablation.log'))
    for name, (shape, flags) in REFINES.items():
        print(f'  [{stub}] 3. refine {name}', flush=True)
        run([py, os.path.join(HERE, 'sample_refine_2d.py'), '--src', os.path.join(out, 'betas_ablation', shape),
             '--out', os.path.join(out, 'refine2d', f'batch_{name}'), '--trial-root', base] + BASE + flags,
            os.path.join(out, f'batch_3_refine_{name}.log'))
    print(f'  [{stub}] done in {(time.time() - t0) / 60:.1f} min', flush=True)


def collect(rep, out_root):
    stub = os.path.basename(rep)[:-4]
    subject = stub.split('_')[0]
    out = os.path.join(out_root, stub)
    rows = []
    info = dict(rep=stub, subject=subject)
    gt = os.path.join(out, 'GT3D', subject, stub + '.npz')
    if os.path.exists(gt):
        d = np.load(gt, allow_pickle=True)
        info.update(n_frames=int(d['h36m_2d'].shape[1]), stature_m=float(d['stature_m']),
                    frames_old_gate=int(d['frame_usable'].sum()))
    summ = os.path.join(out, 'GT3D', subject, 'session_summary.json')
    if os.path.exists(summ):
        with open(summ) as f:
            s = json.load(f)
        info['calib'] = f"{s.get('calib_rule', '?')} ({s.get('n_calib_frames', '?')} frames)"
        info['session_usable'] = bool(s.get('usable', False))
        info['session_reasons'] = '; '.join(s.get('reasons', []))
        rep_q = next((r for r in s.get('reps', []) if r.get('stub') == stub), None)
        if rep_q:
            info['rep_usable'] = bool(rep_q.get('usable', False))
            info['rep_reasons'] = '; '.join(x for x in rep_q.get('reasons', []) if x != 'session not usable')
    tri = os.path.join(out, 'pnp_no_hips', subject, stub, 'Analysis', 'H36M', 'openpose_tri_h36m.npz')
    if os.path.exists(tri):
        k = np.load(tri, allow_pickle=True)['kps3d']
        info['scored_joints'] = int(np.isfinite(k).all(-1).sum())
    fb = os.path.join(out, 'betas_ablation', 'bones_kid', subject, stub, 'Analysis', 'mesh', 'openpose', '1_final_betas.npz')
    if os.path.exists(fb):
        b = np.load(fb, allow_pickle=True)['betas']
        info['kid_weight'] = float(b[-1]) if len(b) > 10 else float('nan')
        info['segment_rms_mm'] = float(np.load(fb, allow_pickle=True)['segment_rms_mm'])
    variants = [('PnP motionbert', os.path.join(out, 'betas_ablation', 'motionbert')),
                ('PnP bones_kid', os.path.join(out, 'betas_ablation', 'bones_kid'))]
    variants += [(f'refined {n}', os.path.join(out, 'refine2d', f'batch_{n}')) for n in REFINES]
    for label, vdir in variants:
        m = metrics(vdir, subject, stub)
        row = dict(info, variant=label)
        if m:
            for c in COLS:
                row[c] = float(np.nanmean(m[c]))
                for cam, v in zip(m['cameras'], m[c]):
                    row[f'{c}_cam{cam}'] = float(v)
        rows.append(row)
    return rows


def summary_text(rows):
    reps = sorted({r['rep'] for r in rows})
    variants = []
    for r in rows:
        if r['variant'] not in variants:
            variants.append(r['variant'])
    lines = ['Korea sample reps: error against the all-camera triangulated OpenPose target, per-joint gated; mean over cameras (mm)', '']
    hdr = f'{"rep":14} {"sess":>5} {"frames":>6} {"joints":>6} {"kid w":>6} {"stature":>7}  ' + ''.join(f'{v:>22}' for v in variants)
    lines += [hdr, f'{"":14} {"":>5} {"":>6} {"":>6} {"":>6} {"":>7}  ' + ''.join(f'{"placed / N / PA":>22}' for _ in variants)]
    by = {(r['rep'], r['variant']): r for r in rows}
    means = {v: {c: [] for c in COLS} for v in variants}
    for rep in reps:
        r0 = by[(rep, variants[0])]
        sess = 'ok' if r0.get('session_usable') else ('BAD' if 'session_usable' in r0 else '?')
        line = (f'{rep:14} {sess:>5} {r0.get("n_frames", float("nan")):6.0f} {r0.get("scored_joints", float("nan")):6.0f} '
                f'{r0.get("kid_weight", float("nan")):6.2f} {r0.get("stature_m", float("nan")):7.3f}  ')
        for v in variants:
            r = by.get((rep, v), {})
            if 'placed' in r:
                line += f'{r["placed"]:7.1f}{r["n"]:7.1f}{r["pa"]:7.1f} '
                for c in COLS:
                    means[v][c].append(r[c])
            else:
                line += f'{"(not run)":>22}'
        lines.append(line)
    line = f'{"MEAN":14} {"":>5} {"":>6} {"":>6} {"":>6} {"":>7}  '
    for v in variants:
        line += ''.join(f'{np.mean(means[v][c]):7.1f}' for c in COLS) + ' ' if means[v]['placed'] else f'{"":>22}'
    lines += ['', line]
    line = f'{"MEDIAN":14} {"":>5} {"":>6} {"":>6} {"":>6} {"":>7}  '
    for v in variants:
        line += ''.join(f'{np.median(means[v][c]):7.1f}' for c in COLS) + ' ' if means[v]['placed'] else f'{"":>22}'
    lines.append(line)
    lines += ['', 'per camera, MPJPE placed (mm):', f'{"rep":14} ' + ''.join(f'{v:>30}' for v in variants)]
    for rep in reps:
        line = f'{rep:14} '
        for v in variants:
            r = by.get((rep, v), {})
            cams = sorted(k for k in r if k.startswith('placed_cam'))
            line += f'{"  ".join(f"{k[-1]}: {r[k]:5.1f}" for k in cams):>30}' if cams else f'{"":>30}'
        lines.append(line)
    lines += ['', 'session quality (build_child_gt gates; BAD = the builder would not have saved this rep):']
    for rep in reps:
        r0 = by[(rep, variants[0])]
        if 'session_usable' in r0:
            lines.append(f'  {rep:14} {r0.get("calib", "")}; session: ' + (r0['session_reasons'] or 'ok')
                         + ('; rep: ' + r0['rep_reasons'] if r0.get('rep_reasons') else ''))
    lines += ['', 'variants: PnP = step_4 rigid placement, hips excluded; refined = sample_refine_2d (ray_const, feet pinned, floor hinge);',
              '  motionbert: MotionBERT shape, hips out of the 2D term; bones_tri: adult SMPL fitted to bone lengths, hips/spine/nose/head out,',
              '  floor-contact; bones_kid: AGORA kid blend fitted to bone lengths, spine/nose/head out, floor-contact (the child recipe).',
              'kid w = fitted kid-template weight (0 adult, 1 infant template); joints = scored target joints after per-joint gating.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--folder', default=os.path.join(HERE, 'data', 'sampleKorea'))
    ap.add_argument('--gt3d', default=DEFAULT_GT3D)
    ap.add_argument('--calib-data', default=DEFAULT_ALIGNED,
                    help='no calibrated twin: calibrate the session from every rep of the subject here')
    ap.add_argument('--reps', default=None, help='comma-separated stubs; default every B*.npz in --folder')
    ap.add_argument('--force', action='store_true', help='redo the layout and MotionBERT too')
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    reps = sorted(glob.glob(os.path.join(args.folder, 'B*.npz')))
    if args.reps:
        want = set(args.reps.split(','))
        reps = [r for r in reps if os.path.basename(r)[:-4] in want]
    out_root = os.path.join(args.folder, 'out')
    os.makedirs(out_root, exist_ok=True)
    print(f'{len(reps)} reps, GT3D {args.gt3d} ({"mounted" if os.path.isdir(args.gt3d) else "NOT reachable: self-calibration"})', flush=True)
    failed = []
    if not args.summary_only:
        for rep in reps:
            print(f'\n### {os.path.basename(rep)[:-4]}', flush=True)
            try:
                process(rep, out_root, args.gt3d, args.calib_data, args.force)
            except Exception as e:                       # keep going, report at the end
                traceback.print_exc()
                failed.append((os.path.basename(rep)[:-4], str(e)))
    rows = []
    for rep in reps:
        rows += collect(rep, out_root)
    text = summary_text(rows)
    if failed:
        text += '\n\nFAILED:\n' + '\n'.join(f'  {s}: {e}' for s, e in failed)
    with open(os.path.join(out_root, 'batch_summary.txt'), 'w') as f:
        f.write(text + '\n')
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out_root, 'batch_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print('\n' + text)
    print(f'\nwritten: {os.path.join(out_root, "batch_summary.txt")} and .csv')


if __name__ == '__main__':
    main()
