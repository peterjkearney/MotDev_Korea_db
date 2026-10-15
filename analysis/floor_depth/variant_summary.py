#!/usr/bin/env python3
"""variant_summary.py -- compare step_4b variants (one OUT_DIR copy each) on the same trials and config.

Per camera: placed and PA MPJPE (step_8), grounded-ankle depth error (placed vs target, camera z), the lean
along the line of sight (shoulder minus ankle depth bias, all frames; + = top of the body too far from the
camera), and the final 2D reprojection RMS when the run's log is given (run_override.py --verbose).

    python3 ../analysis/floor_depth/variant_summary.py --config C12 \\
        ray_const=<tree1>[:<log1>] ray=<tree2>[:<log2>] ...
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, '.')
from utils.smpl_lite import J  # noqa: E402

SHOULDERS, ANKLES = ('Thorax', 'RShoulder', 'LShoulder'), ('RAnkle', 'LAnkle')


def rms_from_log(path):
    out, user = {}, None
    if not path or not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r'##### (B\d+)', line)
        if m:
            user = m.group(1)
        m = re.match(r'\s+(\w+): stage 2 .*reprojection RMS ([\d.]+) px', line)
        if m:
            out[(user, m.group(1))] = float(m.group(2))
    return out


def cam_rows(tree, cfg, rms):
    rows = {}
    for mp in sorted(glob.glob(f'{tree}/*/*/Analysis/diagnostics/openpose/{cfg}/error_metrics_tri.npz')):
        u, a = mp[len(tree) + 1:].split(os.sep)[:2]
        m = np.load(mp)
        names = list(m['joint_names'])
        tgt = np.load(f'{tree}/{u}/{a}/Analysis/H36M/openpose_tri_h36m.npz')['kps3d'].astype(float)
        for k, cam in enumerate(m['cameras']):
            cam = str(cam)
            z = m['bias_xyz'][k, :, 2]
            lean = np.nanmean([z[names.index(j)] for j in SHOULDERS]) - np.nanmean([z[names.index(j)] for j in ANKLES])
            p = np.load(f'{tree}/{u}/{a}/Analysis/PnP/openpose/{cfg}/{cam}_pnp.npz', allow_pickle=True)
            R, t = p['cam_L_ext'][:3, :3], p['cam_L_ext'][:3, 3]
            placed = p['kps_H36M_placed'].astype(float) * 1000.0
            dz = []
            for col, an in enumerate(ANKLES):
                g = p['foot_grounded'][:, col] & np.isfinite(tgt[:, J[an]]).all(1) & np.isfinite(placed[:, J[an]]).all(1)
                dz.append(placed[g, J[an], 2] - (tgt[g, J[an]] @ R.T + t)[:, 2])
            rows[(u, a, cam)] = (float(m['err_placed_all'][k]), float(m['pa_mpjpe'][k]),
                                 float(np.median(np.concatenate(dz))), float(lean), rms.get((u, cam), np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='C12')
    ap.add_argument('variants', nargs='+', help='label=tree[:log]')
    args = ap.parse_args()
    data = {}
    for v in args.variants:
        label, rest = v.split('=', 1)
        tree, _, log = rest.partition(':')
        data[label] = cam_rows(tree, args.config, rms_from_log(log))
    labels = list(data)
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    cols = ('placed', 'PA', 'ankle dz', 'lean', '2D px')
    print(f'{args.config}; each cell: ' + ' / '.join(labels))
    print(f'{"trial cam":22s} ' + ' | '.join(f'{c:>{6 * len(labels)}s}' for c in cols))
    for key in keys:
        cells = []
        for i, c in enumerate(cols):
            fmt = '+6.0f' if c in ('ankle dz', 'lean') else '6.1f'
            cells.append(''.join(format(data[l][key][i], fmt) for l in labels))
        print(f'{key[1]:15s} cam{key[2]:3s} ' + ' | '.join(cells))
    print(f'\nmean over {len(keys)} cameras (|ankle dz|, |lean|):')
    for l in labels:
        a = np.array([data[l][k] for k in keys])
        print(f'  {l:14s} placed {a[:, 0].mean():5.1f}  PA {a[:, 1].mean():4.1f}  |ankle dz| {np.abs(a[:, 2]).mean():4.0f}  '
              f'|lean| {np.abs(a[:, 3]).mean():4.0f}  lean mean {a[:, 3].mean():+4.0f}  2D {np.nanmean(a[:, 4]):.2f} px')


if __name__ == '__main__':
    main()
