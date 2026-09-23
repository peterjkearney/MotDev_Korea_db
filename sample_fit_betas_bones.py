#!/usr/bin/env python3
"""sample_fit_betas_bones.py -- SMPL betas from the bone lengths of a 3D H36M skeleton.

MotionBERT's betas are regressed from bounding-box-normalised 2D and come out
as a near-constant mean shape for every subject.  A triangulated (or mocap)
skeleton carries metric bone lengths, so body shape can be solved for directly:

    SMPL rest-pose H36M joints are LINEAR in the betas,
        J(beta) = J_regressor_h36m @ (v_template + shapedirs @ beta)
    so each segment length ||J_a(beta) - J_b(beta)|| is a mildly non-linear
    function of beta, and 10 betas are fitted to ~15 median segment lengths by
    damped least squares (ridge prior toward the mean shape).

Standalone: touches nothing in pipeline/.  Reads a mocap_h36m.npz or
openpose_tri_h36m.npz (kps3d (T,17,3) in mm, H36M-17 order) and prints the fitted
betas, the T-pose mesh height they imply, and per-segment fit residuals.

    python sample_fit_betas_bones.py --skel path/to/openpose_tri_h36m.npz --stature-m 1.89
    python sample_fit_betas_bones.py --skel mocap_h36m.npz --skel openpose_tri_h36m.npz \
        --compare-betas 00_final_betas.npz --stature-m 1.89
    python sample_fit_betas_bones.py --skel tri.npz --use-stature 1.89   # pin mesh height too

Batch over every BioCV trial (Colab: the Drive folder step_1b wrote into), one row per
trial x skeleton, plus a stature-vs-fitted-beta_00 scatter:

    python sample_fit_betas_bones.py --batch /content/drive/MyDrive/MotorDevelopment/Data/BioCV \
        --trial-dir /content/BioCV --csv betas_from_bones.csv --plot betas_from_bones.png

The SMPL model folder is MotionBERT's data/mesh: $MB_MESH_DIR, else $MB_DIR/data/mesh,
else ../MotionBERT or ../../Python_jetson/MotionBERT next to this script.
"""
import argparse
import os
import pickle

import numpy as np
from scipy.optimize import least_squares



def _find_mesh_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get('MB_MESH_DIR'),
             os.path.join(os.environ['MB_DIR'], 'data', 'mesh') if os.environ.get('MB_DIR') else None,
             os.path.join(here, '..', 'MotionBERT', 'data', 'mesh'),
             os.path.join(here, '..', '..', 'Python_jetson', 'MotionBERT', 'data', 'mesh'),
             os.path.join(here, '..', 'Python_jetson', 'MotionBERT', 'data', 'mesh')]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'SMPL_NEUTRAL.pkl')):
            return os.path.abspath(c)
    raise SystemExit('SMPL_NEUTRAL.pkl not found -- set MB_MESH_DIR to MotionBERT/data/mesh')


MB_MESH = _find_mesh_dir()

H36M = ['Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
J = {n: i for i, n in enumerate(H36M)}

# Segments to match.  Spine / Nose / Head are excluded from the triangulated target
# (no OpenPose joint there), so the torso is taken Hip->Thorax directly, and the two
# widths are added because they are the most shape-informative segments available.
SEGMENTS = [('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
            ('Hip', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
            ('Hip', 'Thorax'),
            ('Thorax', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
            ('Thorax', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
            ('RHip', 'LHip'), ('RShoulder', 'LShoulder')]


# ----------------------------------------------------------------------------- SMPL (linear part)
def load_smpl(n_betas=10):
    import warnings
    with open(os.path.join(MB_MESH, 'SMPL_NEUTRAL.pkl'), 'rb') as f, warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)      # scipy.sparse.csc path inside the pkl
        m = pickle.load(f, encoding='latin1')
    v_t = np.asarray(m['v_template'], dtype=np.float64)                 # (6890,3)  metres
    S = np.asarray(m['shapedirs'], dtype=np.float64)[:, :, :n_betas]    # (6890,3,10)
    Jr = np.load(os.path.join(MB_MESH, 'J_regressor_h36m_correct.npy')).astype(np.float64)  # (17,6890)
    J0 = Jr @ v_t                                                        # (17,3)
    A = np.einsum('jv,vck->jck', Jr, S)                                  # (17,3,10)
    return dict(v_t=v_t, S=S, J0=J0, A=A)


def rest_joints_mm(smpl, beta):
    return 1000.0 * (smpl['J0'] + smpl['A'] @ beta)


def mesh_height_m(smpl, beta):
    """Crown-to-sole height of the T-pose mesh (y-up canonical frame), as step_3 measures it."""
    v = smpl['v_t'] + smpl['S'] @ beta
    return float(v[:, 1].max() - v[:, 1].min())


def seg_lengths_mm(smpl, beta, segs):
    j = rest_joints_mm(smpl, beta)
    return np.array([np.linalg.norm(j[J[a]] - j[J[b]]) for a, b in segs])


# ----------------------------------------------------------------------------- data side
def data_seg_lengths(kps3d, segs):
    """Median (and IQR) over frames of each segment's length, frames where both joints exist."""
    med, iqr, n = [], [], []
    for a, b in segs:
        d = np.linalg.norm(kps3d[:, J[a]] - kps3d[:, J[b]], axis=-1)
        d = d[np.isfinite(d)]
        med.append(np.median(d) if d.size else np.nan)
        iqr.append((np.percentile(d, 75) - np.percentile(d, 25)) if d.size else np.nan)
        n.append(d.size)
    return np.array(med), np.array(iqr), np.array(n)


# ----------------------------------------------------------------------------- fit
def fit_betas(smpl, target_mm, segs, lam=1.0, sigma_mm=10.0, stature_m=None, w_stature=1.0):
    """Damped least squares.  Residuals are (fitted - target)/sigma per segment, sqrt(lam)*beta
    as a ridge prior, and optionally (mesh_height - stature)/sigma."""
    ok = np.isfinite(target_mm)
    segs_ok = [s for s, k in zip(segs, ok) if k]
    tgt = target_mm[ok]
    n_b = smpl['A'].shape[-1]

    def resid(beta):
        r = (seg_lengths_mm(smpl, beta, segs_ok) - tgt) / sigma_mm
        parts = [r, np.sqrt(lam) * beta]
        if stature_m is not None:
            parts.append(np.array([w_stature * 1000.0 * (mesh_height_m(smpl, beta) - stature_m) / sigma_mm]))
        return np.concatenate(parts)

    sol = least_squares(resid, np.zeros(n_b), method='lm')
    # conditioning: singular values of the segment Jacobian at the solution (per unit beta, in mm)
    Jac = sol.jac[:len(tgt)] * sigma_mm
    sv = np.linalg.svd(Jac, compute_uv=False)
    return sol.x, sv


# ----------------------------------------------------------------------------- batch
def _stature_m(user, out_dir, trial_dir):
    import json
    for root in (trial_dir, out_dir):
        if not root:
            continue
        for p in (os.path.join(root, user, 'user_meta.json'), os.path.join(root, 'user_meta.json')):
            if os.path.exists(p):
                with open(p) as f:
                    return float(json.load(f)['stature_m'])
    return float('nan')


def run_batch(args, smpl):
    import csv
    import glob
    rows = []
    for kind in ('mocap', 'triangulated'):
        fn = 'mocap_h36m.npz' if kind == 'mocap' else 'openpose_tri_h36m.npz'
        for path in sorted(glob.glob(os.path.join(args.batch, '*', '*', 'Analysis', 'H36M', fn))):
            action = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
            user = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path)))))
            kps = np.load(path, allow_pickle=True)['kps3d'].astype(np.float64)
            tgt, _, n = data_seg_lengths(kps, SEGMENTS)
            if np.isfinite(tgt).sum() < 6:
                print(f'skip {user}/{action} {kind}: only {np.isfinite(tgt).sum()} segments with data')
                continue
            beta, _ = fit_betas(smpl, tgt, SEGMENTS, lam=args.lam, sigma_mm=args.sigma_mm)
            fit = seg_lengths_mm(smpl, beta, SEGMENTS)
            row = dict(user=user, action=action, skeleton=kind, n_frames=int(n.max()),
                       stature_mm=1000 * _stature_m(user, args.batch, args.trial_dir),
                       mesh_height_mm=1000 * mesh_height_m(smpl, beta),
                       rms_segment_mm=float(np.sqrt(np.nanmean((fit - tgt) ** 2))))
            row.update({f'beta_{i:02d}': float(b) for i, b in enumerate(beta)})
            rows.append(row)
            print(f'{user:8s} {action:14s} {kind:12s} beta_00 {beta[0]:+.3f}  mesh {row["mesh_height_mm"]:.0f} mm'
                  f'  stature {row["stature_mm"]:.0f} mm  RMS {row["rms_segment_mm"]:.1f} mm')
    if not rows:
        raise SystemExit(f'no mocap_h36m.npz / openpose_tri_h36m.npz under {args.batch}/*/*/Analysis/H36M')
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.csv} ({len(rows)} rows)')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        colors = {'mocap': '#2a78d6', 'triangulated': '#e8842c'}
        for kind in ('mocap', 'triangulated'):
            r = [x for x in rows if x['skeleton'] == kind]
            if not r:
                continue
            st = np.array([x['stature_mm'] for x in r])
            axes[0].scatter(st, [x['beta_00'] for x in r], s=30, alpha=0.75, color=colors[kind], label=kind)
            axes[1].scatter(st, [x['mesh_height_mm'] for x in r], s=30, alpha=0.75, color=colors[kind], label=kind)
        lo, hi = 1550, 1950
        axes[1].plot([lo, hi], [lo, hi], color='#999', lw=1, ls='--')
        axes[0].set_xlabel('stature (mm)'); axes[0].set_ylabel('fitted beta_00')
        axes[1].set_xlabel('stature (mm)'); axes[1].set_ylabel('T-pose mesh height from fitted betas (mm)')
        axes[0].set_title('SMPL beta_00 fitted to bone lengths', loc='left', fontsize=10)
        axes[1].set_title('implied mesh height vs measured stature', loc='left', fontsize=10)
        for ax in axes:
            ax.grid(True, color='#e6e6e3'); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f'wrote {args.plot}')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skel', action='append', default=[],
                    help='mocap_h36m.npz or openpose_tri_h36m.npz (repeatable); kps3d (T,17,3) mm')
    ap.add_argument('--stature-m', type=float, default=None, help='measured stature, for reporting only')
    ap.add_argument('--use-stature', type=float, default=None,
                    help='also pin the T-pose mesh height to this stature (metres) in the fit')
    ap.add_argument('--lam', type=float, default=1.0, help='ridge weight toward beta=0 (default 1)')
    ap.add_argument('--sigma-mm', type=float, default=10.0, help='segment-length noise scale (default 10 mm)')
    ap.add_argument('--n-betas', type=int, default=10)
    ap.add_argument('--compare-betas', action='append', default=[],
                    help="MotionBERT {cam}_final_betas.npz to report alongside (repeatable)")
    ap.add_argument('--batch', default=None, help='OUT_DIR: fit every {user}/{action}/Analysis/H36M skeleton under it')
    ap.add_argument('--trial-dir', default=None, help='batch: where {user}/user_meta.json lives if not under --batch')
    ap.add_argument('--csv', default=None, help='batch: write one row per trial x skeleton here')
    ap.add_argument('--plot', default=None, help='batch: stature vs fitted beta_00 / mesh height scatter PNG')
    args = ap.parse_args()
    if not args.skel and not args.batch:
        ap.error('give --skel <npz> (repeatable) or --batch <OUT_DIR>')

    smpl = load_smpl(args.n_betas)
    if args.batch:
        run_batch(args, smpl)
        return
    print(f'SMPL neutral mean shape: T-pose height {mesh_height_m(smpl, np.zeros(args.n_betas)):.3f} m')
    if args.stature_m:
        print(f'subject stature: {args.stature_m:.3f} m')

    for path in args.skel:
        z = np.load(path, allow_pickle=True)
        kps = z['kps3d'].astype(np.float64)
        names = [str(n) for n in z['joint_names']] if 'joint_names' in z.files else H36M
        assert names == H36M, f'{path}: joint order {names} != H36M-17'
        tgt, iqr, n = data_seg_lengths(kps, SEGMENTS)

        beta, sv = fit_betas(smpl, tgt, SEGMENTS, lam=args.lam, sigma_mm=args.sigma_mm,
                             stature_m=args.use_stature)
        fit = seg_lengths_mm(smpl, beta, SEGMENTS)
        mean = seg_lengths_mm(smpl, np.zeros(args.n_betas), SEGMENTS)

        print(f'\n=== {path}')
        print(f'    {kps.shape[0]} frames, {np.isfinite(kps).all(-1).any(-1).sum()} with any joint')
        print(f'    fitted betas : ' + ' '.join(f'{b:+.3f}' for b in beta))
        h = mesh_height_m(smpl, beta)
        line = f'    T-pose mesh height from fitted betas: {h:.3f} m'
        if args.stature_m:
            line += f'   (stature {args.stature_m:.3f} m, diff {1000 * (h - args.stature_m):+.0f} mm)'
        print(line)
        print(f'    segment-Jacobian singular values (mm per unit beta): '
              + ' '.join(f'{s:.1f}' for s in sv))
        print(f'    {"segment":22s} {"target":>8s} {"IQR":>6s} {"n":>5s} {"mean-shape":>11s} {"fitted":>8s} {"resid":>7s}')
        for (a, b), t, q, k, m0, f in zip(SEGMENTS, tgt, iqr, n, mean, fit):
            print(f'    {a + "-" + b:22s} {t:8.1f} {q:6.1f} {k:5d} {m0:11.1f} {f:8.1f} {f - t:+7.1f}')
        rms = np.sqrt(np.nanmean((fit - tgt) ** 2))
        rms0 = np.sqrt(np.nanmean((mean - tgt) ** 2))
        print(f'    RMS segment residual: mean shape {rms0:.1f} mm  ->  fitted {rms:.1f} mm')

    for path in args.compare_betas:
        b = np.load(path)['betas'].astype(np.float64)[:args.n_betas]
        print(f'\n--- MotionBERT {path}')
        print(f'    betas        : ' + ' '.join(f'{x:+.3f}' for x in b))
        print(f'    T-pose mesh height: {mesh_height_m(smpl, b):.3f} m')
        f = seg_lengths_mm(smpl, b, SEGMENTS)
        print(f'    (its segments vs the last skeleton above: RMS {np.sqrt(np.nanmean((f - tgt) ** 2)):.1f} mm)')


if __name__ == '__main__':
    main()
