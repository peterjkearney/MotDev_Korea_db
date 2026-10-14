#!/usr/bin/env python3
"""render_shape_tpose.py -- one subject's three body shapes side by side in the SMPL T-pose, front view.

    MotionBERT   median of the per-frame pass-1 betas over every camera (step_2a)
    adult        step_2c's 10 betas fitted to the triangulated bone lengths
    kid          step_2c's 10 betas + AGORA kid-template weight

Each mesh is drawn at the size step_4b uses it: MotionBERT's scaled to the subject's stature (as step_3
scales it), the two fits at their native metric size (step_4b does not rescale them).  All stand on one floor.  The mesh's H36M joints (the checkpoint's
regressor) are drawn over it, and a grey ghost skeleton built from the target's median segment
lengths (shapes.npz target_mm, metric) is laid along each mesh's own bone directions, from the
ankles up -- where the ghost's hips/shoulders/wrists part from the mesh's is where the shape is wrong.
The RMS label is the displayed mesh against those target segments.

    reads   {rep}/step4b/{user}/shapes.npz, {rep}/step4b/{user}/{rep}/Analysis/mesh/openpose/*_betas.npz,
            stature from *_mesh_pose.npz
    writes  {rep}/shape_tpose.png

    python3 tools/render_shape_tpose.py                      # first sample rep
    python3 tools/render_shape_tpose.py --rep B023_GMS_2_2
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from utils.smpl_lite import H36M, J, SEGMENTS, load_linear, _load_pkl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'data', 'sampleKorea', 'out')

# H36M bones drawn on the mesh; the ghost is SEGMENTS chained from the hip (Spine/Nose/Head have no target)
BONES = [('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('Hip', 'LHip'), ('LHip', 'LKnee'),
         ('LKnee', 'LAnkle'), ('Hip', 'Spine'), ('Spine', 'Thorax'), ('Thorax', 'Nose'), ('Nose', 'Head'),
         ('Thorax', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
         ('Thorax', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist')]
CHAIN = [s for s in SEGMENTS if s not in (('RHip', 'LHip'), ('RShoulder', 'LShoulder'))]   # parent -> child
SIDE_COL = {'L': '#2f6fd6', 'R': '#d64a2f', 'C': '#222222'}


def side(name):
    return name[0] if name[0] in 'LR' and name[1].isupper() else 'C'


def rest_mesh(lin, shape, Jh, stature_m):
    """T-pose vertices and H36M joints for a shape, soles at y=0, pelvis at x=0; scaled to stature_m
    when given, else native size."""
    v = lin['v_t'] + lin['S'] @ np.asarray(shape, dtype=np.float64)
    s = stature_m / (v[:, 1].max() - v[:, 1].min()) if stature_m else 1.0
    v = v * s
    j = Jh @ v
    off = np.array([j[J['Hip'], 0], v[:, 1].min(), 0.0])
    return v - off, j - off, s


def ghost(j_mesh, target_mm):
    """Target segment lengths laid along the mesh's bone directions, ankles on the mesh's ankles."""
    tgt = dict(zip(SEGMENTS, np.asarray(target_mm) / 1000.0))
    g = {'Hip': j_mesh[J['Hip']].copy()}
    for a, b in CHAIN:
        d = j_mesh[J[b]] - j_mesh[J[a]]
        g[b] = g[a] + d / np.linalg.norm(d) * tgt[(a, b)]
    drop = np.mean([g[k][1] - j_mesh[J[k]][1] for k in ('RAnkle', 'LAnkle')])
    return {k: p - np.array([0.0, drop, 0.0]) for k, p in g.items()}


def draw_mesh(ax, v, faces):
    tri = v[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    vis = n[:, 2] > 0                                            # viewer on +z (SMPL faces +z)
    tri, n = tri[vis], n[vis]
    order = np.argsort(tri[:, :, 2].mean(1))                     # painter: far first
    tri, n = tri[order], n[order]
    light = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    base = np.array([0.80, 0.78, 0.74])
    cols = np.clip(shade[:, None] * base, 0, 1)
    ax.add_collection(PolyCollection(tri[:, :, :2], facecolors=cols, edgecolors='none', zorder=1))


def draw_skeleton(ax, pts, col=None, lw=2.5, ms=5, z=3, alpha=1.0, bones=BONES, ls='-'):
    for a, b in bones:
        if a in pts and b in pts:
            c = col or SIDE_COL[side(b)]
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color=c, lw=lw, ls=ls,
                    zorder=z, alpha=alpha, solid_capstyle='round')
    for k, p in pts.items():
        c = col or SIDE_COL[side(k)]
        ax.plot(p[0], p[1], 'o', ms=ms, color=c, mec='white', mew=0.8, zorder=z + 1, alpha=alpha)


def seg_mm(j):
    return np.array([1000 * np.linalg.norm(j[J[a]] - j[J[b]]) for a, b in SEGMENTS])


def leg_torso(seg):
    s = dict(zip(SEGMENTS, seg))
    leg = np.mean([s[('RHip', 'RKnee')] + s[('RKnee', 'RAnkle')], s[('LHip', 'LKnee')] + s[('LKnee', 'LAnkle')]])
    return leg / s[('Hip', 'Thorax')]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of {user}_{action} rep folders')
    ap.add_argument('--rep', default=None, help='rep folder (default: the first)')
    ap.add_argument('--out', default=None, help='default: {rep}/shape_tpose.png')
    args = ap.parse_args()

    rep = args.rep or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))[0]
    user = rep.split('_', 1)[0]
    base = os.path.join(args.root, rep, 'step4b', user)
    shapes = np.load(os.path.join(base, 'shapes.npz'), allow_pickle=True)
    mesh_dir = os.path.join(base, rep, 'Analysis', 'mesh', 'openpose')
    beta_files = sorted(glob.glob(os.path.join(mesh_dir, '*_betas.npz')))
    beta_files = [p for p in beta_files if not p.endswith('_final_betas.npz')]
    mb = np.median(np.concatenate([np.load(p)['betas'] for p in beta_files]), axis=0)
    stature = float(np.load(glob.glob(os.path.join(mesh_dir, '*_mesh_pose.npz'))[0])['stature_m'])
    target = shapes['target_mm']
    tgt_ratio = leg_torso(target)

    lin10, lin_kid = load_linear(10), load_linear(10, kid=True)
    from utils.smpl_lite import load_h36m_regressor
    Jh = load_h36m_regressor()
    faces = np.asarray(_load_pkl()['f'], dtype=np.int64)

    panels = [('MotionBERT (median betas, %d cams)' % len(beta_files), lin10, mb, stature),
              ('Fitted adult (10 betas)', lin10, shapes['adult'], None),
              ('Fitted kid (10 betas + kid template, w=%.2f)' % shapes['kid'][-1], lin_kid, shapes['kid'], None)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6), facecolor='white')
    for ax, (title, lin, shape, to_h) in zip(axes, panels):
        v, j, s = rest_mesh(lin, shape, Jh, to_h)
        draw_mesh(ax, v, faces)
        g = ghost(j, target)
        draw_skeleton(ax, g, col='#8a8a8a', lw=5, ms=7, z=2, alpha=0.55, bones=CHAIN)
        draw_skeleton(ax, {n: j[i] for i, n in enumerate(H36M)})
        seg = seg_mm(j)
        ok = np.isfinite(target)
        rms = np.sqrt(np.mean((seg[ok] - target[ok]) ** 2))
        h = v[:, 1].max()
        size = (f'native height {h / s:.2f} m  ×{s:.2f} → stature {h:.2f} m' if to_h else
                f'native metric size, height {h:.2f} m (stature {stature:.2f} m)')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.text(0.5, -0.02, f'{size}\n'
                            f'segment RMS vs target {rms:.0f} mm   leg/torso {leg_torso(seg):.2f} '
                            f'(target {tgt_ratio:.2f})',
                transform=ax.transAxes, ha='center', va='top', fontsize=10.5)
        ax.axhline(0, color='#bbbbbb', lw=1, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(-0.55 * stature, 0.55 * stature)
        ax.set_ylim(-0.03 * stature, 1.05 * stature)
        ax.axis('off')

    handles = [plt.Line2D([], [], color=SIDE_COL['L'], lw=2.5, marker='o', label='H36M left'),
               plt.Line2D([], [], color=SIDE_COL['R'], lw=2.5, marker='o', label='H36M right'),
               plt.Line2D([], [], color='#8a8a8a', lw=5, alpha=0.55, label='target bone lengths (triangulated), from the ankles')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f'{rep}: T-pose body shapes at the size step_4b uses them (front view)', y=0.94, fontsize=14)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = args.out or os.path.join(args.root, rep, 'shape_tpose.png')
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == '__main__':
    main()
