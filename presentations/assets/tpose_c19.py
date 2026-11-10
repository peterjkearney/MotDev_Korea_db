"""Slide-2 T-pose panel: MotionBERT / adult fit / current-recipe child shape (kid_oprg, C19), B010 dev tree."""
import glob
import os
import sys

import numpy as np

REPO = '/Users/peterkearney/Documents/MotorDev/Python/Korea_db'
sys.path.insert(0, os.path.join(REPO, 'pipeline'))
sys.path.insert(0, os.path.join(REPO, 'pipeline', 'tools'))
from utils.smpl_lite import H36M, J, load_linear, _load_pkl, load_h36m_regressor, joint_regressor  # noqa: E402
from render_shape_tpose import rest_mesh, ghost, draw_mesh, draw_skeleton, seg_mm, leg_torso, CHAIN, SIDE_COL  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(REPO, 'data', 'korea_dev_colab', 'B010')
REP = 'B010_GMS_1_7'
OUT = sys.argv[1]

shapes = np.load(os.path.join(ROOT, 'shapes.npz'), allow_pickle=True)
mesh_dir = os.path.join(ROOT, REP, 'Analysis', 'mesh', 'openpose')
bf = [p for p in sorted(glob.glob(os.path.join(mesh_dir, '*_betas.npz'))) if not p.endswith('_final_betas.npz')]
mb = np.median(np.concatenate([np.load(p)['betas'] for p in bf]), axis=0)
stature = float(np.load(glob.glob(os.path.join(mesh_dir, '*_mesh_pose.npz'))[0])['stature_m'])
target = shapes['target_mm']
tgt_ratio = leg_torso(target)

lin10, lin_kid = load_linear(10), load_linear(10, kid=True)
Jh = load_h36m_regressor()
kid = shapes['kid_oprg']
Jopr = joint_regressor('opr', kw=float(kid[-1]))
faces = np.asarray(_load_pkl()['f'], dtype=np.int64)

panels = [('MotionBERT (adult body)', lin10, mb, stature, Jh, None),
          ('Adult model fitted to the child', lin10, shapes['adult'], None, Jh, None),
          ('Child model (current recipe)', lin_kid, kid, None, Jopr, float(shapes['kid_oprg_rms_mm']))]

fig, axes = plt.subplots(1, 3, figsize=(18, 7.4), facecolor='white')
for ax, (title, lin, shape, to_h, Jr, rms_fit) in zip(axes, panels):
    v, _, s = rest_mesh(lin, shape, Jh, to_h)
    j = Jr @ v                                         # joints of the set this shape is used with
    draw_mesh(ax, v, faces)
    draw_skeleton(ax, ghost(j, target), col='#8a8a8a', lw=5, ms=7, z=2, alpha=0.55, bones=CHAIN)
    draw_skeleton(ax, {n: j[i] for i, n in enumerate(H36M)})
    seg = seg_mm(j)
    ok = np.isfinite(target)
    rms = rms_fit if rms_fit is not None else np.sqrt(np.mean((seg[ok] - target[ok]) ** 2))
    h = v[:, 1].max()
    size = f'scaled ×{s:.2f} to the child\'s height' if to_h else f'fitted height {h:.2f} m'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.text(0.5, -0.02, f'{size}\nbone-length error {rms:.0f} mm   leg/torso {leg_torso(seg):.2f} '
                        f'(child {tgt_ratio:.2f})', transform=ax.transAxes, ha='center', va='top', fontsize=13)
    ax.axhline(0, color='#bbbbbb', lw=1, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(-0.55 * stature, 0.55 * stature)
    ax.set_ylim(-0.03 * stature, 1.05 * stature)
    ax.axis('off')

handles = [plt.Line2D([], [], color=SIDE_COL['L'], lw=2.5, marker='o', label='model joints, left'),
           plt.Line2D([], [], color=SIDE_COL['R'], lw=2.5, marker='o', label='model joints, right'),
           plt.Line2D([], [], color='#8a8a8a', lw=5, alpha=0.55, label="child's measured bone lengths")]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.99))
fig.tight_layout(rect=(0, 0.05, 1, 0.94))
fig.savefig(OUT, dpi=130)
print(OUT, 'stature', stature, 'kid w', kid[-1], 'girth b1 kid/kid_oprg', shapes['kid'][0], kid[0])
