"""topdown.py -- the top-down (X vs depth) skeleton panel shared by step_5 and tools/threshold_sweep.

The static part (axes, grid, camera marker and field-of-view cone, title, legend) is drawn by
matplotlib ONCE; the skeletons are composited onto a copy per frame with cv2, through the
captured data->pixel transform, so a frame costs a few dozen line draws rather than a figure
re-render.
"""
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

H36M_LIMBS = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
_INK, _GRID = '#4b5563', '#c7cdd4'


def hex_bgr(h, fade=0.0):
    """matplotlib colour -> cv2 BGR; fade in [0,1) mixes towards white (cv2 has no alpha)."""
    r, g, b = matplotlib.colors.to_rgb(h)
    return tuple(int(255 * (c * (1 - fade) + fade)) for c in (b, g, r))


def robust_limits(xz_arrays, x_min_half=0.5, z_min=1.0):
    """Fixed axis limits from every skeleton's (…,2) X/Z over the clip, by percentile so one
    wild frame does not shrink everything else."""
    xs = np.concatenate([a[..., 0].ravel() for a in xz_arrays])
    zs = np.concatenate([a[..., 1].ravel() for a in xz_arrays])
    xs, zs = xs[np.isfinite(xs)], zs[np.isfinite(zs)]
    if xs.size == 0 or zs.size == 0:
        return (-1.0, 1.0), (-0.3, 2.0)
    xh = max(float(np.percentile(np.abs(xs), 99)) * 1.25, x_min_half)
    return (-xh, xh), (-0.3, max(float(np.percentile(zs, 99)) * 1.15, z_min))


class TopDown:
    """step_5's top-down panel for several skeletons: static axes drawn once by matplotlib,
    skeletons composited per frame with cv2 through the captured data->pixel transform."""

    def __init__(self, width, height, half_fov, xlim, zlim, title, legend, dpi=100):
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor('white')
        fig.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.06)     # title + legend sit above the axes
        cone = zlim[1] * 1.05
        for sgn in (-1, 1):
            ax.plot([0, sgn * cone * np.sin(half_fov)], [0, cone * np.cos(half_fov)], color=_INK, lw=1.1, ls='--', alpha=0.6)
        ax.plot(0, 0, marker='^', markersize=8, color=_INK)
        ax.text(0, -0.22, 'camera', color=_INK, fontsize=8, ha='center')
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect('equal')
        ax.grid(True, color=_GRID, lw=0.6, alpha=0.7)
        ax.set_xlabel('X, camera-space (m)', color=_INK, fontsize=9)
        ax.set_ylabel('Z, depth (m)', color=_INK, fontsize=9)
        fig.suptitle(title, color='#14181d', fontsize=10, y=0.992)
        for sp in ax.spines.values():
            sp.set_color(_GRID)
        ax.tick_params(colors=_INK, labelsize=8)
        handles = [plt.Line2D([0], [0], color=c, marker='o', lw=2, markersize=5, alpha=al, label=lab)
                    for lab, c, al in legend]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.962), frameon=False,
                   fontsize=8, labelcolor=_INK)
        fig.canvas.draw()                      # the transform is only valid after the draw
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3][:, :, ::-1].copy()
        self.base = cv2.resize(img, (width, height)) if img.shape[:2] != (height, width) else img
        self.tf, self.fig_h = ax.transData, img.shape[0]
        plt.close(fig)

    def _px(self, xz):
        X, Y = self.tf.transform(xz)
        return int(round(X)), int(round(self.fig_h - Y))

    def render(self, skeletons):
        """skeletons: [(xz (17,2), valid (17,), bgr, thickness)] drawn in order."""
        img = self.base.copy()
        for xz, valid, bgr, th in skeletons:
            fin = valid & np.isfinite(xz).all(axis=-1)
            for a, b in H36M_LIMBS:
                if fin[a] and fin[b]:
                    cv2.line(img, self._px(xz[a]), self._px(xz[b]), bgr, th, cv2.LINE_AA)
            for j in np.where(fin)[0]:
                cv2.circle(img, self._px(xz[j]), th + 1, bgr, -1, cv2.LINE_AA)
        return img
