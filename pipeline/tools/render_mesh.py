#!/usr/bin/env python3
"""render_mesh.py -- the placed SMPL mesh of several configurations, side by side, as a video.

One row per configuration (step_4b's {config}/{cam}_pnp.npz), two columns:

    left    the original camera: the mesh over the (undistorted) video frame, or over the lab
            floor grid when there is no video (Korea)
    right   a virtual camera orbited --azimuth degrees about the vertical axis through the subject,
            same intrinsics and distance: what the original camera cannot see -- depth

Both views carry the ground-truth skeleton (mocap, or the triangulated target -- this camera's
leave-one-out version when the file has one, as step_8 scores it) and the per-frame MPJPE of that configuration against it.  The mesh is the one each configuration
optimised: its own shape (betas, kid weight) through its own rotations, rigidly fitted to its
placed SMPL-24 joints (the fit is exact up to float error -- the residual is printed).
--smooth names configurations drawn and scored from their RTS-smoothed skeleton instead
(the 'pnp' placement's kps_*_smooth; refined configurations store the same skeleton in both).
--com adds each mesh's centre of mass (smpl_lite.body_points), its drop to the floor, and the
base of support -- the hull of the grounded feet's sole points, grounded as step_4b's contact
detector decided (foot_grounded, from any of the configurations that stored it).  The drop
point is green inside the support, amber within --com-margin of it, red beyond.

    reads   {OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{config}/{cam}_pnp.npz
            {OUT_DIR}/{user}/{action}/Analysis/keypoints/{detector}/{cam}_2d.npz   frame index
            {OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz (or openpose_tri_h36m.npz)
            {TRIAL_DIR}/{user}/{action}/{cam}.mp4
    writes  {OUT_DIR}/{user}/{action}/Analysis/diagnostics/{detector}/{cam}_mesh_{configs}.mp4

Needs pyrender + trimesh (motEnv has them) and the SMPL pickle smpl_lite finds.

    python3 tools/render_mesh.py --user P03 --action P03_CMJM_01 --cam 07 --configs C0,C5,C8 --smooth C0
    python3 tools/render_mesh.py --user B010 --action B010_GMS_1_1 --cam 1 --gt tri --smooth C0
    python3 tools/render_mesh.py --user B010 --action B010_GMS_1_1 --cam 1 --gt tri --configs C8,C10 --com
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import TRIAL_DIR, analysis_dir, twod_path, pnp_path, mocap_path, tri_target_path  # noqa: E402
from utils.smpl_lite import H36M, MiniSMPL, body_points, kabsch  # noqa: E402

if sys.platform.startswith('linux') and 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'                      # headless (Colab / Jetson); macOS uses pyglet
import pyrender  # noqa: E402
import trimesh  # noqa: E402

# one colour per row (RGB 0-1), in --configs order
ROW_COLOURS = [(0.86, 0.38, 0.33), (0.92, 0.68, 0.24), (0.31, 0.63, 0.86), (0.45, 0.75, 0.45), (0.70, 0.50, 0.85)]
BONES = [('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('Hip', 'LHip'), ('LHip', 'LKnee'),
         ('LKnee', 'LAnkle'), ('Hip', 'Spine'), ('Spine', 'Thorax'), ('Thorax', 'Nose'), ('Nose', 'Head'),
         ('Thorax', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
         ('Thorax', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist')]
BONES = [(H36M.index(a), H36M.index(b)) for a, b in BONES]
CV_TO_GL = np.diag([1.0, -1.0, -1.0])                            # OpenCV camera axes -> OpenGL's


def placed_vertices(npz, which='placed', chunk=64):
    """(T,6890,3) camera-space metres: the configuration's mesh, rigidly fitted per frame to its
    SMPL-24 joints (which: 'placed' or 'smooth'), and the same fit applied to body_points
    (T,9,3: COM, then the 8 sole points).  NaN rows where placement failed."""
    smpl = MiniSMPL(npz['shape'], full=True)
    moments = torch.tensor(body_points(npz['shape']), dtype=torch.float64)
    rot = torch.tensor(npz['rotmats'], dtype=torch.float64)
    scale = float(npz['scale_factor'])
    S = npz[f'kps_SMPL24_{which}'].astype(np.float64)
    ok = npz['pnp_ok'] & ~np.isnan(S).any((1, 2))
    T = len(rot)
    V = np.full((T, smpl.n, 3), np.nan, np.float32)
    B = np.full((T, len(moments), 3), np.nan)
    worst = 0.0
    for a in range(0, T, chunk):
        with torch.no_grad():
            A = smpl.bone_transforms(rot[a:a + chunk])
            v = smpl.vertices(rot[a:a + chunk], A)
            j = torch.einsum('jn,tnc->tjc', smpl.Jr, v)
            p = torch.einsum('pkj,tkij->tpi', moments, A[:, :, :3, :])
        v, j, p = v.numpy() * scale, j.numpy() * scale, p.numpy() * scale
        for k in range(len(v)):
            i = a + k
            if not ok[i]:
                continue
            R, t = kabsch(j[k], S[i])
            worst = max(worst, float(np.abs(j[k] @ R.T + t - S[i]).max()))
            V[i] = v[k] @ R.T + t
            B[i] = p[k] @ R.T + t
    return V, B, smpl.faces, ok, worst


def support_geometry(B_cam, grounded, L_ext):
    """Per frame: the COM's drop point on the floor and the base-of-support hull (lab floor points,
    metres, z = 0) of the grounded feet, and the COM's signed distance OUTSIDE the hull (m, < 0
    inside; NaN when no foot is grounded).  B_cam (T,9,3) camera metres; grounded (T,2) R, L."""
    R, t = L_ext[:3, :3], L_ext[:3, 3] / 1000.0
    lab = (B_cam - t) @ R
    T = len(lab)
    drop = np.concatenate([lab[:, 0, :2], np.zeros((T, 1))], 1)
    hulls, excess = [None] * T, np.full(T, np.nan)
    for i in range(T):
        on = np.repeat(grounded[i], 4)
        if not on.any() or np.isnan(lab[i]).any():
            continue
        xy = lab[i, 1:, :2][on].astype(np.float32)
        hull = cv2.convexHull(xy)[:, 0]
        hulls[i] = np.concatenate([hull.astype(np.float64), np.zeros((len(hull), 1))], 1)
        if len(hull) >= 3:
            excess[i] = -cv2.pointPolygonTest(hull, tuple(map(float, drop[i, :2])), True)
        else:
            excess[i] = float(np.linalg.norm(drop[i, :2] - hull.mean(0)))
    return lab[:, 0], drop, hulls, excess


def lab_to_cam(X_lab_m, L_ext):
    return X_lab_m @ L_ext[:3, :3].T + L_ext[:3, 3] / 1000.0


def draw_support(img, Kv, M, L_ext, com_lab, drop, hull, excess, margin_m):
    """COM dot, dashed drop to the floor, the support hull, the drop point coloured by excess."""
    if hull is not None:
        px = proj(Kv, apply(M, lab_to_cam(hull, L_ext)))
        cv2.polylines(img, [np.round(px).astype(np.int32)], True, (235, 235, 235), 2, cv2.LINE_AA)
    if np.isnan(com_lab).any():
        return
    c, f = (proj(Kv, apply(M, lab_to_cam(x[None], L_ext)))[0] for x in (com_lab, drop))
    n = max(2, int(np.linalg.norm(c - f) // 8))
    for k in range(0, n, 2):
        p0, p1 = c + (f - c) * k / n, c + (f - c) * (k + 1) / n
        cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    col = ((200, 200, 200) if not np.isfinite(excess) else (80, 200, 80) if excess <= 0 else
           (40, 170, 240) if excess <= margin_m else (60, 60, 230))
    cv2.circle(img, tuple(np.round(f).astype(int)), 6, col, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(c).astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(c).astype(int)), 6, (0, 0, 0), 1, cv2.LINE_AA)


def side_transform(L_ext, centre_lab_mm, azimuth_deg):
    """4x4 (metres) taking original-camera coordinates to a virtual camera orbited azimuth_deg
    about the lab vertical (Z) through centre_lab_mm."""
    Lm = L_ext.copy()
    Lm[:3, 3] /= 1000.0
    c = centre_lab_mm / 1000.0
    a = np.radians(-azimuth_deg)                                 # turning the world -a == orbiting the camera +a
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    orbit = np.eye(4)
    orbit[:3, :3], orbit[:3, 3] = Rz, c - Rz @ c
    return Lm @ orbit @ np.linalg.inv(Lm)


def apply(M, X):
    return X @ M[:3, :3].T + M[:3, 3]


def proj(K, X):
    """Pinhole (no distortion) projection, camera-space (...,3) -> (...,2) px; NaN behind the camera."""
    z = np.where(X[..., 2] > 1e-3, X[..., 2], np.nan)
    return np.stack([K[0, 0] * X[..., 0] / z + K[0, 2], K[1, 1] * X[..., 1] / z + K[1, 2]], -1)


def crop_box(points_px, panel_wh, pad=0.12):
    """(x0, y0, s): the square-pixel crop covering points_px with padding, at the panel's aspect;
    panel pixel = s * (image pixel - (x0, y0))."""
    p = points_px.reshape(-1, 2)
    p = p[~np.isnan(p).any(1)]
    (x0, y0), (x1, y1) = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    aspect = panel_wh[0] / panel_wh[1]
    w, h = max(w, h * aspect), max(h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w / 2, cy - h / 2, panel_wh[0] / w


def panel_K(K, box):
    x0, y0, s = box
    Kp = K.copy()
    Kp[0, 0] *= s
    Kp[1, 1] *= s
    Kp[0, 2] = s * (K[0, 2] - x0)
    Kp[1, 2] = s * (K[1, 2] - y0)
    return Kp


def draw_skeleton(img, pts, ok, colour=(255, 255, 255), thick=2):
    for a, b in BONES:
        if ok[a] and ok[b] and not np.isnan(pts[[a, b]]).any():
            pa, pb = tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int))
            cv2.line(img, pa, pb, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.line(img, pa, pb, colour, thick, cv2.LINE_AA)


def floor_grid(K, M, centre_lab_mm, L_ext, wh, half_m=2.5, step_m=0.5):
    """Background for a virtual view: dark gradient + the lab floor (z=0) as a grid."""
    w, h = wh
    img = np.tile(np.linspace(52, 28, h, dtype=np.float32)[:, None, None], (1, w, 3)).astype(np.uint8)
    Lm = L_ext.copy()
    Lm[:3, 3] /= 1000.0
    c = centre_lab_mm / 1000.0
    ticks = np.arange(-half_m, half_m + 1e-9, step_m)
    for along in (0, 1):
        for v in ticks:
            u = np.linspace(-half_m, half_m, 60)
            P = np.zeros((60, 3))
            P[:, along], P[:, 1 - along] = c[along] + u, c[1 - along] + v
            X = apply(M, apply(Lm, P))
            px = proj(K, X)
            good = ~np.isnan(px).any(1)
            if good.sum() > 1:
                cv2.polylines(img, [np.round(px[good]).astype(np.int32)], False, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def error_strip(err, a, b, wh, ymax, colour):
    """(image, x-of-row function): per-frame MPJPE over rows a..b as a filled curve; the playhead is
    drawn per frame."""
    w, h = wh
    img = np.full((h, w, 3), 22, np.uint8)
    for mm in range(200, int(ymax), 200):                        # gridline every 200 mm
        y = int(h - 4 - (h - 8) * mm / ymax)
        cv2.line(img, (0, y), (w, y), (55, 55, 55), 1)
    cv2.putText(img, f'MPJPE over the trial, 0-{ymax:.0f} mm', (6, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (140, 140, 140), 1, cv2.LINE_AA)
    xs = lambda i: int(round((i - a) * (w - 1) / max(b - 1 - a, 1)))
    bgr = tuple(int(255 * c) for c in colour[::-1])
    seg = []
    for i in range(a, b):
        e = err[i]
        if np.isfinite(e):
            seg.append((xs(i), int(h - 4 - (h - 8) * min(e, ymax) / ymax)))
        elif seg:
            cv2.polylines(img, [np.array(seg, np.int32)], False, bgr, 2, cv2.LINE_AA)
            seg = []
    if seg:
        cv2.polylines(img, [np.array(seg, np.int32)], False, bgr, 2, cv2.LINE_AA)
    return img, xs


def put(img, text, org, scale=0.55, colour=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


class PanelRenderer:
    """pyrender scene for one panel size; render(verts, faces, K, colour) -> (rgb uint8, alpha 0-1)."""

    def __init__(self, wh):
        self.r = pyrender.OffscreenRenderer(*wh)
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
        self.scene.add(pyrender.DirectionalLight(intensity=2.5), pose=np.eye(4))      # from the viewer
        top = np.eye(4)
        top[:3, :3] = cv2.Rodrigues(np.array([-np.pi / 2.5, 0, 0]))[0]
        self.scene.add(pyrender.DirectionalLight(intensity=1.5), pose=top)            # from above
        self.cam_node = None
        self.mesh_node = None

    def render(self, verts_cv, faces, K, colour):
        if self.cam_node is not None:
            self.scene.remove_node(self.cam_node)
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
            self.mesh_node = None
        self.cam_node = self.scene.add(pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                                                 znear=0.05, zfar=100.0), pose=np.eye(4))
        if verts_cv is not None:
            tm = trimesh.Trimesh(verts_cv @ CV_TO_GL, faces, process=False)
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*colour, 1.0), metallicFactor=0.0,
                                                     roughnessFactor=0.65)
            self.mesh_node = self.scene.add(pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True))
        rgba, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return rgba[..., :3], rgba[..., 3:].astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--user', required=True)
    ap.add_argument('--action', required=True)
    ap.add_argument('--cam', required=True)
    ap.add_argument('--configs', default='C0,C5,C8', help='comma-separated step_4b configurations, one row each')
    ap.add_argument('--detector', default='openpose')
    ap.add_argument('--gt', choices=('mocap', 'tri'), default='mocap', help='skeleton drawn and scored against')
    ap.add_argument('--all-cameras', action='store_true',
                    help='score against the all-camera triangulation even when a leave-one-out one exists')
    ap.add_argument('--smooth', default='', help='configurations drawn from their RTS-smoothed skeleton, e.g. C0')
    ap.add_argument('--com', action='store_true', help='draw the centre of mass, its floor drop and the base of support')
    ap.add_argument('--com-margin', type=float, default=0.03, help='metres outside the support drawn amber, not red')
    ap.add_argument('--azimuth', type=float, default=90.0, help='orbit of the right-hand view, degrees about vertical')
    ap.add_argument('--panel', default='480x600', help='panel size WxH')
    ap.add_argument('--mesh-alpha', type=float, default=0.8, help='mesh opacity over the video')
    ap.add_argument('--fps', type=float, default=None, help='output frame rate (default: the analysis rate)')
    ap.add_argument('--frames', default=None, help='a:b row range, for a quick look')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    pw, ph = (int(x) for x in args.panel.split('x'))
    configs = [c.strip() for c in args.configs.split(',') if c.strip()]
    smooth = {c.strip() for c in args.smooth.split(',') if c.strip()}

    tw = np.load(twod_path(args.user, args.action, args.detector, args.cam))
    sfi = tw['source_frame_idx'].astype(int)
    rows, grounded = [], None
    for cid in configs:
        d = np.load(pnp_path(args.user, args.action, args.detector, args.cam, cid))
        which = 'smooth' if cid in smooth else 'placed'
        V, B, faces, ok, worst = placed_vertices(d, which)
        if grounded is None and 'foot_grounded' in d.files:
            grounded = d['foot_grounded'].astype(bool)
        print(f'{cid}: mesh fitted to {which} joints, worst residual {1000 * worst:.2f} mm, {ok.sum()}/{len(ok)} frames')
        cfg = json.loads(str(d['config_json'])) if 'config_json' in d.files else {}
        note2 = (f'grounded ankle held {1000 * float(d["floor_contact_m"]):.0f} mm above the floor'
                 if cfg.get('w_floor_contact', 0) > 0 and 'floor_contact_m' in d.files else
                 'RTS-smoothed root' if which == 'smooth' else '')
        rows.append(dict(cid=cid, V=V, B=B, ok=ok, H=d[f'kps_H36M_{which}'].astype(np.float64), note=cfg.get('note', ''),
                         note2=note2))
    K, L_ext, dist = d['cam_K'], d['cam_L_ext'], d['cam_dist_cv']
    fps = float(d['fps'])
    T = min(len(sfi), *(len(r['V']) for r in rows))

    # ground truth into the original camera, metres (step_8's convention)
    g = np.load(mocap_path(args.user, args.action) if args.gt == 'mocap' else tri_target_path(args.user, args.action))
    gt_lab = g['kps3d'].astype(np.float64)
    gt_name = args.gt
    if 'kps3d_loo' in g.files and not args.all_cameras:          # triangulated WITHOUT this camera, as step_8
        gt_lab = g['kps3d_loo'][[str(c) for c in g['cameras']].index(args.cam)].astype(np.float64)
        gt_name = 'leave-one-out tri'
    elif args.gt == 'tri':
        gt_name = 'all-camera tri'
    gt_cam = np.full((T, 17, 3), np.nan)
    inside = sfi[:T] < len(gt_lab)
    gt_cam[inside] = (gt_lab[sfi[:T][inside]] @ L_ext[:3, :3].T + L_ext[:3, 3]) / 1000.0
    gt_ok = g['valid_joint_mask'] if 'valid_joint_mask' in g.files else np.ones(17, bool)
    centre = np.nanmean(gt_lab[:, 0], 0)                         # orbit about the mean GT pelvis (lab mm)
    for r in rows:
        e = np.linalg.norm(r['H'][:T] - gt_cam, axis=-1)[:, gt_ok] * 1000.0
        n = (~np.isnan(e)).sum(1)
        r['err'] = np.where(n > 0, np.nansum(e, 1) / np.maximum(n, 1), np.nan)
        r['mean'] = float(np.nanmean(e))
        print(f'{r["cid"]}: MPJPE vs {gt_name} {r["mean"]:.1f} mm')

    if args.com:
        if grounded is None:
            raise SystemExit('--com: none of these configurations stored foot_grounded -- rerun one with step_4b')
        for r in rows:
            r['com'], r['drop'], r['hulls'], r['excess'] = support_geometry(r['B'][:T], grounded[:T], L_ext)
            ex = r['excess'][np.isfinite(r['excess'])]
            r['com_note'] = (f'COM out >{1000 * args.com_margin:.0f} mm: {(ex > args.com_margin).sum()}/{len(ex)} stance, '
                             f'median {1000 * np.median(ex):.0f} mm' if len(ex) else 'no grounded frames')
            print(f'{r["cid"]}: {r["com_note"]}')

    a, b = (int(x) for x in args.frames.split(':')) if args.frames else (0, T)
    strip_h = 56
    ymax = 100 * np.ceil(np.nanpercentile(np.concatenate([r['err'][a:b] for r in rows]), 99) / 100)
    for ri, r in enumerate(rows):
        r['strip'], xs = error_strip(r['err'], a, b, (pw, strip_h), ymax, ROW_COLOURS[ri % len(ROW_COLOURS)])

    # the two views: transforms from original-camera coordinates, crops from everything drawn
    # (the side view's crop fills the panel above its error strip)
    views = [np.eye(4), side_transform(L_ext, centre, args.azimuth)]
    boxes, Ks = [], []
    for vi, M in enumerate(views):
        pts = [proj(K, apply(M, gt_cam[:, gt_ok]))]
        pts += [proj(K, apply(M, r['V'][:T:5, ::20].astype(np.float64))) for r in rows]
        box = crop_box(np.concatenate([p.reshape(-1, 2) for p in pts]), (pw, ph - (strip_h if vi else 0)))
        boxes.append(box)
        Ks.append(panel_K(K, box))
    side_bg = floor_grid(Ks[1], views[1], centre, L_ext, (pw, ph))
    video = os.path.join(TRIAL_DIR, args.user, args.action, f'{args.cam}.mp4')
    cam_grid = None if os.path.exists(video) else floor_grid(Ks[0], views[0], centre, L_ext, (pw, ph))
    to_panel = [np.float32([[b[2], 0, -b[2] * b[0]], [0, b[2], -b[2] * b[1]]]) for b in boxes]

    title_h = 58
    W, Hh = 2 * pw + 6, title_h + len(rows) * (ph + 6)
    out = args.out or os.path.join(analysis_dir(args.user, args.action), 'diagnostics', args.detector,
                                   f'{args.cam}_mesh_{"_".join(configs)}{"_gtall" if args.all_cameras else ""}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps or fps, (W, Hh))
    cap = cv2.VideoCapture(video) if cam_grid is None else None
    rend = PanelRenderer((pw, ph))
    vid_i, frame = -1, None
    for i in range(a, b):
        if cap is None:
            cam_bg = cam_grid
        else:
            while vid_i < sfi[i]:
                okr, frame = cap.read()
                vid_i += 1
                if not okr:
                    raise SystemExit(f'video ended at frame {vid_i}, row {i} needs {sfi[i]}')
            und = cv2.undistort(frame, K, dist)       # onto the same pinhole the mesh is drawn with
            cam_bg = cv2.warpAffine(und, to_panel[0], (pw, ph), flags=cv2.INTER_AREA, borderValue=(30, 30, 30))
        canvas = np.full((Hh, W, 3), 18, np.uint8)
        put(canvas, f'{args.user} {args.action}  |  {args.detector} 2D  |  cam {args.cam}  |  '
                    f'white: {gt_name} skeleton', (10, 22), 0.55)
        put(canvas, f'{i / fps:5.2f} s', (W - 80, 22), 0.55)
        put(canvas, f'camera {args.cam} view' + (' (no video)' if cap is None else ''), (pw // 2 - 70, 49), 0.6,
            (190, 190, 190))
        put(canvas, f'orbited {args.azimuth:.0f} deg about vertical', (pw + 6 + pw // 2 - 125, 49), 0.6,
            (190, 190, 190))
        for ri, r in enumerate(rows):
            y0 = title_h + ri * (ph + 6)
            col = ROW_COLOURS[ri % len(ROW_COLOURS)]
            for vi, M in enumerate(views):
                bg = cam_bg.copy() if vi == 0 else side_bg.copy()
                verts = apply(M, r['V'][i].astype(np.float64)) if r['ok'][i] else None
                rgb, al = rend.render(verts, faces, Ks[vi], col)
                al = al * (args.mesh_alpha if vi == 0 else 1.0)
                img = (bg * (1 - al) + rgb[..., ::-1] * al).astype(np.uint8)
                if not np.isnan(gt_cam[i]).all():
                    draw_skeleton(img, proj(Ks[vi], apply(M, gt_cam[i])), gt_ok)
                if args.com and r['ok'][i]:
                    draw_support(img, Ks[vi], M, L_ext, r['com'][i], r['drop'][i], r['hulls'][i], r['excess'][i],
                                 args.com_margin)
                if vi == 0:
                    put(img, r['cid'], (10, 26), 0.8, tuple(int(255 * c) for c in col[::-1]), 2)
                    put(img, r['note'][:56], (10, 50), 0.45)
                    if r['note2']:
                        put(img, r['note2'], (10, 70), 0.45, (190, 190, 190))
                    put(img, f'trial MPJPE {r["mean"]:.0f} mm', (10, ph - 14), 0.55)
                    if args.com:
                        put(img, r['com_note'], (10, ph - 36), 0.45)
                else:
                    e = r['err'][i]
                    put(img, f'MPJPE {e:.0f} mm' if np.isfinite(e) else 'MPJPE --', (10, 26), 0.7)
                    if args.com:
                        x = r['excess'][i]
                        put(img, 'COM: no foot grounded' if not np.isfinite(x) else
                            f'COM {1000 * x:.0f} mm outside support' if x > 0 else f'COM inside support ({-1000 * x:.0f} mm in)',
                            (10, 50), 0.5)
                    strip = r['strip'].copy()
                    cv2.line(strip, (xs(i), 0), (xs(i), strip_h), (255, 255, 255), 1)
                    img[ph - strip_h:] = strip
                x0 = vi * (pw + 6)
                canvas[y0:y0 + ph, x0:x0 + pw] = img
        writer.write(canvas)
        if (i - a) % 50 == 0:
            print(f'  row {i}/{b}', flush=True)
    writer.release()
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
