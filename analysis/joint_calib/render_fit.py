#!/usr/bin/env python3
"""render_fit.py -- one rep of smpl_op_fit's SMPL body against its triangulated OpenPose, as a video.

Picks a subject (or takes --subject): with --pick-group A, the subject of that set whose hip error under the
kid-weight vertex regressor (vertex_regressor.py --out npz) is closest to the set's median -- a TYPICAL case.
The rep with the most usable frames is re-posed frame by frame with the subject's fitted body held fixed
(smpl_op_fit.fit_subject(fixed=...): shape and offsets from fit_v3, pose only), then drawn in two panels as
tools/render_mesh.py draws them:

    left    camera 1's view (the Korea videos are rendered stick figures, so a floor grid stands in for the frame)
    right   a virtual camera orbited --azimuth degrees about the vertical through the child

Both: the mesh (translucent), the triangulated BODY_25 skeleton (white), and at the hips and shoulders
    white   the triangulated OpenPose keypoint (the target)
    yellow  SMPL-24's joint
    green   the kid-weight regressor's prediction
plus the per-frame hip error (green to white, mm) in the corner.

    python analysis/joint_calib/render_fit.py --gt3d "/Volumes/Extreme SSD/MotorDevelopment/Korea/A/GT3D" \\
        --fits analysis/joint_calib/out/fit_v3 --vreg analysis/joint_calib/out/vreg_v3.npz --pick-group A
"""
import argparse
import glob
import os
import sys
import types

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline', 'tools'))
import smpl_op_fit as F  # noqa: E402
import vertex_regressor as VR  # noqa: E402
from utils.smpl_lite import MiniSMPL, rodrigues, SMPL24  # noqa: E402
from utils.openpose import BODY25_NAMES  # noqa: E402
from render_mesh import PanelRenderer, crop_box, panel_K, proj, apply, put  # noqa: E402

B = {n: i for i, n in enumerate(BODY25_NAMES)}
S24 = {n: i for i, n in enumerate(SMPL24)}
BONES25 = [('Neck', 'MidHip'), ('MidHip', 'RHip'), ('MidHip', 'LHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
           ('LHip', 'LKnee'), ('LKnee', 'LAnkle'), ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'),
           ('RElbow', 'RWrist'), ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
           ('Neck', 'Nose'), ('Nose', 'REye'), ('REye', 'REar'), ('Nose', 'LEye'), ('LEye', 'LEar'),
           ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'), ('RBigToe', 'RSmallToe'),
           ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'), ('LBigToe', 'LSmallToe')]
BONES25 = [(B[a], B[b]) for a, b in BONES25]
MARK = ('LHip', 'RHip', 'LShoulder', 'RShoulder')
MESH_RGB = (0.45, 0.62, 0.86)


def pick_subject(fits, vreg, group):
    """The subject of `group` whose hip |error| under the kid-weight regressor is closest to the group median."""
    z = np.load(vreg)
    cand = {k: z[f'{k}_vertices'] for k in VR.KEYPOINTS}
    rng = np.random.default_rng(0)
    rows = []
    for p in sorted(glob.glob(os.path.join(fits, f'{group}*.npz'))):
        s = VR.subject_meshes(p, cand, 120, rng)
        e = [VR.score(s, k, VR.predict(s, k, (z[f'{k}_weights_kw_w'], z[f'{k}_weights_kw_u']), []))[1]
             for k in ('LHip', 'RHip')]
        rows.append((s['name'], float(np.mean(e))))
    err = np.array([r[1] for r in rows])
    med = np.median(err)
    i = int(np.argmin(np.abs(err - med)))
    print(f'set {group}: {len(rows)} subjects, hip |error| (kid-weight regressor) median {med:.1f} mm, '
          f'IQR {np.percentile(err, 25):.1f}-{np.percentile(err, 75):.1f}; picked {rows[i][0]} ({rows[i][1]:.1f} mm)')
    return rows[i][0]


def load_rep(gt3d, subject, rep=None):
    reps = []
    for p in sorted(glob.glob(os.path.join(gt3d, subject, f'{subject}_*.npz'))):
        z = np.load(p)
        if not bool(z['usable']):
            continue
        X = z['X_body25_cam'][0].astype(float)
        X[~z['valid_body25']] = np.nan
        full = np.isfinite(X[:, [B[k] for k in F.CORE25 + ['Neck']]]).all((1, 2))
        keep = np.where(z['frame_usable'] & full)[0]
        reps.append((os.path.basename(p)[:-4], keep, X, z))
    if rep:
        reps = [r for r in reps if r[0] == rep]
    stub, keep, X, z = max(reps, key=lambda r: len(r[1]))
    return stub, keep, X[keep], z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--gt3d', required=True, help="the set's GT3D folder")
    ap.add_argument('--fits', default='analysis/joint_calib/out/fit_v3')
    ap.add_argument('--vreg', default='analysis/joint_calib/out/vreg_v3.npz')
    ap.add_argument('--subject', default=None)
    ap.add_argument('--pick-group', default='A')
    ap.add_argument('--rep', default=None, help='default: the rep with the most usable frames')
    ap.add_argument('--azimuth', type=float, default=90.0)
    ap.add_argument('--panel', default='640x720')
    ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--out-dir', default='analysis/joint_calib/out/render')
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    subject = args.subject or pick_subject(args.fits, args.vreg, args.pick_group)
    fit = np.load(os.path.join(args.fits, f'{subject}.npz'), allow_pickle=True)
    stub, keep, X, z = load_rep(args.gt3d, subject, args.rep)
    print(f'{stub}: {len(keep)} usable frames; re-posing with {subject}\'s fitted body')

    fargs = types.SimpleNamespace(shoulder_bone=str(fit['shoulder_bone']), two_bone=bool(fit['two_bone']),
                                  robust_mm=30.0, lam_beta=1.0, lam_torso=100.0, lam_pose=1.0, sigma_offset_mm=100.0,
                                  sigma_child_mm=50.0, iters=300)
    fixed = dict(betas=fit['betas_raw'], offsets={str(g): d / 1000.0 for g, d in zip(fit['groups'], fit['offsets_rest_mm'])})
    model, _ = F.fit_subject(X, bool(fit['kid']), fargs, print, fixed=fixed)

    # full posed mesh, SMPL-24 joints, regressor predictions (camera-1 frame, metres)
    shape = fit['shape']
    smpl = MiniSMPL(shape, full=True)
    with torch.no_grad():
        aa = torch.cat([model.root[:, None], model.pose * model.pose_mask], 1)
        R = rodrigues(aa)
        A = smpl.bone_transforms(R)
        V = smpl.vertices(R, A).numpy() + model.trans.detach().numpy()[:, None]
    J24 = np.einsum('jn,tnc->tjc', smpl.Jr.numpy(), V)
    vz = np.load(args.vreg)
    kw = float(shape[-1]) if len(shape) > 10 else 0.0
    pred = {k: np.einsum('tnc,n->tc', V[:, vz[f'{k}_vertices']], vz[f'{k}_weights_kw_w'] + kw * vz[f'{k}_weights_kw_u'])
            for k in MARK}
    joint = {k: J24[:, S24[f'{k[0]}_{k[1:]}']] for k in MARK}
    hip_err = np.mean([np.linalg.norm(pred[k] - X[:, B[k]], axis=-1) for k in ('LHip', 'RHip')], 0) * 1000.0
    hip_err_joint = np.mean([np.linalg.norm(joint[k] - X[:, B[k]], axis=-1) for k in ('LHip', 'RHip')], 0) * 1000.0
    print(f'hip |error| over the rep: regressor median {np.nanmedian(hip_err):.1f} mm, '
          f'SMPL-24 joint {np.nanmedian(hip_err_joint):.1f} mm')

    # views: camera 1, and an orbit about the floor's vertical through the child
    K = z['K_1080'][0]
    up = z['R_world_to_floor'][2]                                  # floor +z in camera-1 coordinates
    centre = np.nanmedian(X[:, B['MidHip']], 0)
    a = np.radians(args.azimuth)
    u = up / np.linalg.norm(up)
    Ku = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    Rot = np.eye(3) + np.sin(a) * Ku + (1 - np.cos(a)) * Ku @ Ku
    M_side = np.eye(4)
    M_side[:3, :3], M_side[:3, 3] = Rot, centre - Rot @ centre     # rotate the world about the vertical axis
    views = [('camera 1', np.eye(4)), (f'orbit {args.azimuth:.0f} deg', M_side)]
    pw, ph = (int(v) for v in args.panel.split('x'))
    boxes = []
    for _, Mv in views:
        pts = proj(K, apply(Mv, V[:, ::25].reshape(-1, 3)))
        boxes.append(crop_box(pts, (pw, ph), pad=0.15))
    rend = PanelRenderer((pw, ph))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f'{stub}_fit_az{args.azimuth:.0f}.mp4')
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (2 * pw, ph))
    faces = smpl.faces
    for t in range(len(X)):
        panels = []
        for (label, Mv), box in zip(views, boxes):
            Kp = panel_K(K, box)
            bg = np.tile(np.linspace(52, 28, ph, dtype=np.float32)[:, None, None], (1, pw, 3)).astype(np.uint8)
            rgb, alpha = rend.render(apply(Mv, V[t]), faces, Kp, MESH_RGB)
            img = (bg * (1 - 0.7 * alpha) + rgb[..., ::-1] * 0.7 * alpha).astype(np.uint8)
            p = proj(Kp, apply(Mv, X[t]))
            for i, j in BONES25:
                if np.isfinite(p[[i, j]]).all():
                    cv2.line(img, tuple(np.round(p[i]).astype(int)), tuple(np.round(p[j]).astype(int)),
                             (255, 255, 255), 2, cv2.LINE_AA)
            for k in MARK:
                for P, col, r in ((X[t, B[k]], (255, 255, 255), 6), (joint[k][t], (40, 220, 250), 5),
                                  (pred[k][t], (90, 220, 90), 5)):
                    q = proj(Kp, apply(Mv, P[None]))[0]
                    if np.isfinite(q).all():
                        cv2.circle(img, tuple(np.round(q).astype(int)), r, col, -1, cv2.LINE_AA)
                        cv2.circle(img, tuple(np.round(q).astype(int)), r, (0, 0, 0), 1, cv2.LINE_AA)
            put(img, f'{stub}  {label}', (10, 22))
            panels.append(img)
        frame = np.hstack(panels)
        put(frame, f'frame {keep[t]}  hip error: regressor {hip_err[t]:.0f} mm, SMPL-24 joint {hip_err_joint[t]:.0f} mm',
            (10, ph - 14), 0.5)
        put(frame, 'white: OpenPose (target)   yellow: SMPL-24 joint   green: kid-weight regressor', (10, 46), 0.45)
        vw.write(frame)
    vw.release()
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
