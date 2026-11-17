#!/usr/bin/env python3
"""render_pseudo_gt.py -- a saved pseudo_gt.py fit drawn over its triangulated OpenPose, as a video + contact sheet.

Panels as analysis/joint_calib/render_fit.py draws them (camera 1, then a virtual camera orbited about the vertical
through the child for each --azimuth), each with

    the mesh (translucent), the triangulated BODY_25 skeleton (white), and
    red rings   target keypoints dropped by GT3D's gates on this frame (the fit has no data there)

The corner reports the frame's median keypoint residual and how many keypoints it had, and which limbs the
keypoints do not determine on that frame (obs_swing; the pose there is the fit's guess).  Look for what the numbers
cannot show: forearm / thigh twist, hands, the feet's direction, a pelvis turned the wrong way, flicker.

    python3 training/render_pseudo_gt.py --fit <dir>/B011_GMS_3_2_lam0.1.npz --azimuth 90,180
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'training'))
sys.path.insert(0, os.path.join(_ROOT, 'analysis', 'joint_calib'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline', 'tools'))
import pseudo_gt as PG  # noqa: E402
from render_fit import BONES25, MESH_RGB  # noqa: E402
from utils.smpl_lite import MiniSMPL, rodrigues  # noqa: E402
from render_mesh import PanelRenderer, crop_box, panel_K, proj, apply, put  # noqa: E402

S24 = PG.F.S24
UNSEEN = {'L arm': [S24['L_Shoulder'], S24['L_Elbow']], 'R arm': [S24['R_Shoulder'], S24['R_Elbow']],
          'L leg': [S24['L_Hip'], S24['L_Knee']], 'R leg': [S24['R_Hip'], S24['R_Knee']],
          'head': [S24['Neck'], S24['Head']], 'trunk': [S24['Pelvis'], S24['Spine3']]}
FFMPEG = os.environ.get('FFMPEG', '/Users/peterkearney/Documents/ffmpeg/bin/ffmpeg')


def orbit(up, centre, deg):
    a = np.radians(deg)
    u = up / np.linalg.norm(up)
    Ku = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    Rot = np.eye(3) + np.sin(a) * Ku + (1 - np.cos(a)) * Ku @ Ku
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rot, centre - Rot @ centre
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--fit', required=True, help="pseudo_gt.py output ({rep}_lam{x}.npz)")
    ap.add_argument('--root', default=PG.KORE_ROOT)
    ap.add_argument('--azimuth', default='90', help='comma list of orbit angles (deg) for the extra panels')
    ap.add_argument('--panel', default='480x600')
    ap.add_argument('--fps', type=float, default=15.0)
    ap.add_argument('--sheet', type=int, default=8, help='frames in the contact sheet PNG (0: none)')
    ap.add_argument('--out-dir', default=None, help="default: the fit's folder")
    args = ap.parse_args()

    f = np.load(args.fit, allow_pickle=True)
    stub = os.path.basename(args.fit).rsplit('_lam', 1)[0]
    tag = os.path.basename(args.fit)[:-4]
    X, z = PG.load_rep(os.path.join(args.root, stub[0], 'GT3D'), stub)
    Xall = z['X_body25_cam'][0].astype(float)
    Xall[~z['valid_body25']] = np.nan                       # what GT3D triangulated, before the bone gate
    smpl = MiniSMPL(f['shape'], full=True)
    with torch.no_grad():
        R = rodrigues(torch.cat([torch.tensor(f['root'])[:, None], torch.tensor(f['pose'])], 1))
        V = smpl.vertices(R, smpl.bone_transforms(R)).numpy() + f['trans'][:, None]

    K = z['K_1080'][0]
    up = z['R_world_to_floor'][2]
    centre = np.nanmedian(V[:, :, :].mean(1), 0)
    views = [('camera 1', np.eye(4))] + [(f'orbit {float(a):.0f} deg', orbit(up, centre, float(a)))
                                         for a in args.azimuth.split(',')]
    pw, ph = (int(v) for v in args.panel.split('x'))
    boxes = [crop_box(proj(K, apply(Mv, V[:, ::25].reshape(-1, 3))), (pw, ph), pad=0.15) for _, Mv in views]
    rend = PanelRenderer((pw, ph))
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.fit))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{tag}.mp4')
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (len(views) * pw, ph))
    T = len(X)
    pick = set(np.linspace(0, T - 1, args.sheet).round().astype(int)) if args.sheet else set()
    sheet = []
    for t in range(T):
        panels = []
        for (label, Mv), box in zip(views, boxes):
            Kp = panel_K(K, box)
            bg = np.tile(np.linspace(52, 28, ph, dtype=np.float32)[:, None, None], (1, pw, 3)).astype(np.uint8)
            rgb, alpha = rend.render(apply(Mv, V[t]), smpl.faces, Kp, MESH_RGB)
            img = (bg * (1 - 0.7 * alpha) + rgb[..., ::-1] * 0.7 * alpha).astype(np.uint8)
            p = proj(Kp, apply(Mv, X[t]))
            for i, j in BONES25:
                if np.isfinite(p[[i, j]]).all():
                    cv2.line(img, tuple(np.round(p[i]).astype(int)), tuple(np.round(p[j]).astype(int)),
                             (255, 255, 255), 2, cv2.LINE_AA)
            gone = np.isfinite(Xall[t]).all(-1) & ~np.isfinite(X[t]).all(-1)
            q = proj(Kp, apply(Mv, Xall[t]))
            for i in np.where(gone)[0]:
                cv2.circle(img, tuple(np.round(q[i]).astype(int)), 7, (40, 40, 230), 2, cv2.LINE_AA)
            put(img, label, (10, 22))
            panels.append(img)
        frame = np.hstack(panels)
        put(frame, f'{tag}  frame {t}/{T}  residual {f["frame_residual_mm"][t]:.0f} mm, '
                   f'{int(f["n_keypoints"][t])} keypoints', (10, ph - 14), 0.5)
        if 'obs_swing' in f.files:
            guess = [g for g, js in UNSEEN.items() if not f['obs_swing'][t, js].all()]
            if guess:
                put(frame, 'NOT OBSERVED (pose is a guess): ' + ', '.join(guess), (10, ph - 36), 0.5)
        vw.write(frame)
        if t in pick:
            sheet.append(frame)
    vw.release()
    if os.path.exists(FFMPEG):                       # OpenCV writes MPEG-4 Part 2, which VS Code cannot play
        tmp = out[:-4] + '_mp4v.mp4'
        os.replace(out, tmp)
        r = os.system(f'"{FFMPEG}" -loglevel error -y -i "{tmp}" -c:v libx264 -pix_fmt yuv420p -crf 20 "{out}"')
        if r == 0:
            os.remove(tmp)
        else:
            os.replace(tmp, out)
    print(f'wrote {out}')
    if sheet:
        rows = [np.hstack(sheet[i:i + 2]) if i + 1 < len(sheet) else
                np.hstack([sheet[i], np.zeros_like(sheet[i])]) for i in range(0, len(sheet), 2)]
        png = os.path.join(out_dir, f'{tag}_sheet.png')
        cv2.imwrite(png, cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        print(f'wrote {png}')


if __name__ == '__main__':
    main()
