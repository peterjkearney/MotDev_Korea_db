#!/usr/bin/env python3
"""pseudo_gt.py -- SMPL pose for EVERY frame of a Korea rep, for fine-tuning MotionBERT-Mesh (step 2).

smpl_op_fit.py fitted each child's body (shape + hip/shoulder/neck/mid-hip offsets, out/fit_v3) to 300 frames
pooled across reps.  Here that body is held fixed and only the pose is fitted, frame by frame through a whole rep,
to the all-camera triangulated BODY_25 (GT3D X_body25_cam[0]: camera 1's frame, metres at the cohort stature):

    data     the same keypoints, weights and robust loss as smpl_op_fit; a joint is used where GT3D kept it
             (valid_body25) and, for points behind an H36M joint, where the per-joint bone gate kept that joint
             (valid_h36m_gated), so swapped / stretched joints drop out joint by joint, not frame by frame
    smooth   --lam-smooth x mean squared 2nd difference (mm/frame^2) of the SMPL-24 joints, and --lam-smooth-rot x
             that of the axis-angles (rad^2, scaled by 1e4) -- MotionBERT's loss weights joint velocity heavily,
             so frame-independent jitter would be taught as motion; it also carries frames with few keypoints

Frames are never dropped; each gets n_keypoints and its median residual so the dataset step can gate them, and
obs_swing / obs_full (T,24): which SMPL-24 rotations the frame's keypoints actually determine (OBSERVE below) --
elsewhere the pose is the prior's and the smoothing's guess, and the fine-tuning loss must not be taught it.
Scored against the leave-one-camera-out targets (X_h36m_cam_loo, the triangulation without camera c), next to
the all-camera triangulation itself: the fit should sit as close to them as the triangulation does.

Prototype (a few reps, a sweep of smoothing values, scored and saved side by side):

    python3 training/pseudo_gt.py --reps B010_GMS_1_7,B011_GMS_3_2 --lam-smooth 0,0.1,0.3 --out-dir <dir>

Full run (every usable rep of the train and dev children that have a fit_v3 body -> {set}/PseudoGT/{child}/{rep}.npz
beside GT3D; done reps skipped, files written by rename so a disconnect cannot leave a half file; one JSON line per
rep in {set}/PseudoGT/_log/; --shard k/n for parallel processes):

    python3 training/pseudo_gt.py --split train,dev --dry-run
    python3 training/pseudo_gt.py --split train,dev --shard 0/4 --threads 2      # x4, k = 0..3

Colab needs: this repo; MB_DIR (SMPL_NEUTRAL.pkl in data/mesh, and smpl_kid_template.npy there or via
SMPL_KID_TEMPLATE / MODELS_DIR, as the pipeline already finds it); the GT3D folders under --root
(.../Korea/{B,C}/GT3D); and analysis/joint_calib/out/fit_v3 (git-ignored: upload it, pass --fits).
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'analysis', 'joint_calib'))
sys.path.insert(0, os.path.join(_ROOT, 'pipeline'))
import smpl_op_fit as F  # noqa: E402
from utils.openpose import BODY25_NAMES, H36M_FROM_BODY25, body25_to_h36m_3d  # noqa: E402
from utils.smpl_lite import rodrigues  # noqa: E402

B = {n: i for i, n in enumerate(BODY25_NAMES)}
KORE_ROOT = '/Volumes/Sandisk_SSD/MotorDevelopment/Korea'
# Which SMPL-24 rotations the keypoints pin down, per frame.  A joint's rotation sets the direction of the bone(s)
# below it (its SWING) and the turn about that bone (its TWIST).  Swing is observed when the keypoints at both ends
# of the child bone are; twist needs a third point off that bone (for the hip and shoulder: the ankle / wrist, and
# only while the knee / elbow is bent enough to give a plane).  Spine1-3, neck vs head and collar vs shoulder share
# what they move, so the split between them is the prior's; never observed: forearm twist (pronation), wrists, hands,
# toes.  Over a gap the fit's limbs are a guess: hiding a whole arm on dev reps, 1-frame gaps came back within
# ~2 deg, 3-frame gaps 5-14 deg off (wrist 11-22 mm) -- so only --gap-frames 1 counts as seen.
OBSERVE = {   # SMPL-24 joint: (swing keypoints, twist keypoints or None = twist observed with swing, bend joint)
    'Pelvis': (['LHip', 'RHip', 'Neck'], None, None),
    'Spine1': (['LHip', 'RHip', 'LShoulder', 'RShoulder', 'Neck'], None, None),
    'Spine2': (['LHip', 'RHip', 'LShoulder', 'RShoulder', 'Neck'], None, None),
    'Spine3': (['LHip', 'RHip', 'LShoulder', 'RShoulder', 'Neck'], None, None),
    'Neck': (['Neck', 'Nose'], ['REye', 'LEye', 'REar', 'LEar'], None),
    'Head': (['Neck', 'Nose'], ['REye', 'LEye', 'REar', 'LEar'], None),
    **{f'{s}_Hip': ([f'{s[0]}Hip', f'{s[0]}Knee'], [f'{s[0]}Ankle'], f'{s}_Knee') for s in ('L', 'R')},
    **{f'{s}_Knee': ([f'{s[0]}Knee', f'{s[0]}Ankle'], [f'{s[0]}Heel', f'{s[0]}BigToe'], None) for s in ('L', 'R')},
    **{f'{s}_Ankle': ([f'{s[0]}Ankle', f'{s[0]}BigToe'], [f'{s[0]}Heel', f'{s[0]}SmallToe'], None) for s in ('L', 'R')},
    **{f'{s}_Collar': (['Neck', f'{s[0]}Shoulder'], None, None) for s in ('L', 'R')},
    **{f'{s}_Shoulder': ([f'{s[0]}Shoulder', f'{s[0]}Elbow'], [f'{s[0]}Wrist'], f'{s}_Elbow') for s in ('L', 'R')},
    **{f'{s}_Elbow': ([f'{s[0]}Elbow', f'{s[0]}Wrist'], [], None) for s in ('L', 'R')},  # [] = twist never
}
BEND_DEG = 25.0      # knee / elbow flexion below which the hip / shoulder twist is not determined
OBS_REPORT = {g: [F.S24[j] for j in js] for g, js in (
    ('root', ['Pelvis']), ('spine', ['Spine1', 'Spine2', 'Spine3']), ('head', ['Neck', 'Head']),
    ('hips', ['L_Hip', 'R_Hip']), ('knees', ['L_Knee', 'R_Knee']), ('ankles', ['L_Ankle', 'R_Ankle']),
    ('collars', ['L_Collar', 'R_Collar']), ('shoulders', ['L_Shoulder', 'R_Shoulder']),
    ('elbows', ['L_Elbow', 'R_Elbow']), ('wrists+hands+feet', ['L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand',
                                                               'L_Foot', 'R_Foot']))}


def seen(X, gap):
    """(T,25) keypoint present, with interior gaps of <= gap frames counted as present."""
    p = np.isfinite(X).all(-1)
    if gap <= 0:
        return p
    out = p.copy()
    for j in range(p.shape[1]):
        idx = np.where(p[:, j])[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap + 1:
                out[a + 1:b, j] = True
    return out


def observed_mask(X, J24, gap):
    """-> swing (T,24), full (T,24) bool: SMPL-24 rotations the frame's keypoints determine (see OBSERVE)."""
    T = len(X)
    p = seen(X, gap)
    have = lambda names: p[:, [B[n] for n in names]].all(1)               # noqa: E731
    swing = np.zeros((T, 24), bool)
    full = np.zeros((T, 24), bool)
    for j, (sw, tw, bend) in OBSERVE.items():
        k = F.S24[j]
        swing[:, k] = have(sw)
        if tw is None:
            full[:, k] = swing[:, k]
        elif tw:
            twist = have(tw) if j not in ('Neck', 'Head') else p[:, [B[n] for n in tw]].sum(1) >= 2
            if bend:
                c = F.S24[bend]
                kid = F.S24[bend.replace('Knee', 'Ankle').replace('Elbow', 'Wrist')]
                u = J24[:, c] - J24[:, F.S24[j]]
                v = J24[:, kid] - J24[:, c]
                cosang = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
                twist &= np.degrees(np.arccos(np.clip(cosang, -1, 1))) >= BEND_DEG
            full[:, k] = swing[:, k] & twist
    return swing, full
# keypoints the model puts out, by group, for the residual report
GROUPS = {'limbs': list(F.ANCHOR_JOINTS), 'hips_sh': ['RHip', 'LHip', 'RShoulder', 'LShoulder', 'Neck', 'MidHip'],
          'face': list(F.FACE_VERTICES), 'feet': list(F.OPENPOSE_FOOT_VERTICES)}


def load_rep(gt3d, rep):
    z = np.load(os.path.join(gt3d, rep[:4], f'{rep}.npz'))
    X = z['X_body25_cam'][0].astype(float)
    ok = z['valid_body25'].copy()
    dropped = z['valid_h36m'] & ~z['valid_h36m_gated']                # the bone gate's per-joint verdict
    for k, idx in H36M_FROM_BODY25.items():
        for i in idx:
            ok[:, i] &= ~dropped[:, k]
    X[~ok] = np.nan
    return X, z


def fill_time(X):
    """NaNs filled along time per keypoint (linear inside, nearest at the ends) -- initialisation only."""
    X = X.copy()
    t = np.arange(len(X))
    for j in range(X.shape[1]):
        good = np.isfinite(X[:, j]).all(-1)
        if good.sum() == 0:
            continue
        for c in range(3):
            X[~good, j, c] = np.interp(t[~good], t[good], X[good, j, c])
    return X


def fit_rep(X, fit, args, log, init=None):
    """fit_v3 body fixed; root, translation and pose fitted on every frame with the smoothness terms."""
    T = len(X)
    names = list(F.WEIGHT)
    tgt = torch.tensor(np.stack([X[:, B[n]] for n in names], 1))
    ok = torch.isfinite(tgt).all(-1)
    tgt = torch.nan_to_num(tgt)
    w = torch.tensor([F.WEIGHT[n] for n in names], dtype=torch.float64)[None] * ok
    model = F.SMPLFit(T, bool(fit['kid']), str(fit['shoulder_bone']), bool(fit['two_bone']))
    Xi = fill_time(X)
    aa, mid = F.init_root(Xi)
    with torch.no_grad():
        model.betas.copy_(torch.tensor(fit['betas_raw']))
        for g, d in zip(fit['groups'], fit['offsets_rest_mm']):
            model.offsets[str(g)].copy_(torch.tensor(d / 1000.0))
        model.root.copy_(torch.tensor(aa))
        _, J_rest = model.rest()
        model.trans.copy_(torch.tensor(mid) - rodrigues(model.root) @ J_rest[0])
        for nm, sgn in (('L_Shoulder', -1.0), ('R_Shoulder', 1.0)):
            model.pose[:, F.S24[nm] - 1, 2] = sgn * np.radians(75.0)
    torso = torch.tensor([n in ('LHip', 'RHip', 'LShoulder', 'RShoulder', 'Neck', 'MidHip', 'LKnee', 'RKnee')
                          for n in names], dtype=torch.float64)
    torso_idx = torch.tensor([F.S24[n] - 1 for n in F.TORSO_POSE])

    def d2(x):
        return x[2:] - 2 * x[1:-1] + x[:-2]

    def loss(scale_mm, torso_only, smooth):
        kp, J24, _ = model()
        P = torch.stack([kp[n] for n in names], 1)
        r2 = ((P - tgt) ** 2).sum(-1)
        ww = w * (torso[None] if torso_only else 1.0)
        data = (ww * F.gm(r2, scale_mm / 1000.0)).sum() / ww.sum() * 1e6 * (args.robust_mm / scale_mm) ** 2
        pose = model.pose
        prior = args.lam_torso * (pose[:, torso_idx] ** 2).sum() / T + args.lam_pose * (pose ** 2).sum() / T
        sm = torch.zeros((), dtype=torch.float64)
        if smooth and T > 2:
            sm = args.lam_smooth * (d2(J24 * 1000.0) ** 2).sum(-1).mean() \
                + args.lam_smooth_rot * 1e4 * (d2(torch.cat([model.root[:, None], pose], 1)) ** 2).sum(-1).mean()
        return data + prior + sm, data, sm

    t0 = time.time()
    rt, rtp = [model.root, model.trans], [model.root, model.trans, model.pose]
    # coarse to fine frame by frame, THEN the smoothness terms: switched on from the start they hold every frame
    # near the (arms-down) initial pose and the fit never reaches the data.  init: a frame-by-frame solution
    # (another sweep value) to start the smoothing stage from.
    stages = [(rt, 60, 300.0, True, False), (rtp, 100, 300.0, False, False), (rtp, 100, 100.0, False, False),
              (rtp, args.iters, args.robust_mm, False, False)]
    if init is not None:
        with torch.no_grad():
            for k in ('root', 'trans', 'pose'):
                getattr(model, k).copy_(torch.tensor(init[k]))
        stages = []
    if args.lam_smooth > 0 or args.lam_smooth_rot > 0:
        stages.append((rtp, args.iters, args.robust_mm, False, True))
    for stage, (params, iters, scale, torso_only, smooth) in enumerate(stages):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            L, _, _ = loss(scale, torso_only, smooth)
            L.backward()
            return L
        opt.step(closure)
    with torch.no_grad():
        L, D, S = loss(args.robust_mm, False, True)
    log(f'    loss {L.item():.1f} (data {D.item():.1f}, smooth {S.item():.1f}) [{time.time() - t0:.0f}s]')
    return model, names, time.time() - t0


def model_body25(kp, T):
    """model keypoints -> (T,25,3) in BODY_25 order (NaN where the model has no such point)."""
    out = np.full((T, 25, 3), np.nan)
    for n, x in kp.items():
        out[:, B[n]] = x.detach().numpy()
    return out


def score(model, names, X, z, gap=1):
    with torch.no_grad():
        kp, J24, _ = model()
    T = len(X)
    P25 = model_body25(kp, T)
    res = np.linalg.norm(P25 - X, axis=-1) * 1000.0                         # (T,25), NaN where unobserved
    n_kp = np.isfinite(X[:, [B[n] for n in names]]).all(-1).sum(1)
    frame_res = np.nanmedian(np.where(np.isfinite(res[:, [B[n] for n in names]]), res[:, [B[n] for n in names]],
                                      np.nan), 1) if T else np.zeros(0)
    groups = {g: float(np.nanmedian(res[:, [B[n] for n in ns]])) for g, ns in GROUPS.items()}

    # H36M (OpenPose-style, as the targets are built) from the model, into each camera vs the LOO target
    H, _ = body25_to_h36m_3d(P25, np.isfinite(P25).all(-1))
    R, t = z['R_world_to_cam'], z['t_world_to_cam']
    ev = z['eval_joint_mask_h36m']
    fit_err, tri_err = [], []
    for c in range(len(R)):
        Hc = ((H - t[0]) @ R[0]) @ R[c].T + t[c]
        loo = z['X_h36m_cam_loo'][c].astype(float)
        m = z['valid_h36m_loo_gated'][c] & z['valid_h36m_gated'] & ev[None]
        fit_err.append(np.linalg.norm(Hc - loo, axis=-1)[m] * 1000.0)
        tri_err.append(np.linalg.norm(z['X_h36m_cam'][c].astype(float) - loo, axis=-1)[m] * 1000.0)
    fit_err, tri_err = np.concatenate(fit_err), np.concatenate(tri_err)

    # jitter: 2nd difference of the H36M joints, fit vs the all-camera triangulation (eval joints, mm/frame^2)
    tri = z['X_h36m_cam'][0].astype(float)
    tri[~z['valid_h36m_gated']] = np.nan
    acc_fit = np.linalg.norm(H[2:] - 2 * H[1:-1] + H[:-2], axis=-1)[:, ev] * 1000.0
    acc_tri = np.linalg.norm(tri[2:] - 2 * tri[1:-1] + tri[:-2], axis=-1)[:, ev] * 1000.0
    swing, full = observed_mask(X, J24.numpy(), gap)
    return dict(groups=groups, res=res, n_kp=n_kp, frame_res=frame_res, obs_swing=swing, obs_full=full,
                loo_fit=float(np.mean(fit_err)), loo_tri=float(np.mean(tri_err)),
                loo_fit_med=float(np.median(fit_err)), loo_tri_med=float(np.median(tri_err)),
                acc_fit=float(np.nanmedian(acc_fit)), acc_tri=float(np.nanmedian(acc_tri)), H=H)


def split_reps(root, splits_path, sets, split_names, fits):
    """(set, subject, rep) for every usable rep of the subjects in the named splits that have a fit_v3 body."""
    sp = json.load(open(splits_path))
    out, no_body = [], []
    for st in sets:
        subs = sorted({u for nm in split_names for u in sp['sets'][st][nm]}
                      | ({u for u in sp['criteria'].get('force_dev', []) if u[0] == st} if 'dev' in split_names else set()))
        for u in subs:
            if not os.path.exists(os.path.join(fits, f'{u}.npz')):
                no_body.append(u)
                continue
            # the session summary lists the usable (and saved) reps: opening every rep file instead to read its
            # flag costs minutes on a Drive mount (checked: identical lists on all B/C sessions, 2026-11-17)
            summ = os.path.join(root, st, 'GT3D', u, 'session_summary.json')
            if not os.path.exists(summ):
                continue
            d = json.load(open(summ))
            if not d.get('usable'):
                continue
            out += [(st, u, r['stub']) for r in sorted(d['reps'], key=lambda r: r['stub'])
                    if r.get('usable') and r.get('saved', True)]
    return out, no_body


def save_fit(path, model, fit, s, args, a, secs, extra):
    """np.savez to a temporary name, then rename, so an interrupted run never leaves a file --skip-done trusts."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + '.part.npz'
    with torch.no_grad():
        np.savez(tmp, root=model.root.numpy(), trans=model.trans.numpy(),
                 pose=(model.pose * model.pose_mask).numpy(), betas_raw=fit['betas_raw'],
                 shape=fit['shape'], kid=fit['kid'], groups=fit['groups'],
                 offsets_rest_mm=fit['offsets_rest_mm'], frame='camera 1 (GT3D X_body25_cam[0]), metres',
                 n_keypoints=s['n_kp'], frame_residual_mm=s['frame_res'], residual_mm=s['res'],
                 H36M_model=s['H'].astype(np.float32), obs_swing=s['obs_swing'], obs_full=s['obs_full'],
                 gap_frames=args.gap_frames, lam_smooth=a.lam_smooth, lam_smooth_rot=a.lam_smooth_rot,
                 robust_mm=args.robust_mm, lam_torso=args.lam_torso, lam_pose=args.lam_pose, seconds=secs,
                 **extra)
    os.replace(tmp, path)


def summary_row(rep, lam, s, secs):
    return dict(rep=rep, lam=lam, secs=round(secs, 1), frames=int(len(s['n_kp'])),
                **{k: s[k] for k in ('groups', 'loo_fit', 'loo_tri', 'loo_fit_med', 'loo_tri_med', 'acc_fit', 'acc_tri')},
                obs_swing={g: round(float(s['obs_swing'][:, i].mean()), 3) for g, i in OBS_REPORT.items()},
                obs_full={g: round(float(s['obs_full'][:, i].mean()), 3) for g, i in OBS_REPORT.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reps', default=None, help='prototype mode: comma list of rep stubs, e.g. B010_GMS_1_7')
    ap.add_argument('--split', default=None, help="full mode: comma list of splits.json splits, e.g. 'train,dev'")
    ap.add_argument('--sets', default='B,C', help='full mode: Korea sets')
    ap.add_argument('--splits-json', default=os.path.join(_ROOT, 'pipeline', 'datasets', 'korea', 'splits.json'))
    ap.add_argument('--root', default=KORE_ROOT, help='holds {set}/GT3D/')
    ap.add_argument('--out-root', default=None, help='full mode: writes {out_root}/{set}/PseudoGT/{subject}/{rep}.npz '
                                                     '(default --root)')
    ap.add_argument('--fits', default=os.path.join(_ROOT, 'analysis', 'joint_calib', 'out', 'fit_v3'))
    ap.add_argument('--lam-smooth', default='0.1', help="comma list to sweep, each 'joint' or 'joint:rot' (joint "
                                                        "acceleration in mm^2 units; rot = --lam-smooth-rot); full mode: one value")
    ap.add_argument('--lam-smooth-rot', type=float, default=0.0, help='axis-angle acceleration (x1e4)')
    ap.add_argument('--robust-mm', type=float, default=30.0)
    ap.add_argument('--lam-torso', type=float, default=100.0)
    ap.add_argument('--lam-pose', type=float, default=1.0)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--gap-frames', type=int, default=1, help='keypoint gaps this short still count as observed')
    ap.add_argument('--shard', default=None, help="full mode: 'k/n', every n-th rep from the k-th (parallel processes)")
    ap.add_argument('--no-skip-done', action='store_true', help='full mode: refit reps whose output exists')
    ap.add_argument('--dry-run', action='store_true', help='full mode: list the reps and stop')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out-dir', default=None, help='prototype mode: output folder')
    args = ap.parse_args()
    if bool(args.reps) == bool(args.split):
        raise SystemExit('give --reps (prototype) or --split (full run)')
    torch.set_num_threads(args.threads or max(1, os.cpu_count() - 1))

    if args.split:
        if ',' in args.lam_smooth:
            raise SystemExit('full mode takes one --lam-smooth value')
        jobs, no_body = split_reps(args.root, args.splits_json, args.sets.split(','), args.split.split(','), args.fits)
        out_root = args.out_root or args.root
        jobs = [(st, u, r, os.path.join(out_root, st, 'PseudoGT', u, f'{r}.npz')) for st, u, r in jobs]
        if args.shard:
            k, n = (int(x) for x in args.shard.split('/'))
            jobs = jobs[k::n]
        todo = [j for j in jobs if args.no_skip_done or not os.path.exists(j[3])]
        print(f'{len(jobs)} reps ({args.split}, sets {args.sets}{", shard " + args.shard if args.shard else ""}), '
              f'{len(jobs) - len(todo)} already done, {len(todo)} to fit; no fit_v3 body (skipped): '
              f'{", ".join(no_body) or "none"}', flush=True)
        if args.dry_run:
            for st, u, r, p in todo:
                print(f'  {r} -> {p}')
            return
        lam, lam_rot = (float(x) for x in (args.lam_smooth.split(':') + ['0'])[:2])
        a = types.SimpleNamespace(**{**vars(args), 'lam_smooth': lam, 'lam_smooth_rot': lam_rot})
        a0 = types.SimpleNamespace(**{**vars(args), 'lam_smooth': 0.0, 'lam_smooth_rot': 0.0})
        t_all, n_frames = time.time(), 0
        for i, (st, u, rep, path) in enumerate(todo, 1):
            try:
                X, z = load_rep(os.path.join(args.root, st, 'GT3D'), rep)
                fit = np.load(os.path.join(args.fits, f'{u}.npz'), allow_pickle=True)
                base, names, secs = fit_rep(X, fit, a0, lambda m: None)
                model = base
                if lam > 0 or lam_rot > 0:
                    init = {k: getattr(base, k).detach().numpy() for k in ('root', 'trans', 'pose')}
                    model, names, s2 = fit_rep(X, fit, a, lambda m: None, init=init)
                    secs += s2
                s = score(model, names, X, z, args.gap_frames)
                save_fit(path, model, fit, s, args, a, secs, dict(subject=u, rep=rep, fit_file=f'{u}.npz'))
                row = summary_row(rep, args.lam_smooth, s, secs)
                log = os.path.join(out_root, st, 'PseudoGT', '_log', f'shard{(args.shard or "0/1").replace("/", "of")}.jsonl')
                os.makedirs(os.path.dirname(log), exist_ok=True)
                with open(log, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                n_frames += len(X)
                el = time.time() - t_all
                print(f'[{i}/{len(todo)}] {rep}: {len(X)} fr, {secs:.0f}s, residual limbs {s["groups"]["limbs"]:.1f} mm, '
                      f'vs LOO {s["loo_fit"]:.1f} (tri {s["loo_tri"]:.1f}) | {el / 3600:.2f} h, '
                      f'{el / n_frames:.2f} s/frame', flush=True)
            except Exception as e:                                  # one bad rep must not end a long run
                print(f'[{i}/{len(todo)}] {rep}: FAILED {type(e).__name__}: {e}', flush=True)
        return

    rows = []
    for rep in args.reps.split(','):
        X, z = load_rep(os.path.join(args.root, rep[0], 'GT3D'), rep)
        fit = np.load(os.path.join(args.fits, f'{rep[:4]}.npz'), allow_pickle=True)
        print(f'{rep}: {len(X)} frames, fit_v3 body {rep[:4]} (median residual there '
              f'{np.median(fit["residual_mm"]):.0f} mm)', flush=True)
        # the frame-by-frame solution first; every smoothing value starts from it
        base, _, base_secs = fit_rep(X, fit, types.SimpleNamespace(**{**vars(args), 'lam_smooth': 0.0,
                                                                     'lam_smooth_rot': 0.0}), print)
        init = {k: getattr(base, k).detach().numpy() for k in ('root', 'trans', 'pose')}
        for item in args.lam_smooth.split(','):
            lam, lam_rot = (float(x) for x in (item.split(':') + ['0'])[:2])
            a = types.SimpleNamespace(**{**vars(args), 'lam_smooth': lam, 'lam_smooth_rot': lam_rot})
            if lam == 0 and lam_rot == 0:
                model, names, secs = base, list(F.WEIGHT), base_secs
            else:
                model, names, secs = fit_rep(X, fit, a, print, init=init)
                secs += base_secs
            s = score(model, names, X, z, args.gap_frames)
            rows.append(summary_row(rep, item, s, secs))
            g = s['groups']
            print(f'  lam {item}: residual mm limbs {g["limbs"]:.1f} hips/sh {g["hips_sh"]:.1f} face {g["face"]:.1f} '
                  f'feet {g["feet"]:.1f} | vs LOO mean fit {s["loo_fit"]:.1f} / tri {s["loo_tri"]:.1f} '
                  f'(median {s["loo_fit_med"]:.1f} / {s["loo_tri_med"]:.1f}) | accel fit {s["acc_fit"]:.1f} '
                  f'tri {s["acc_tri"]:.1f} mm/fr^2 | frames <8 kp {int((s["n_kp"] < 8).sum())}/{len(X)}', flush=True)
            print('    observed % swing/full: ' + '  '.join(
                f'{g} {100 * s["obs_swing"][:, idx].mean():.0f}/{100 * s["obs_full"][:, idx].mean():.0f}'
                for g, idx in OBS_REPORT.items()), flush=True)
            if args.out_dir:
                save_fit(os.path.join(args.out_dir, f'{rep}_lam{item}.npz'), model, fit, s, args, a, secs,
                         dict(subject=rep[:4], rep=rep, fit_file=f'{rep[:4]}.npz'))
    if args.out_dir:
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
