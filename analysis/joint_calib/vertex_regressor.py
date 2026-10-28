#!/usr/bin/env python3
"""vertex_regressor.py -- OpenPose hips and shoulders as fixed vertex regressors on SMPL, tested across groups.

smpl_op_fit.py models each OpenPose hip / shoulder as an SMPL-24 joint plus one offset per subject.  Offsets
differ between adults and children (hip lateral and forward), and within children they trade with shape.  A vertex
regressor -- the keypoint as a fixed convex combination of mesh vertices, as SMPL's own H36M regressor is -- would
let shape carry the keypoint: girth pushes the surface out, the kid template moves it with child proportions.

Per keypoint (LHip, RHip, LShoulder, RShoulder), weights w >= 0, sum 1, over the vertices within --radius of the
SMPL-24 joint on the mean template, fitted by least squares to smpl_op_fit.py's posed meshes (its per-frame pose and
shape, held fixed) against the triangulated keypoint.  Each subject counts once (rows / sqrt(frames)) and scale-free
(rows * OP_SIZE_MM / size), so 63 children do not swamp 11 adults by count and 2x-smaller children are not
down-weighted by size.

Tested by stratified k-fold cross-validation BY SUBJECT (every subject scored by a model that never saw them;
every fold keeps its share of adults and of each child set A/B/C), then on the evaluation children (--heldout,
never trained on).  With --splits (datasets/korea/splits.json) the children come from the split instead: train
children are the calibration set (cross-validated within), dev children are the evaluation set, and test children
are scored only with --test (once, when the recipe is settled); children in no split (set A) are left out.
--loso adds leave-one-child-set-out, an extrapolation stress test.  Scored per test subject as the median residual (data - prediction) in the pelvis (hips) / thorax (shoulders) bone
frame, (lateral, up, forward), lateral away from the midline, scaled to OP_SIZE_MM; compared with
    joint      the SMPL-24 joint itself (no offset)
    offset     joint + the training group's mean size-scaled offset (smpl_op_fit's model, one offset for all)
    regressor  this

    python3 analysis/joint_calib/vertex_regressor.py <fit_dir> [<fit_dir> ...] --heldout B010,B011,B014,B023 [--out w.npz]
    python3 analysis/joint_calib/vertex_regressor.py <fit_dir> --splits pipeline/datasets/korea/splits.json [--out w.npz]
<fit_dir>: smpl_op_fit.py --out-dir outputs ({subject}.npz with pose/root/trans/X); P* adults, B* children.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pipeline'))
from utils.smpl_lite import MiniSMPL, SMPL24, OP_SIZE_MM, _load_pkl, rodrigues  # noqa: E402
from utils.openpose import BODY25_NAMES  # noqa: E402

B = {n: i for i, n in enumerate(BODY25_NAMES)}
S24 = {n: i for i, n in enumerate(SMPL24)}
KEYPOINTS = {'LHip': ('L_Hip', 'Pelvis'), 'RHip': ('R_Hip', 'Pelvis'),
             'LShoulder': ('L_Shoulder', 'Spine3'), 'RShoulder': ('R_Shoulder', 'Spine3')}
COMP = ('lat', 'up', 'fwd')


def candidates(radius_m):
    m = _load_pkl()
    v = np.asarray(m['v_template'])
    Jr = m['J_regressor']
    Jr = Jr.toarray() if hasattr(Jr, 'toarray') else np.asarray(Jr)
    J = Jr @ v
    return {k: np.where(np.linalg.norm(v - J[S24[j]], axis=1) < radius_m[k[1:]])[0] for k, (j, _) in KEYPOINTS.items()}


def subject_meshes(path, cand, max_frames, rng):
    """Posed candidate vertices, the four base joints, the two carrying bones' rotations and the data for one
    smpl_op_fit output, at most max_frames frames, float32 (a few hundred subjects fit in memory)."""
    z = np.load(path, allow_pickle=True)
    T = len(z['root'])
    keep = np.sort(rng.choice(T, max_frames, replace=False)) if T > max_frames else np.arange(T)
    smpl = MiniSMPL(z['shape'], full=True)
    aa = torch.tensor(np.concatenate([z['root'][keep, None], z['pose'][keep]], 1))
    with torch.no_grad():
        R = rodrigues(aa)
        A = smpl.bone_transforms(R)
        v = smpl.vertices(R, A).numpy()
        J24 = np.einsum('jn,tnc->tjc', smpl.Jr.numpy(), v) + z['trans'][keep, None]
        v = v + z['trans'][keep, None]
    G = A[:, :, :3, :3].numpy()
    name = os.path.basename(path)[:-4]
    shape = z['shape']
    return dict(name=name, group='adult' if name.startswith('P') else name[0], size=float(z['size_mm']),
                kw=float(shape[-1]) if len(shape) > 10 else 0.0, X=z['X'][keep].astype(np.float32),
                V={k: v[:, c].astype(np.float32) for k, c in cand.items()},
                J={j: J24[:, S24[j]].astype(np.float32) for j, _ in KEYPOINTS.values()},
                G={b: G[:, S24[b]].astype(np.float32) for b in ('Pelvis', 'Spine3')},
                offs=dict(zip(z['groups'], z['offsets_rest_mm'])))


def local(sub, k, d):
    """(T,3) lab residuals -> (lat, up, fwd) in the carrying bone's frame, mm at OP_SIZE_MM."""
    _, bone = KEYPOINTS[k]
    loc = np.einsum('tji,tj->ti', sub['G'][bone], d) * 1000.0 * OP_SIZE_MM / sub['size']
    if k[0] == 'R':
        loc[:, 0] *= -1
    return loc


def _design(subs, k, fit_frames, rng, with_kw):
    rows, rhs = [], []
    for s in subs:
        x = s['X'][:, B[k]].astype(float)
        ok = np.where(np.isfinite(x).all(1))[0]
        if len(ok) > fit_frames:
            ok = np.sort(rng.choice(ok, fit_frames, replace=False))
        Vk = s['V'][k][ok].astype(float)                            # (T,n,3)
        sc = OP_SIZE_MM / s['size'] / np.sqrt(len(ok))              # each subject once, scale-free
        M = np.transpose(Vk, (0, 2, 1)).reshape(-1, Vk.shape[1]) * sc
        rows.append(np.hstack([M, s['kw'] * M]) if with_kw else M)
        rhs.append(x[ok].reshape(-1) * sc)
    return np.concatenate(rows), np.concatenate(rhs)


KW_REF = 0.6                       # the kid weight the child regressor is anchored at (set B's median)


def fit_weights(subs, k, ridge, fit_frames, rng, with_kw=False, sum_w=1e3, kw_mode='convex', ridge_u=1e-6):
    """Convex weights w (>= 0, sum 1).  with_kw: two convex regressors, W0 at kid weight 0 (adult) and W1 at KW_REF,
    blended linearly in the body's kid weight -- W(kw) = W0 + (kw / KW_REF)(W1 - W0) -- so every body between
    the two gets a true weighted average of surface points (an unconstrained correction went dense, +-14 weights).
    Returned as (w, u) with W(kw) = w + kw u."""
    M, y = _design(subs, k, fit_frames, rng, with_kw)
    n = M.shape[1] // (2 if with_kw else 1)
    if not with_kw:
        M = np.vstack([M, sum_w * np.ones((1, n)), np.sqrt(ridge) * np.eye(n)])
        w, _ = nnls(M, np.r_[y, sum_w, np.zeros(n)], maxiter=50 * n)
        return w / w.sum()
    if kw_mode == 'dense':
        # w >= 0 (sum 1) plus a free correction u (sum 0) scaled by kid weight, ridge_u on u: u's differences
        # of vertices act as directions that turn with the surface -- a pose-following offset -- but large
        # cancelling weights are fragile, so ridge_u is chosen by cross-validation
        from scipy.optimize import lsq_linear
        cons = np.zeros((2, 2 * n))
        cons[0, :n], cons[1, n:] = sum_w, sum_w
        reg = np.sqrt(np.r_[ridge * np.ones(n), ridge_u * np.ones(n)])
        A = np.vstack([M, cons, np.diag(reg)])
        lb = np.r_[np.zeros(n), -np.inf * np.ones(n)]
        sol = lsq_linear(A, np.r_[y, sum_w, 0.0, np.zeros(2 * n)], bounds=(lb, np.inf * np.ones(2 * n)),
                         method='bvls', max_iter=200)
        return sol.x[:n], sol.x[n:]
    # _design's columns are [V, kw V]: re-express them on (W0, W1): V W0 (1 - kw/r) + V W1 kw/r
    Va, Vk = M[:, :n], M[:, n:]
    A = np.hstack([Va - Vk / KW_REF, Vk / KW_REF])
    cons = np.zeros((2, 2 * n))
    cons[0, :n], cons[1, n:] = sum_w, sum_w
    A = np.vstack([A, cons, np.sqrt(ridge) * np.eye(2 * n)])
    W, _ = nnls(A, np.r_[y, sum_w, sum_w, np.zeros(2 * n)], maxiter=50 * 2 * n)
    W0, W1 = W[:n] / W[:n].sum(), W[n:] / W[n:].sum()
    return W0, (W1 - W0) / KW_REF


def predict(sub, k, model, train):
    joint, bone = KEYPOINTS[k]
    J = sub['J'][joint].astype(float)
    if isinstance(model, str) and model == 'joint':
        return J
    if isinstance(model, str) and model.startswith('offset'):
        g = 'hip' if 'Hip' in k else 'shoulder'
        D = np.array([t['offs'][g] / t['size'] for t in train])  # scale-free rest offsets
        kw = np.array([t['kw'] for t in train])
        if model == 'offset_kw' and kw.std() > 1e-3:              # offset linear in kid weight
            c = np.linalg.lstsq(np.c_[np.ones(len(kw)), kw], D, rcond=None)[0]
            d = c[0] + c[1] * sub['kw']
        else:
            d = D.mean(0)
        d = d * sub['size'] / 1000.0
        if k[0] == 'R':
            d = d * np.array([-1.0, 1.0, 1.0])
        return J + np.einsum('tij,j->ti', sub['G'][bone].astype(float), d)
    if isinstance(model, tuple):
        w, u = model
        return np.einsum('tnc,n->tc', sub['V'][k].astype(float), w + sub['kw'] * u)
    return np.einsum('tnc,n->tc', sub['V'][k].astype(float), model)


def score(sub, k, pred):
    d = sub['X'][:, B[k]].astype(float) - pred
    ok = np.isfinite(d).all(1)
    loc = local(sub, k, d[ok])
    return np.median(loc, 0), np.median(np.linalg.norm(loc, axis=1))


MODELS = ('joint', 'offset', 'offset_kw', 'regressor', 'regressor_kw')


def report(title, test, train, W, Wkw):
    """Hips and shoulders, L/R pooled (lateral mirrored), per test group."""
    print(f'\n{title}: train {len(train)} ({_count(train)}), test {len(test)} ({_count(test)})')
    print('  median residual (data - model) mm at size 1423, mean over test subjects (lat, up, fwd); |res| mean')
    for part in ('Hip', 'Shoulder'):
        for model in MODELS:
            if model == 'regressor_kw' and Wkw is None:
                continue
            cells = []
            for g in sorted({s['group'] for s in test}):
                tg = [s for s in test if s['group'] == g]
                res = []
                for s in tg:
                    for k in ('L' + part, 'R' + part):
                        m = W[k] if model == 'regressor' else Wkw[k] if model == 'regressor_kw' else model
                        res.append(score(s, k, predict(s, k, m, train)))
                b = np.mean([r[0] for r in res], 0)
                cells.append(f'{g}: ' + ','.join(f'{x:+4.0f}' for x in b) + f' |{np.mean([r[1] for r in res]):4.1f}|')
            print(f'  {part:8s} {model:12s} ' + '   '.join(cells))


def report_oof(title, test, oof):
    """report() with each test subject scored by its own fold's model."""
    print(f'\n{title}: {len(test)} subjects ({_count(test)})')
    print('  median residual (data - model) mm at size 1423, mean over test subjects (lat, up, fwd); |res| mean')
    has_kw = all(v[2] is not None for v in oof.values())
    for part in ('Hip', 'Shoulder'):
        for model in MODELS:
            if model == 'regressor_kw' and not has_kw:
                continue
            cells = []
            for g in sorted({s['group'] for s in test}):
                res = []
                for s in (s for s in test if s['group'] == g):
                    tr, W, Wkw = oof[s['name']]
                    for k in ('L' + part, 'R' + part):
                        m = W[k] if model == 'regressor' else Wkw[k] if model == 'regressor_kw' else model
                        res.append(score(s, k, predict(s, k, m, tr)))
                b = np.mean([r[0] for r in res], 0)
                cells.append(f'{g}: ' + ','.join(f'{x:+4.0f}' for x in b) + f' |{np.mean([r[1] for r in res]):4.1f}|')
            print(f'  {part:8s} {model:12s} ' + '   '.join(cells))


def _count(subs):
    g = {}
    for s in subs:
        g[s['group']] = g.get(s['group'], 0) + 1
    return ' '.join(f'{k} {v}' for k, v in sorted(g.items()))


def fit_all(train, args, rng):
    W = {k: fit_weights(train, k, args.ridge, args.fit_frames, rng) for k in KEYPOINTS}
    kw = np.array([s['kw'] for s in train])
    Wkw = ({k: fit_weights(train, k, args.ridge, args.fit_frames, rng, with_kw=True, kw_mode=args.kw_mode,
                           ridge_u=args.ridge_u) for k in KEYPOINTS}
           if kw.std() > 1e-3 and not args.no_kw else None)
    return W, Wkw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('fit_dirs', nargs='+')
    ap.add_argument('--heldout', default='B010,B011,B014,B023', help='evaluation children, never trained on')
    ap.add_argument('--splits', default=None, help='splits.json: train children calibrate, dev evaluate (overrides --heldout)')
    ap.add_argument('--test', action='store_true', help='--splits: also score the test children (once, at the end)')
    ap.add_argument('--radius-hip-mm', type=float, default=150.0)
    ap.add_argument('--radius-shoulder-mm', type=float, default=120.0)
    ap.add_argument('--ridge', type=float, default=1e-6)
    ap.add_argument('--max-frames', type=int, default=120, help='frames kept per subject (memory)')
    ap.add_argument('--fit-frames', type=int, default=60, help='frames per subject in each weight fit')
    ap.add_argument('--no-kw', action='store_true', help='skip the kid-weight regressor (slow on many subjects)')
    ap.add_argument('--folds', type=int, default=5, help='cross-validation folds (by subject, stratified by group)')
    ap.add_argument('--loso', action='store_true', help='also leave-one-child-set-out (extrapolation stress test)')
    ap.add_argument('--kw-mode', choices=('convex', 'dense'), default='convex',
                    help="kid-weight regressor: 'convex' blends two convex regressors (safe); 'dense' adds a free "
                         "correction u (pose-following, fragile unless --ridge-u is large enough)")
    ap.add_argument('--ridge-u', type=float, default=1e-6, help="'dense': ridge on the correction u")
    ap.add_argument('--exclude-groups', default=None, help="comma list of subject-name prefixes to leave out, e.g. 'A'")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cand = candidates({'Hip': args.radius_hip_mm / 1000.0, 'Shoulder': args.radius_shoulder_mm / 1000.0})
    print('candidate vertices: ' + ', '.join(f'{k} {len(c)}' for k, c in cand.items()))
    paths = sorted(p for d in args.fit_dirs for p in glob.glob(os.path.join(d, '*.npz')))
    drop = tuple(args.exclude_groups.split(',')) if args.exclude_groups else ()
    paths = [p for p in paths if os.path.basename(p)[0] not in drop]
    subs = [subject_meshes(p, cand, args.max_frames, rng) for p in paths]
    adults = [s for s in subs if s['group'] == 'adult']
    if args.splits:
        import json
        with open(args.splits) as f:
            sp = json.load(f)
        kids = [s for s in subs if s['name'] in sp['train']]
        heldout = [s for s in subs if s['name'] in sp['dev']]
        test = [s for s in subs if s['name'] in sp['test']] if args.test else []
    else:
        held = set(args.heldout.split(','))
        kids = [s for s in subs if s['group'] != 'adult' and s['name'] not in held]
        heldout, test = [s for s in subs if s['name'] in held], []
    sets = sorted({s['group'] for s in kids})
    print(f'{len(adults)} adults, {len(kids)} calibration children ({_count(kids)}), {len(heldout)} held out')
    for g in ['adult'] + sets:
        kw = np.array([s['kw'] for s in subs if s['group'] == g])
        print(f'  {g}: kid weight median {np.median(kw):.2f} range {kw.min():.2f}-{kw.max():.2f}')

    # 1. stratified k-fold by SUBJECT: every subject tested once, by a model that never saw them; each fold
    #    keeps its share of every group (adults, A, B, C) -- the in-range question deployment asks
    pool = adults + kids
    fold = np.zeros(len(pool), int)
    for g in {s['group'] for s in pool}:
        idx = rng.permutation([i for i, s in enumerate(pool) if s['group'] == g])
        fold[idx] = np.arange(len(idx)) % args.folds
    oof = {}                                                       # subject -> (train set, W, Wkw) of its fold
    for f in range(args.folds):
        tr = [s for s, q in zip(pool, fold) if q != f]
        W, Wkw = fit_all(tr, args, rng)
        for s, q in zip(pool, fold):
            if q == f:
                oof[s['name']] = (tr, W, Wkw)
        print(f'  fold {f + 1}/{args.folds} fitted', flush=True)
    report_oof(f'1. {args.folds}-fold cross-validation, stratified by group (out-of-fold)', pool, oof)
    if args.loso and len(sets) > 1:                                # 2. does each set sit on the others' trend?
        for g in sets:
            tr = adults + [s for s in kids if s['group'] != g]
            W, Wkw = fit_all(tr, args, rng)
            report(f'2. adults + other sets -> set {g} (extrapolation stress test)',
                   [s for s in kids if s['group'] == g], tr, W, Wkw)
    W_all, Wkw_all = fit_all(adults + kids, args, rng)
    report('3. adults + all children -> ' + ('dev children' if args.splits else 'held-out children'),
           heldout, adults + kids, W_all, Wkw_all)
    if test:
        report('4. adults + all children -> TEST children', test, adults + kids, W_all, Wkw_all)
    report('3. adults + all children -> training groups (in-sample)', adults + kids, adults + kids, W_all, Wkw_all)
    if args.out:
        extra = {}
        if Wkw_all is not None:
            extra = {**{f'{k}_weights_kw_w': Wkw_all[k][0] for k in KEYPOINTS},
                     **{f'{k}_weights_kw_u': Wkw_all[k][1] for k in KEYPOINTS}}
        np.savez(args.out, **{f'{k}_vertices': cand[k] for k in KEYPOINTS}, **{f'{k}_weights': W_all[k] for k in KEYPOINTS},
                 **extra, subjects=np.array([s['name'] for s in adults + kids]))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
