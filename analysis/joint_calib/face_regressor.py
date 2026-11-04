#!/usr/bin/env python3
"""face_regressor.py -- OpenPose's nose and ears as vertex regressors on SMPL, as vertex_regressor.py does the hips.

smplx's single face vertices (smpl_op_fit.FACE_VERTICES) miss OpenPose's points on children: the ears sit 6-11 mm
higher and 10-12 mm further out, the nose 5-7 mm further back (face_check.py), which tips the nose -> mid-ears line
4-7.5 deg against OpenPose's.  Here each of Nose, LEar, REar is a weighted average of the template vertices within
--radius of its smplx vertex, fitted (vertex_regressor.fit_weights) to smpl_op_fit.py's posed meshes against the
triangulated keypoint, with and without the kid-weight term.

Caveat (circularity): smpl_op_fit oriented every head through the smplx vertices themselves, so part of their
mismatch was absorbed into the fitted head pose and is invisible here; scores on its meshes cannot show that part.
The pipeline's head pitch with these points (step_4b, head posed without smpl_op_fit) is the independent test.

Train / evaluate as vertex_regressor.py --splits: 5-fold CV by subject within adults + train children, then fitted
on all of them and scored on the dev children.  Scored per subject in the Head bone's frame (lat away from the
midline, up, fwd; mm at OP_SIZE_MM), plus the nose -> mid-ears line: elevation and length, mesh minus target.

    python analysis/joint_calib/face_regressor.py analysis/joint_calib/out/fit_v3 \\
        --splits pipeline/datasets/korea/splits.json --out models/openpose_face_regressor.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'pipeline'))
import vertex_regressor as VR  # noqa: E402
from smpl_op_fit import FACE_VERTICES  # noqa: E402
from utils.smpl_lite import MiniSMPL, SMPL24, OP_SIZE_MM, _load_pkl, rodrigues  # noqa: E402

FACE = ('Nose', 'LEar', 'REar')
HEAD = SMPL24.index('Head')


def candidates(radius_mm):
    v = np.asarray(_load_pkl()['v_template'])
    return {k: np.where(np.linalg.norm(v - v[FACE_VERTICES[k]], axis=1) < radius_mm[k[1:] if k != 'Nose' else k] / 1000.0)[0]
            for k in FACE}


def subject_meshes(path, cand, max_frames, rng):
    """vertex_regressor.subject_meshes' layout for the face: candidate vertices, the smplx vertex (the baseline),
    the Head bone's rotation."""
    z = np.load(path, allow_pickle=True)
    T = len(z['root'])
    keep = np.sort(rng.choice(T, max_frames, replace=False)) if T > max_frames else np.arange(T)
    smpl = MiniSMPL(z['shape'], full=True)
    aa = torch.tensor(np.concatenate([z['root'][keep, None], z['pose'][keep]], 1))
    with torch.no_grad():
        R = rodrigues(aa)
        A = smpl.bone_transforms(R)
        v = smpl.vertices(R, A).numpy() + z['trans'][keep, None]
    name = os.path.basename(path)[:-4]
    shape = z['shape']
    return dict(name=name, group='adult' if name.startswith('P') else name[0], size=float(z['size_mm']),
                kw=float(shape[-1]) if len(shape) > 10 else 0.0, X=z['X'][keep].astype(np.float32),
                V={k: v[:, c].astype(np.float32) for k, c in cand.items()},
                smplx={k: v[:, FACE_VERTICES[k]].astype(np.float32) for k in FACE},
                G=A[:, HEAD, :3, :3].numpy().astype(np.float32))


def predict(s, k, model):
    if model is None:
        return s['smplx'][k].astype(float)
    w = model[0] + s['kw'] * model[1] if isinstance(model, tuple) else model
    return np.einsum('tnc,n->tc', s['V'][k].astype(float), w)


def head_local(s, d, ok=None):
    G = s['G'].astype(float) if ok is None else s['G'][ok].astype(float)
    return np.einsum('tji,tj->ti', G, d) * 1000.0 * OP_SIZE_MM / s['size']


def score(s, models):
    """{keypoint: median (lat, up, fwd) residual}, |residual| median, and the nose -> mid-ears elevation / length
    error (mesh - target, deg / mm at OP_SIZE_MM) for one subject under {keypoint: model}."""
    P = {k: predict(s, k, models[k]) for k in FACE}
    out = {}
    for k in FACE:
        d = s['X'][:, VR.B[k]].astype(float) - P[k]
        ok = np.isfinite(d).all(1)
        loc = head_local(s, d[ok], ok)
        if k[0] == 'R':
            loc[:, 0] *= -1
        out[k] = (np.median(loc, 0), np.median(np.linalg.norm(loc, axis=1)))
    vm = head_local(s, (P['LEar'] + P['REar']) / 2 - P['Nose'])
    vt = head_local(s, s['X'][:, [VR.B['LEar'], VR.B['REar']]].astype(float).mean(1) - s['X'][:, VR.B['Nose']])
    ok = np.isfinite(vt).all(1)
    el = lambda v: np.degrees(np.arctan2(v[:, 1], -v[:, 2]))  # noqa: E731
    out['elev'] = float(np.median(el(vm[ok]) - el(vt[ok])))
    out['len'] = float(np.median(np.linalg.norm(vm[ok], axis=1) - np.linalg.norm(vt[ok], axis=1)))
    return out


def report(title, rows):
    """rows: {model label: [(subject, score dict)]}."""
    print(f'\n{title}')
    print('  median residual (data - model) mm at size 1423, head frame (lat, up, fwd), mean over subjects; |res|;'
          '  nose -> mid-ears elevation / length error (mesh - target), mean [IQR] over subjects')
    for label, rs in rows.items():
        for g in sorted({s['group'] for s, _ in rs}):
            sc = [r for s, r in rs if s['group'] == g]
            cells = []
            for part, ks in (('Nose', ('Nose',)), ('Ears', ('LEar', 'REar'))):
                b = np.mean([r[k][0] for r in sc for k in ks], 0)
                cells.append(f'{part} ' + ','.join(f'{x:+5.1f}' for x in b) + f' |{np.mean([r[k][1] for r in sc for k in ks]):4.1f}|')
            e = np.array([r['elev'] for r in sc])
            ln = np.array([r['len'] for r in sc])
            print(f'  {label:13s} {g:5s} ' + '   '.join(cells) +
                  f'   elev {e.mean():+5.1f} [{np.percentile(e, 25):+.1f},{np.percentile(e, 75):+.1f}] deg'
                  f'   len {ln.mean():+5.1f} mm')


def fit_all(train, args, rng):
    W = {k: VR.fit_weights(train, k, args.ridge, args.fit_frames, rng) for k in FACE}
    Wkw = {k: VR.fit_weights(train, k, args.ridge, args.fit_frames, rng, with_kw=True, kw_mode='dense',
                             ridge_u=args.ridge_u) for k in FACE}
    return W, Wkw


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('fit_dir')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--radius-nose-mm', type=float, default=30.0)
    ap.add_argument('--radius-ear-mm', type=float, default=45.0)
    ap.add_argument('--ridge', type=float, default=1e-6)
    ap.add_argument('--ridge-u', type=float, default=1e-4)
    ap.add_argument('--max-frames', type=int, default=120)
    ap.add_argument('--fit-frames', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cand = candidates({'Nose': args.radius_nose_mm, 'Ear': args.radius_ear_mm})
    print('candidate vertices: ' + ', '.join(f'{k} {len(c)}' for k, c in cand.items()))
    with open(args.splits) as f:
        sp = json.load(f)
    paths = sorted(glob.glob(os.path.join(args.fit_dir, '*.npz')))
    subs = [subject_meshes(p, cand, args.max_frames, rng) for p in paths
            if os.path.basename(p)[0] == 'P' or os.path.basename(p)[:-4] in sp['train'] + sp['dev']]
    pool = [s for s in subs if s['group'] == 'adult' or s['name'] in sp['train']]
    dev = [s for s in subs if s['name'] in sp['dev']]
    print(f'{len(pool)} training subjects ({VR._count(pool)}), {len(dev)} dev children')

    fold = np.zeros(len(pool), int)
    for g in {s['group'] for s in pool}:
        idx = rng.permutation([i for i, s in enumerate(pool) if s['group'] == g])
        fold[idx] = np.arange(len(idx)) % args.folds
    rows = {'smplx vertex': [], 'regressor': [], 'regressor_kw': []}
    for f in range(args.folds):
        W, Wkw = fit_all([s for s, q in zip(pool, fold) if q != f], args, rng)
        for s, q in zip(pool, fold):
            if q == f:
                rows['smplx vertex'].append((s, score(s, {k: None for k in FACE})))
                rows['regressor'].append((s, score(s, W)))
                rows['regressor_kw'].append((s, score(s, Wkw)))
        print(f'  fold {f + 1}/{args.folds} fitted', flush=True)
    report(f'1. {args.folds}-fold cross-validation within train (out-of-fold)', rows)

    W, Wkw = fit_all(pool, args, rng)
    report('2. trained on all of train -> dev children',
           {'smplx vertex': [(s, score(s, {k: None for k in FACE})) for s in dev],
            'regressor': [(s, score(s, W)) for s in dev],
            'regressor_kw': [(s, score(s, Wkw)) for s in dev]})
    if args.out:
        np.savez(args.out, **{f'{k}_vertices': cand[k] for k in FACE}, **{f'{k}_weights': W[k] for k in FACE},
                 **{f'{k}_weights_kw_w': Wkw[k][0] for k in FACE}, **{f'{k}_weights_kw_u': Wkw[k][1] for k in FACE},
                 subjects=np.array([s['name'] for s in pool]))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
