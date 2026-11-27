"""skeleton.py -- re-triangulate one Korea rep on demand for the review viewer.

Every rep that reached a calibrated camera setup (usable or not) can be rebuilt exactly from what GT3D keeps:
the setup's cameras and final sync offsets (session_cameras.npz), the chosen person (session_summary.json) and
the synced tracks (Aligned/).  build_child_gt calibrates with fit_k1=False, so the pinhole cameras reproduce
its triangulation exactly (checked 2026-11-20: n_usable_frames identical on 98.7% of reps, +-1 frame on the
rest because the summary rounds the session bone medians to 4 decimals).

rep_skeleton(set, stub)  -> per-frame reprojection into each camera, a side view (floor frame, looking across
                            camera 1's line of sight, so depth errors show), and per-bone length deviations
setup_bone_stats(set, stub) -> per-bone CV over every rep of the stub's camera setup: left, right, pooled,
                            robust, and the left-right difference -- what sits behind the session 'long-bone CV'
Reps that failed sync, never calibrated, or were excluded have no cameras: {'available': False, 'why': ...}.
"""
import json, os, sys, threading, warnings
import numpy as np
from scipy.spatial.transform import Rotation

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'korea_gt'))
import build_child_gt as B   # noqa: E402
import camcal as cc          # noqa: E402

ROOT = B.KOREA_ROOT
LONG = list(B.LONG_BONES)                       # torso, upperarm, forearm, thigh, shank -- the session CV gate
# camcal.BONES order, with a side label for the panel
BONES = [(a, b, k, 'L' if a in (5, 6, 12, 13) or b in (5, 12) else ('R' if k != 'torso' else '-'))
         for a, b, k in cc.BONES]
_lock = threading.Lock()
_sess, _reps, _setup_stats = {}, {}, {}


def _session(S, subj):
    key = (S, subj)
    with _lock:
        if key in _sess:
            return _sess[key]
    G = f'{ROOT}/{S}/GT3D/{subj}'
    summ = json.load(open(f'{G}/session_summary.json')) if os.path.exists(f'{G}/session_summary.json') else None
    cams = dict(np.load(f'{G}/session_cameras.npz')) if os.path.exists(f'{G}/session_cameras.npz') else None
    with _lock:
        _sess[key] = (summ, cams)
    return summ, cams


def _loaded(S, subj):
    key = (S, subj)
    with _lock:
        if key in _reps:
            return _reps[key]
    reps = {r['stub']: r for r in B.load_reps(f'{ROOT}/{S}/Aligned', subj)}
    with _lock:
        _reps[key] = reps
    return reps


def _locate(S, stub):
    """(summary, setup summary, rep record, cameras dict, setup index) or (None, why)."""
    subj = stub.split('_')[0]
    summ, cams = _session(S, subj)
    if summ is None:
        return None, 'no GT3D session for this child'
    if summ.get('status') != 'ok' or cams is None:
        return None, 'session never calibrated: ' + '; '.join(summ.get('reasons', []))
    rec = next((r for r in summ.get('reps', []) if r['stub'] == stub), None)
    if rec is None:
        return None, 'rep not in a calibrated setup (failed sync or excluded action)'
    n_set = int(cams['n_setups'])
    s = next((k for k in range(1, n_set + 1) if stub in [str(x) for x in cams[f'stubs_s{k}']]), None)
    if s is None:
        return None, 'rep not assigned to a camera setup'
    st = summ['setups'][s - 1] if n_set > 1 else summ
    return (summ, st, rec, cams, s), ''


def _camset(cams, s):
    return cc.CameraSet(f=cams[f'f_s{s}'].astype(float), k1=np.zeros(3), k2=np.zeros(3),
                        rvec=Rotation.from_matrix(cams[f'R_world_to_cam_s{s}']).as_rotvec(),
                        t=cams[f't_world_to_cam_s{s}'].astype(float),
                        cx=cams[f'cx_s{s}'].astype(float), cy=cams[f'cy_s{s}'].astype(float))


def _triangulate(S, stub, loc):
    summ, st, rec, cams, s = loc
    rep = dict(_loaded(S, stub.split('_')[0])[stub])
    p = rec.get('person')
    if p and 'person_chosen' in p and rep['people']:
        c = rep['people'][p['person_chosen']]
        rep.update(raw_xy=c['raw_xy'], raw_conf=c['raw_conf'])
    i = [str(x) for x in cams[f'stubs_s{s}']].index(stub)
    lag = cams[f'lag_final_s{s}'][i]
    rep['lag'] = (int(lag[0]), int(lag[1]))
    xy, conf, start = B.rep_span(rep)
    vis = (conf > B.GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    cs = _camset(cams, s)
    X, ok, used, err = cc.triangulate_robust(cs, xy, vis, B.GATES['reproj_thresh_px'])
    # used (T,3,25) views kept; err (T,3,25) px residual (NaN unused); vis (T,3,25) detected above the conf threshold
    return cs, X, ok, start, rep['lag'], used, err, vis, conf, xy


# Worst residual above which a 3-view joint is flagged.  3-view joints in set C: median 6.0 px, p75 8.7, p90 11.6
# (view_agreement.py, 2026-11-26); the triangulation's own cut is 15 px.
HIGH_RESID_PX = 10.0


def _nanmax(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmax(a, axis=axis)


def _devs(X, ok, med):
    """(T, n_bones) signed length deviation from the setup median, NaN where a bone end is missing."""
    out = np.full((len(X), len(BONES)), np.nan)
    for j, (a, b, k, _) in enumerate(BONES):
        m = ok[:, a] & ok[:, b]
        L = np.linalg.norm(X[:, a] - X[:, b], axis=1)
        out[m, j] = L[m] / med[k] - 1
    return out


def _jsonable(a, nd):
    a = np.round(np.asarray(a, float), nd).astype(object)
    a[~np.isfinite(np.asarray(a, float))] = None
    return a.tolist()


def rep_skeleton(S, stub):
    loc, why = _locate(S, stub)
    if loc is None:
        return {'available': False, 'why': why}
    summ, st, rec, cams, s = loc
    cs, X, ok, start, lag, used, err, vis, conf, xy_ = _triangulate(S, stub, loc)
    T = len(X)
    med = st['session_bone_median']
    # reprojection into each camera, as a fraction of the 1920x1080 image (the videos are scaled renders of it)
    P2 = np.full((3, T, 25, 2), np.nan)
    flat = np.where(ok[..., None], X, np.nan).reshape(-1, 3)
    for c in range(3):
        uv = cs.project(np.nan_to_num(flat), c).reshape(T, 25, 2)
        uv[~ok] = np.nan
        P2[c] = uv / np.array(cs.image_size, float)
    # side view: floor frame (z up), horizontal axis = camera 1's line of sight projected onto the floor
    Rw, org = cams[f'R_world_to_floor_s{s}'], cams[f'floor_origin_s{s}']
    Xf = (X - org) @ Rw.T
    axis = Rw @ cs.R[0][2]
    d = np.array([axis[0], axis[1], 0.0])
    d /= np.linalg.norm(d)
    side = np.stack([Xf @ d, Xf[..., 2]], axis=-1)            # (T,25,2): depth away from camera 1, height
    side[~ok] = np.nan
    cam1_depth = float((((-org) @ Rw.T) @ d))                  # camera 1 centre (world origin) along that axis
    dev = _devs(X, ok, med)
    # the rep gate: each long bone's median over the rep (both sides) vs the setup median
    rep_gate = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k in cc.BONE_KEYS:
            cols = [j for j, bb in enumerate(BONES) if bb[2] == k]
            rep_gate[k] = float(np.nanmedian(dev[:, cols])) if np.isfinite(dev[:, cols]).any() else None
    return {
        'available': True, 'setup': s, 'T': T,
        'start': [int(x) for x in start],          # data row i == camera c video frame start[c] + i
        'lag': list(lag),
        'bones': [[a, b, k, side_] for a, b, k, side_ in BONES],
        'long_bones': LONG,
        'session_bone_median': med,
        'session_long_cv': st.get('long_bone_cv_pct'),
        'rep_bone_dev_gate': B.GATES['max_rep_bone_dev'], 'joint_bone_dev_gate': B.GATES['joint_bone_dev'],
        'rep_dev': rep_gate,
        'proj': _jsonable(P2, 4), 'side': _jsonable(side, 3), 'cam1_depth': cam1_depth,
        'dev': _jsonable(dev, 3),
        # camera agreement per joint: views the triangulation kept, and the worst reprojection residual among them (px,
        # 1080p).  3 views agreeing within a few px = position checked from three directions; 2 views = no redundancy
        # against an error along the line joining those two cameras.
        'views': np.where(ok, used.sum(axis=1), 0).astype(int).tolist(),
        'resid': _jsonable(np.where(ok, _nanmax(err, axis=1), np.nan), 1),
        'high_resid_px': HIGH_RESID_PX,
        # per joint, per camera: 1 = used, 2 = dropped (confident but disagreed with the other two by > 15 px),
        # 3 = low confidence (OpenPose gave a point, conf <= the 0.5 threshold -- the stick-figure render still draws
        # it), 0 = no point in this camera's synced track (OpenPose missed it, or the person track lost the child)
        # -- only for triangulated joints, else all 0
        'cam_state': np.where(ok[:, None, :], np.where(used, 1, np.where(vis, 2, np.where(
            (conf > 0) & ~((xy_[..., 0] == 0) & (xy_[..., 1] == 0)), 3, 0))), 0).transpose(0, 2, 1).astype(int).tolist(),
        'conf_thresh': B.GATES['conf_thresh'],
    }


def setup_bone_stats(S, stub):
    """Per-bone CV over every rep of this stub's camera setup (all triangulated frames, no gate)."""
    loc, why = _locate(S, stub)
    if loc is None:
        return {'available': False, 'why': why}
    summ, st, rec, cams, s = loc
    key = (S, stub.split('_')[0], s)
    with _lock:
        if key in _setup_stats:
            return _setup_stats[key]
    lens = {(k, sd): [] for _, _, k, sd in BONES}
    stubs = [str(x) for x in cams[f'stubs_s{s}']]
    for sb in stubs:
        l2, _ = _locate(S, sb)
        if l2 is None:
            continue
        _, X, ok, *_ = _triangulate(S, sb, l2)
        for a, b, k, sd in BONES:
            m = ok[:, a] & ok[:, b]
            lens[(k, sd)].append(np.linalg.norm(X[m, a] - X[m, b], axis=1))
    cv = lambda x: 100 * np.std(x) / np.mean(x) if len(x) > 2 else None
    rcv = lambda x: 100 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.median(x) if len(x) > 2 else None
    rows = []
    for k in cc.BONE_KEYS:
        sides = sorted({sd for (kk, sd) in lens if kk == k})
        per = {sd: np.concatenate(lens[(k, sd)]) if lens[(k, sd)] else np.zeros(0) for sd in sides}
        pooled = np.concatenate(list(per.values()))
        row = dict(bone=k, long=k in LONG, n=int(len(pooled)), pooled_cv=cv(pooled), robust_cv=rcv(pooled),
                   L_cv=cv(per['L']) if 'L' in per else None, R_cv=cv(per['R']) if 'R' in per else None,
                   LR_diff=(100 * (np.median(per['L']) / np.median(per['R']) - 1)
                            if 'L' in per and len(per['L']) > 2 and len(per['R']) > 2 else None))
        rows.append({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in row.items()})
    longs = [r['pooled_cv'] for r in rows if r['long'] and r['pooled_cv'] is not None]
    rlongs = [r['robust_cv'] for r in rows if r['long'] and r['robust_cv'] is not None]
    # The session gate's CV is measured on the calibration frames (clean, spread-selected, bundle-adjusted), which
    # are not saved.  Over ALL rep frames a few outlier frames inflate the plain CV (C220 usable: 10.3 vs session
    # 4.4); the robust CV (1.4826 MAD / median) ignores them and lands near the session figure (5.8; B356 9.1 vs
    # 11.2) -- so it is the headline, plain CV the secondary.
    out = {'available': True, 'setup': s, 'n_reps': len(stubs), 'rows': rows,
           'median_long_pooled_cv': round(float(np.median(longs)), 2) if longs else None,
           'median_long_robust_cv': round(float(np.median(rlongs)), 2) if rlongs else None,
           'session_long_cv': st.get('long_bone_cv_pct'), 'gate': B.GATES['max_long_bone_cv_pct']}
    with _lock:
        _setup_stats[key] = out
    return out


if __name__ == '__main__':          # quick check against a saved GT3D rep
    S, stub = sys.argv[1], sys.argv[2]
    loc, why = _locate(S, stub)
    cs, X, ok, start, lag, *_ = _triangulate(S, stub, loc)
    p = f'{ROOT}/{S}/GT3D/{stub.split("_")[0]}/{stub}.npz'
    if os.path.exists(p):
        z = np.load(p)
        Xs, oks = z['X_body25_cam'][0], z['valid_body25']
        both = ok & oks
        print('saved rep: valid masks equal', bool((ok == oks).all()),
              ' max |dX| mm', float(np.nanmax(np.abs(X[both] - Xs[both])) * 1000))
    r = rep_skeleton(S, stub)
    print({k: r[k] for k in ('T', 'start', 'lag', 'rep_dev', 'session_long_cv')})
    st = setup_bone_stats(S, stub)
    print('setup median long CV: robust', st['median_long_robust_cv'], 'plain', st['median_long_pooled_cv'],
          '| session', st['session_long_cv'])
    for row in st['rows']:
        print(row)
