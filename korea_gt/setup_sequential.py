#!/usr/bin/env python3
"""setup_sequential.py -- PROTOTYPE: find a session's camera setups rep by rep, with a looser calibration-frame rule.

Not wired into build_child_gt; reads Aligned/ and writes nothing but an optional JSON report.

The build's find_setups calibrates the whole session, then (if that fails) each ACTION, and merges actions whose
cameras explain each other's reps by error size.  B015 showed three failure modes: a camera moving during reps
(1_1, 1_2), a move part-way through an action (between 1_2 and 1_3), and a ~30 px move (action 3) merged because the
error-size tolerance is loose.  And its strict calibration-frame rule (>= 20 of 25 joints conf > 0.6 in all 3 views)
found no frame at all in action 3.

Here:
  0. LOOSE calibration rule: a frame counts if >= MIN_JOINTS3 joints are conf > 0.6 in ALL THREE cameras; only those
     3-view points are used; after a first fit, points with a residual > OUTLIER_PX in any view are dropped and the
     cameras refitted.  At most PER_REP_CAP candidate frames per rep, so no rep dominates a setup.
  1. MOVING-CAMERA check, per rep, without any reference: calibrate from the rep alone, then follow each camera's
     offset (its point minus the other two cameras' triangulation, median per window) through N_WIN windows.  A
     stationary camera's offset stays put; a moving one wanders.  Slow joints only (SLOW_PX), so sync error
     during fast movement can't pass for a moving camera.  Reps whose offsets wander > MOVE_PX are dropped.
  2. SEQUENTIAL segments WITHIN each action (rep order is trusted there; order across actions is not): seed a setup with the first stationary rep; test each next rep
     by calibrating it together with the setup's reps (person choice + sync refined first): if every rep's
     per-camera whole-body offset is within AGREE_PX under the joint cameras, add it; if not, look at the following rep -- if that agrees, the first was an
     outlier (set aside); if it doesn't either, close the setup and start a new one from the first disagreeing rep.
  3. JOIN segments from different actions whose cameras agree, in any order, refit.
  4. Report each setup's quality (the build's session gates), each rep's error and offsets under its setup, and a
     strict-vs-loose comparison on the largest setup: held-out-camera error (each camera predicted from the other
     two) over all its reps' frames.

    python korea_gt/setup_sequential.py --set B --subject B015 [--json out.json]
"""
import argparse, json, os, sys, time, warnings
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_child_gt as B     # noqa: E402
import camcal as cc            # noqa: E402

CONF = 0.6
MIN_JOINTS3 = 6          # joints conf > CONF in all 3 cameras for a frame to count
PER_REP_CAP = 60         # candidate frames per rep at least; scales up to SETUP_FRAMES / n_reps, so a small setup
                         # (one rep) uses nearly all its frames while a big one stays balanced across reps
SLOW_CAL_PX = 5.0        # calibration points only from joints moving slower than this (px/frame) in ALL 3 cameras:
                         # sync error scales with speed (a frame off at 30 px/frame = 30 px); looser than SLOW_PX
                         # so the points still spread over the image
SETUP_FRAMES = 300       # frames per setup calibration
MIN_FRAMES_REP = 20      # a single-rep calibration needs this many frames
OUTLIER_PX = 15.0
N_WIN = 8
MOVE_PX = 25.0           # a camera's windowed offset wandering more than this = moving   (to tune)
AGREE_PX = 8.0           # a rep's whole-body offset per camera within this = same setup   (to tune)
LOO_CONF = 0.5
SLOW_PX = 3.0            # windowed offsets use only joints moving slower than this (px/frame) in that camera: in a
                         # jump's take-off every joint moves one way, so a frame of sync error reads as an offset
                         # (B015_GMS_3_4: last eighth at 34 px/frame gave a false 'moving camera')
BODY = list(range(15))
ALIGNED = 'Aligned'      # folder under {root}/{set} holding align_cameras' reps (--aligned; GT3D_v5 uses Aligned_v5)
ALIGNED_PATH = None      # full path to that folder, when the caller (build_child_gt --data) knows it


# ---------------------------------------------------------------- calibration ----

def image_speed(xy, conf):
    """(T, 3, 25) px/frame of every joint in every camera (central difference; NaN where not seen)."""
    p = np.where(((conf > LOO_CONF) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0)))[..., None], xy, np.nan)
    v = np.full(p.shape[:3], np.nan)
    v[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=-1) / 2
    return v


def candidates_loose(reps, slow_px=SLOW_CAL_PX, cap=None):
    xy_out, vis_out, conf_out, meta = [], [], [], []
    cap = cap or max(PER_REP_CAP, SETUP_FRAMES // max(len(reps), 1))
    for r in reps:
        xy, conf, _ = B.rep_span(r)
        if not len(xy):
            continue
        vis = (conf > CONF) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        v3 = vis.all(axis=1)                                   # (T, 25) seen confidently by all 3
        if slow_px:
            v3 &= np.nan_to_num(image_speed(xy, conf), nan=np.inf).max(axis=1) < slow_px
        good = np.where(v3.sum(1) >= MIN_JOINTS3)[0]
        if not len(good):
            continue
        keep = good[::max(1, int(np.ceil(len(good) / cap)))]
        xy_out.append(xy[keep]); conf_out.append(conf[keep])
        vis_out.append(np.repeat(v3[keep][:, None, :], 3, axis=1))
        meta.extend((r['stub'].split('_')[0], r['stub'], int(t)) for t in keep)
    if not meta:
        raise cc.InsufficientData('no frame with enough 3-view confident joints')
    return cc.Observations(xy=np.concatenate(xy_out).astype(float), vis=np.concatenate(vis_out),
                           conf=np.concatenate(conf_out).astype(float), meta=meta, subjects=[meta[0][0]],
                           subject_of_frame=np.zeros(len(meta), int))


F_BORROW_TOL = 0.05     # a setup borrowing focal lengths may move them this fraction (a reframe could touch the zoom)


def _fit(obs, cams0=None, f_ref=None):
    """Bundle adjustment.  f_ref (3,): focal lengths borrowed from the session's main setup -- the same cameras,
    so a small or moved setup only has to find where they stand, not their lenses too (one jump can't pin a focal
    length: single-rep rolls of 17-83 deg on B015)."""
    fb = cc.focal_bounds_for_camera('sony-rx100', B.IMAGE_SIZE[0])
    if f_ref is not None:
        f_ref = np.asarray(f_ref, float)
        fb = (f_ref * (1 - F_BORROW_TOL), f_ref * (1 + F_BORROW_TOL))
    if cams0 is None:
        f0 = cc.init_focal_from_spec(fb, 3, 'wide') if f_ref is None else f_ref.copy()
        base12 = float(np.linalg.norm(cc.DEFAULT_LAYOUT[1] - cc.DEFAULT_LAYOUT[0]))
        cams0 = cc.init_from_essential(obs, f0, base12, image_size=B.IMAGE_SIZE, verbose=False)
    opts = cc.BAOptions(fit_k1=False, f_bounds=fb, use_bone_prior=False, verbose=0, max_nfev=200)
    return cc.solve(cams0, obs, opts=opts, sigma_px=4.0, verbose=False)


def calibrate_loose(reps, n_frames=SETUP_FRAMES, min_frames=MIN_FRAMES_REP, f_ref=None):
    cand = candidates_loose(reps)
    obs = cc.select_spread(cand, n_frames=n_frames, min_frames=min_frames, min_views=3)
    cams, L, X, _, prob = _fit(obs, f_ref=f_ref)
    # drop 3-view points any camera disagrees with, refit from the first fit
    X3, _ = cc.triangulate(cams, obs.xy, obs.vis)
    e = cc.reprojection_per_view(cams, X3, obs.xy, obs.vis)            # (F,3,25)
    bad = (np.nan_to_num(e, nan=0) > OUTLIER_PX).any(axis=1)          # (F,25)
    n_drop = int((bad & obs.vis[:, 0]).sum())
    if n_drop:
        obs.vis = obs.vis & ~bad[:, None, :]
        cams, L, X, _, prob = _fit(obs, cams, f_ref=f_ref)
    cal = _finish(obs, cams, L, X, prob, rule=f'loose (dropped {n_drop} pts)' + (', borrowed f' if f_ref is not None else ''))
    cal['reps'] = reps                                   # for quality()'s fixed point set
    return cal


def calibrate_strict(reps):
    args = SimpleNamespace(calib_stride=6, calib_max_frames=800, calib_corr=6000)
    return B.calibrate('x', reps, args)


def _finish(obs, cams, L, X, prob, rule):
    normal, offset, inl, _ = cc.fit_floor(X, prob, obs, thresh=0.04)
    cams, X, L, _, s, info = cc.rescale_to_convergence(cams, X, L, prob, normal, offset, B.ANCHOR_DISTANCE, 0,
                                                       'ground', verbose=False)
    return dict(obs=obs, cams=cams, X=X, L=L, prob=prob, normal=normal, rule=rule, offset=offset * s,
                floor_inlier_frac=float(inl.mean()), axis_miss_m=info['miss'])


FIXED_CAP = 400          # frames per rep for the fixed point set


def fixed_points(reps):
    """Every confident 3-view point of these reps (conf > CONF in all 3 cameras, any speed) -- a set the choice of
    calibration frames can't change."""
    return candidates_loose(reps, slow_px=None, cap=FIXED_CAP)


def quality(cal):
    """The build's session gates for this calibration, plus the reprojection-vs-noise-floor ratio measured on a FIXED
    point set.  The build measures the floor on the calibration's own points, so choosing cleaner calibration points
    (slow joints) lowers the floor while the fit stays put and the ratio fails for the wrong reason (B015 setup 1:
    1.45 -> 1.7 with identical held-out error).  Here both the fit (median reprojection per camera, worst camera)
    and the floor (epipolar_floor) are measured on all confident 3-view points of the setup's reps."""
    q = B.setup_quality(cal)
    q['ratio_calib_pts'] = q.pop('ratio')
    if cal.get('reps'):
        fx = fixed_points(cal['reps'])
        X3, _ = cc.triangulate(cal['cams'], fx.xy, fx.vis)
        e = cc.reprojection_per_view(cal['cams'], X3, fx.xy, fx.vis)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            per_cam = [float(np.nanmedian(e[:, c][fx.vis[:, c]])) for c in range(3)]
        fit_fx = max(per_cam)
        floor = B.epipolar_floor(fx)
        q['fit_fixed_px'], q['floor_fixed_px'] = fit_fx, floor
        q['fit_fixed_per_cam'] = [round(x, 3) for x in per_cam]
        q['ratio'] = fit_fx / floor if floor > 0 else np.nan
        q['ok'] = bool(q['ratio'] <= B.GATES['max_reproj_over_epipolar_fixed'] and q['roll'] <= B.GATES['max_abs_roll_deg']
                       and q['cv'] <= B.GATES['max_long_bone_cv_pct'])
    q['f'] = [round(float(x)) for x in cal['cams'].f]
    rp = cc.reprojection_errors(cal['cams'], cal['prob'], cal['X'])
    q['calib_median_px'] = [round(rp[c]['median'], 2) for c in range(3)]
    q['n_frames'] = int(cal['obs'].xy.shape[0])
    q['rule'] = cal['rule']
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in q.items()}


# ---------------------------------------------------------------- offsets ----

def loo(cams, xy, conf, c):
    """(T, 15, 2): camera c's point minus the projection of the other two cameras' triangulation (agreeing pair)."""
    T = len(xy)
    present = (conf > LOO_CONF) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    o = [k for k in range(3) if k != c]
    v = np.zeros_like(present)
    v[:, o] = present[:, o]
    Xp, okp = cc.triangulate(cams, xy, v)
    okp &= v[:, o[0]] & v[:, o[1]]
    e = cc.reprojection_per_view(cams, Xp, xy, v)
    okp &= (np.nan_to_num(e[:, o[0]], nan=1e9) <= OUTLIER_PX) & (np.nan_to_num(e[:, o[1]], nan=1e9) <= OUTLIER_PX)
    uv = cams.project(np.nan_to_num(Xp.reshape(-1, 3)), c).reshape(T, 25, 2)
    d = xy[:, c] - uv
    d[~(okp & present[:, c])] = np.nan
    return d[:, BODY]


def _med(d, min_n=30):
    d = d.reshape(-1, 2)
    d = d[np.isfinite(d).all(1)]
    return (np.median(d, axis=0), len(d)) if len(d) >= min_n else (np.array([np.nan, np.nan]), len(d))


def offsets(cams, rep):
    """Per camera: whole-rep offset (x, y), its size, scatter, and the windowed offsets."""
    xy, conf, _ = B.rep_span(rep)
    out = []
    for c in range(3):
        d = loo(cams, xy, conf, c)
        b, n = _med(d)
        flat = d.reshape(-1, 2); flat = flat[np.isfinite(flat).all(1)]
        scatter = float(np.median(np.linalg.norm(flat - b, axis=1))) if n >= 30 and np.isfinite(b).all() else np.nan
        p = np.where(((xy[:, c, BODY] == 0).all(-1) | (conf[:, c, BODY] <= LOO_CONF))[..., None], np.nan,
                     xy[:, c, BODY])
        speed = np.full(p.shape[:2], np.nan)
        speed[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=-1) / 2
        ds = np.where((speed < SLOW_PX)[..., None], d, np.nan)
        edges = np.linspace(0, len(d), N_WIN + 1).astype(int)
        wins = [_med(ds[a:z], 15)[0] for a, z in zip(edges[:-1], edges[1:])]
        out.append(dict(b=b, size=float(np.hypot(*b)) if np.isfinite(b).all() else np.nan, n=n, scatter=scatter,
                        wins=np.array(wins)))
    return out


def wander(offs):
    """Largest distance of any window's offset from the camera's whole-rep offset, per camera."""
    w = []
    for o in offs:
        dd = np.linalg.norm(o['wins'] - o['b'], axis=1)
        w.append(float(np.nanmax(dd)) if np.isfinite(dd).any() else np.nan)
    return w


# ---------------------------------------------------------------- bystanders ----

FEET = [11, 14, 19, 22]  # ankles, big toes: a standing person's feet stay put even while they gesture
BYST_MIN_SEEN = 0.5      # a bystander's feet must be seen in this share of the rep
BYST_STILL_PX = 8.0      # a bystander whose feet stay within this all rep: that camera did not move
BYST_MOVE_PX = 15.0      # a bystander shift at least this, in step with the child's offset drift: the camera moved
BYST_CORR = 0.6          # ... 'in step': correlation of the two window series


def bystander_windows(rep, c, start, T):
    """Each other person in camera c (align_cameras' tracks, not the child's): their feet's median position per
    window, relative to their whole-rep median -- (N_WIN, 2) arrays.  A camera turning shifts everyone in it."""
    path = os.path.join(ALIGNED_PATH or f"{B.KOREA_ROOT}/{rep['stub'][0]}/{ALIGNED}", f"{rep['stub']}.npz")
    if not os.path.exists(path):
        return []
    a = np.load(path)
    if f'tracks_xy{c + 1}' not in a:
        return []
    tx, ts = a[f'tracks_xy{c + 1}'], a[f'tracks_score{c + 1}']
    child = {int(a['track'][c])}
    if 'candidates' in a:
        child |= {int(row[1 + c]) for row in a['candidates']}
    out = []
    for k in range(len(tx)):
        if k in child or np.array_equal(tx[k], rep['raw_xy'][c]):
            continue
        xy, sc = tx[k][start:start + T], ts[k][start:start + T]
        if len(xy) < T:
            continue
        ok = (sc[:, FEET] > 0.5) & ~(xy[:, FEET] == 0).all(-1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            f = np.nanmean(np.where(ok[..., None], xy[:, FEET], np.nan), axis=1)
        if np.isfinite(f).all(1).mean() < BYST_MIN_SEEN:
            continue
        e = np.linspace(0, T, N_WIN + 1).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            w = np.array([np.nanmedian(f[a0:z], axis=0) for a0, z in zip(e[:-1], e[1:])])
            out.append(w - np.nanmedian(f, axis=0))
    return out


def moving_cameras(rep, offs):
    """Per camera: (moving?, how decided).  Bystanders first, the child's own offsets as the fallback.

    - a bystander shifting >= BYST_MOVE_PX in step with the child's offset drift (corr >= BYST_CORR) -> moving
    - else a bystander standing still (<= BYST_STILL_PX all rep) -> not moving, whatever the child's offsets say
    - else (no bystander, or bystanders who moved on their own) -> the child's wander > MOVE_PX
    """
    _, _, start = B.rep_span(rep)
    T = len(B.rep_span(rep)[0])
    wand = wander(offs)
    out = []
    for c in range(3):
        child = offs[c]['wins'] - offs[c]['b']
        bys = bystander_windows(rep, c, int(start[c]), T)
        verdict = None
        notes = []
        for bw in bys:
            amp = float(np.nanmax(np.linalg.norm(bw, axis=1)))
            m = np.isfinite(bw).all(1) & np.isfinite(child).all(1)
            corr = np.nan
            if m.sum() >= 4:
                x, y = child[m].ravel(), bw[m].ravel()
                if np.linalg.norm(x) > 0 and np.linalg.norm(y) > 0:
                    corr = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
            notes.append(f'bystander {amp:.0f}px corr {corr:+.2f}')
            if amp >= BYST_MOVE_PX and np.isfinite(corr) and corr >= BYST_CORR:
                verdict = (True, 'bystander moves with the child')
            elif amp <= BYST_STILL_PX and verdict is None:
                verdict = (False, 'bystander still')
        if verdict is None:
            mv = bool(np.isfinite(wand[c]) and wand[c] > MOVE_PX)
            verdict = (mv, f"child wander {wand[c]:.0f}px" + (' (no bystander)' if not bys else ' (bystanders moved on their own)'))
        out.append((verdict[0], verdict[1] + ('; ' + ', '.join(notes) if notes else '')))
    return out


# ---------------------------------------------------------------- the walk ----

def prepare(rep, cams):
    """Person choice and sync against these cameras (in place); returns the rep's sync error."""
    B.choose_person(cams, rep)
    _, e = B.refine_sync(cams, rep)
    return e


def _worst(cams, r):
    offs = offsets(cams, r)
    s = [o['size'] for o in offs]
    return (float(np.nanmax(s)) if np.isfinite(s).any() else np.inf), offs


def joint_test(members, cands, cal):
    """Can ONE set of cameras explain both the setup's reps and the candidates?

    Testing a candidate against the setup's cameras extrapolates them -- and a setup seeded by one jump is too weak
    for that (single-rep rolls of 17-83 deg on B015).  Instead calibrate members + candidates together and require
    every camera's whole-body offset, for every rep, within AGREE_PX.  A moved camera can't be fitted by one
    position for both groups, so one side keeps an offset.  The joint calibration becomes the setup's if they agree.
    Returns (ok, worst offset among candidates, worst among members, first candidate's offsets, prepared
    candidates, joint calibration).
    """
    trials = [dict(r) for r in cands]
    for t in trials:
        prepare(t, cal['cams'])
    try:
        cal2 = calibrate_loose(members + trials)
    except (cc.InsufficientData, RuntimeError, ValueError):
        return False, np.inf, np.inf, None, trials, None
    new = [_worst(cal2['cams'], t) for t in trials]
    old = [_worst(cal2['cams'], m)[0] for m in members]
    wn, wo = max(w for w, _ in new), max(old)
    return wn <= AGREE_PX and wo <= AGREE_PX, wn, wo, new[0][1], trials, cal2


def fmt_offs(offs):
    return ' '.join(f"c{c + 1}({o['b'][0]:+.0f},{o['b'][1]:+.0f})" if np.isfinite(o['size']) else f'c{c + 1}(-)'
                    for c, o in enumerate(offs))


def run(S, subject, verbose=True):
    t0 = time.time()
    data = f'{B.KOREA_ROOT}/{S}/{ALIGNED}'
    reps = [r for r in B.load_reps(data, subject)
            if r['stub'].split('_')[2] not in B.EXCLUDE_ACTIONS_BY_SET[S]]
    reps.sort(key=lambda r: (B._action(r['stub']), int(r['stub'].split('_')[3])))
    log = print if verbose else (lambda *a, **k: None)
    report = dict(subject=subject, reps={}, setups=[])

    # 1. moving-camera check, each rep on its own calibration
    log(f'== {subject}: {len(reps)} reps.  1. moving-camera check (own-rep calibration, {N_WIN} windows)')
    still = []
    for r in reps:
        rec = report['reps'].setdefault(r['stub'], {})
        try:
            cal = calibrate_loose([r])
        except (cc.InsufficientData, RuntimeError, ValueError) as e:
            rec['moving'] = None; rec['why'] = f'cannot self-calibrate: {str(e)[:60]}'
            log(f"  {r['stub']}: cannot self-calibrate ({str(e)[:50]}) -- kept for the walk")
            still.append(r)
            continue
        prepare(r, cal['cams'])
        offs = offsets(cal['cams'], r)
        w = wander(offs)
        moving = bool(np.nanmax(w) > MOVE_PX) if np.isfinite(w).any() else False
        rec.update(moving=moving, wander=[round(x, 1) for x in w], self_fit=quality(cal)['calib_median_px'])
        log(f"  {r['stub']}: wander c1-3 {' '.join(f'{x:5.1f}' for x in w)} px  self-fit "
            f"{rec['self_fit']}  {'MOVING -> dropped' if moving else ''}")
        if not moving:
            still.append(r)

    # 2. sequential walk
    log(f'\n2. sequential walk (agree: every camera offset <= {AGREE_PX} px)')
    setups, cur, pending = [], None, None

    def new_setup(r):
        cal = calibrate_loose([r])
        prepare(r, cal['cams'])
        return dict(members=[r], cal=cal, outliers=[])

    def add(st, r_trial):
        st['members'].append(r_trial)
        st['cal'] = calibrate_loose(st['members'])

    def close():
        nonlocal cur, pending
        if cur is not None:
            setups.append(cur)
            log(f"  -> close setup {len(setups)} ({len(cur['members'])} reps)")
        cur = None
        if pending is not None:                          # an action's last rep disagreed: its own one-rep segment
            try:
                setups.append(new_setup(pending))
                log(f"  -> {pending['stub']} kept as its own one-rep segment (setup {len(setups)})")
            except (cc.InsufficientData, RuntimeError, ValueError):
                report['reps'][pending['stub']]['walk'] = 'disagreed, cannot calibrate alone'
            pending = None

    prev_action = None
    for r in still:
        a = r['stub'].split('_')[2]
        if a != prev_action and prev_action is not None:
            log(f'  -- action {prev_action} ends: order across actions is unknown, so its segments close here')
            close()
        prev_action = a
        if cur is None:
            try:
                cur = new_setup(r)
                log(f"  seed setup {len(setups) + 1}: {r['stub']}")
            except (cc.InsufficientData, RuntimeError, ValueError):
                log(f"  {r['stub']}: cannot seed (too few frames) -- skipped")
                report['reps'][r['stub']]['walk'] = 'cannot seed'
            continue
        ok, wn, wo, offs, trials, cal2 = joint_test(cur['members'], [r], cur['cal'])
        log(f"  {r['stub']} + setup {len(setups) + 1} jointly: worst offset {wn:5.1f} px (rep) / {wo:5.1f} px "
            f"(setup's reps)  {fmt_offs(offs) if offs else ''}  -> {'agree' if ok else 'DISAGREE'}")
        if ok:
            if pending is not None:
                cur['outliers'].append(pending['stub'])
                report['reps'][pending['stub']]['walk'] = 'outlier (next rep agreed)'
                log(f"    {pending['stub']} set aside as an outlier")
                pending = None
            cur['members'].append(trials[0]); cur['cal'] = cal2
        elif pending is None:
            pending = r
        else:                                            # two in a row: the cameras moved
            setups.append(cur)
            log(f"  -> close setup {len(setups)} ({len(cur['members'])} reps); new setup from {pending['stub']}")
            cur = new_setup(pending)
            pending = None
            ok, wn, wo, offs, trials, cal2 = joint_test(cur['members'], [r], cur['cal'])
            log(f"  {r['stub']} + setup {len(setups) + 1} jointly: worst offset {wn:5.1f} / {wo:5.1f} px  "
                f"-> {'agree' if ok else 'DISAGREE'}")
            if ok:
                cur['members'].append(trials[0]); cur['cal'] = cal2
            else:
                pending = r
    close()

    # 3. join segments that agree, across actions in any order (actions may not have been filmed in numbered
    #    order; within an action, rep order is trusted and segments are never re-joined there)
    log('\n3. join segments from different actions whose cameras agree')
    merged = True
    while merged and len(setups) > 1:
        merged = False
        for i in range(len(setups)):
            for j in range(i + 1, len(setups)):
                a, b = setups[i], setups[j]
                acts = lambda st: {r['stub'].split('_')[2] for r in st['members']}
                if acts(a) & acts(b):                    # same action: split by the walk on purpose
                    continue
                ok, wn, wo, _, trials, cal2 = joint_test(a['members'], b['members'], a['cal'])
                log(f"  setups {i + 1}+{j + 1} jointly: worst offset {wn:.1f} / {wo:.1f} px")
                if ok:
                    a['members'] += trials; a['outliers'] += b['outliers']
                    a['cal'] = cal2
                    setups.pop(j); merged = True
                    log(f'    merged')
                    break
            if merged:
                break

    # 4. report
    log('\n4. setups')
    for k, st in enumerate(setups):
        q = quality(st['cal'])
        st_rep = dict(members=[r['stub'] for r in st['members']], outliers=st['outliers'], quality=q, reps={})
        log(f"  setup {k + 1}: {', '.join(m.split('_', 2)[2] for m in st_rep['members'])}"
            f"{'  outliers ' + ', '.join(st['outliers']) if st['outliers'] else ''}")
        log(f"    {q}")
        for r in st['members']:
            trial = dict(r)
            err = prepare(trial, st['cal']['cams'])
            offs = offsets(st['cal']['cams'], trial)
            xy, conf, _ = B.rep_span(trial)
            vis = (conf > B.GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
            _, _, _, e = cc.triangulate_robust(st['cal']['cams'], xy, vis, B.GATES['reproj_thresh_px'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = [round(float(np.nanmedian(e[:, c])), 2) for c in range(3)]
            ratio = max(m / c for m, c in zip(med, q['calib_median_px']))
            st_rep['reps'][r['stub']] = dict(median_reproj_px=med, ratio=round(ratio, 2),
                                             offsets=[round(o['size'], 1) for o in offs])
            log(f"    {r['stub']}: reproj {med} (ratio {ratio:.2f}, worst {max(med):.1f} px)  {fmt_offs(offs)}")
        report['setups'].append(st_rep)

    # strict vs loose on the largest setup: held-out-camera error over its reps' frames
    big = max(setups, key=lambda s: len(s['members']))
    log(f"\n5. strict vs loose calibration on the largest setup ({len(big['members'])} reps): "
        f"held-out-camera error, px (each camera predicted from the other two)")
    cmp = {}
    for name, fn in (('strict', calibrate_strict), ('loose', calibrate_loose)):
        try:
            cal = fn([dict(r) for r in big['members']])
        except (cc.InsufficientData, RuntimeError, ValueError) as e:
            log(f'  {name}: cannot calibrate ({str(e)[:60]})'); continue
        per_action = {}
        for r in big['members']:
            trial = dict(r); prepare(trial, cal['cams'])
            xy, conf, _ = B.rep_span(trial)
            for c in range(3):
                d = loo(cal['cams'], xy, conf, c).reshape(-1, 2)
                d = d[np.isfinite(d).all(1)]
                per_action.setdefault(r['stub'].split('_')[2], []).append(np.linalg.norm(d, axis=1))
        q = quality(cal)
        res = {a: round(float(np.median(np.concatenate(v))), 2) for a, v in per_action.items()}
        cmp[name] = dict(quality=q, heldout_by_action=res,
                         heldout_all=round(float(np.median(np.concatenate(sum(per_action.values(), [])))), 2))
        log(f"  {name:6s}: frames {q['n_frames']}, calib median {q['calib_median_px']}, roll {q['roll']}, "
            f"bone CV {q['cv']}%  | held-out median {cmp[name]['heldout_all']} px, by action {res}")
    report['strict_vs_loose'] = cmp
    report['seconds'] = round(time.time() - t0)
    log(f'\n({report["seconds"]} s)')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='B')
    ap.add_argument('--subject', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--aligned', default=ALIGNED, help='folder under {root}/{set} with the aligned reps')
    a = ap.parse_args()
    globals()['ALIGNED'] = a.aligned
    warnings.simplefilter('ignore', RuntimeWarning)
    rep = run(a.set, a.subject)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(B.jsonable(rep), f, indent=1)


if __name__ == '__main__':
    main()
