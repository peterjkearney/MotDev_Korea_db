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
     stationary camera's offset stays put; a moving one wanders.  Reps whose offsets wander > MOVE_PX are dropped.
  2. SEQUENTIAL segments WITHIN each action (rep order is trusted there; order across actions is not): seed a setup with the first stationary rep; test each next rep
     against the setup's cameras (person choice + sync refined first) by its per-camera whole-body offset; if every
     camera is within AGREE_PX, add it and refit; if not, look at the following rep -- if that agrees, the first was an
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
PER_REP_CAP = 60         # candidate frames per rep (before select_spread)
SETUP_FRAMES = 300       # frames per setup calibration
MIN_FRAMES_REP = 20      # a single-rep calibration needs this many frames
OUTLIER_PX = 15.0
N_WIN = 8
MOVE_PX = 25.0           # a camera's windowed offset wandering more than this = moving   (to tune)
AGREE_PX = 8.0           # a rep's whole-body offset per camera within this = same setup   (to tune)
LOO_CONF = 0.5
BODY = list(range(15))


# ---------------------------------------------------------------- calibration ----

def candidates_loose(reps):
    xy_out, vis_out, conf_out, meta = [], [], [], []
    for r in reps:
        xy, conf, _ = B.rep_span(r)
        if not len(xy):
            continue
        vis = (conf > CONF) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        v3 = vis.all(axis=1)                                   # (T, 25) seen confidently by all 3
        good = np.where(v3.sum(1) >= MIN_JOINTS3)[0]
        if not len(good):
            continue
        keep = good[::max(1, int(np.ceil(len(good) / PER_REP_CAP)))]
        xy_out.append(xy[keep]); conf_out.append(conf[keep])
        vis_out.append(np.repeat(v3[keep][:, None, :], 3, axis=1))
        meta.extend((r['stub'].split('_')[0], r['stub'], int(t)) for t in keep)
    if not meta:
        raise cc.InsufficientData('no frame with enough 3-view confident joints')
    return cc.Observations(xy=np.concatenate(xy_out).astype(float), vis=np.concatenate(vis_out),
                           conf=np.concatenate(conf_out).astype(float), meta=meta, subjects=[meta[0][0]],
                           subject_of_frame=np.zeros(len(meta), int))


def _fit(obs, cams0=None):
    fb = cc.focal_bounds_for_camera('sony-rx100', B.IMAGE_SIZE[0])
    if cams0 is None:
        f0 = cc.init_focal_from_spec(fb, 3, 'wide')
        base12 = float(np.linalg.norm(cc.DEFAULT_LAYOUT[1] - cc.DEFAULT_LAYOUT[0]))
        cams0 = cc.init_from_essential(obs, f0, base12, image_size=B.IMAGE_SIZE, verbose=False)
    opts = cc.BAOptions(fit_k1=False, f_bounds=fb, use_bone_prior=False, verbose=0, max_nfev=200)
    return cc.solve(cams0, obs, opts=opts, sigma_px=4.0, verbose=False)


def calibrate_loose(reps, n_frames=SETUP_FRAMES, min_frames=MIN_FRAMES_REP):
    cand = candidates_loose(reps)
    obs = cc.select_spread(cand, n_frames=n_frames, min_frames=min_frames, min_views=3)
    cams, L, X, _, prob = _fit(obs)
    # drop 3-view points any camera disagrees with, refit from the first fit
    X3, _ = cc.triangulate(cams, obs.xy, obs.vis)
    e = cc.reprojection_per_view(cams, X3, obs.xy, obs.vis)            # (F,3,25)
    bad = (np.nan_to_num(e, nan=0) > OUTLIER_PX).any(axis=1)          # (F,25)
    n_drop = int((bad & obs.vis[:, 0]).sum())
    if n_drop:
        obs.vis = obs.vis & ~bad[:, None, :]
        cams, L, X, _, prob = _fit(obs, cams)
    return _finish(obs, cams, L, X, prob, rule=f'loose (dropped {n_drop} pts)')


def calibrate_strict(reps):
    args = SimpleNamespace(calib_stride=6, calib_max_frames=800, calib_corr=6000)
    return B.calibrate('x', reps, args)


def _finish(obs, cams, L, X, prob, rule):
    normal, offset, inl, _ = cc.fit_floor(X, prob, obs, thresh=0.04)
    cams, X, L, _, s, info = cc.rescale_to_convergence(cams, X, L, prob, normal, offset, B.ANCHOR_DISTANCE, 0,
                                                       'ground', verbose=False)
    return dict(obs=obs, cams=cams, X=X, L=L, prob=prob, normal=normal, rule=rule, offset=offset * s,
                floor_inlier_frac=float(inl.mean()), axis_miss_m=info['miss'])


def quality(cal):
    q = B.setup_quality(cal)
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
        edges = np.linspace(0, len(d), N_WIN + 1).astype(int)
        wins = [_med(d[a:z], 15)[0] for a, z in zip(edges[:-1], edges[1:])]
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


# ---------------------------------------------------------------- the walk ----

def prepare(rep, cams):
    """Person choice and sync against these cameras (in place); returns the rep's sync error."""
    B.choose_person(cams, rep)
    _, e = B.refine_sync(cams, rep)
    return e


def test(rep, cal):
    trial = dict(rep)
    err = prepare(trial, cal['cams'])
    offs = offsets(cal['cams'], trial)
    worst = np.nanmax([o['size'] for o in offs]) if any(np.isfinite(o['size']) for o in offs) else np.inf
    return worst <= AGREE_PX, worst, err, offs, trial


def fmt_offs(offs):
    return ' '.join(f"c{c + 1}({o['b'][0]:+.0f},{o['b'][1]:+.0f})" if np.isfinite(o['size']) else f'c{c + 1}(-)'
                    for c, o in enumerate(offs))


def run(S, subject, verbose=True):
    t0 = time.time()
    data = f'{B.KOREA_ROOT}/{S}/Aligned'
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
        ok, worst, err, offs, trial = test(r, cur['cal'])
        log(f"  {r['stub']} vs setup {len(setups) + 1}: worst offset {worst:5.1f} px, err {err:4.1f} px  "
            f"{fmt_offs(offs)}  -> {'agree' if ok else 'DISAGREE'}")
        if ok:
            if pending is not None:
                cur['outliers'].append(pending['stub'])
                report['reps'][pending['stub']]['walk'] = 'outlier (next rep agreed)'
                log(f"    {pending['stub']} set aside as an outlier")
                pending = None
            add(cur, trial)
        elif pending is None:
            pending = r
        else:                                            # two in a row: the cameras moved
            setups.append(cur)
            log(f"  -> close setup {len(setups)} ({len(cur['members'])} reps); new setup from {pending['stub']}")
            cur = new_setup(pending)
            pending = None
            ok, worst, err, offs, trial = test(r, cur['cal'])
            log(f"  {r['stub']} vs setup {len(setups) + 1}: worst offset {worst:5.1f} px  {fmt_offs(offs)}  "
                f"-> {'agree' if ok else 'DISAGREE'}")
            if ok:
                add(cur, trial)
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
                w_ab = [test(r, a['cal'])[1] for r in b['members']]
                w_ba = [test(r, b['cal'])[1] for r in a['members']]
                log(f"  setups {i + 1}+{j + 1}: median worst offset {np.median(w_ab):.1f} / {np.median(w_ba):.1f} px")
                if np.median(w_ab) <= AGREE_PX and np.median(w_ba) <= AGREE_PX:
                    a['members'] += b['members']; a['outliers'] += b['outliers']
                    a['cal'] = calibrate_loose(a['members'])
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
    a = ap.parse_args()
    warnings.simplefilter('ignore', RuntimeWarning)
    rep = run(a.set, a.subject)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(B.jsonable(rep), f, indent=1)


if __name__ == '__main__':
    main()
