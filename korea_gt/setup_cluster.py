#!/usr/bin/env python3
"""setup_cluster.py -- PROTOTYPE: find a session's camera setups by clustering reps on their camera offsets.

The cheaper alternative to setup_sequential.py (one calibration per added rep).  Shares its calibration rule
(>= 6 joints conf > 0.6 in all 3 cameras, outlier points dropped and refitted) and its offset measure (each camera's
point minus the other two cameras' triangulation; median over the rep = its whole-body offset).

  1. Calibrate ALL reps together; each rep's 6-number offset (x, y per camera) under that compromise calibration.
  2. Within each action (rep order trusted), cut the reps into runs where neighbours' offsets differ by more than
     CLUSTER_PX; group the runs ACROSS actions, in any order (action order is not trusted), by average linkage.
  3. Loop: calibrate each group on its own; give every rep to the group whose cameras explain it best (worst camera
     offset smallest, and <= AGREE_PX to join; a rep stays in its own multi-rep group up to KEEP_PX); a group whose own cameras leave members beyond AGREE_PX is split, those
     members re-clustered; reps no group explains are clustered among themselves into new groups.  Within an
     action a camera can't go back: an A..B..A label sequence sets the B run aside as outliers.  Until no rep moves
     (or MAX_PASSES).
  4. Moving-camera check against each rep's group cameras (bystanders where visible, else the child's own offsets on
     slow joints); moving reps dropped, groups refitted -- except a rep with reps of its own setup both before and
     after it in the same action (the camera can't have moved and come back).  The largest group (>= MAIN_MIN_REPS)
     calibrates its focal lengths; every other group borrows them (+-5%).
  5. Report like setup_sequential, plus any action whose reps hop between groups (rep order within an action is
     trusted, so one action should split at most at clean cut points).

    python korea_gt/setup_cluster.py --set B --subject B015 [--json out.json]
"""
import argparse, json, os, sys, time, warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_child_gt as B      # noqa: E402
import camcal as cc             # noqa: E402
import setup_sequential as Q    # noqa: E402

CLUSTER_PX = 15.0        # offset-vector distance for the first grouping (and re-grouping of leftovers)
AGREE_PX = Q.AGREE_PX   # to JOIN a group
MAIN_MIN_REPS = 3       # the group lending its focal lengths must have at least this many reps
KEEP_PX = 12.0          # to STAY in the group it is in (hysteresis: B015 action 3 reps at 8.3-8.7 px against their own
                        # 4-rep group were thrown out at 8 px, regrouped as one-rep groups -- which always fit
                        # themselves -- and the action fell apart)
MOVE_PX = Q.MOVE_PX
MAX_PASSES = 6
CAL_ERRORS = (cc.InsufficientData, RuntimeError, ValueError, np.linalg.LinAlgError)


def offset_vec(cams, r):
    offs = Q.offsets(cams, r)
    v = np.concatenate([o['b'] for o in offs])
    return v, offs


def group(vecs, thresh=CLUSTER_PX):
    """Average-linkage clusters of offset vectors (NaN cameras ignored pairwise). Returns labels 0..k-1."""
    n = len(vecs)
    if n == 1:
        return np.zeros(1, int)
    V = np.array(vecs)
    D = []
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(V[i]) & np.isfinite(V[j])
            D.append(np.linalg.norm(V[i][m] - V[j][m]) * np.sqrt(6 / max(m.sum(), 1)) if m.any() else 1e3)
    return fcluster(linkage(np.array(D), 'average'), thresh, 'distance') - 1


def action_of(stub):
    return stub.split('_')[2]


def vdist(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.linalg.norm(a[m] - b[m]) * np.sqrt(6 / max(m.sum(), 1)) if m.any() else 1e3


def action_runs(stubs, vecs, thresh=CLUSTER_PX):
    """Cut each action's reps (in rep order) into runs wherever neighbouring reps' offsets differ by > thresh."""
    runs = []
    for a in sorted({action_of(s) for s in stubs}, key=lambda a: B._action(f'x_x_{a}')):
        cur = []
        for s in [s for s in stubs if action_of(s) == a]:
            if cur and vdist(vecs[cur[-1]], vecs[s]) > thresh:
                runs.append(cur); cur = []
            cur.append(s)
        runs.append(cur)
    return runs


def enforce_order(assign, stubs):
    """Within an action a camera can't go back: in a label sequence A..B..A the shorter run in between is set aside.
    `assign` maps stub -> group (or None).  Returns (assign, set-aside stubs)."""
    out, aside = dict(assign), []
    for a in {action_of(s) for s in stubs}:
        seq = [s for s in stubs if action_of(s) == a and out.get(s) is not None]
        while True:
            runs = []
            for s in seq:
                if runs and runs[-1][0] == out[s]:
                    runs[-1][1].append(s)
                else:
                    runs.append([out[s], [s]])
            labels = [k for k, _ in runs]
            bad = [i for i in range(1, len(runs) - 1) if labels[i] != labels[i - 1] and labels[i - 1] in labels[i + 1:]]
            if not bad:
                break
            i = min(bad, key=lambda i: len(runs[i][1]))
            for s in runs[i][1]:
                out[s] = None; aside.append(s)
            seq = [s for s in seq if out.get(s) is not None]
    return out, aside


def run(S, subject, verbose=True, cheap=True, reps=None):
    """`reps`: already loaded (build_child_gt passes its own, actions already excluded); default load from
    {root}/{S}/{Q.ALIGNED}.  The report's '_internal' holds the final groups, their calibrations, the rep dicts
    (person and sync as last prepared) and the dropped reps, for the build; main() leaves it out of the JSON."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    if reps is None:
        reps = [r for r in B.load_reps(f'{B.KOREA_ROOT}/{S}/{Q.ALIGNED}', subject)
                if action_of(r['stub']) not in B.EXCLUDE_ACTIONS_BY_SET[S]]
    reps.sort(key=lambda r: (B._action(r['stub']), int(r['stub'].split('_')[3])))
    by = {r['stub']: r for r in reps}
    n_cal = 0

    # 1. compromise calibration over everything
    cal_all = Q.calibrate_loose(reps); n_cal += 1
    for r in reps:
        Q.prepare(r, cal_all['cams'])
    vecs = {}
    log(f'== {subject}: {len(reps)} reps.  1. one calibration over all reps: {Q.quality(cal_all)}')
    for r in reps:
        v, offs = offset_vec(cal_all['cams'], r)
        vecs[r['stub']] = v
        log(f"  {r['stub']}: {Q.fmt_offs(offs)}")

    # 2. first grouping: runs of consecutive reps within each action, then runs grouped across actions (any order)
    runs = action_runs(list(by), vecs)
    log(f'\n2. runs within actions (cut where neighbours differ by > {CLUSTER_PX} px): ' + ' | '.join(
        ','.join(s.split('_', 2)[2] for s in r) for r in runs))
    lab = group([np.nanmedian(np.array([vecs[s] for s in r]), axis=0) for r in runs], CLUSTER_PX)
    groups = {}
    for r, k in zip(runs, lab):
        groups.setdefault(int(k), []).extend(r)
    groups = list(groups.values())
    log(f'   grouped across actions: ' + ' | '.join(','.join(s.split('_', 2)[2] for s in g) for g in groups))

    # 3. refine
    cals, dropped = [], {}
    q_all = Q.quality(cal_all)
    worst_all = {s: Q._worst(cal_all['cams'], by[s])[0] for s in by}
    one_group = len(groups) == 1 and max(worst_all.values()) <= AGREE_PX
    report_path = 'cheap' if (cheap and q_all['ok'] and one_group) else 'full'
    log(f"   cheap path: gates {'pass' if q_all['ok'] else 'FAIL'}, {len(groups)} group(s), worst rep offset "
        f"{max(worst_all.values()):.1f} px (join <= {AGREE_PX}) -> {report_path}")

    def calibrate_groups(gs):
        """Largest group first, focal lengths free; every other group borrows them (same cameras, same lenses)."""
        nonlocal n_cal
        out_g, out_c = [], []
        f_main = None
        for g in sorted(gs, key=len, reverse=True):
            try:
                c = Q.calibrate_loose([by[s] for s in g], f_ref=f_main); n_cal += 1
                if f_main is None and len(g) >= MAIN_MIN_REPS:
                    f_main = c['cams'].f.copy()
                out_g.append(g); out_c.append(c)
            except CAL_ERRORS as e:
                log(f"    group {','.join(s.split('_', 2)[2] for s in g)} cannot be calibrated ({str(e)[:50]})")
                for s in g:
                    dropped[s] = 'its group cannot be calibrated'
        return out_g, out_c

    for p in range(0 if report_path == 'cheap' else MAX_PASSES):
        groups, cals = calibrate_groups(groups)
        if not groups:
            break
        W = {}                                           # (stub, group) -> worst camera offset
        for s in by:
            if s in dropped:
                continue
            for k, c in enumerate(cals):
                W[s, k] = Q._worst(c['cams'], by[s])[0]
        assign, left = {}, []
        home = {s: k for k, g in enumerate(groups) for s in g}
        for s in by:
            if s in dropped:
                continue
            h = home.get(s)
            if h is not None and len(groups[h]) > 1 and W[s, h] <= KEEP_PX:
                assign[s] = h                            # stays: still explained by its own (multi-rep) group
                continue
            k = min(range(len(groups)), key=lambda k: W[s, k])
            if W[s, k] <= AGREE_PX:
                assign[s] = k
            else:
                left.append(s)
        assign, aside = enforce_order(assign, list(by))
        for s in aside:
            log(f'    {s}: set aside -- its action returns to an earlier setup after it (a camera cannot go back)')
            dropped[s] = 'outlier within its action (the action returns to the setup either side of it)'
        new = [[s for s in by if assign.get(s) == k] for k in range(len(groups))]
        log(f'\n3. pass {p + 1}: ' + ' | '.join(
            f"[{','.join(s.split('_', 2)[2] for s in g)}] fit {Q.quality(c)['fit_px']}"
            for g, c in zip(groups, cals)))
        for s in by:
            if s not in dropped:
                log(f"    {s}: worst offset per group " + ' '.join(f'{W[s, k]:6.1f}' for k in range(len(groups)))
                    + ('   -> none' if s in left else ''))
        # MERGE groups whose cameras explain each other's reps (every rep of each within KEEP_PX under the other's
        # cameras -- the 'stay' threshold: merged groups stay together; at AGREE_PX B231 missed by 0.3-0.9 px).  Reassignment alone never undoes a bad first split: B231's whole-session calibration was bent by a
        # real move (action 4 from 4_2), the bend gave the other reps position-dependent offsets of 30-100 px, and the
        # one setup they share (the build: 3.5-4.6 px) came out as four groups that each fit the others' reps <= 8 px.
        idx = [k for k in range(len(groups)) if new[k]]
        par = {k: k for k in idx}
        def root(k):
            while par[k] != k:
                k = par[k]
            return k
        for a_ in idx:
            for b_ in idx:
                if a_ < b_ and root(a_) != root(b_) and \
                        all(W[s, b_] <= KEEP_PX for s in new[a_]) and all(W[s, a_] <= KEEP_PX for s in new[b_]):
                    par[root(b_)] = root(a_)
        merged = {}
        for k in idx:
            merged.setdefault(root(k), []).extend(new[k])
        if len(merged) < len(idx):
            log('    merged (cameras explain each other\'s reps): ' + ' | '.join(
                ','.join(s.split('_', 2)[2] for s in g) for g in merged.values() if len(g) > 0))
        new = [sorted(g, key=list(by).index) for g in merged.values()]
        if left:                                         # leftovers: runs within actions, grouped across actions
            lv = {}
            for s in left:
                k = min(range(len(groups)), key=lambda k: W[s, k])
                lv[s] = offset_vec(cals[k]['cams'], by[s])[0]
            lruns = action_runs(left, lv)
            lab = group([np.nanmedian(np.array([lv[s] for s in r]), axis=0) for r in lruns], CLUSTER_PX)
            extra = {}
            for r, k in zip(lruns, lab):
                extra.setdefault(int(k), []).extend(r)
            new += list(extra.values())
            log('    leftovers regrouped: ' + ' | '.join(','.join(s.split('_', 2)[2] for s in g)
                                                     for g in extra.values()))
        if sorted(map(sorted, new)) == sorted(map(sorted, groups)):
            log('    no rep moved: converged')
            break
        groups = new

    if report_path == 'cheap':
        groups, cals = [list(by)], [cal_all]

    # 4. moving-camera check against each rep's group cameras
    log(f'\n4. moving-camera check under each rep\'s group cameras: bystanders where visible, else the child\'s '
        f'own offsets (slow joints, wander > {MOVE_PX} px)')
    moving = []
    where = {s: k for k, g in enumerate(groups) for s in g}
    for g, c in zip(groups, cals):
        for s in g:
            mc = Q.moving_cameras(by[s], Q.offsets(c['cams'], by[s]))
            mv = any(m for m, _ in mc)
            # Sandwiched: an earlier AND a later rep of the same action sit in the same setup.  The camera can't have
            # moved and come back exactly, so this rep has that setup too -- keep it, whatever the offsets say.
            same = [t for t in by if action_of(t) == action_of(s)]
            i = same.index(s)
            sandwiched = (any(where.get(t) == where[s] for t in same[:i])
                          and any(where.get(t) == where[s] for t in same[i + 1:]))
            if mv and sandwiched:
                log(f"  {s}:  would be MOVING, but reps before and after it share its setup -> kept")
                mv = False
            else:
                log(f"  {s}:{'  MOVING -> dropped' if mv else ''}")
            for k, (m, why) in enumerate(mc):
                log(f"      cam{k + 1}: {'MOVING' if m else 'still '}  {why}")
            if mv:
                moving.append(s)
    if moving:
        groups = [[s for s in g if s not in moving] for g in groups]
        groups = [g for g in groups if g]
        for s in moving:
            dropped[s] = 'camera moving during the rep'
        groups, cals = calibrate_groups(groups)

    # 5. report
    log('\n5. setups')
    report = dict(subject=subject, setups=[], dropped=dropped, path=report_path, whole_session=q_all)
    for k, (g, c) in enumerate(zip(groups, cals)):
        q = Q.quality(c)
        log(f"  setup {k + 1}: {', '.join(s.split('_', 2)[2] for s in g)}\n    {q}")
        rows = {}
        for s in g:
            r = dict(by[s]); Q.prepare(r, c['cams'])
            xy, conf, _ = B.rep_span(r)
            vis = (conf > B.GATES['conf_thresh']) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
            _, _, _, e = cc.triangulate_robust(c['cams'], xy, vis, B.GATES['reproj_thresh_px'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = [round(float(np.nanmedian(e[:, j])), 2) for j in range(3)]
            w, offs = Q._worst(c['cams'], r)
            rows[s] = dict(median_reproj_px=med, worst_offset=round(w, 1))
            log(f"    {s}: reproj {med} (worst {max(med):.1f} px)  {Q.fmt_offs(offs)}")
        report['setups'].append(dict(members=g, quality=q, reps=rows))
    # rep order within an action: do its reps hop between setups?
    where = {s: k for k, g in enumerate(groups) for s in g}
    for a in sorted({action_of(s) for s in by}, key=lambda a: B._action(f'x_x_{a}')):
        seq = [where.get(s, '-') for s in by if action_of(s) == a]
        kept = [x for x in seq if x != '-']                     # dropped reps don't break a run
        runs = [x for i, x in enumerate(kept) if i == 0 or x != kept[i - 1]]
        flag = '  <- hops between setups' if len(runs) != len(set(runs)) else ''
        log(f"  action {a}: {' '.join(str(x + 1) if x != '-' else '-' for x in seq)}{flag}")
    for s, why in dropped.items():
        log(f'  dropped {s}: {why}')
    report['n_calibrations'] = n_cal
    report['seconds'] = round(time.time() - t0)
    log(f"\n({n_cal} calibrations, {report['seconds']} s)")
    report['_internal'] = dict(groups=groups, cals=cals, by=by, dropped=dropped)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='B')
    ap.add_argument('--subject', required=True)
    ap.add_argument('--json', default=None)
    ap.add_argument('--aligned', default=Q.ALIGNED, help='folder under {root}/{set} with the aligned reps')
    ap.add_argument('--no-cheap', action='store_true', help='always run the full clustering')
    a = ap.parse_args()
    Q.ALIGNED = a.aligned
    warnings.simplefilter('ignore', RuntimeWarning)
    rep = run(a.set, a.subject, cheap=not a.no_cheap)
    rep.pop('_internal', None)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(B.jsonable(rep), f, indent=1)


if __name__ == '__main__':
    main()
