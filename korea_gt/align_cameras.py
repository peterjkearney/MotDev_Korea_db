#!/usr/bin/env python3
"""align_cameras.py -- put the three cameras of every Korea rep on one clock.

The cameras were not started together, so each camera's OpenPose track starts at a
different moment.  Per rep:

  0. find the child in each camera (--people track).  OpenPose detects every frame on its
     own and lists the people in it in no fixed order, so "the first person listed" can
     switch between the child and an adult from one frame to the next (B237_GMS_2_4 camera 3:
     two people every frame, 44 switches).  Each camera's detections are linked into tracks,
     one person each, and the combination of one track per camera whose movement agrees best
     across the three cameras is taken as the child.  Combinations nearly as good are saved as
     candidates for build_child_gt.py to settle by body size (an adult moving with the child
     correlates too).  Before this, reps with switches failed the sync 73% of the time vs 6%,
  1. reduce each camera's track to a 1-D motion signal,
  2. find the best offset for every camera pair (1-2, 1-3, 2-3) by masked
     cross-correlation, within +-max_lag frames, scoring only offsets where both signals
     overlap by min(60, 80% of the shortest camera's frames) frames, at least 30,
  3. accept the rep only if every pair correlates (>= min_corr) and the three
     offsets agree: lag23 must equal lag13 - lag12 to within max_mismatch frames,
  4. trim every camera to the common span and save it.

The signal (--signal):
  vertical  confidence-weighted SIGNED vertical image velocity, averaged over all
            joints and smoothed over 3 frames (default).  The cameras are level,
            at about the same height and tilted alike, so up/down motion in the
            room is up/down in every image whichever way the camera faces;
            horizontal motion is not, and speed throws the sign away.
  speed     confidence-weighted joint speed -- the original notebook's signal
            (align_cameras.ipynb / merge_cameras.ipynb).  With --max-lag 30 it
            reproduces the original Aligned/ exactly.
Measured over the 1,551 three-camera reps: vertical syncs 1,343 vs 1,177 and its
weakest camera pair correlates at 0.91 vs 0.80 (median); on reps both sync, the
offsets agree within a frame 89% of the time.

The offsets are whole frames from one signal, so about half are a frame off the
best geometric fit.  build_child_gt.py refines them once the cameras are
calibrated; for that it needs each camera's UNTRIMMED track, so it is saved too.

Each {stub}.npz holds:
    xy1..3, score1..3                      the synced, trimmed tracks (T,25,2) / (T,25)
    raw_xy1..3, raw_score1..3, frame_index1..3   each camera's full track and JSON frame numbers
    lag12, lag13                           the offsets used (camera 1 frame i+lag1c == camera c frame i)
    start                                  (3,) first raw row of each camera in the synced span
plus sync_status.csv in --out: every rep, its offsets, correlations and outcome.

    python align_cameras.py --set B --out "/Volumes/Sandisk_SSD/MotorDevelopment/Korea/B/Aligned"
    python align_cameras.py --set A --out ...     # JSONs from {--root}/A
    python align_cameras.py --out ... --signal speed --max-lag 30     # the original behaviour
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

# Korea data root holding the age sets A/, B/, C/ (per-camera OpenPose JSONs) and {set}_video/
KOREA_ROOT = os.environ.get('KOREA_ROOT', '/Volumes/Sandisk_SSD/MotorDevelopment/Korea')

SYNC = dict(
    conf_thresh=0.3,      # a joint counts in the signal above this confidence
    min_overlap=60,       # frames both signals must have valid to compare an offset (the cap; see below)
    # Short reps: the required overlap is min(min_overlap, min_overlap_frac x the shortest camera's frames), never
    # below min_overlap_floor (= build_child_gt's min_usable_frames).  60 came untested from the original notebook
    # and rejected every rep under 60 frames ('too short': C 556 reps, many jumps) and reps where the child leaves one
    # camera early (C224_GMS_3_2: child in cam3 for 57 of 67 frames -> a still bystander's track was the only one
    # long enough, corr 0.29).  A flat lower minimum is worse: at 30, short overlaps at extreme offsets correlate by
    # chance and moved 88 synced reps to absurd lags (A077_GMS_4_2 (0,0) -> (0,-34)).  Relative, reps >= 75 frames
    # keep 60 exactly; measured 2026-11-27 over A/B/C: +681 synced reps (corr >= 0.8 on the weakest pair for 550),
    # 25 synced reps change (23 by 1-2 frames), 1 lost.  --min-overlap-frac 0 restores the fixed 60.
    min_overlap_frac=0.8,
    min_overlap_floor=30,
    min_corr=0.3,         # every camera pair must correlate at least this well
    max_mismatch=1,       # frames: lag23 vs lag13 - lag12
)


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def _smooth3(x):
    """Centred 3-frame mean, edges averaged over what exists."""
    if len(x) < 3:
        return x
    s = np.convolve(x, np.ones(3), 'same')
    n = np.convolve(np.ones_like(x), np.ones(3), 'same')
    return s / n


def signal_speed(xy, conf, thresh=SYNC['conf_thresh']):
    """The original: confidence-weighted mean joint speed (px/frame)."""
    valid = conf > thresh
    step = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    ok = valid[1:] & valid[:-1]
    w = np.where(ok, conf[1:], 0.0)
    den = w.sum(axis=1)
    sig = np.where(den > 0, (step * w).sum(axis=1) / np.maximum(den, 1e-9), 0.0)
    return sig, den > 0


def signal_vertical(xy, conf, thresh=SYNC['conf_thresh']):
    """Confidence-weighted mean SIGNED vertical velocity, 3-frame smoothed."""
    valid = (conf > thresh) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
    dy = np.diff(xy[..., 1], axis=0)
    ok = valid[1:] & valid[:-1]
    w = np.where(ok, conf[1:], 0.0)
    den = w.sum(axis=1)
    sig = np.where(den > 0, (dy * w).sum(axis=1) / np.maximum(den, 1e-9), 0.0)
    return _smooth3(sig), den > 0


SIGNALS = {'vertical': signal_vertical, 'speed': signal_speed}


def normalise(sig, valid):
    sig = sig.copy()
    if valid.sum() > 1 and sig[valid].std() > 0:
        sig[valid] = (sig[valid] - sig[valid].mean()) / sig[valid].std()
    return sig, valid


# --------------------------------------------------------------------------
# Offsets
# --------------------------------------------------------------------------

def rep_min_overlap(lengths):
    """Overlap a rep's offsets need, from its cameras' frame counts (SYNC: cap, fraction, floor)."""
    if SYNC['min_overlap_frac'] <= 0:
        return SYNC['min_overlap']
    return max(SYNC['min_overlap_floor'],
               min(SYNC['min_overlap'], int(np.ceil(SYNC['min_overlap_frac'] * min(lengths)))))


def masked_xcorr(a, b, va, vb, max_lag, min_overlap=None):
    """[(lag, pearson r)]; positive lag => a is delayed relative to b (a[i+lag] matches b[i]).
    min_overlap None -> SYNC['min_overlap'], read at call time."""
    if min_overlap is None:
        min_overlap = SYNC['min_overlap']
    na, nb = len(a), len(b)
    out = []
    for lag in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -lag), min(nb, na - lag)
        if hi - lo < min_overlap:
            continue
        y, my = b[lo:hi], vb[lo:hi]
        x, mx = a[lo + lag:hi + lag], va[lo + lag:hi + lag]
        m = mx & my
        if m.sum() < min_overlap:
            continue
        xv, yv = x[m] - x[m].mean(), y[m] - y[m].mean()
        den = np.sqrt((xv ** 2).sum() * (yv ** 2).sum())
        if den > 0:
            out.append((lag, float((xv * yv).sum() / den)))
    return out


def find_offsets(xys, confs, signal='vertical', max_lag=45, min_overlap=None):
    """Pairwise best offsets and the accept / reject decision for one rep."""
    sigs = [normalise(*SIGNALS[signal](x, c)) for x, c in zip(xys, confs)]
    out, best = {}, {}
    for key, (a, b) in {'12': (0, 1), '13': (0, 2), '23': (1, 2)}.items():
        res = masked_xcorr(sigs[a][0], sigs[b][0], sigs[a][1], sigs[b][1], max_lag, min_overlap)
        best[key] = max(res, key=lambda r: r[1]) if res else None
        out[f'lag{key}'] = best[key][0] if best[key] else np.nan
        out[f'corr{key}'] = best[key][1] if best[key] else np.nan
    weak = [k for k, b in best.items() if b is None or b[1] < SYNC['min_corr']]
    if weak:
        out.update(status='weak correlation', detail='pairs ' + ','.join(weak))
        return out
    mismatch = out['lag23'] - (out['lag13'] - out['lag12'])
    out['mismatch'] = mismatch
    if abs(mismatch) > SYNC['max_mismatch']:
        out.update(status='offsets disagree', detail=f'{mismatch:+d} frames')
        return out
    out.update(status='OK', detail='')
    return out


def span(lengths, lag12, lag13):
    """First row of each camera in the common span, and its length."""
    s1 = max(0, lag12, lag13)
    start = np.array([s1, s1 - lag12, s1 - lag13])
    return start, int(min(n - s for n, s in zip(lengths, start)))


# --------------------------------------------------------------------------
# One rep
# --------------------------------------------------------------------------

def load_rep(json_dir, stub):
    """Per camera: every person OpenPose listed, per frame.

    Returns (people, fidx): people[c][t] is a list of (xy (25,2), score (25,)) -- OpenPose
    detects each frame on its own, so list position carries no identity from frame to frame --
    and fidx[c] the JSON frame numbers.  Raises on a missing camera.
    """
    people, fidx = [], []
    for c in (1, 2, 3):
        path = os.path.join(json_dir, f'{stub}_{c}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'missing camera {c}')
        with open(path) as f:
            data = json.load(f)['data']
        people.append([[(np.asarray(s['pose'], float).reshape(25, 2), np.asarray(s['score'], float))
                        for s in fr['skeleton']] for fr in data])
        fidx.append(np.array([fr['frame_index'] for fr in data]))
    return people, fidx


# --------------------------------------------------------------------------
# People: tracks per camera, and which track is the child
# --------------------------------------------------------------------------

TRACK = dict(
    conf_thresh=0.3,      # joints used to compare two detections
    max_step_px=100.0,    # largest frame-to-frame move of one person (mean over shared joints)
    step_per_gap_px=15.0, # allowance per missed frame when bridging a gap
    max_gap=10,           # frames a person may go undetected and still continue the same track
    merge_gap=45,         # frames: join a track that ends to one that starts later, if they line up
    min_frames=30,        # a track this short is never the child's
    max_tracks=3,         # longest tracks kept as candidates, per camera
    ambiguous_margin=0.15,  # a runner-up combination this close to the best is a second candidate
    # Second join pass, for one person whose track split at a fast move (a jump's take-off, a catch): more than
    # max_step_px between frames, so the first join refused it (B011_GMS_3_1 cam 1: one person every frame but
    # one, split at frame 111/112).  Joined only where nobody else is around the junction, so it can't splice two
    # people.  Set False to reproduce the earlier behaviour.
    join_jumps=True,
    jump_max_px=300.0,    # unpredicted jump allowed at the junction (mean joint distance)
    jump_clear=10,        # frames either side of the junction that must show no third track ...
    jump_other_min=10,    # ... of at least this many frames (a one-frame stray detection is not a competitor)
    jump_size_ratio=1.5,  # the two pieces' median body heights within this factor of each other
)


def _seen(xy, sc):
    return (sc > TRACK['conf_thresh']) & ~((xy[:, 0] == 0) & (xy[:, 1] == 0))


def _distance(a, b):
    """Mean distance over joints both detections see; centroid distance if fewer than 4."""
    (xa, sa), (xb, sb) = a, b
    ma, mb = _seen(xa, sa), _seen(xb, sb)
    both = ma & mb
    if both.sum() >= 4:
        return float(np.linalg.norm(xa[both] - xb[both], axis=1).mean())
    if ma.sum() and mb.sum():
        return float(np.linalg.norm(xa[ma].mean(0) - xb[mb].mean(0)))
    return np.inf


def build_tracks(frames):
    """Link one camera's per-frame detections into tracks of one person each.

    frames[t] = list of (xy, score).  Each frame's detections are matched to the tracks seen
    in the last max_gap frames (Hungarian on mean joint distance, gated by how far a person can
    move in that time); the unmatched start new tracks.  Tracks that end where a later one
    begins are then joined.  Returns a list of (xy (T,25,2), score (T,25)), zeros where absent,
    longest first.
    """
    from scipy.optimize import linear_sum_assignment
    T = len(frames)
    tracks = []                                   # each: dict(rows={t: (xy, sc)}, last=t)
    for t, dets in enumerate(frames):
        dets = [d for d in dets if _seen(*d).sum() >= 4]
        live = [k for k, tr in enumerate(tracks) if t - tr['last'] <= TRACK['max_gap']]
        assigned = set()
        if live and dets:
            cost = np.full((len(live), len(dets)), 1e9)
            for i, k in enumerate(live):
                tr = tracks[k]
                gap = t - tr['last']
                gate = TRACK['max_step_px'] + TRACK['step_per_gap_px'] * (gap - 1)
                for j, d in enumerate(dets):
                    dist = _distance(tr['rows'][tr['last']], d)
                    if dist <= gate:
                        cost[i, j] = dist
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e9:
                    tracks[live[i]]['rows'][t] = dets[j]
                    tracks[live[i]]['last'] = t
                    assigned.add(j)
        for j, d in enumerate(dets):
            if j not in assigned:
                tracks.append(dict(rows={t: d}, first=t, last=t))

    # join a track that ends to one that starts afterwards, if they line up
    tracks.sort(key=lambda tr: tr['first'])
    merged = []
    for tr in tracks:
        best = None
        for m in merged:
            gap = tr['first'] - m['last']
            if 0 < gap <= TRACK['merge_gap']:
                dist = _distance(m['rows'][m['last']], tr['rows'][tr['first']])
                if dist <= TRACK['max_step_px'] + TRACK['step_per_gap_px'] * (gap - 1) and (best is None or dist < best[0]):
                    best = (dist, m)
        if best:
            best[1]['rows'].update(tr['rows'])
            best[1]['last'] = tr['last']
        else:
            merged.append(tr)

    if TRACK['join_jumps']:
        merged = join_jumps(merged)

    out = []
    for tr in sorted(merged, key=lambda tr: -len(tr['rows'])):
        xy, sc = np.zeros((T, 25, 2)), np.zeros((T, 25))
        for t, (x, s) in tr['rows'].items():
            xy[t], sc[t] = x, s
        out.append((xy, sc))
    return out


def _height(tr):
    h = []
    for x, s in tr['rows'].values():
        m = _seen(x, s)
        if m.sum() >= 4:
            h.append(np.ptp(x[m][:, 1]))
    return float(np.median(h)) if h else np.nan


def _predicted(tr, gap):
    """The track's last detection carried forward by its last velocity (per joint), `gap` frames on."""
    t1 = tr['last']
    x1, s1 = tr['rows'][t1]
    prev = [t for t in tr['rows'] if t1 - 3 <= t < t1]
    if not prev:
        return x1, s1
    t0 = max(prev)
    x0, s0 = tr['rows'][t0]
    v = np.where((_seen(x0, s0) & _seen(x1, s1))[:, None], (x1 - x0) / (t1 - t0), 0.0)
    return x1 + v * gap, s1


def join_jumps(tracks):
    """Join a track that ends to one that starts within merge_gap frames when the first join refused it only for
    distance, and nothing else could be confused with it: no third track (of jump_other_min+ frames) present within
    jump_clear frames of the junction, bodies of compatible size, and the start either where the first track was heading (its last velocity
    carried forward) or within jump_max_px.  Repeats until nothing joins."""
    tracks = sorted(tracks, key=lambda tr: tr['first'])
    while True:
        best = None
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                gap = b['first'] - a['last']
                if i == j or not 0 < gap <= TRACK['merge_gap']:
                    continue
                lo, hi = a['last'] - TRACK['jump_clear'], b['first'] + TRACK['jump_clear']
                if any(k not in (i, j) and len(tr['rows']) >= TRACK['jump_other_min']
                       and any(lo <= t <= hi for t in tr['rows']) for k, tr in enumerate(tracks)):
                    continue
                ha, hb = _height(a), _height(b)
                if not (np.isfinite(ha) and np.isfinite(hb) and max(ha, hb) <= TRACK['jump_size_ratio'] * min(ha, hb)):
                    continue
                first = b['rows'][b['first']]
                d_pred = _distance(_predicted(a, gap), first)
                d_raw = _distance(a['rows'][a['last']], first)
                gate = TRACK['max_step_px'] + TRACK['step_per_gap_px'] * (gap - 1)
                if d_pred <= gate or d_raw <= TRACK['jump_max_px']:
                    if best is None or gap < best[0]:
                        best = (gap, i, j)
        if best is None:
            return tracks
        _, i, j = best
        a, b = tracks[i], tracks[j]
        a['rows'].update(b['rows'])
        a['last'] = b['last']
        tracks.pop(j)


def first_listed(frames):
    """The original rule: whoever OpenPose lists first, every frame."""
    T = len(frames)
    xy, sc = np.zeros((T, 25, 2)), np.zeros((T, 25))
    for t, dets in enumerate(frames):
        if dets:
            xy[t], sc[t] = dets[0]
    return xy, sc


def candidate_tracks(frames):
    """A camera's candidate people: the series itself if it never shows two people at once,
    else its longest tracks (at least min_frames long)."""
    if max((len(d) for d in frames), default=0) <= 1:
        return [first_listed(frames)], 1
    tr = [t for t in build_tracks(frames) if (t[1] > 0).any(axis=1).sum() >= TRACK['min_frames']]
    return tr[:TRACK['max_tracks']], max(len(d) for d in frames)


def choose_people(people, signal, max_lag, min_overlap=None):
    """The combination of one track per camera whose movement agrees best across cameras.

    Every combination is scored by the sum of its three pairwise correlations; the best
    combination that also passes the offset checks wins.  Runners-up that pass and score within
    ambiguous_margin of it, with a different person in some camera, are kept as candidates for
    build_child_gt.py to settle by body size.  Returns (tracks per camera, chosen index per camera,
    offsets dict of the winner, candidate list, max people per camera).
    """
    import itertools
    cands, maxp = zip(*[candidate_tracks(f) for f in people])
    if not all(cands):
        return None
    sigs = [[normalise(*SIGNALS[signal](x, s)) for x, s in cam] for cam in cands]
    results = []
    for combo in itertools.product(*[range(len(c)) for c in cands]):
        xys = [cands[c][i][0] for c, i in enumerate(combo)]
        scs = [cands[c][i][1] for c, i in enumerate(combo)]
        off = find_offsets(xys, scs, signal, max_lag, min_overlap)
        score = np.nansum([off['corr12'], off['corr13'], off['corr23']])
        results.append((off['status'] == 'OK', score, combo, off))
    results.sort(key=lambda r: (r[0], r[1]), reverse=True)
    ok, best_score, best, off = results[0]
    alts = [(s, c, o) for passed, s, c, o in results[1:]
            if passed and ok and best_score - s <= TRACK['ambiguous_margin'] and c != best]
    return cands, best, off, [(best_score, best, off)] + alts, maxp


def process(task):
    stub, args = task
    TRACK['join_jumps'] = not getattr(args, 'no_join_jumps', False)   # workers may be spawned: set it here
    SYNC['min_overlap_frac'] = getattr(args, 'min_overlap_frac', SYNC['min_overlap_frac'])
    row = dict(stub=stub)
    try:
        people, fidx = load_rep(args.json_dir, stub)
    except FileNotFoundError as e:
        row.update(status=str(e))
        return row
    except (IndexError, KeyError, ValueError) as e:
        row.update(status='unreadable skeleton', detail=repr(e)[:80])
        return row
    mo = rep_min_overlap([len(f) for f in people])
    row['min_overlap'] = mo
    if min(len(f) for f in people) < mo:
        row.update(status='too short', detail=f'{min(len(f) for f in people)} frames in the shortest camera')
        return row
    try:
        if args.people == 'first':
            cands = [[first_listed(f)] for f in people]
            combo, maxp = (0, 0, 0), [max(len(d) for d in f) for f in people]
            off = find_offsets([c[0][0] for c in cands], [c[0][1] for c in cands], args.signal, args.max_lag, mo)
            options = [(0.0, combo, off)]
        else:
            chosen = choose_people(people, args.signal, args.max_lag, mo)
            if chosen is None:
                row.update(status='no person track', detail='a camera has no track of 30+ frames')
                return row
            cands, combo, off, options, maxp = chosen
        row.update(off)
    except Exception as e:                       # one odd rep must not end the batch
        row.update(status='sync error', detail=repr(e)[:80])
        return row
    row.update(frames_raw=min(len(f) for f in people), max_people='/'.join(str(m) for m in maxp),
               n_tracks='/'.join(str(len(c)) for c in cands), track='/'.join(str(i) for i in combo),
               n_candidates=len(options), ambiguous=len(options) > 1)
    if row['status'] != 'OK':
        return row
    xys = [cands[c][i][0] for c, i in enumerate(combo)]
    scs = [cands[c][i][1] for c, i in enumerate(combo)]
    l12, l13 = int(row['lag12']), int(row['lag13'])
    start, T = span([len(x) for x in xys], l12, l13)
    row['frames'] = T
    # frames in the synced span where all three cameras' signals are valid: below 60 = synced on a short overlap
    # (allowed for short reps since the relative min_overlap) -> worth a look in the viewer
    valid = [SIGNALS[args.signal](x, s)[1] for x, s in zip(xys, scs)]      # one shorter than the track (a diff)
    vs = [valid[c][start[c]:start[c] + T] for c in range(3)]
    n = min(len(v) for v in vs)
    row['sync_overlap'] = int(np.logical_and.reduce([v[:n] for v in vs]).sum())
    out = dict(lag12=l12, lag13=l13, start=start, signal=np.array(args.signal), people=np.array(args.people),
               track=np.array(combo))
    for c in range(3):
        out[f'xy{c + 1}'] = xys[c][start[c]:start[c] + T]
        out[f'score{c + 1}'] = scs[c][start[c]:start[c] + T]
        out[f'raw_xy{c + 1}'] = xys[c]
        out[f'raw_score{c + 1}'] = scs[c]
        out[f'frame_index{c + 1}'] = fidx[c]
        # every candidate person, for build_child_gt.py's body-size check on ambiguous reps
        out[f'tracks_xy{c + 1}'] = np.stack([x for x, _ in cands[c]]).astype(np.float32)
        out[f'tracks_score{c + 1}'] = np.stack([s for _, s in cands[c]]).astype(np.float32)
    # candidate combinations: (score, track1, track2, track3, lag12, lag13), best first
    out['candidates'] = np.array([[s, *c, o['lag12'], o['lag13']] for s, c, o in options], float)
    np.savez(os.path.join(args.out, stub + '.npz'), **out)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=KOREA_ROOT, help='Korea data root (env KOREA_ROOT) holding A/, B/, C/')
    ap.add_argument('--set', choices=['A', 'B', 'C'], default='B', help='age set: A 20-35, B 36-53, C 54-71 months')
    ap.add_argument('--json-dir', default=None,
                    help='per-camera OpenPose JSONs ({stub}_{cam}.json); default {root}/{set}')
    ap.add_argument('--out', default=None, help='output folder for the synced reps; default {root}/{set}/Aligned')
    ap.add_argument('--signal', choices=sorted(SIGNALS), default='vertical')
    ap.add_argument('--people', choices=['track', 'first'], default='track',
                    help="'track': follow each person over time and keep the one whose movement agrees across "
                         "cameras (default); 'first': whoever OpenPose lists first, every frame (the original)")
    ap.add_argument('--max-lag', type=int, default=45, help='largest offset searched, frames (the original used 30)')
    ap.add_argument('--subjects', default=None, help='comma-separated; default all')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min-overlap-frac', type=float, default=SYNC['min_overlap_frac'],
                    help='short reps need min(60, this x the shortest camera\'s frames) overlapping frames, at least '
                         f"{SYNC['min_overlap_floor']}; 0 = a fixed 60 for every rep (the earlier behaviour)")
    ap.add_argument('--no-join-jumps', action='store_true',
                    help='skip the second join pass (tracks split at a fast move stay split): the earlier behaviour')
    args = ap.parse_args()
    TRACK['join_jumps'] = not args.no_join_jumps
    SYNC['min_overlap_frac'] = args.min_overlap_frac
    args.json_dir = args.json_dir or os.path.join(args.root, args.set)
    args.out = args.out or os.path.join(args.root, args.set, 'Aligned')
    os.makedirs(args.out, exist_ok=True)

    stubs = sorted({'_'.join(f[:-5].split('_')[:-1]) for f in os.listdir(args.json_dir)
                    if f.endswith('.json') and not f.startswith('.')})
    if args.subjects:
        want = set(args.subjects.split(','))
        stubs = [s for s in stubs if s.split('_')[0] in want]
    print(f'{len(stubs)} reps, signal {args.signal}, max lag {args.max_lag} -> {args.out}', flush=True)

    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(process, [(s, args) for s in stubs], chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(stubs):
                ok = sum(r['status'] == 'OK' for r in rows)
                print(f'PROGRESS {i}/{len(stubs)} reps ({time.time() - t0:.0f}s), {ok} synced', flush=True)

    df = pd.DataFrame(rows).sort_values('stub')
    df.to_csv(os.path.join(args.out, 'sync_status.csv'), index=False)
    print(df.status.value_counts().to_string())
    print(f"{(df.status == 'OK').sum()} reps from {df[df.status == 'OK'].stub.str.split('_').str[0].nunique()} "
          f"children written to {args.out}")


if __name__ == '__main__':
    main()
