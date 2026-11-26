#!/usr/bin/env python3
"""build_review_sheet.py -- one row per raw Korea rep, for walking through the stick-figure videos by eye.

Every rep in the raw OpenPose JSONs is listed, whichever stage dropped it:
  sync        align_cameras.py could not sync the three cameras (Aligned/sync_status.csv)
  excluded    set A's stair actions, left out of build_child_gt on purpose
  calibration the session never calibrated (too few frames/correspondences; session_summary status 'skipped')
  setup       the session / camera setup failed its gates (bone CV, reprojection, roll, floor, stature)
  rep         the rep failed its own gates inside a usable setup
  usable      in the training/eval data

`group` orders the review: near-misses (the recoverable kinds from the 2026-11-20 recovery study) first, then
the other rejections, usable reps last as reference.  `bad_cams` names the camera(s) over the 1.5x reprojection
rep gate.  Videos are OpenPose renders (no RGB): {set}_video/{action}/{train|valid|test}/{stub}_{cam}.mp4,
camera 1 front, 2 right 45 deg, 3 left 45 deg.  `notes` and `verdict` are empty columns for the reviewer.

    python analysis/rep_review/build_review_sheet.py                 # A, B, C -> {root}/rep_review.csv
    python analysis/rep_review/build_review_sheet.py --sets B,C --out x.csv
Refuses to overwrite an existing sheet (it holds your notes) unless --force.
"""
import argparse, glob, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

ROOT = os.environ.get('KOREA_ROOT', '/Volumes/Sandisk_SSD/MotorDevelopment/Korea')
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TASKS = {'A': {1: 'stairs up', 2: 'stairs down', 3: 'overhead throw', 4: 'one foot 1 s'},
         'B': {1: 'one foot >3 s', 2: 'hop one foot', 3: 'two-foot jump', 4: 'catch ball'},
         'C': {1: 'stop rolling ball', 2: 'bounce ball', 3: 'jump over rope', 4: 'jump rope'}}
EXCLUDED = {'A': ('1', '2'), 'B': (), 'C': ()}

# review order
GROUPS = [
    '01 setup: bone CV <=10% only',
    '02 setup: stature unmeasurable only',
    '03 rep: one camera off (reprojection)',
    '04 rep: bone lengths off only',
    '05 rep: too few usable frames only',
    '06 rep: camera moved / set aside',
    '07 rep: several rep gates',
    '08 setup: bone CV >10% only',
    '09 setup: reprojection vs epipolar only',
    '10 setup: camera roll / floor only',
    '11 setup: several setup gates',
    '12 calibration: too few frames',
    '13 sync: offsets disagree',
    '14 sync: weak correlation',
    '15 sync: too short',
    '16 sync: no person track',
    '17 excluded action (stairs)',
    '18 usable',
]


def setup_group(reasons):
    t = ' '.join(reasons)
    kinds = set()
    for k, pat in [('cv', 'long-bone CV'), ('stature', 'stature unmeasurable'), ('reproj', 'epipolar floor'),
                   ('roll', 'camera roll'), ('floor', 'floor inliers')]:
        if pat in t:
            kinds.add(k)
    cvs = [float(x) for x in re.findall(r'long-bone CV ([\d.]+)%', t)]
    if kinds == {'cv'}:
        return GROUPS[0] if max(cvs) <= 10 else GROUPS[7]
    if kinds == {'stature'}:
        return GROUPS[1]
    if kinds == {'reproj'}:
        return GROUPS[8]
    if kinds and kinds <= {'roll', 'floor'}:
        return GROUPS[9]
    return GROUPS[10]


def rep_kinds(reasons):
    k = set()
    for x in reasons:
        if 'not usable' in x:
            continue
        if x.startswith('bone lengths'): k.add('bone')
        elif x.startswith('reprojection'): k.add('reproj')
        elif 'usable frames' in x or 'usable fraction' in x: k.add('frames')
        elif 'moved' in x or 'cannot explain' in x or 'fits none' in x: k.add('moved')
        else: k.add('other')
    return k


def label_of(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get('label', '')
    except Exception:
        return ''


def build_set(S, splits):
    raw = sorted(glob.glob(f'{ROOT}/{S}/{S}*_GMS_*_*_[123].json'))
    stubs = sorted({os.path.basename(p)[:-7] for p in raw if not os.path.basename(p).startswith('._')})
    cam1 = {s: f'{ROOT}/{S}/{s}_1.json' for s in stubs}
    with ThreadPoolExecutor(8) as ex:
        labels = dict(zip(stubs, ex.map(lambda s: label_of(cam1[s]), stubs)))

    vids = {}
    for p in glob.glob(f'{ROOT}/{S}_video/**/*.mp4', recursive=True):
        n = os.path.basename(p)
        if not n.startswith('._'):
            vids.setdefault(n[:-4].replace('[SHANA]', ''), p)

    sync = pd.read_csv(f'{ROOT}/{S}/Aligned/sync_status.csv').set_index('stub')

    # GT3D per session: rep records + setup verdicts
    gt = {}
    sess = {}
    for f in glob.glob(f'{ROOT}/{S}/GT3D/*/session_summary.json'):
        d = json.load(open(f))
        sess[d['subject']] = d
        setups = d.get('setups') or [d]
        for r in d.get('reps', []):
            st = setups[r['setup'] - 1] if r.get('setup') and len(setups) >= r['setup'] else d
            gt[r['stub']] = (r, st)

    rows = []
    for s in stubs:
        child, _, action, rep = s.split('_')
        row = dict(set=S, stub=s, child=child, action=int(action), task=TASKS[S].get(int(action), ''),
                   rep=int(rep), label=labels.get(s, ''), split=splits.get(child, 'none'))
        sy = sync.loc[s] if s in sync.index else None
        row['sync_status'] = sy['status'] if sy is not None else 'missing'
        row['sync_detail'] = '' if sy is None or pd.isna(sy['detail']) else sy['detail']
        if sy is not None:
            cs = [sy[c] for c in ('corr12', 'corr13', 'corr23') if pd.notna(sy[c])]
            row['sync_min_corr'] = round(min(cs), 2) if cs else np.nan
            row['frames_raw'] = sy['frames_raw']
        if row['sync_status'] != 'OK':
            stage, grp = 'sync', next((g for g in GROUPS if g.endswith('sync: ' + str(row['sync_status']))), '16 sync: no person track')
        elif action in EXCLUDED[S]:
            stage, grp = 'excluded', GROUPS[16]
        elif s in gt:
            r, st = gt[s]
            row.update(n_frames=r['n_frames'], n_usable_frames=r['n_usable_frames'],
                       bone_dev=r.get('bone_dev'), reproj_over_session=r.get('reproj_over_session'),
                       setup=r.get('setup'), session_bone_cv=st.get('long_bone_cv_pct'),
                       reasons='; '.join(r['reasons']), setup_reasons='; '.join(st.get('reasons', [])))
            cal = np.array(st.get('calib_median_reproj_px') or [np.nan] * 3, float)
            med = np.array(r.get('median_reproj_px') or [np.nan] * 3, float)
            with np.errstate(invalid='ignore', divide='ignore'):
                row['bad_cams'] = ','.join(str(c + 1) for c in range(3) if med[c] / cal[c] > 1.5)
            if r['usable']:
                stage, grp = 'usable', GROUPS[17]
            elif not st.get('usable', False):
                stage, grp = 'setup', setup_group(st.get('reasons', []))
                if rep_kinds(r['reasons']):
                    grp = GROUPS[10] if grp in (GROUPS[0], GROUPS[1]) else grp   # not a clean near-miss
            else:
                k = rep_kinds(r['reasons'])
                stage = 'rep'
                grp = {frozenset({'reproj'}): GROUPS[2], frozenset({'bone'}): GROUPS[3],
                       frozenset({'frames'}): GROUPS[4], frozenset({'moved'}): GROUPS[5]}.get(frozenset(k), GROUPS[6])
        else:
            d = sess.get(child)
            stage, grp = 'calibration', GROUPS[11]
            row['setup_reasons'] = '; '.join(d['reasons']) if d else 'no GT3D session'
        row['stage_failed'], row['group'] = stage, grp
        for c in (1, 2, 3):
            p = vids.get(f'{s}_{c}', '')
            row[f'video_cam{c}'] = p
            row[f'open_cam{c}'] = f'=HYPERLINK("file://{p}","cam{c}")' if p else ''
        row['notes'] = ''
        row['verdict'] = ''
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='A,B,C')
    ap.add_argument('--out', default=f'{ROOT}/rep_review.csv')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.force:
        sys.exit(f'{a.out} exists (may hold your notes); pass --force to overwrite')
    sp = json.load(open(f'{REPO}/pipeline/datasets/korea/splits.json'))
    splits = {c: k for k in ('train', 'dev', 'test') for c in sp[k]}
    df = pd.concat([build_set(S, splits) for S in a.sets.split(',')], ignore_index=True)
    order = ['set', 'group', 'stub', 'child', 'task', 'action', 'rep', 'label', 'split', 'stage_failed',
             'reasons', 'setup_reasons', 'bad_cams', 'sync_status', 'sync_detail', 'sync_min_corr',
             'frames_raw', 'n_frames', 'n_usable_frames', 'bone_dev', 'reproj_over_session', 'session_bone_cv',
             'setup', 'open_cam1', 'open_cam2', 'open_cam3', 'notes', 'verdict',
             'video_cam1', 'video_cam2', 'video_cam3']
    df = df[[c for c in order if c in df.columns]].sort_values(['set', 'group', 'child', 'action', 'rep'])
    df.to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(df)} reps')
    print(df.pivot_table(index='group', columns='set', values='stub', aggfunc='size', fill_value=0).to_string())
    miss = (df[['video_cam1', 'video_cam2', 'video_cam3']] == '').sum()
    print('reps missing a video per camera:', miss.to_dict())


if __name__ == '__main__':
    main()
