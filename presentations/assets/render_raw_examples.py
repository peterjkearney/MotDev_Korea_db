"""Clips for the deck's raw-data slides, drawn from the Korea OpenPose JSONs / Aligned tracks (thicker and clearer
than the dataset's own stick-figure videos).

    good      B010_GMS_1_1: the three cameras after syncing
    switch    B237_GMS_2_4 camera 3: raw 'first person listed' (jumps between child and adult) vs the tracked child
    handheld  B286_GMS_1_1: camera 1 (handheld) next to camera 2 (fixed), with a trail of the mid-hip

    python render_raw_examples.py OUT_DIR
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = '/Volumes/Sandisk_SSD/MotorDevelopment/Korea/B'
OUT = sys.argv[1]
W, H = 640, 360                   # panel size; keypoints are in 1920x1080
SC = W / 1920.0
BG = (0x1C, 0x16, 0x11)           # BGR of #11161C
COL = {'L': (214, 111, 47), 'R': (47, 74, 214), 'C': (230, 230, 230)}   # BGR: blue left, red right
GREY = (110, 110, 110)
BONES = [(1, 8), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13),
         (13, 14), (1, 0), (11, 22), (11, 24), (14, 19), (14, 21)]
SIDE = {2: 'R', 3: 'R', 4: 'R', 9: 'R', 10: 'R', 11: 'R', 22: 'R', 24: 'R',
        5: 'L', 6: 'L', 7: 'L', 12: 'L', 13: 'L', 14: 'L', 19: 'L', 21: 'L'}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FPS = 30


def panel():
    return np.full((H, W, 3), BG, np.uint8)


def draw(img, xy, sc, colour=None, thick=4, conf=0.2):
    ok = sc > conf
    p = (xy * SC).astype(int)
    for a, b in BONES:
        if ok[a] and ok[b]:
            c = colour or COL[SIDE.get(b, 'C')]
            cv2.line(img, tuple(p[a]), tuple(p[b]), c, thick, cv2.LINE_AA)
    for j in np.flatnonzero(ok[:15]):
        cv2.circle(img, tuple(p[j]), thick, colour or COL[SIDE.get(j, 'C')], -1, cv2.LINE_AA)


def label(img, text, sub=None, colour=(240, 240, 240)):
    cv2.putText(img, text, (16, 34), FONT, 0.75, colour, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (16, 62), FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


def writer(name, n_panels):
    path = os.path.join(OUT, name)
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{W * n_panels + 8 * (n_panels - 1)}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                         stdin=subprocess.PIPE)
    return p, path


def hstack(panels):
    gap = np.full((H, 8, 3), (60, 60, 60), np.uint8)
    out = [panels[0]]
    for q in panels[1:]:
        out += [gap, q]
    return np.hstack(out)


def finish(p, path, poster_frame):
    p.stdin.close()
    p.wait()
    cv2.imwrite(path.replace('.mp4', '.jpg'), poster_frame)
    print(path)


def raw_people(stub, cam):
    with open(os.path.join(ROOT, f'{stub}_{cam}.json')) as f:
        data = json.load(f)['data']
    return [[(np.asarray(s['pose'], float).reshape(25, 2), np.asarray(s['score'], float)) for s in fr['skeleton']]
            for fr in data]


# ---------------- good: three synced cameras
z = np.load(os.path.join(ROOT, 'Aligned', 'B010_GMS_1_1.npz'), allow_pickle=True)
names = ['camera 1 (front)', 'camera 2 (right 45)', 'camera 3 (left 45)']
p, path = writer('raw_good_B010.mp4', 3)
poster = None
T = len(z['xy1'])
for t in range(T):
    ps = []
    for c in (1, 2, 3):
        im = panel()
        draw(im, z[f'xy{c}'][t], z[f'score{c}'][t])
        label(im, names[c - 1])
        ps.append(im)
    fr = hstack(ps)
    p.stdin.write(fr.tobytes())
    if t == T // 2:
        poster = fr
finish(p, path, poster)

# ---------------- switch: raw first-listed vs tracked child, camera 3
stub = 'B237_GMS_2_4'
z = np.load(os.path.join(ROOT, 'Aligned', f'{stub}.npz'), allow_pickle=True)
people = raw_people(stub, 3)
fidx = z['frame_index3']
start3 = int(z['start'][2])
first_xy = np.array([fr[0][0] if fr else np.zeros((25, 2)) for fr in people])      # whoever OpenPose lists first
first_sc = np.array([fr[0][1] if fr else np.zeros(25) for fr in people])
p, path = writer('raw_switch_B237.mp4', 2)
best, best_d = None, -1
prev = None
for t in range(len(first_xy)):
    a, b = panel(), panel()
    for xy, sc in (people[t] if t < len(people) else []):
        draw(a, xy, sc, colour=GREY, thick=2)
    draw(a, first_xy[t], first_sc[t], colour=(60, 80, 235), thick=5)
    label(a, 'Raw: first person OpenPose lists', 'grey = everyone detected')
    i = t - start3
    if 0 <= i < len(z['xy3']):
        draw(b, z['xy3'][i], z['score3'][i], colour=(110, 200, 95), thick=5)
    label(b, 'After tracking: the child')
    fr = hstack([a, b])
    p.stdin.write(fr.tobytes())
    if 0 <= i < len(z['xy3']) and first_sc[t][8] > 0.3 and z['score3'][i][8] > 0.3:
        d = np.linalg.norm(first_xy[t][8] - z['xy3'][i][8])   # raw pick vs tracked child: largest = wrong person
        if d > best_d:
            best, best_d = fr, d
finish(p, path, best if best is not None else fr)

# ---------------- handheld: camera 1 vs camera 2 with mid-hip trails
stub = 'B286_GMS_1_1'
cams = {c: raw_people(stub, c) for c in (1, 2)}
T = min(len(v) for v in cams.values())
trail = {1: [], 2: []}
p, path = writer('raw_handheld_B286.mp4', 2)
poster = None
for t in range(T):
    ps = []
    for c, title, sub in ((1, 'Camera 1: handheld', 'trail = mid-hip, last 2 s'),
                          (2, 'Camera 2: on a tripod', 'trail = mid-hip, last 2 s')):
        im = panel()
        dets = cams[c][t]
        if dets:
            xy, sc = dets[0]
            draw(im, xy, sc)
            if sc[8] > 0.3:
                trail[c].append(tuple((xy[8] * SC).astype(int)))
        pts = trail[c][-60:]
        for k in range(1, len(pts)):
            cv2.line(im, pts[k - 1], pts[k], (0, 200, 255), 2, cv2.LINE_AA)
        label(im, title, sub, colour=(120, 170, 255) if c == 1 else (240, 240, 240))
        ps.append(im)
    fr = hstack(ps)
    p.stdin.write(fr.tobytes())
    if t == int(T * 0.8):
        poster = fr
finish(p, path, poster)
