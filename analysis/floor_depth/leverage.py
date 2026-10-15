"""Depth bias vs floor leverage (camera height / distance) and the effect of the per-subject contact height.
lev numbers for B010/B011/B014 are the 2026-10-14 re-levelled run (same code; the tree was lost with the scratchpad)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.calibration import load_calib
from utils.smpl_lite import J
from floorlib import grounded_ankles

S = sys.argv[1]
REP = {'B010': 'B010_GMS_1_1', 'B011': 'B011_GMS_2_3', 'B014': 'B014_GMS_1_2', 'B023': 'B023_GMS_2_2'}
LEV = {  # config -> cam -> (placed mm, depth bias mm), re-levelled, contact 25 mm
    'B010': {'C8': {'2': (64.5, 42), '1': (78.6, 74), '3': (79.6, 64)}, 'C12': {'2': (47.3, 13), '1': (45.2, 23), '3': (59.6, 25)},
             'C13': {'2': (48.2, 5), '1': (43.5, 24), '3': (57.0, 19)}},
    'B011': {'C8': {'3': (40.0, -17), '1': (58.2, -4), '2': (49.1, -16)}, 'C12': {'3': (64.4, -53), '1': (51.1, -1), '2': (47.2, -19)},
             'C13': {'3': (55.0, -43), '1': (52.7, -4), '2': (47.6, -25)}},
    'B014': {'C8': {'3': (45.7, -29), '1': (32.1, 1), '2': (69.0, -67)}, 'C12': {'3': (43.8, -39), '1': (30.7, -18), '2': (53.4, -51)},
             'C13': {'3': (52.1, -49), '1': (45.8, -42), '2': (76.8, -75)}},
}


def metrics(tree, u, c):
    p = glob.glob(f'{S}/{tree}/{u}/{REP[u]}/Analysis/diagnostics/openpose/{c}/error_metrics_tri.npz')
    if not p:
        return None
    m = np.load(p[0])
    return {str(k): (float(e), float(b)) for k, e, b in zip(m['cameras'], m['err_placed_all'], np.nanmean(m['bias_xyz'], 1)[:, 2])}


rows = []
for u, rep in REP.items():
    X = np.load(f'{S}/korea_lev/{u}/{rep}/Analysis/H36M/openpose_tri_h36m.npz')['kps3d'].astype(float)
    allX = np.concatenate([np.load(p)['kps3d'].astype(float) for p in glob.glob(f'{S}/korea_lev/{u}/*/Analysis/H36M/openpose_tri_h36m.npz')])
    dh = np.median(grounded_ankles(allX)[:, 2]) - 25.0
    hip = np.nanmedian(X[:, J['Hip']], 0)
    lev = LEV.get(u) or {c: metrics('korea_lev', u, c) for c in ('C8', 'C12', 'C13')}
    for cam in ('1', '2', '3'):
        _, _, K, L, _ = load_calib(f'{S}/korea_lev/{u}/{rep}/{cam}.mp4-mocAligned.calib')
        C = -L[:3, :3].T @ L[:3, 3]
        H, D = C[2], np.linalg.norm((hip - C)[:2])
        for c in ('C8', 'C12', 'C13'):
            a, b = lev[c][cam], metrics('korea_ctc', u, c)[cam]
            rows.append((u, cam, c, H, D, dh, a[0], a[1], b[0], b[1]))

print('per camera: H = camera height, D = horizontal distance to the hip (mm, 1.02 m-child units); leverage H/D;')
print('dh = planted-ankle median - 25 mm; predicted depth shift -dh*D/H; observed = bias(contact) - bias(25 mm)\n')
print(f'{"rep":5s}{"cam":>4s}{"cfg":>5s}{"H":>6s}{"D":>6s}{"H/D":>6s}{"dh":>5s} | {"25mm: placed  bias":>19s} | {"own h: placed  bias":>20s} | {"pred":>5s} {"obs":>5s}')
for u, cam, c, H, D, dh, e0, b0, e1, b1 in rows:
    print(f'{u:5s}{cam:>4s}{c:>5s}{H:6.0f}{D:6.0f}{H / D:6.2f}{dh:5.0f} | {e0:9.1f} {b0:+7.0f}   | {e1:9.1f} {b1:+7.0f}    | {-dh * D / H:+5.0f} {b1 - b0:+5.0f}')

r = np.array([x[3:] for x in rows], dtype=float)
print('\nmean placed over 12 cameras:  ' + '   '.join(
    f'{c}: {r[[x[2] == c for x in rows], 3].mean():.1f} -> {r[[x[2] == c for x in rows], 5].mean():.1f}' for c in ('C8', 'C12', 'C13')))
lever = r[:, 0] / r[:, 1]
for c in ('C8', 'C12', 'C13'):
    k = np.array([x[2] == c for x in rows])
    print(f'{c}: |depth bias| (25 mm) vs leverage H/D: corr {np.corrcoef(lever[k], np.abs(r[k, 4]))[0, 1]:+.2f}; '
          f'predicted vs observed shift: corr {np.corrcoef(-r[k, 2] * r[k, 1] / r[k, 0], r[k, 6] - r[k, 4])[0, 1]:+.2f}')
