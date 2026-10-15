"""Per contact weight: placed / PA (step_8), grounded placed-ankle depth error and height, final 2D RMS (log)."""
import glob, os, re, sys
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.smpl_lite import J
from floorlib import grounded_ankles

S = sys.argv[1]
REP = {'B010': 'B010_GMS_1_1', 'B011': 'B011_GMS_2_3', 'B014': 'B014_GMS_1_2', 'B023': 'B023_GMS_2_2'}
W = ('3e3', '3e4', '3e5')


def rms_from_log(w):
    out, user = {}, None
    for line in open(f'{S}/w{w}.log'):
        m = re.match(r'##### (B\d+)', line)
        if m:
            user = m.group(1)
        m = re.match(r'\s+(\d): stage 2 .*reprojection RMS ([\d.]+) px', line)
        if m:
            out[(user, m.group(1))] = float(m.group(2))
    return out


def ankle_stats(tree, u, cam, h):
    base = f'{S}/{tree}/{u}/{REP[u]}/Analysis'
    tgt = np.load(f'{base}/H36M/openpose_tri_h36m.npz')['kps3d'].astype(float)
    p = np.load(f'{base}/PnP/openpose/C12/{cam}_pnp.npz', allow_pickle=True)
    R, t = p['cam_L_ext'][:3, :3], p['cam_L_ext'][:3, 3]
    placed = p['kps_H36M_placed'].astype(float) * 1000.0
    dz, hz = [], []
    for col, a in enumerate(('RAnkle', 'LAnkle')):
        g = p['foot_grounded'][:, col] & np.isfinite(tgt[:, J[a]]).all(1) & np.isfinite(placed[:, J[a]]).all(1)
        dz.append(placed[g, J[a], 2] - (tgt[g, J[a]] @ R.T + t)[:, 2])
        hz.append(((placed[g, J[a]] - t) @ R)[:, 2])
    return np.median(np.concatenate(dz)), np.median(np.concatenate(hz))


rms = {w: rms_from_log(w) for w in W}
print('C12, re-levelled, per-child contact height.  Each cell: weight 3e3 / 3e4 / 3e5')
print(f'{"rep":5s}{"cam":>4s} | {"placed mm":>17s} | {"PA mm":>17s} | {"ankle depth err":>17s} | {"ankle height (h)":>22s} | {"2D RMS px":>17s}')
agg = {w: [] for w in W}
for u, rep in REP.items():
    allX = np.concatenate([np.load(p)['kps3d'].astype(float) for p in glob.glob(f'{S}/korea_w3e3/{u}/*/Analysis/H36M/openpose_tri_h36m.npz')])
    h = np.median(grounded_ankles(allX)[:, 2])
    ms = {w: np.load(f'{S}/korea_w{w}/{u}/{rep}/Analysis/diagnostics/openpose/C12/error_metrics_tri.npz') for w in W}
    for k, cam in enumerate(ms[W[0]]['cameras']):
        cam = str(cam)
        pl = [float(ms[w]['err_placed_all'][k]) for w in W]
        pa = [float(ms[w]['pa_mpjpe'][k]) for w in W]
        an = [ankle_stats(f'korea_w{w}', u, cam, h) for w in W]
        rr = [rms[w].get((u, cam), np.nan) for w in W]
        for i, w in enumerate(W):
            agg[w].append((pl[i], pa[i], abs(an[i][0]), rr[i]))
        f = lambda v, fmt: ' / '.join(format(x, fmt) for x in v)
        print(f'{u:5s}{cam:>4s} | {f(pl, "5.1f"):>17s} | {f(pa, "5.1f"):>17s} | {f([a[0] for a in an], "+5.0f"):>17s} | '
              f'{f([a[1] for a in an], "4.0f"):>16s} ({h:2.0f}) | {f(rr, "5.2f"):>17s}')
print('\nmean over 12 cameras:')
for w in W:
    a = np.array(agg[w])
    print(f'  w {w}: placed {a[:, 0].mean():5.1f}  PA {a[:, 1].mean():4.1f}  |ankle depth err| {a[:, 2].mean():4.0f}  2D RMS {np.nanmean(a[:, 3]):.2f} px')
