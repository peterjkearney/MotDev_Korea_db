import glob, os, sys, numpy as np
d = sys.argv[1]
names = ['hip lat', 'hip up', 'hip fwd', 'sh lat', 'sh up', 'sh fwd']
K, A = [], []
for p in sorted(glob.glob(os.path.join(d, '*.npz'))):
    z = np.load(p, allow_pickle=True)
    s = os.path.basename(p)[:-4]
    o = z['offsets_local_mm'][:2].ravel() / float(z['size_mm']) * 1423.0     # scale-free, at adult size
    rn = list(z['residual_names']); r = z['residual_mm']
    q = np.mean([r[rn.index(x)] for x in ('LKnee', 'RKnee', 'LElbow', 'RElbow', 'LWrist', 'RWrist')])
    row = dict(s=s, o=o, kw=float(z['shape'][-1]) if len(z['shape']) > 10 else 0.0, size=float(z['size_mm']),
               n=int(z['n_frames']), q=q, b0=float(z['shape'][0]))
    (K if s.startswith('B') else A).append(row)
kw = np.array([r['kw'] for r in K]); O = np.array([r['o'] for r in K]); OA = np.array([r['o'] for r in A])
q = np.array([r['q'] for r in K])
print(f'children n={len(K)}: kid weight median {np.median(kw):.2f} range {kw.min():.2f}-{kw.max():.2f} (IQR {np.percentile(kw,25):.2f}-{np.percentile(kw,75):.2f}); '
      f'frames median {np.median([r["n"] for r in K]):.0f}; anchor resid median {np.median(q):.1f} mm (max {q.max():.0f})')
print(f'adults   n={len(A)}')
print('\noffsets scaled to size 1423 mm (lat, up, fwd): mean [sd] -- children, adults; children slope per +0.1 kid weight [95% CI], r')
for i, n in enumerate(names):
    X = np.c_[np.ones(len(kw)), kw]; c = np.linalg.lstsq(X, O[:, i], rcond=None)[0]
    e = O[:, i] - X @ c; se = np.sqrt(e @ e / (len(kw) - 2) / ((kw - kw.mean()) ** 2).sum())
    r = np.corrcoef(kw, O[:, i])[0, 1]
    pred0 = c[0]
    print(f'  {n:8s} kids {O[:, i].mean():+6.1f} [{O[:, i].std():4.1f}]   adults {OA[:, i].mean():+6.1f} [{OA[:, i].std():4.1f}]   '
          f'slope {0.1 * c[1]:+5.1f} [{0.1 * (c[1] - 2 * se):+5.1f},{0.1 * (c[1] + 2 * se):+5.1f}] r {r:+.2f}   kid line at w=0: {pred0:+6.1f}')
bad = [r['s'] for r in K if r['q'] > 15]
print('\nchildren with anchor residual > 15 mm (fit or data suspect):', bad)
