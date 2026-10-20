import glob, os, sys, numpy as np
d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, '*.npz'))):
    z = np.load(p, allow_pickle=True)
    rows.append((os.path.basename(p)[:-4], z))
groups = list(rows[0][1]['groups'])
print(f'{"subj":5s} {"size":>5s} {"kidw":>5s} ' + ' '.join(f'{g:^20s}' for g in groups) + '   anchor-resid(knee,ankle,elbow,wrist)')
agg = {}
for s, z in rows:
    L = z['offsets_local_mm']; sz = float(z['size_mm'])
    kw = z['shape'][-1] if len(z['shape']) > 10 else 0
    rn = list(z['residual_names']); r = z['residual_mm']
    an = [np.mean([r[rn.index(sd + j)] for sd in 'LR']) for j in ('Knee', 'Ankle', 'Elbow', 'Wrist')]
    print(f'{s:5s} {sz:5.0f} {kw:5.2f} ' + ' '.join('(' + ','.join(f'{x:+4.0f}' for x in L[i]) + ')   ' for i in range(len(groups)))
          + '  ' + ','.join(f'{x:.0f}' for x in an))
    agg.setdefault('kid' if kw > 0.3 else 'adult', []).append((sz, L))
for k, v in agg.items():
    sz = np.array([a for a, _ in v]); L = np.array([b for _, b in v])
    sc = L / sz[:, None, None] * 1423.0
    print(f'{k} n={len(v)} mean size {sz.mean():.0f}')
    for i, g in enumerate(groups):
        print(f'   {g:10s} mm ' + ','.join(f'{x:+5.1f}' for x in L[:, i].mean(0)) + ' sd ' + ','.join(f'{x:4.1f}' for x in L[:, i].std(0))
              + '  | @1423 ' + ','.join(f'{x:+5.1f}' for x in sc[:, i].mean(0)) + ' sd ' + ','.join(f'{x:4.1f}' for x in sc[:, i].std(0)))
