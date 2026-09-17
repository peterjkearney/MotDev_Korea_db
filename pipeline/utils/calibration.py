import numpy as np
import cv2

def load_calib(path):
    with open(path) as f:
        lines = [l.strip() for l in f]
    lines = [l for l in lines if l != '']
    w, h = int(lines[0]), int(lines[1])
    K = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)])
    L = np.array([[float(x) for x in lines[i].split()] for i in range(5, 9)])
    dist = np.array([float(x) for x in lines[9].split()])
    return w, h, K, L, dist

def reproject(pts3d_cam, K, dist_cv):
    pts2d, _ = cv2.projectPoints(pts3d_cam.astype(np.float64),
                                  np.zeros(3), np.zeros(3), K, dist_cv)
    return pts2d.reshape(-1, 2)

def save_calib(path, w, h, K, L, dist):
    """Write the same text format load_calib reads (blank lines are ignored
    by the reader, so the layout matches BioCV's files exactly)."""
    K, L, dist = np.asarray(K, float), np.asarray(L, float), np.asarray(dist, float).ravel()
    with open(path, 'w') as f:
        f.write(f'{int(w)}\n{int(h)}\n')
        for row in K:
            f.write(' '.join(f'{v:.10g}' for v in row) + ' \n')
        f.write('\n')
        for row in L:
            f.write(' '.join(f'{v:.10g}' for v in row) + ' \n')
        f.write('\n')
        f.write(' '.join(f'{v:.10g}' for v in dist) + ' \n')
