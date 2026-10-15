#!/usr/bin/env python3
"""anatomical_copy.py -- score an OpenPose-style configuration against mocap fairly.

A configuration run with joints='op' / 'op_hs' outputs OpenPose-style H36M joints (hips 25 mm lateral and 26 mm
behind the joint centre, ...), which mocap's joint centres would penalise for the definition alone.  This writes a
copy of each {cam}_pnp.npz under config '{config}a' with kps_H36M_placed / _smooth rebuilt from the placed SMPL-24
joints (smpl_lite.SMPL24_FOR_H36M, Hip / Thorax / Spine midpoints; Nose / Head NaN), so step_8 --config {config}a
scores the same body at its joint centres.

    python3 ../analysis/joint_calib/anatomical_copy.py <tree> C15 [C16 ...]       (run from pipeline/)
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, '.')
from utils.smpl_lite import J, SMPL24, SMPL24_FOR_H36M  # noqa: E402


def h36m_from_smpl24(S):
    H = np.full(S.shape[:-2] + (17, 3), np.nan, dtype=S.dtype)
    for h, s in SMPL24_FOR_H36M.items():
        H[..., J[h], :] = S[..., SMPL24.index(s), :]
    H[..., J['Hip'], :] = 0.5 * (H[..., J['RHip'], :] + H[..., J['LHip'], :])
    H[..., J['Thorax'], :] = 0.5 * (H[..., J['RShoulder'], :] + H[..., J['LShoulder'], :])
    H[..., J['Spine'], :] = 0.5 * (H[..., J['Hip'], :] + H[..., J['Thorax'], :])
    return H


def main():
    tree, configs = sys.argv[1], sys.argv[2:]
    for cfg in configs:
        n = 0
        for p in glob.glob(os.path.join(tree, '*', '*', 'Analysis', 'PnP', '*', cfg, '*_pnp.npz')):
            d = dict(np.load(p, allow_pickle=True))
            d['kps_H36M_placed'] = h36m_from_smpl24(d['kps_SMPL24_placed'])
            d['kps_H36M_smooth'] = h36m_from_smpl24(d['kps_SMPL24_smooth'])
            out = os.path.join(os.path.dirname(os.path.dirname(p)), cfg + 'a', os.path.basename(p))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.savez(out, **d)
            n += 1
        print(f'{cfg} -> {cfg}a: {n} files')


if __name__ == '__main__':
    main()
