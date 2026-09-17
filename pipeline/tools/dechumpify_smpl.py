#!/usr/bin/env python3
"""Rewrite SMPL_NEUTRAL.pkl so it contains only plain numpy arrays.

The stock SMPL pickle stores `shapedirs` as a chumpy object and the two
J_regressors as scipy.sparse matrices, so `pickle.load` has to import chumpy
to reconstruct it. chumpy is unmaintained and does not import on Python >= 3.11
(it calls the removed `inspect.getargspec`), and `scipy.sparse.csc` is itself
deprecated. Both are only needed to *read* the file, so we read it once here
without chumpy installed -- via an unpickler that swaps in a stub class -- and
write a version that needs neither.

smplx calls `to_np()` on these fields, which densifies sparse input anyway, so
the rewritten file is equivalent for model construction.

    python3 tools/dechumpify_smpl.py \
        ../MotionBERT/data/mesh/SMPL_NEUTRAL.pkl \
        ../MotionBERT/data/mesh/SMPL_NEUTRAL.pkl
"""
import argparse
import pickle
import shutil

import numpy as np


class _ChStub:
    """Stands in for chumpy.ch.Ch while unpickling.

    chumpy's Ch keeps its value in the 'x' key of its pickled state
    (Ch.dterms == ['x']), so capturing that is enough to recover the array.
    """
    def __setstate__(self, state):
        self.x = state['x']


class _NoChumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split('.')[0] == 'chumpy':
            return _ChStub
        return super().find_class(module, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()

    with open(args.src, 'rb') as f:
        raw = _NoChumpyUnpickler(f, encoding='latin1').load()

    clean = {}
    for key, val in raw.items():
        if isinstance(val, _ChStub):
            clean[key] = np.asarray(val.x)
            print(f'  {key:20s} chumpy  -> ndarray {clean[key].shape}')
        elif 'scipy.sparse' in str(type(val)):
            clean[key] = np.asarray(val.todense())
            print(f'  {key:20s} sparse  -> dense   {clean[key].shape}')
        else:
            clean[key] = val

    if args.src == args.dst:                      # keep the original around
        shutil.copyfile(args.src, args.src + '.orig')
        print(f'  original saved to {args.src}.orig')

    with open(args.dst, 'wb') as f:
        pickle.dump(clean, f, protocol=4)
    print(f'wrote {args.dst}')


if __name__ == '__main__':
    main()
