# OpenPose-on-SMPL calibration outputs (2026-10-16)

Saved from the session scratchpad so the work can resume. `*.npz` are git-ignored.

- `fit_1b/`, `fit_2b/`: smpl_op_fit.py one-bone / `--two-bone`, 11 BioCV adults (P*) + 4 Korea sample children
  (out_v4 loader). No saved poses (older script version).
- `fit_v2/`: one-bone, 11 adults + 63 set-B children (`gt3d` loader, GT3D_v4, B010/B011/B014/B023 excluded), with
  per-frame pose/root/trans/X saved. This is the calibration set.
- `fit_v2_dev/`: the 4 held-out sample children, same settings.
- `vreg.npz`: vertex_regressor.py weights (all / adults-only / kids-only training).
- `scripts/`: scratch helpers. `summ.py <dir>` offsets table; `kidw.py <dir>` kid-weight analysis (needs fit_v2
  layout); `diag.py` single-subject fit + segment-length diagnostics (run from repo root).

Rerun: `python analysis/joint_calib/smpl_op_fit.py gt3d /Volumes/Expansion/MotorDevelopment/Korea/B/GT3D_v4
--exclude B010,B011,B014,B023 --out-dir <dir>` (~1 min/child) and `... biocv data/sampleBiocv --out-dir <dir>`
(P16/P17/P26/P28 2d.npz were added to data/sampleBiocv; calibs come from the Expansion drive).
Findings and open decisions: see the memory note openpose-joint-calibration-plan.
