# PnP_depth pipeline — child study copy

A working copy of `Python_jetson/PnP_depth_clean`, amended for two work
streams.  The original is untouched; MotionBERT and the model weights are
still read from their original location (`config.py`).

| stream | subjects | 2D input | ground truth |
|---|---|---|---|
| 1 | adults, BioCV | OpenPose (was YOLO) | mocap **and** triangulated OpenPose |
| 2 | children, Korea | OpenPose (from the dataset) | triangulated OpenPose only |

The point of running stream 1 against a *triangulated* target as well as
mocap: Korea has no mocap, so children can only be scored against
triangulated OpenPose.  Scoring the adults the same way gives a like-for-like
baseline, and the adults' mocap-vs-triangulated gap says what that yardstick
itself costs.  The child effect is the difference between the two
triangulated-target results.

## What changed

- **`step_2b` — one body shape per camera.**  It used to average SMPL betas
  across every camera present; now each camera's skeleton uses only that
  camera's betas (`{cam}_final_betas.npz`).  `step_3` reads them per camera.
  Apply to both streams so adults and children get the same treatment.
- **`step_1_openpose_2d.py`** replaces YOLO `step_1` for BioCV: reads
  OpenPose `--write_json` output, picks the subject per frame by proximity
  to projected mocap, writes the same `{cam}_2d.npz` (plus `body25_2d`).
- **`step_1b_triangulate_2d.py`** builds the triangulated-OpenPose target
  on BioCV from the known calibration: `openpose_tri_h36m.npz`, in
  `mocap_h36m.npz`'s layout.  Targets are leave-one-out — the target for
  camera *c* is triangulated without camera *c*, so the input never shapes
  the target it is scored against.
- **`step_1_korea_2d.py`** lays the Korea `GT3D/` output out as trials
  (`user` = subject, `action` = rep, cameras `1,2,3`), with calibs,
  `user_meta.json` and leave-one-out targets, so steps 2a–8 run unchanged.
- **`step_8`** takes `--gt mocap|triangulated`, adds PA-MPJPE, N-MPJPE,
  per-bone proportion ratios and errors as % of stature; writes
  `error_metrics[_tri].npz` so both ground truths can coexist.
- **`utils/openpose.py`** — BODY_25 → H36M-17 with the same midpoint
  conventions as `coco2h36m`, except that a composite joint with a missing
  part is itself missing.  (OpenPose writes missing joints as (0,0,0);
  averaging that into a midpoint would pass the PnP confidence gate.)
- **Spine, Nose and Head are excluded from the triangulated targets' scoring
  mask.**  All three are synthesised, not detected.  Spine matters: the
  pelvis–thorax midpoint sits ~90 mm from mocap's T10-based Spine on P08
  while every detected joint is within 7 mm — a convention, not an error.
- `tools/run_batch.py` knows the new steps, discovers cameras without
  videos, and writes `results_table[_tri].csv`.
- **`step_2a` fills missing joints before MotionBERT sees them**
  (`utils/missing_joints.py`).  OpenPose's missing joints are `(0,0,0)`, and
  composites (Hip, Thorax, Spine, Head) go missing with any of their parts.
  `crop_scale` leaves zeros out of its bounding box but still normalises
  them, so MotionBERT saw the joint in the top-left corner of the crop, and
  through the 81-frame window each one disturbed up to 40 frames either side.
  On User03 camera 06 this, not the detector, was most of the OpenPose-vs-YOLO
  gap.  Missing joints now take the nearest seen position at confidence 0.1
  (below PnP's 0.4 gate); the `{cam}_2d.npz` files are unchanged.  YOLO never
  emits zeros, so its results do not move; **every OpenPose result, Korea
  included, needs rerunning from step_2a.**  `--missing zero` restores the old
  behaviour (ladder rung C).
- **`--json-frames` is `auto` now** (step_1_openpose_2d, run_batch).  The
  JSONs from `run_openpose.py` are numbered by the decimated 60 fps video, but
  both defaults were `original`, so JSON *n* was read as native frame *n*:
  only the first third of the rows found a JSON at all, and those were
  3.3x time-stretched, so the subject matched the projected mocap on a
  handful of frames.  `auto` reads the layout off the JSON dir (`frames.json`,
  else the JSON count against the video's) and the step refuses a reading
  under which most rows have no JSON.  Any OpenPose `{cam}_2d.npz` made
  before this is wrong and must be regenerated from the JSONs.

## Layout

```
{TRIAL_DIR}/{user}/user_meta.json                   {"stature_m": ...}
{TRIAL_DIR}/{user}/{cam}.mp4-mocAligned.calib       w, h, K, L_ext (lab mm -> cam mm), dist
{TRIAL_DIR}/{user}/{action}/markers.c3d             BioCV only
{TRIAL_DIR}/{user}/{action}/openpose/{cam}/*_keypoints.json   BioCV, OpenPose output
{TRIAL_DIR}/{user}/{action}/Analysis/keypoints/{cam}_2d.npz            the 2D steps 2a-8 read
{TRIAL_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_2d.npz   step_1_openpose_2d's own copy
{TRIAL_DIR}/{user}/{action}/Analysis/keypoints/yolo/{cam}_2d.npz       step_1_extract_2d's own copy
{TRIAL_DIR}/{user}/{action}/Analysis/keypoints/mocap_h36m.npz          (step_0)
{TRIAL_DIR}/{user}/{action}/Analysis/keypoints/openpose_tri_h36m.npz   (step_1b / step_1_korea)
```

Both 2D steps used to write only `keypoints/{cam}_2d.npz`, so running one
detector silently replaced the other's files.  Each now keeps its own copy in
`keypoints/{detector}/` and then copies it to `keypoints/{cam}_2d.npz` to
make it the one the later steps use; `--no-activate` skips that copy.

Set `BIOCV_ROOT` (= `TRIAL_DIR`) and `BIOCV_RESULTS` in the environment;
Colab: `%env BIOCV_ROOT=/content/data/BioCV`.

## Stream 1 — adults, OpenPose

OpenPose itself is not pip-installable: it is a C++/CUDA build, done once
per Colab session.  `tools/install_openpose_colab.sh` is the standard recipe
(untested here -- see its notes on the two usual failure points: cuDNN, and
the CMU model server, which has been unreliable for years; the BODY_25
weights may need fetching from a mirror).  Everything after it is batched:

```
bash tools/install_openpose_colab.sh /content/openpose        # 20-40 min, once per session
python3 tools/run_openpose.py --openpose-bin /content/openpose/build/examples/openpose/openpose.bin \
                              --model-folder /content/openpose/models/ [--users P08 --actions ...]
```

For every `{user}/{action}/{cam}.mp4` it decimates the video to the 60 fps
frames step_1 would use (3.3x fewer for OpenPose), runs the binary with
BODY_25 and `--write_json`, and writes `{trial}/openpose/{cam}/`.  Each
finished camera is copied to `RESULTS_DIR` (Drive) immediately, and at the
start of a run anything already complete there is copied back and skipped --
`TRIAL_DIR` is the VM's local disk and does not survive the runtime, so this
is what makes a disconnect cost only the camera in flight.  **Run it before
`run_batch.py` in every session**, even when all cameras are done; that is
what restores the JSONs for `step_1op`.  Measured: ~77 s per camera on a T4
(cuDNN-free build), ~12 min per 9-camera trial; `--users/--actions/--cameras`
pick a subset.

Then, per trial (or via `run_batch.py`, below):

```
python3 step_0_load_mocap.py     --user P08 --action P08_CMJM_01
python3 step_1_openpose_2d.py    --user P08 --action P08_CMJM_01 --json-frames decimated
python3 step_1b_triangulate_2d.py --user P08 --action P08_CMJM_01     # prints tri-vs-mocap cost
python3 step_2a_extract_betas.py --user P08 --action P08_CMJM_01     # GPU
python3 step_2b_finalise_betas.py --user P08 --action P08_CMJM_01
python3 step_3_extract_3d.py     --user P08 --action P08_CMJM_01     # GPU
python3 step_4_PnP.py            --user P08 --action P08_CMJM_01
python3 step_8_spider_error.py   --user P08 --action P08_CMJM_01 --gt mocap
python3 step_8_spider_error.py   --user P08 --action P08_CMJM_01 --gt triangulated
```

`--json-frames original` if OpenPose was run on the native 200 Hz video
instead.  Frame lists come from the video (`frames.json`, written by
`run_openpose.py`, or recomputed from the video for older runs) -- never
from the mocap, which desynchronised trials whose mocap and video lengths
differ.  `tools/check_sync.py` prints the per-trial counts and flags any
trial where the stages disagree.  Batch:

```
python3 tools/run_batch.py --gt triangulated \
    --steps step_0,step_1op,step_1t,step_2a,step_2b,step_3,step_4,step_8
python3 tools/run_batch.py --gt mocap --steps step_8      # second table, same runs
```

## Stream 2 — children, Korea

Copy `Korea/B/GT3D/` (from `build_child_gt.py`) to Colab, then:

```
python3 step_1_korea_2d.py --gt3d /content/data/Korea/GT3D      # writes into $BIOCV_ROOT
python3 tools/run_batch.py --gt triangulated \
    --steps step_2a,step_2b,step_3,step_4,step_8
```

Only reps `build_child_gt.py` marked usable are laid out
(`--include-unusable` to override).  Frames that failed its per-frame gates
are NaN in the target, so `step_8` skips them without knowing why.

## Scoring

`step_8` prints and saves, per camera, for `placed` and `smooth`:

- **MPJPE** (mm and % of stature) — absolute placement.  Score `placed`;
  `smooth` depends on metre-valued smoother settings.
- **PA-MPJPE** — pose shape only (rotation, translation, scale removed).
- **N-MPJPE** — scale removed, placement kept.
- **bone ratios** — median predicted length / median target length per bone
  (thighs, shanks, upper arms, forearms, hip and shoulder width), plus the
  same as fractions of stature.  >1 means the model makes that bone too long.
  This is the direct test of whether limb proportions survive lifting, and
  where an adult prior on a child body should show as a *systematic* bias.
- **bias_xyz** — mean signed error per joint, for direction.

Compare child (triangulated) against adult (triangulated); read adult
(mocap) vs adult (triangulated) as the yardstick cost.  Same detector, same
target construction, same per-camera betas on both sides.

## Why did a number change?  The ladder

```
python3 tools/ladder.py --user User08 --action P08_CMJM_01 --camera 07
```

Builds variant copies of one trial under `{TRIAL_DIR}/../ladder` and runs
each through steps 2a-8, so adjacent rungs differ in exactly one thing:
A YOLO + averaged betas (the original recipe); B YOLO + per-camera betas;
C OpenPose with missing joints zeroed (the pipeline before step_2a filled
them); D OpenPose with missing joints filled from the nearest seen frame (the
pipeline now); E D + a betas gate on the limb joints
only.  All scored against mocap on the one camera, placed and smoothed.
Rung F, printed first, needs no lifting: the raw 2D against projected mocap,
YOLO vs OpenPose.  Rungs A/B regenerate YOLO 2D with `step_1_extract_2d.py`
(needs ultralytics and the model); `--rungs C,D,E` skips them.  Everything
lands in the work root, so `run_batch.py` never sees the variants.

## Looking at a trial

```
python3 tools/render_compare.py --user B010 --action B010_GMS_1_1 --camera 1
python3 tools/render_compare.py --user User08 --action P08_CMJM_01 --camera 07 --gt mocap
```

Writes `Analysis/diagnostics/compare_{cam}[_tri].mp4`: the camera view with
OpenPose's 2D, the placed MotionBERT skeleton and the ground truth projected
onto it (over the source video if it is next to the trial), plus side and
top-down views of the two 3D skeletons.  The all-camera skeleton is drawn by
default because it exists on more frames; `--gt-source loo` draws the
leave-one-out target step_8 actually scores.  It only needs `Analysis/`, so
pointing `BIOCV_ROOT` at `RESULTS_DIR` renders straight from the synced
outputs without re-running anything.

## OpenPose vs YOLO through MotionBERT: threshold sweep

```
python3 tools/threshold_sweep.py [--user User03 --action P03_CMJM_01 --camera 06] \
        [--thresholds 0,0.2,0.4,0.6,0.8] [--missing zero,fill]
```

Standalone: one camera, OpenPose and YOLO 2D side by side, each run through
MotionBERT, single-camera betas, SMPL, PnP and the smoother inside the script
(nothing under the trial's `Analysis/` is touched).  Joints whose 2D
confidence is below the threshold are treated as missing before MotionBERT
sees them, encoded either as `(0,0,0)` or filled from the nearest frame where
the joint was kept.  One video per threshold and encoding, rendered locally
and copied to `{RESULTS_DIR}/threshold_sweep/{user}/{action}/`: OpenPose 2D +
its H36M over the video (top left), YOLO 2D + its H36M (bottom left), and
step_5's top-down view with mocap and both H36M skeletons (right).
`{cam}_summary.csv` has MPJPE / PA-MPJPE against mocap, the frames that
passed the betas gate and PnP successes per run.  Threshold 0 is the plain
OpenPose-vs-YOLO comparison.  It reads `keypoints/openpose/` and
`keypoints/yolo/` (see Layout); add the detector a trial was not run with
using that step's `--no-activate`.

## Floors, measured by feeding the target back through the pipeline

With a *perfect* skeleton (the target itself), PnP placement from one
camera's 2D still lands 30 mm from the target on BioCV (3 px synthetic
noise) and 34–47 mm on Korea B010 (real OpenPose) — 2–5% of stature.  That
is the single-view placement floor; PA-MPJPE and bone ratios are exactly 0
and 1.00 in the same test.  A model cannot beat these numbers, so read
absolute results against them.

## Caveats

- Steps 2a and 3 need CUDA; everything else is numpy/cv2.
- The RTS smoother (`step_4`) can wander badly across runs of frames where
  PnP failed (seen: 339 mm vs 47 mm placed on one Korea camera in the
  plumbing test).  Real MotionBERT output has no missing joints, so this is
  less likely in use, but it is another reason to score `placed`.
- `run_pipeline.sh` is the Jetson/docker runner and is unchanged; use
  `tools/run_batch.py` on Colab.
- Korea 2D is 1920×1080 (`K_1080`); the Korea videos are anonymised renders,
  so YOLO cannot be run on them and nothing here reads them.
