# child_lifting — the PnP_depth pipeline for the child study

Started as a copy of `Python_jetson/PnP_depth_clean`, amended for two work
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

## Code layout

```
korea_gt/               (repo root) Korea raw -> GT3D: camera sync, self-calibration (camcal),
                        triangulated ground truth (build_child_gt).  Runs once, before the pipeline.
pipeline/
  config.py             where things are: data roots, MotionBERT, one path function per output
  formulations.py       which placement methods step_4b runs and step_8 scores (C0-C9, L1-L6, R1)
  step_2a ... step_8    the lifter and its scoring, the same for both cohorts
  utils/                shared maths (SMPL, triangulation, metrics, smoother, OpenPose conversion)
  tools/                cohort-independent: results_table, compare_configs, dechumpify_smpl
  datasets/biocv/       adults' inputs: mocap (step_0), YOLO / OpenPose 2D (step_1*), triangulated
                        target (step_1b), the OpenPose runner and its checks, the step_7 render
  datasets/korea/       children's inputs: step_1_korea_2d (GT3D -> trials), run_korea (end to end)
analysis/               (repo root) adults vs children from the results tables
```

Run everything from `pipeline/`; the scripts under `datasets/` find `config` and `utils`
themselves, so `python3 datasets/korea/step_1_korea_2d.py ...` works from there.

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
  `mocap_h36m.npz`'s layout.  It holds the all-camera triangulation and, per
  camera, a leave-one-out one.  **Scoring and renders use the all-camera
  solution**: the question is how close lifting + placement gets to the truth,
  and that is the best estimate of it -- with three cameras and estimated
  calibration on Korea the leave-one-out target is too weak to be the
  yardstick.  `--tri loo` (step_5, step_8) selects it where the circularity of
  scoring a camera against a target its own 2D helped build matters more.
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
  behaviour.
- **OpenPose output is one JSON per video, with its frame numbers.**
  `run_openpose.py` decimates the video before OpenPose sees it, so OpenPose's
  own per-frame files are numbered by the decimated video -- and
  `step_1_openpose_2d` defaulted to reading JSON *n* as native
  frame *n*.  Only the first third of the rows found a JSON at all, 3.3x
  time-stretched, so the subject matched the projected mocap on a handful of
  frames.  The per-video `{cam}_openpose.json` now carries `native_frame_idx`,
  and `--json-frames` is gone.  Any OpenPose `{cam}_2d.npz` from the old
  results folder is wrong; re-import the JSONs and rerun step 1op.

## Layout

Inputs are read from `BIOCV_ROOT` (= `TRIAL_DIR`), the copy of BioCV unzipped
on the VM's local disk.  Outputs go to `BIOCV_OUT` (= `OUT_DIR`, default
`/content/drive/MyDrive/MotorDevelopment/Data/BioCV`), the original BioCV
folder on Drive, inside each action's `Analysis/`.  Nothing is written
locally, so nothing needs syncing and nothing is lost with the runtime.  Each
output's location is one function in `config.py`, used by the step that
writes it and every step that reads it.

```
{TRIAL_DIR}/{user}/{cam}.mp4-mocAligned.calib       w, h, K, L_ext (lab mm -> cam mm), dist
{TRIAL_DIR}/{user}/user_meta.json                   {"stature_m": ...}
{TRIAL_DIR}/{user}/{action}/markers.c3d, {cam}.mp4

{OUT_DIR}/{user}/{action}/Analysis/H36M/mocap_h36m.npz                        step_0
{OUT_DIR}/{user}/{action}/Analysis/keypoints/yolo/{cam}_2d.npz               step_1_extract_2d
{OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_openpose.json     tools/run_openpose
{OUT_DIR}/{user}/{action}/Analysis/keypoints/openpose/{cam}_2d.npz            step_1_openpose_2d
{OUT_DIR}/{user}/{action}/Analysis/H36M/openpose_tri_h36m.npz, tri_vs_mocap.npz   step_1b_triangulate_2d
{OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_betas.npz               step_2a  (--detector)
{OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_final_betas.npz         step_2b
{OUT_DIR}/{user}/{action}/Analysis/mesh/{detector}/{cam}_mesh_pose.npz           step_3
{OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{cam}_pnp.npz                  step_4
{OUT_DIR}/{user}/{action}/Analysis/diagnostics/{cam}_pnp_depth_vs_mocap.mp4        step_5 (both detectors)
{OUT_DIR}/{user}/{action}/Analysis/features/{detector}/{cam}_features.npz        step_6
{OUT_DIR}/{user}/{action}/Analysis/diagnostics/{detector}/error_metrics[_tri].npz, spider_error_*.png   step_8 (--gt)
```

Steps 0 to 8 are moved, except the step_7 render (2c and 4b, the shape fit and
the configuration runner, were added on the new layout).  From 2a on, `--detector
openpose|yolo` (default openpose) says whose 2D a step works from, and its
outputs go to a matching subfolder.  Run with no `--user` / `--action` they
do every trial, skip what is already done (`--force` to redo) and carry on
past a failure, so re-running after a disconnect resumes; `--dry-run` lists
the work.  **Still on the old layout:** the `step_7` render and
`datasets/biocv/check_sync.py`.

## Stream 1 — adults, OpenPose

OpenPose itself is not pip-installable: it is a C++/CUDA build, done once
per Colab session.  `datasets/biocv/install_openpose_colab.sh` is the standard recipe
(untested here -- see its notes on the two usual failure points: cuDNN, and
the CMU model server, which has been unreliable for years; the BODY_25
weights may need fetching from a mirror).  Everything after it is batched:

```
bash datasets/biocv/install_openpose_colab.sh /content/openpose        # 20-40 min, once per session
python3 datasets/biocv/run_openpose.py --openpose-bin /content/openpose/build/examples/openpose/openpose.bin \
                              --model-folder /content/openpose/models/ [--users P08 --actions ...]
```

For every `{user}/{action}/{cam}.mp4` it decimates the video to the 60 fps
frames step_1 would use (3.3x fewer for OpenPose), runs the binary with
BODY_25 and `--write_json` into a temp dir, and folds the per-frame files into
ONE `{cam}_openpose.json` per video under `OUT_DIR`, with the native frame
number of every entry.  A camera is done when that file exists, so re-running
the same command after a disconnect resumes, losing at most the camera in
flight.  Per-frame JSON dirs from the old results folder need not be
recomputed: `--import-from /content/drive/MyDrive/MotorDevelopment/results`
folds every complete one into the new file (no OpenPose build needed for
that).  Measured: ~77 s per camera on a T4 (cuDNN-free build), ~12 min per
9-camera trial; `--users/--actions/--cameras` pick a subset.

Then, per trial (or with no `--user` / `--action` for every trial):

```
python3 datasets/biocv/step_0_load_mocap.py     --user P08 --action P08_CMJM_01
python3 datasets/biocv/step_1_openpose_2d.py    --user P08 --action P08_CMJM_01
python3 datasets/biocv/step_1b_triangulate_2d.py --user P08 --action P08_CMJM_01     # prints tri-vs-mocap cost
python3 step_2a_extract_betas.py --user P08 --action P08_CMJM_01     # GPU
python3 step_2b_finalise_betas.py --user P08 --action P08_CMJM_01
python3 step_3_extract_3d.py     --user P08 --action P08_CMJM_01     # GPU
python3 step_4_PnP.py            --user P08 --action P08_CMJM_01
python3 step_8_spider_error.py   --user P08 --action P08_CMJM_01 --gt mocap
python3 step_8_spider_error.py   --user P08 --action P08_CMJM_01 --gt triangulated
```

The JSON records which native frame each entry is, so the 2D, the video and
the mocap share one clock whether OpenPose ran on the decimated or the native
video.  `datasets/biocv/check_sync.py` prints the per-trial counts and flags any
trial where the stages disagree.

## Stream 2 — children, Korea

The steps are not BioCV-specific: they take every path from `config.py`, which
reads `BIOCV_ROOT` / `BIOCV_OUT` from the environment, and find their work by
looking for files under the output root.  So the children run through the same
commands once the output root points at a Korea folder and the GT3D files
(from `build_child_gt.py`) are laid out in it as trials:

```
%env BIOCV_OUT=/content/drive/MyDrive/MotorDevelopment/Data/Korea
python3 datasets/korea/step_1_korea_2d.py --gt3d /content/drive/MyDrive/.../Korea/B/GT3D    # user = subject, action = rep, cameras 1,2,3
python3 step_2a_extract_betas.py ; python3 step_2b_finalise_betas.py ; python3 step_3_extract_3d.py
python3 step_4_PnP.py ; python3 step_6_extract_features.py
python3 step_8_spider_error.py --gt triangulated
python3 step_5_mocap_comparison.py
```

There is no local copy of anything for this dataset, so the layout step puts
the calibs and `user_meta.json` in the output root too, where `config` looks
second.  Everything skips what is already done.

Only reps `build_child_gt.py` marked usable, of subjects it marked usable, are
laid out (`--include-unusable` to override).  The target is gated PER JOINT
(`--gate joint`, the default): every cleanly triangulated joint is kept, and a
bone further than 25% from the session median -- the label-swap signature --
loses its two joints.  The builder's whole-frame gate (`--gate frame`) threw
away two thirds of the frames on a typical rep.

### Configurations: the child recipe and its ablations

`step_4` is one recipe (MotionBERT's shape rescaled to the stature, rigid PnP
per frame).  `formulations.py` lists the alternatives that were worked out on the
sample reps, as a CHAIN from the original pipeline to the child recipe (C0-C9,
one change per step), a LEAVE-ONE-OUT set from the recipe (L1-L6) and
references; `python3 formulations.py` prints the legend.  The child recipe (C8) is:
the AGORA kid-blend shape fitted to the subject's bone lengths, MotionBERT's
rotations as the initialisation only, the global rotation frozen to the
per-frame ray orientation plus one constant per trial, the translation and the
23 joint angles refined against the 2D (spine/nose/head left out, whose H36M
and OpenPose definitions differ), feet pinned while grounded, a floor hinge,
and the grounded ankle held at its height above the calibrated floor.

```
python3 step_2c_fit_shape.py ; python3 step_2c_fit_shape.py --cohort    # shapes per subject, then the cohort median
python3 step_4b_refine.py [--configs chain|loo|C8,C9] [--device cuda]   # every trial x configuration, skip-if-done
python3 step_8_spider_error.py --gt triangulated --config all           # score each (PnP/{det}/{config}/ -> diagnostics/{det}/{config}/)
python3 tools/results_table.py --gt triangulated                        # the long table: one row per trial x camera x config
python3 tools/compare_configs.py [--baseline C8]                        # wide views per metric, subject medians, paired differences
python3 datasets/korea/run_korea.py --gt3d .../Korea/B/GT3D                      # all of the above in order, resumable
```

Layout additions:

```
{OUT_DIR}/{user}/shapes.npz, {OUT_DIR}/cohort_shapes.npz                       step_2c
{OUT_DIR}/{user}/{action}/Analysis/PnP/{detector}/{config}/{cam}_pnp.npz         step_4b (step_4's format + the config)
{OUT_DIR}/{user}/{action}/Analysis/diagnostics/{detector}/{config}/error_metrics_tri.npz   step_8 --config
{OUT_DIR}/results_table_tri.csv, configs_wide_*.csv, configs_by_subject.csv, configs_summary.txt
```

`utils/smpl_lite.py` holds the SMPL pieces the shape fit and the refinement
need (linear shape model, kid blend, skinning at the regressor vertices);
the kid template is AGORA's `smpl_kid_template.npy`, a licensed asset kept out
of git like the SMPL pickle and the MotionBERT weights: put it next to
`SMPL_NEUTRAL.pkl` in `MotionBERT/data/mesh` on Drive (or in the repo's local
`models/`, or point `SMPL_KID_TEMPLATE` at it).  Aggregate camera -> trial -> subject -> cohort; the
subject is the unit of evidence, and the paired differences per subject
against the recipe are what decide a configuration.

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

## Looking at a trial

```
python3 step_5_mocap_comparison.py --user User03 --action P03_CMJM_01 --cameras 06,07      # BioCV
python3 step_5_mocap_comparison.py --user B010 --action B010_GMS_1_1                        # Korea
```

Left, the camera's view with the smoothed PnP-placed skeleton from each
detector (OpenPose blue, YOLO green) over the source video; right, top-down
in the camera's frame with mocap, the all-camera triangulated target and the
same skeletons, fixed axes, clip-level MPJPE in the legend.  Korea has no
footage and no mocap: the left panel is then a blank canvas of the camera's
size carrying the dataset's OpenPose keypoints (yellow) and the triangulated
target projected into the camera (orange) as well, and the error is against
the triangulated target.  `--draw-2d` / `--draw-target` add those layers
over a video too.  Written to `Analysis/diagnostics/{cam}_pnp_depth_vs_mocap.mp4`.

## Checking OpenPose against the mocap by eye

```
python3 datasets/biocv/render_openpose_check.py --user User03 --action P03_CMJM_01 --camera 01 [--slow 4]
```

The source video, cropped to the subject, with every person OpenPose found
(the one the matcher picks in orange, others grey), the confidence written
beside each joint (red under `--conf`), the projected mocap in green, and a
status line per frame: people found, the pick, its distance over the joints
it has, and ACCEPTED / REJECTED under step_1_openpose_2d's rule.  Written to
`Analysis/diagnostics/openpose/{cam}_openpose_check.mp4`.

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
- Korea 2D is 1920×1080 (`K_1080`); the Korea videos are anonymised renders,
  so YOLO cannot be run on them and nothing here reads them.
