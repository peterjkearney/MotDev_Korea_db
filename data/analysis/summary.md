# Adults vs children through the MotionBERT pipeline

Target: leave-one-out triangulated OpenPose, both cohorts. Unit of analysis: subject (camera -> trial -> subject medians; cohort medians with IQR across subjects).

## Coverage

| cohort | subjects | trials | camera rows | rows dropped (shape from <30 frames) |
|---|---|---|---|---|
| adult | 7 | 16 | 110 | 16 (15%) |
| child | 36 | 338 | 998 | 45 (5%) |

## Errors, % of stature (median [IQR] across subjects)

| measure | adult | child | floor adult / child |
|---|---|---|---|
| placement error (MPJPE, no alignment) | 11.10 [10.51-11.26] (n=7) | 9.97 [9.15-11.81] (n=36) | 1.8 / 4.0 |
| pose-shape error (PA-MPJPE) | 4.33 [4.27-4.34] (n=7) | 3.97 [3.69-4.18] (n=36) | 0 / 0 |
| scale-free placement (N-MPJPE) | 5.42 [5.16-5.50] (n=7) | 5.14 [4.58-5.61] (n=36) | 0 / 0 |

The floor is what a *perfect* skeleton scores through single-view PnP; PA-MPJPE and bone ratios have no such floor.

## Bone length ratios, predicted / target (median [IQR] across subjects)

Read child against adult, not against 1.0: OpenPose joints and SMPL-regressed joints are different anatomical points, and that offset is the same in both cohorts.

| bone | adult | child | child - adult | Mann-Whitney p |
|---|---|---|---|---|
| thigh | 1.05 [1.05-1.06] (n=7) | 1.26 [1.23-1.29] (n=36) | +0.203 | 6.21e-08 |
| shank | 0.99 [0.99-1.00] (n=7) | 1.16 [1.11-1.23] (n=36) | +0.167 | 3.65e-05 |
| upperarm | 1.04 [1.01-1.06] (n=7) | 1.06 [1.03-1.08] (n=36) | +0.015 | 0.176 |
| forearm | 1.10 [1.05-1.15] (n=7) | 1.07 [1.05-1.09] (n=36) | -0.037 | 0.223 |
| hip_width | 1.38 [1.38-1.39] (n=7) | 1.30 [1.26-1.35] (n=36) | -0.076 | 0.00379 |
| shoulder_width | 0.93 [0.89-0.93] (n=7) | 0.82 [0.80-0.86] (n=36) | -0.104 | 7.45e-07 |

## Per-joint error, % of stature (variant `smooth_all` -- the per-joint columns follow run_batch's --joint-variant, so these are the *smoothed* skeleton unless it was changed)

| joint | adult | child |
|---|---|---|
| RHip | 11.6 | 10.0 |
| RKnee | 10.2 | 8.2 |
| RAnkle | 11.9 | 7.1 |
| LHip | 12.1 | 10.1 |
| LKnee | 10.3 | 8.0 |
| LAnkle | 10.7 | 7.5 |
| LShoulder | 11.3 | 10.7 |
| LElbow | 9.9 | 10.6 |
| LWrist | 11.7 | 11.8 |
| RShoulder | 12.1 | 10.6 |
| RElbow | 9.9 | 9.9 |
| RWrist | 11.7 | 10.4 |
| Hip | 11.7 | 9.4 |
| Thorax | 11.8 | 11.0 |

## Placement error by camera, % of stature (median over trials)

|   camera |   adult |   child |
|---------:|--------:|--------:|
|        0 |    14.2 |   nan   |
|        1 |   nan   |    11.3 |
|        2 |    10.9 |    10.3 |
|        3 |     8.9 |     9.5 |
|        4 |     8.4 |   nan   |
|        6 |    12.3 |   nan   |
|        7 |     9.4 |   nan   |
|        8 |    11.6 |   nan   |

Korea camera 2 was the least stable calibration parameter; BioCV cameras differ in distance and angle.

## SMPL shape parameters (betas), per-subject medians

| | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 | mean abs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| adult | -0.26 | +0.09 | +0.02 | +0.04 | +0.04 | -0.02 | -0.01 | -0.00 | +0.05 | +0.01 | 0.05 |
| child | -0.27 | -0.02 | -0.00 | +0.02 | +0.01 | -0.03 | -0.01 | -0.01 | -0.04 | +0.00 | 0.04 |

beta_0 is broadly overall size/height in SMPL; large magnitudes mean the fit is pushed toward the edge of the adult shape space.

