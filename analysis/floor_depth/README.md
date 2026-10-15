# Floor and depth diagnostics (Korea)

Experiments from 2026-10-14/15 on why step_4b's placed skeleton is off in depth (same sign on every camera of a
rep).  All of them work on a COPY of an OUT_DIR (e.g. `data/sampleKorea/out_v4`) -- `relevel.py` rewrites
targets and calibs, `run_override.py` overwrites a configuration's outputs.  Run from `pipeline/`.

| script | what it does |
|---|---|
| `floorlib.py` | planted-ankle samples (ankle speed < 3 mm/frame, within 30 mm of the rep's low ankle) and the plane through them |
| `relevel.py <tree> B023 ...` | rotates each subject's lab frame so the planted-ankle plane is level (flat-floor assumption); targets and calibs get the same rigid transform, camera-frame data unchanged |
| `run_override.py` | step_4b with formulations settings overridden (`--set w_floor_contact=3e4,placement=ray`, `BALANCE`), `--contact planted` = the subject's planted-ankle median height |
| `rayplane.py <S> <tree> <cfg>` | grounded ankles: the camera's own 2D ankle ray intersected with the floor at the contact height vs step_4b's placed ankle, both against the target |
| `variant_summary.py` | several trees side by side: placed, PA, grounded-ankle depth error, lean along the line of sight, 2D RMS |
| `leverage.py`, `weight_summary.py` | records of the 2026-10-15 runs (hard-coded tree names and numbers): depth shift vs -dh*D/H, contact weight 3e3/3e4/3e5 |

Findings so far (4 sample reps, C12): the GT floor is tilted against the planted-ankle plane (B023 4.4 deg);
planted ankles sit 13-27 mm above it, not 25; a contact-height change dh shifts depth by about -dh*D/H (camera
height / distance); the ray-plane foot is within ~20 mm but w_floor_contact 3e3 let the placed foot float 10-23 mm
high (up to 90 mm depth error) -- 3e4 fixes the feet; what remains is lean along each camera's line of sight
(upper body too far from the camera in 11/12 views), invisible to the 2D term.
