#!/usr/bin/env bash
# run_pipeline.sh — run the full PnP_depth_clean pipeline for one user/action.
#
#   ./run_pipeline.sh User28 P28_CMJM_01                    # all 9 cameras, no renders
#   ./run_pipeline.sh User28 P28_CMJM_01 00,06,07           # camera subset
#   RENDER=1 ./run_pipeline.sh User28 P28_CMJM_01 07        # also 5a/5b/7 videos
#
# Notes:
#  * step_0 runs on the HOST (ezc3d lives only in the miniconda env); every
#    other step runs in the motor-dev container. Container steps write files
#    as root -- run step_0 first so Analysis/keypoints_clean is created by
#    you (root can then write inside it; see conversation history on the
#    ownership rules).
#  * {user}/user_meta.json with {"stature_m": ...} must exist (step_3).
#  * Stops at the first failing step (set -e).

set -euo pipefail

SUBJECT=${1:?usage: run_pipeline.sh <user> <action> [cameras] }
ACTION=${2:?usage: run_pipeline.sh <user> <action> [cameras] }
CAMERAS=${3:-00,01,02,03,04,05,06,07,08}
RENDER=${RENDER:-0}                       # RENDER=1 to also produce the videos
ACCEL_STD=${ACCEL_STD:-10}                # RTS process noise for step_4

PIPE_DIR=/ssd/MotorDevelopment/Python/PnP_depth_clean
META=/ssd/MotorDevelopment/external_datasets/BioCV/$SUBJECT/user_meta.json
[ -f "$META" ] || { echo "ERROR: $META missing (step_3 needs stature_m)"; exit 1; }

indocker() {                              # run one step inside the container
    echo; echo "### $* ###"
    docker run --rm --runtime nvidia \
        -v /ssd/MotorDevelopment:/ssd/MotorDevelopment \
        -w "$PIPE_DIR" \
        motor-dev:latest \
        python3 "$@"
}

echo "### step_0_load_mocap.py (host) ###"
( cd "$PIPE_DIR" && \
  LD_LIBRARY_PATH=/ssd/miniconda3/lib/python3.13/site-packages/ezc3d:${LD_LIBRARY_PATH:-} \
  python3 step_0_load_mocap.py --user "$SUBJECT" --action "$ACTION" )

indocker step_1_extract_2d.py       --user "$SUBJECT" --action "$ACTION"
indocker step_2a_extract_betas.py   --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"
indocker step_2b_finalise_betas.py  --user "$SUBJECT" --action "$ACTION"
indocker step_3_extract_3d.py       --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"
indocker step_4_PnP.py              --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS" \
                                    --process-accel-std "$ACCEL_STD"
indocker step_6_extract_features.py --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"
indocker step_8_spider_error.py     --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"

if [ "$RENDER" = "1" ]; then
    indocker step_5_mocap_comparison.py  --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"
    indocker step_7_animate.py            --user "$SUBJECT" --action "$ACTION" --cameras "$CAMERAS"
fi

echo; echo "### pipeline complete for $SUBJECT/$ACTION (cameras $CAMERAS) ###"
