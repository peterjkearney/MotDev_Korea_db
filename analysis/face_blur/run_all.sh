#!/bin/zsh
# Every face-blur step in order; each skips work already done, so re-running resumes.
#   nohup ./run_all.sh > "$FACEBLUR_OUT/run_all.log" 2>&1 &
set -e
cd "$(dirname "$0")"
PY=${PY:-~/anaconda3/envs/motEnv/bin/python}
ACTION=${ACTION:-P03_CMJM_01}

echo "== $(date) step_1 reenc/haar/deface";  $PY step_1_anonymise.py --action $ACTION --methods reenc haar deface --workers 4
echo "== $(date) step_2 reenc/haar/deface";  $PY step_2_detect.py   --action $ACTION --conds reenc haar deface --workers 2
echo "== $(date) step_3 head";               $PY step_3_head.py     --action $ACTION
echo "== $(date) step_1 oracle";             $PY step_1_anonymise.py --action $ACTION --methods oracle --workers 4
echo "== $(date) step_2 oracle";             $PY step_2_detect.py   --action $ACTION --conds oracle --workers 2
echo "== $(date) step_4 lift"                # one process per detector: they share nothing
for d in yolo rtmpose mediapipe; do
  $PY step_4_lift.py --action $ACTION --dets $d --threads 2 > "${FACEBLUR_OUT:-/Volumes/Sandisk_SSD/MotorDevelopment/FaceBlur}/step_4_$d.log" 2>&1 &
done
wait
if [ -f step_5_report.py ]; then
  echo "== $(date) step_5 report";           $PY step_5_report.py   --action $ACTION
fi
echo "== $(date) done"
