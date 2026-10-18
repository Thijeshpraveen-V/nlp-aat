#!/usr/bin/env bash
# Waits for the hp grid to finish, then runs the remaining pipeline (same steps as run_all.sh).
set -euo pipefail
PY=.venv/Scripts/python.exe
while [ ! -f results/runs/hp_lr5e-5_bs16.json ]; do sleep 30; done
sleep 5
ARGS=$("$PY" src/select_config.py | tail -1)
echo "selected: $ARGS"
LR_BS=$(echo "$ARGS" | sed 's/--epochs [0-9]*//')
run() { "$PY" src/finetune.py "$@" 2>&1 | grep -E "^\[|^CV|Error|error" || true; }
run --tag main $ARGS --save_model
run --tag abl_ep5 $LR_BS --epochs 5
run --tag abl_cw $ARGS --class_weights
run --tag abl_os $ARGS --oversample
run --tag abl_trunc $ARGS --truncate
run --tag abl_raw $ARGS --variant raw
run --tag cv_main $ARGS --cv
"$PY" src/benchmark.py 2>&1 | tail -1
"$PY" src/evaluate.py
"$PY" src/error_analysis.py
echo PIPELINE_DONE
