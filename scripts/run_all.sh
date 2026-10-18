#!/usr/bin/env bash
# Reproduce every number and figure from scratch (GPU recommended; ~1.5 h on an RTX 4050).
# Usage (Git Bash / Linux, from the repo root):  bash scripts/run_all.sh
set -euo pipefail
PY=${PY:-.venv/Scripts/python.exe}
[ -x "$PY" ] || PY=.venv/bin/python
[ -f .env ] && { set -a; . ./.env; set +a; }

[ -f "data/raw/Entity Recognition in Resumes.json" ] || \
  "$PY" -m kaggle datasets download -d dataturks/resume-entities-for-ner -p data/raw --unzip

"$PY" src/log_env.py
"$PY" src/data_prep.py                       # Phase 1
"$PY" src/pretrained_eval.py                 # Phase 2A (before)
for bs in 8 16; do for lr in 2e-5 3e-5 5e-5; do  # Phase 2B grid, selected on validation F1
  "$PY" src/finetune.py --tag "hp_lr${lr}_bs${bs}" --lr "$lr" --batch_size "$bs" --epochs 10
done; done
ARGS=$("$PY" src/select_config.py | tail -1)
"$PY" src/finetune.py --tag main $ARGS --save_model   # final "after" model
LR_BS=$(echo "$ARGS" | sed 's/--epochs [0-9]*//')
"$PY" src/finetune.py --tag abl_ep5 $LR_BS --epochs 5  # Phase 3 ablations
"$PY" src/finetune.py --tag abl_cw $ARGS --class_weights
"$PY" src/finetune.py --tag abl_os $ARGS --oversample
"$PY" src/finetune.py --tag abl_trunc $ARGS --truncate
"$PY" src/finetune.py --tag abl_raw $ARGS --variant raw
"$PY" src/finetune.py --tag cv_main $ARGS --cv          # 5-fold CV
"$PY" src/benchmark.py                                   # Phase 4 efficiency (CPU)
"$PY" src/evaluate.py                                    # tables + figures + results/summary.json
"$PY" src/error_analysis.py                              # Phase 5
"$PY" src/make_samples.py                                # demo inputs
