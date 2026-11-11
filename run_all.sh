#!/usr/bin/env bash
# End-to-end reproduction of sub_H_july_trained_resid.csv
# Expects data/train.csv, data/test.csv, data/sample_submission.csv (official Kaggle files) next to this script,
# or DATA_DIR pointing to a folder with them.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-python3}

echo "== 1/3 feature selection (~5 min) =="
$PY select_features.py
echo "== 2/3 train + predict (~10 s) =="
$PY train_predict.py
echo "== 3/3 validation scores (~1 min) =="
$PY validate.py
