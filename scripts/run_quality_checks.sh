#!/usr/bin/env bash
set -euo pipefail

python scripts/check_python.py
python -m pip check
python -m compileall -q \
  nba_win_probability.py \
  enhanced_governance.py \
  challenger_analysis.py \
  ablation_and_timing.py \
  project_runtime.py \
  run_submission.py \
  validate_submission.py \
  research \
  scripts \
  tests
python -m pytest -q
python scripts/check_repository_policy.py
python scripts/validate_committed_artifacts.py
git diff --check

echo "PASS: all supported-environment quality checks completed."
