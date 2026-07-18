# Artifact Manifest

## Official champion workflow

Run:

```bash
python nba_win_probability.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs
```

Generates:

- `outputs/april_predictions.csv`
- `outputs/april_descriptive_metrics.csv`
- `outputs/validation_grid.csv`
- `outputs/march_temporal_check_metrics.csv`
- `outputs/march_calibration_bins.csv`
- `outputs/final_model_coefficients.csv`
- `outputs/model_summary.json`
- `outputs/selected_hyperparameters.json`
- `figures/march_model_comparison.png`
- `figures/march_calibration.png`
- `figures/final_model_coefficients.png`

## Enhanced limitations-driven governance

Run:

```bash
python enhanced_governance.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs   --figure-dir figures
```

Generates:

- `outputs/enhanced_model_comparison.csv`
- `outputs/enhanced_candidate_grid.csv`
- `outputs/enhanced_bootstrap_model_differences.csv`
- `outputs/enhanced_monthly_backtest.csv`
- `outputs/enhanced_march_calibration_summary.csv`
- `outputs/enhanced_march_reliability_bins.csv`
- `outputs/enhanced_march_calibration_bootstrap.csv`
- `outputs/enhanced_feature_correlation.csv`
- `outputs/enhanced_feature_vif.csv`
- `outputs/enhanced_coefficient_stability.csv`
- `outputs/april_model_uncertainty.csv`
- `outputs/enhanced_selection_decision.json`
- `figures/enhanced_candidate_comparison.png`
- `figures/monthly_model_stability.png`
- `figures/coefficient_stability.png`
- `figures/march_calibration_uncertainty.png`

## Optional machine-learning challengers

Install:

```bash
python -m pip install -r requirements-challengers.txt
```

Run:

```bash
python challenger_analysis.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs
```

Generates the standalone tree, residual-XGBoost, ensemble, calibration and
SHAP governance artifacts.

## Feature ablation and local timing

Run:

```bash
python ablation_and_timing.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs   --repeats 20
```

Timing is hardware-dependent and should be regenerated on the interview
machine.

## Core regeneration

Run:

```bash
python run_submission.py   --root .   --data /path/to/nba-win-probability-data.csv
```

This reruns the champion, enhanced governance and validator using core
dependencies only.

## Validator

Run:

```bash
python validate_submission.py   --root .   --data /path/to/nba-win-probability-data.csv
```

The validator generates no model artifact.


## Team-specific home-court research

Run:

```bash
python research/team_specific_home_effects.py   --data /path/to/nba-win-probability-data.csv   --output-dir research/outputs   --figure-dir research/figures
```

Generates:

- `research/outputs/team_specific_home_effect_grid.csv`
- `research/outputs/team_specific_home_effect_summary.csv`
- `research/outputs/team_specific_home_effect_bootstrap.csv`
- `research/outputs/team_specific_home_effect_april_descriptive.csv`
- `research/outputs/team_specific_home_effect_decision.json`
- `research/figures/team_specific_home_effect_comparison.png`


## Repository quality utilities

These scripts generate no model output:

- `scripts/check_python.py`: rejects unsupported Python before numerical
  extensions load.
- `scripts/check_repository_policy.py`: rejects private files, source data,
  caches, compiled files, and obvious credentials.
- `scripts/validate_committed_artifacts.py`: validates committed predictions,
  fair odds, uncertainty, governance tables, JSON, and PNG signatures.
- `scripts/run_quality_checks.sh`: runs the complete data-free quality gate.
- `scripts/bootstrap_macos.sh`: creates a clean Python 3.12 environment and
  executes the quality gate.
