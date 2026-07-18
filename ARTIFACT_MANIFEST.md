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
