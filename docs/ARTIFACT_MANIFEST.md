# Artifact Manifest

## Official ensemble workflow

Run:

```bash
python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

Generates:

- `outputs/data_fingerprint.json`
- `outputs/april_predictions.csv`
- `outputs/april_descriptive_metrics.csv`
- `outputs/april_repricing_backtest.csv`
- `outputs/april_component_dispersion.csv`
- `outputs/single_model_benchmark_april_predictions.csv`
- `outputs/single_model_benchmark_april_metrics.csv`
- `outputs/validation_grid.csv`
- `outputs/ensemble_validation_metrics.csv`
- `outputs/march_temporal_check_metrics.csv`
- `outputs/march_calibration_bins.csv`
- `outputs/ensemble_component_summary.csv`
- `outputs/selected_model.json`
- `outputs/model_summary.json`
- `figures/validation_model_comparison.png`
- `figures/march_model_comparison.png`
- `figures/march_calibration.png`
- `figures/ensemble_mean_coefficients.png`
- `figures/april_component_dispersion.png`

## Promotion governance

Run:

```bash
python model_governance.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs \
  --figure-dir figures
```

Generates:

- `outputs/governance_model_comparison.csv`
- `outputs/governance_candidate_grid.csv`
- `outputs/governance_bootstrap_differences.csv`
- `outputs/governance_monthly_backtest.csv`
- `outputs/governance_march_calibration_summary.csv`
- `outputs/governance_march_calibration_bins.csv`
- `outputs/governance_feature_correlation.csv`
- `outputs/governance_feature_vif.csv`
- `outputs/governance_runtime.csv`
- `outputs/governance_selection_decision.json`
- `figures/governance_model_comparison.png`
- `figures/monthly_model_stability.png`
- `figures/march_calibration_reliability.png`

`governance_runtime.csv` is an environment-specific timing diagnostic. Its
elapsed value may vary across machines without changing any probability.

## Optional challengers

`challenger_analysis.py` generates tree, residual-boosting, probability-blend,
calibration, and SHAP artifacts.

`ablation_and_timing.py` generates feature-ablation and hardware-timing
artifacts.

## Historical and shadow research

`research/single_model_governance.py` and
`research/single_model_outputs/` preserve the prior single-model research
stage.

`research/team_specific_home_effects.py` contains the non-promoted
team-specific venue challenger.

## Quality utilities

- `scripts/check_python.py`
- `scripts/check_repository_policy.py`
- `scripts/validate_committed_artifacts.py`
- `scripts/run_quality_checks.sh`
- `scripts/bootstrap_macos.sh`

These scripts generate no official model price.
