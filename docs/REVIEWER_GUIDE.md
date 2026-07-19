# Reviewer Guide

This page provides a direct screen-share and code-review route.

## Two-minute route

1. Read [`SUMMARY.md`](../SUMMARY.md).
2. Open [`outputs/april_predictions.csv`](../outputs/april_predictions.csv).
3. Confirm the locked architecture in
   [`outputs/selected_model.json`](../outputs/selected_model.json).

## Ten-minute route

### 1. State the target

The model prices the probability that the listed home team wins each April
game from a strict March 31 performance snapshot.

### 2. Show the information timestamp

Open [`nba_win_probability.py`](../nba_win_probability.py) and search:

```text
build_sequential_features
```

Point out the order:

1. Read pregame state.
2. Store the feature row.
3. Observe the result.
4. Update state.

### 3. Show the three sports signals

Search:

```text
feature_values
```

Every component uses the same home-minus-away signals:

- Net wins.
- Cumulative point margin.
- Evidence-weighted recent point margin.

### 4. Show component fitting

Search:

```text
make_model
```

Each component is a training-only standardized, L2-regularized logistic
paired-comparison model. `component_predictions` fits the complete fixed grid.

### 5. Show ensemble aggregation

Search:

```text
component_predictions
ensemble_probability
```

The official price is the arithmetic mean of 40 fixed component
probabilities.

### 6. Show the promotion evidence

Open:

- [`outputs/ensemble_validation_metrics.csv`](../outputs/ensemble_validation_metrics.csv)
- [`outputs/march_temporal_check_metrics.csv`](../outputs/march_temporal_check_metrics.csv)
- [`outputs/governance_monthly_backtest.csv`](../outputs/governance_monthly_backtest.csv)
- [`outputs/governance_bootstrap_differences.csv`](../outputs/governance_bootstrap_differences.csv)

Say:

> The ensemble improves the target-relevant January–February validation score,
> preserves the direction in March governance, and reduces dependence on one
> nearly tied grid point. The gain is modest and not statistically decisive.

### 7. Show interpretability

Open:

- [`outputs/ensemble_component_summary.csv`](../outputs/ensemble_component_summary.csv)
- [`outputs/april_component_dispersion.csv`](../outputs/april_component_dispersion.csv)
- [`outputs/single_model_benchmark_april_predictions.csv`](../outputs/single_model_benchmark_april_predictions.csv)
- [Ensemble Method](ENSEMBLE_METHOD.md)

### 8. Close with production translation

Read:

- [Model Card](MODEL_CARD.md)
- [Limitations and Roadmap](LIMITATIONS_AND_ROADMAP.md)

Explain that the output is a fundamental zero-margin price, not a final
customer price.

## Twenty-minute presentation route

| Minutes | Screen | Message |
|---|---|---|
| 0–2 | `README.md` | Objective, official output, strict cutoff |
| 2–5 | `build_sequential_features` | Leakage control |
| 5–8 | `feature_values` | Sports logic |
| 8–11 | `make_model` | Standardization and regularization |
| 11–13 | `ensemble_probability` | Fixed model averaging |
| 13–16 | Validation and March CSVs | Promotion evidence and uncertainty |
| 16–18 | Component summary and dispersion | Interpretability |
| 18–20 | Limitations document | Sportsbook production roadmap |

## Thirty-minute technical route

1. Run `bash scripts/run_quality_checks.sh`.
2. Run the official model using the supplied CSV.
3. Inspect the strict April output.
4. Walk through sequential state construction.
5. Walk through component fitting and probability averaging.
6. Compare the ensemble with the single benchmark.
7. Review rich challengers and paired bootstrap uncertainty.
8. Discuss season maturity, calibration, and production extensions.

## Official versus benchmark versus research

### Official

- `nba_win_probability.py`
- Uniform 40-component ensemble
- `outputs/april_predictions.csv`

### Benchmark

- Validation-best single component
- Half-life 12, `C = 0.0075`
- Separate April benchmark artifacts

### Governance

- `model_governance.py`
- Rich box-score, schedule, Bradley–Terry, Elo, bootstrap, and monthly analysis

### Research

- Historical single-model governance
- Team-specific home effects
- Optional ML challengers

Research artifacts do not silently replace the official price.
