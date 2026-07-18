# NBA Home-Win Probability Pricing

[![CI](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml)

A leakage-safe, chronologically validated probability-pricing model for the
bet365 NBA technical task.

## Start here

| Time available | Review path |
|---|---|
| 2 minutes | Read [SUMMARY.md](SUMMARY.md) and open [April predictions](outputs/april_predictions.csv) |
| 10 minutes | Follow the [Reviewer Guide](docs/REVIEWER_GUIDE.md) |
| 20 minutes | Read the [Model Story](docs/MODEL_EVOLUTION.md) and inspect [the champion code](nba_win_probability.py) |
| Reproduce | Follow [Reproducibility](docs/REPRODUCIBILITY.md) |
| Audit | Review the [Model Card](docs/MODEL_CARD.md), [limitations](docs/LIMITATIONS_AND_ROADMAP.md), and [artifact manifest](docs/ARTIFACT_MANIFEST.md) |

## Business objective

Estimate the probability that the team listed as `home` wins each April game,
using only information available through March.

The official submission is:

- [`outputs/april_predictions.csv`](outputs/april_predictions.csv)
- Column: `home_win_probability`

All 96 official April prices are frozen at the March 31 information cutoff.

## Selected model

The champion is a strongly regularized logistic paired-comparison model with
three home-minus-away signals:

1. Net-wins differential.
2. Cumulative point-margin differential.
3. Evidence-weighted recent point-margin differential.

Selected hyperparameters:

- Recent-form half-life: **12 games**
- Logistic regularization: **`C = 0.0075`**

The probability model is:

\[
P(\text{home win})
=
\sigma\left(
\beta_0 + \beta_1 z_1 + \beta_2 z_2 + \beta_3 z_3
\right)
\]

where each \(z_j\) is standardized using training-only statistics.

## Why this is not “just basic logistic regression”

The statistical link is logistic, but the model is built around:

- Sequential, pregame-only team states.
- Explicit feature-before-update ordering.
- Evidence-weighted dynamic form.
- Strong L2 shrinkage for correlated signals.
- Chronological hyperparameter selection.
- Proper probability scoring.
- Frozen April information timing.
- Extensive champion-challenger governance.

The feature engineering and information timestamp do most of the work. The
logistic layer converts those signals into a stable, auditable probability.

## Development design

| Period | Role |
|---|---|
| October-December | Fit coefficients |
| January-February | Select half-life, regularization, and architecture |
| March | Later governance and robustness check |
| October-March | Final refit |
| April | Requested forecast period |

March is not described as a pristine untouched test because it informed later
governance decisions.

## March governance result

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Constant home-rate baseline | 0.680580 | 0.243727 | 0.500000 | 60.251% |
| Net-wins-only logistic | 0.520933 | 0.171274 | 0.823684 | 74.059% |
| Cumulative-margin-only logistic | 0.522780 | 0.172852 | 0.816118 | 75.732% |
| **Three-signal champion** | **0.509645** | **0.167618** | **0.823538** | **75.732%** |

The primary metric is forward log loss because the business output is a
probability price, not merely a winner classification.

## Home-court interpretation

The official probability is already specific to the actual home and away
teams in each matchup.

For otherwise equal raw team-strength features:

- Home log odds: **0.221501**
- Home-win probability: **55.515%**
- Home odds multiplier: **1.2479x**

A separate shrunk team-specific venue challenger is retained under
[`research/`](research/) because its small improvement was not statistically
or temporally stable enough to replace the champion.

## Why added complexity was rejected

The champion was compared with:

- XGBoost and residual XGBoost.
- CatBoost, random forest, and ExtraTrees.
- Convex probability ensembles.
- Platt, beta, and isotonic calibration.
- Opponent-adjusted ridge team ratings.
- Pure and Bayesian-shrunken EWMA variants.
- PCA latent-strength models.
- Every two-feature subset.
- Team-specific venue deviations.
- Rich lagged box-score and schedule features.

Some challengers improved isolated periods or secondary metrics, but no model
produced a material, temporally stable improvement in proper probability
scores.

See [Model Evolution](docs/MODEL_EVOLUTION.md) and
[Reviewer Guide](docs/REVIEWER_GUIDE.md).

## Run locally

Supported Python: **3.11-3.13**. Python 3.12.13 is recommended.

```bash
bash scripts/bootstrap_macos.sh
```

Run the official model:

```bash
python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

Run the complete core workflow:

```bash
python run_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

Run the data-free quality gate:

```bash
bash scripts/run_quality_checks.sh
```

## Repository map

```text
.
├── README.md                     # Main landing page
├── SUMMARY.md                    # Brief recruiter summary
├── nba_win_probability.py        # Official champion and April prices
├── enhanced_governance.py        # SRS, shrinkage, PCA, uncertainty
├── challenger_analysis.py        # Tree models, ensembles, calibration, SHAP
├── ablation_and_timing.py        # Feature subsets and timing
├── outputs/                      # Official predictions and audit artifacts
├── figures/                      # Core diagnostics
├── research/                     # Non-promoted research challengers
├── docs/                         # Reviewer and model-governance documentation
├── scripts/                      # Setup, privacy, and integrity utilities
└── tests/                        # Leakage, runtime, configuration, and artifact tests
```

## Important limitations

This is a strong technical-task model, not a complete production NBA pricing
system. The supplied data contains one season and lacks injuries, expected
lineups, player minutes, possession-based ratings, travel context, and market
prices.

Those limitations are documented in
[`docs/LIMITATIONS_AND_ROADMAP.md`](docs/LIMITATIONS_AND_ROADMAP.md).
