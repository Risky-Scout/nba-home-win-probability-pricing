# Model Card

## Model name

Regularized Three-Signal NBA Home-Win Probability Model

## Version

Final technical-task governance version.

## Intended use

Estimate a fundamental pregame home-win probability for NBA games when only
the supplied team-level season history is available.

The model is suitable for:

- The bet365 technical assignment.
- A transparent fundamental probability baseline.
- Model comparison and trader review.
- Demonstrating leakage-safe chronological modeling.
- Converting fair probability into zero-margin decimal odds.

## Not intended for

- Direct production customer pricing without additional information.
- In-play pricing.
- Player prop markets.
- Point-spread or totals pricing.
- Injury-sensitive pricing without lineup data.
- Claims of guaranteed profitability.
- Automated stake or liability decisions.

## Supported runtime

- Python 3.11-3.13.
- Python 3.12.13 recommended.
- Python 3.14 rejected before compiled numerical imports.
- Exact direct dependencies are declared in `pyproject.toml`.
- GitHub Actions tests Python 3.11, 3.12, and 3.13.

## Target

\[
Y=1
\]

when the home team wins, otherwise zero.

## Information cutoff

The official April probability uses a strict March 31 snapshot.

Current-game box-score information affects only later games.

## Features

1. `net_wins_diff`
2. `cumulative_margin_diff`
3. `recent_margin_evidence_diff`

All features are home minus away.

## Model family

L2-regularized logistic regression with training-only standardization.

## Selected hyperparameters

- EWMA half-life: 12 games.
- Logistic `C`: 0.0075.
- Random seed: 365.

## Equal-strength home baseline

- Log odds: 0.221501
- Home-win probability:
  55.515%
- Odds multiplier:
  1.2479x

## Primary model-selection metric

Forward log loss.

## Secondary metrics

- Brier score.
- Reliability by price band.
- Calibration intercept and slope.
- ROC AUC.
- 0.50-threshold accuracy.
- Computational latency.
- Probability and coefficient stability.

## Development design

- Train: October-December.
- Validate: January-February.
- Governance check: March.
- Final fit: October-March.
- Forecast: April.

March is not described as a perfectly untouched test because it later
informed model-governance decisions.

## March evidence

- Log loss: 0.509645
- Brier score: 0.167618
- ROC AUC: 0.823538
- Mean probability: 54.743%
- Actual home-win rate:
  60.251%
- Calibration gap:
  5.508%
- Calibration slope: 1.341

Date-block bootstrap 95% interval for the calibration gap:

[
0.886%,
10.192%
]

## Calibration decision

No additional calibration layer.

Identity, Platt, beta and isotonic mappings were compared prospectively.
Additional mappings did not improve the next-period scores.

## Model uncertainty

The supplemental April uncertainty file contains date-block bootstrap
intervals around coefficient estimation.

Median 90% interval width:

5.650%

These intervals do not cover:

- Injuries.
- Lineups.
- Player minutes.
- Market information.
- Structural model error.
- Data-source error.

## Challenger families tested

- Reduced feature subsets.
- Team-specific venue deviations.
- Pure EWMA.
- Bayesian-shrunken EWMA.
- Opponent-adjusted ridge SRS.
- PCA latent factors.
- Elo.
- XGBoost.
- Residual XGBoost.
- Random forest.
- ExtraTrees.
- CatBoost.
- Convex probability ensembles.
- Platt, beta and isotonic calibration.
- Rich lagged turnover, rebound, foul and schedule features.

## Selection decision

The architecture is retained because no opponent-adjusted, Bayesian-shrunk, pure-EWMA, PCA, or reduced-feature challenger produces a material and temporally stable proper-score improvement.

## Known risks

- One season only.
- No external-season validation.
- High feature correlation.
- No player-level information.
- Cumulative margin is not pace-adjusted.
- March home outcomes were underpriced on average.
- Evidence-weighted EWMA is unusual.
- April outcomes have been viewed descriptively and are not pristine.

## Monitoring requirements in production

- Log loss and Brier score over time.
- Calibration-in-the-large.
- Calibration slope.
- Reliability by price band.
- Prediction-distribution drift.
- Feature freshness.
- Injury and lineup data latency.
- Closing-line value.
- Trader overrides.
- Liability and P&L.
- Challenger-model comparison.
