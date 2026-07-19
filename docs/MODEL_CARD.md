# Model Card

## Model name

Late-Season Hyperparameter-Averaged NBA Home-Win Probability Model

## Version

v1.4 — ensemble champion.

## Intended use

Estimate a fundamental pregame home-win probability for April NBA games when
only the supplied team-level season history is available.

Suitable for:

- The bet365 technical assignment.
- A transparent late-season fundamental probability baseline.
- Model comparison and trader review.
- Demonstrating leakage-safe chronological modeling.
- Converting fair probability into zero-margin decimal odds.

## Not intended for

- Direct customer pricing without additional information.
- In-play pricing.
- Player props.
- Injury-sensitive pricing without lineup data.
- Automated liability or staking decisions.
- Claims of guaranteed profitability.

## Target

\[
Y=1
\]

when the home team wins, otherwise zero.

## Information cutoff

Official April prices use a strict March 31 snapshot.

Current-game box-score values update only later games.

## Features

Every component uses:

1. `net_wins_diff`
2. `cumulative_margin_diff`
3. `recent_margin_evidence_diff`

All features are home minus away.

## Model family

Uniform arithmetic mean of 40 L2-regularized logistic
paired-comparison models.

Hyperparameter set:

- Half-lives: 5, 8, 12, 16, 24 games.
- `C`: 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05.

Weights:

- 0.025 per component.
- Fixed and untuned.

## Selection protocol

- October-December: fit candidate components.
- January-February: compare ensemble and single benchmark.
- March: governance check.
- October-March: final component refit.
- April: requested forecast period and descriptive audit.

March was used for promotion governance, not for component or weight tuning.

The grid and equal-weight rule are fixed in code and were not fitted to April
outcomes. April has been viewed descriptively and is not presented as an
untouched test.

## Primary metric

Forward log loss.

## Secondary metrics

- Brier score.
- Calibration-in-the-large.
- Reliability by price band.
- ROC AUC.
- Accuracy.
- Runtime.
- Component dispersion.

## Pre-April evidence

- Ensemble validation log loss:
  0.627259
- Single validation log loss:
  0.627529
- Ensemble March log loss:
  0.508638
- Single March log loss:
  0.509645

## April descriptive evidence

- Log loss: 0.467607
- Brier score: 0.150287
- ROC AUC: 0.865497
- Accuracy: 80.208%

## Equal-strength home baseline

Average component home-win probability:

55.568%

## Interpretability

The ensemble is interpreted at two levels:

1. Sports-feature level: all components use the same three signals.
2. Component level: each logistic coefficient vector is exported.

The validation-best single component remains the primary coefficient-level
benchmark.

## Uncertainty

`outputs/april_component_dispersion.csv` reports the range of probabilities
across plausible components.

This is specification dispersion, not full predictive uncertainty. It excludes
injuries, lineups, market information, data-source error, and structural model
risk.

## Known risks

- One season only.
- Small differences between ensemble and benchmark.
- Hyperparameter-grid density acts like a discrete prior.
- Features are correlated.
- No injuries, players, possessions, or market prices.
- Late-season competitive regimes may shift.
- April outcomes have been viewed descriptively and are not pristine.

## Monitoring in production

- Log loss and Brier score.
- Calibration-in-the-large and slope.
- Component dispersion.
- Prediction-distribution drift.
- Data freshness.
- Injury and lineup latency.
- Market disagreement.
- Trader overrides.
- Liability and P&L.
- Single benchmark fallback.
