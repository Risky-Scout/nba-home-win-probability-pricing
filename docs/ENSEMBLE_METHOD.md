# Ensemble Method

## Purpose

The official April model averages uncertainty over two hyperparameters:

- The recent-form half-life.
- The L2 regularization strength.

The ensemble is designed for the assignment's late-season target. It does not
change the sports features or add a black-box learner.

## Component grid

Five half-lives:

```text
5, 8, 12, 16, 24 games
```

Eight logistic `C` values:

```text
0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05
```

Total components:

\[
5 \times 8 = 40
\]

Each component uses the same three standardized features and an
L2-regularized logistic link.

## Aggregation

For component probabilities \(p_1,\ldots,p_{40}\):

\[
p_{ensemble} =
\frac{1}{40}
\sum_{k=1}^{40}p_k
\]

Weights are equal and fixed.

No weight is optimized against March or April outcomes.

## Why equal weights

The validation surface is flat. Several nearby hyperparameter choices have
nearly indistinguishable scores.

Selecting the single minimum can overstate certainty about one half-life and
one shrinkage value.

Equal weighting:

- Reduces single-grid-point selection risk.
- Avoids estimating 40 stacking weights from only 399 validation games.
- Keeps every component auditable.
- Preserves low latency.
- Requires no post-hoc calibration layer.

Equal weighting is not assumption-free. The predeclared grid defines the
model-uncertainty set and therefore acts like a discrete prior over plausible
specifications.

## Selection evidence

The ensemble was promoted using pre-April evidence.

| Model | Jan-Feb validation | March governance |
|---|---:|---:|
| Best single | 0.627529 | 0.509645 |
| **Ensemble** | **0.627259** | **0.508638** |

March was used to confirm that the validation gain did not reverse before the
April deployment period. It was not used to tune component weights.

Paired date-block bootstrap intervals include zero in both periods. The
promotion is therefore a target-specific governance decision, not a claim of
statistically decisive superiority.

## Season maturity

Monthly expanding-window results:

| Month | Single | Ensemble | Ensemble minus single |
|---|---:|---:|---:|
| 2025-12 | 0.691675 | 0.710008 | +0.018333 |
| 2026-01 | 0.652248 | 0.653643 | +0.001395 |
| 2026-02 | 0.602651 | 0.600639 | -0.002012 |
| 2026-03 | 0.509645 | 0.508638 | -0.001008 |

The ensemble is worse when team histories are sparse and modestly better once
team-strength states mature.

That pattern is relevant because the requested target is April.

It is not evidence of a universal monotonic improvement rule, and one season
is insufficient to estimate a production maturity threshold.

## April descriptive audit

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Single benchmark | 0.468596 | 0.150628 | 0.868196 | 81.250% |
| **Ensemble** | **0.467607** | **0.150287** | 0.865497 | 80.208% |

April is a descriptive confirmation, not the sole promotion criterion.

## Interpretation

The single component selected on validation is:

- Half-life: 12 games.
- `C = 0.0075`.

It remains the coefficient-level interpretation anchor.

The ensemble component summary reports for every component:

- Validation log loss.
- March log loss.
- Equal-strength home probability.
- Standardized feature coefficients.

The dispersion file reports the range of April prices across plausible
components.

## Runtime

Forty logistic models on approximately 1,100 observations remain
computationally trivial. The ensemble is substantially simpler and faster
than many tree or simulation-based alternatives.

## Governance status

Official April price:

- Uniform 40-component ensemble.

Primary benchmark and fallback:

- Validation-best single component.

Production research:

- Multi-season validation.
- Player and lineup information.
- Maturity-aware blending.
- Market-price benchmarking.
