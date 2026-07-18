# Limitations and Production Roadmap

## 1. One season

The dataset contains one season only.

Consequences:

- No external-season validation.
- Uncertain persistence of the ensemble gain.
- No preseason priors learned across years.
- Limited evidence for a maturity threshold.

Production response:

- Add multiple seasons.
- Use season-level rolling validation.
- Estimate team-strength persistence and home advantage hierarchically.

## 2. Small ensemble advantage

The ensemble improves pre-April validation and March governance, but the gain
is small.

This is not described as statistically conclusive.

Promotion is based on:

- Directional pre-April consistency.
- Relevance to the late-season target.
- Reduced single-grid-point risk.
- Negligible runtime cost.
- Preserved interpretability.

Production response:

- Continue shadow comparison over future seasons.
- Predeclare a minimum materiality threshold.
- Retain the single model as a fallback.

## 3. Hyperparameter-grid prior

Equal weighting is not assumption-free.

The number and spacing of half-lives and `C` values determine the implicit
weight placed on different specifications.

Production response:

- Predeclare the grid.
- Compare with prior-weighted averaging.
- Consider nested temporal stacking only with more data.
- Audit sensitivity to grid density.

## 4. No player-level information

The model lacks:

- Injuries.
- Expected starters.
- Expected minutes.
- Trades.
- Player impact.
- Rotation depth.

This is the largest source of missing independent information.

## 5. No possession adjustment

Cumulative point margin is informative but not possession-normalized.

The source lacks the inputs required for defensible possessions.

Production response:

- Add play-by-play or complete box-score possession fields.
- Model offense, defense, and pace separately.

## 6. Opponent strength

Raw margin is not fully schedule-adjusted.

Ridge SRS, Bradley-Terry, and margin-sensitive Elo challengers were tested.
They did not improve both validation and March governance enough to replace
the ensemble.

## 7. Correlated features

The three signals represent related views of team strength.

L2 regularization stabilizes each component but does not create independent
information.

Production response:

- Add player, schedule, and market signals.
- Monitor component coefficients and dispersion.
- Avoid causal interpretation.

## 8. Calibration

March prices remained conservative on average.

No post-hoc calibration layer is forced because earlier prospective
calibration experiments did not improve later proper scores consistently.

Production response:

- Monitor calibration over larger archives.
- Fit mappings only on genuinely out-of-sample historical prices.

## 9. Late-season regime change

April basketball may involve:

- Resting starters.
- Playoff seeding incentives.
- Tanking.
- Teams eliminated from contention.
- Late injuries and lineup changes.

More history improves team-strength estimation but does not eliminate regime
shift.

## 10. No market prices

The model cannot evaluate:

- Closing-line value.
- Vig-adjusted market consensus.
- Offered-price profitability.
- Exposure or limits.

Production response:

- De-vig market probabilities.
- Blend fundamental and market prices prospectively.
- Separate fair-value estimation from margin and risk controls.

## Prioritized roadmap

1. Player availability and expected minutes.
2. Market-implied probability.
3. Multiple seasons.
4. Possession-based strength.
5. Dynamic hierarchical player/team states.
6. Travel, rest, altitude, and time zones.
7. Mature-season ensemble monitoring.
8. Production calibration and rollback controls.
