# Brief Technical Summary

## Objective

Estimate the home-team win probability for each April NBA game using only
information available through March.

## Official model

A fixed equal-weight ensemble of 40 L2-regularized logistic
paired-comparison models.

All components use:

- Net-wins differential.
- Cumulative point-margin differential.
- Evidence-weighted recent point-margin differential.

The ensemble averages five EWMA half-lives and eight regularization values.
The grid and equal weights are fixed; April outcomes are not used to tune
them.

## Why the ensemble was promoted

The ensemble improves the pre-April January-February validation score and
preserves the direction in the March governance period.

| Model | Jan-Feb log loss | March log loss |
|---|---:|---:|
| Best single component | 0.627529 | 0.509645 |
| **Official ensemble** | **0.627259** | **0.508638** |

The gain is modest and not claimed to be statistically conclusive. Promotion
is justified by target relevance, pre-April directional consistency,
negligible runtime cost, and reduced single-grid-point selection risk.

## April descriptive result

- Log loss: **0.467607**
- Brier score: **0.150287**
- ROC AUC: **0.865497**
- Accuracy: **80.208%**

The single benchmark April log loss is 0.468596.

## Information timing

Current-game box-score values are used only after the prediction timestamp.
All official April features are frozen at March 31.

## Reproduce

```bash
bash scripts/bootstrap_macos.sh

python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Final claim

This is the strongest late-season model among the tested candidates for the
supplied information set. It is not claimed to be a complete production NBA
pricing system.
