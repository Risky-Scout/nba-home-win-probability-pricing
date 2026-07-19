# Brief Technical Summary

## Objective

Estimate the home-team win probability for each April NBA game using only
information available through March.

## Official model

A fixed equal-weight ensemble of 40 L2-regularized logistic
paired-comparison models.

Every component uses the same three home-minus-away signals:

1. Net wins.
2. Cumulative point margin.
3. Evidence-weighted recent point margin.

The components span five recent-form half-lives and eight regularization
values. All weights are fixed at 0.025.

## Why it was selected

The ensemble reduces dependence on one nearly tied grid minimum while
preserving the same interpretable feature family and negligible runtime.

| Model | January–February log loss | March governance log loss |
|---|---:|---:|
| Validation-best single component | 0.627529 | 0.509645 |
| **Official ensemble** | **0.627259** | **0.508638** |

Paired bootstrap intervals include zero. The gain is modest and not
statistically decisive. The decision is based on
target relevance, pre-April directional consistency, specification-risk
reduction, and low operational cost.

## April descriptive audit

| Model | Log loss | Brier | ROC AUC | Accuracy |
|---|---:|---:|---:|---:|
| Single benchmark | 0.468596 | 0.150628 | **0.868196** | **81.250%** |
| **Official ensemble** | **0.467607** | **0.150287** | 0.865497 | 80.208% |

April has been reviewed descriptively and is not represented as a pristine
untouched test.

## Information timing

Current-game box-score values update team state only after the current
prediction row is stored. Every official April performance state is frozen at
March 31.

## Deliverable

- [`outputs/april_predictions.csv`](outputs/april_predictions.csv)
- 96 unique ten-character game identifiers
- `home_win_probability`
- Zero-margin home and away decimal odds

## Reproduce

```bash
bash scripts/bootstrap_macos.sh

python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Final claim

This is the strongest target-specific late-season price among the tested
candidates for the supplied information set. It is not claimed to be a
complete production NBA pricing system or a universally optimal model.
