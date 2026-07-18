# Limitations and Production Research Roadmap

## Current limitations

### 1. One season

The data contains one NBA season.

Consequences:

- No independent-season validation.
- Potential season-specific coefficients.
- Limited evidence about stability at season boundaries.
- No learned preseason prior.

Research response:

- Add multiple seasons.
- Use season-level rolling validation.
- Apply partial pooling across seasons.
- Re-estimate home advantage and strength persistence.

### 2. No injuries or player-level information

NBA prices often move materially when a high-impact player is unavailable.

Missing information includes:

- Injury status.
- Expected starters.
- Expected minutes.
- Player impact ratings.
- Rotation depth.
- Late news.

Research response:

- Build expected-minute projections.
- Use player-level adjusted plus-minus or impact priors.
- Aggregate expected player availability into team strength.
- Timestamp every news input to prevent leakage.

### 3. No pace adjustment

Cumulative point margin is informative but not possession-normalized.

The supplied file lacks field-goal attempts, free throws and offensive
rebounds, so a defensible possession estimate cannot be constructed.

Research response:

- Add play-by-play or box-score possession inputs.
- Model offensive and defensive efficiency separately.
- Retain raw margin as a challenger rather than relabeling it as efficiency.

### 4. Opponent strength

Raw cumulative margin does not distinguish schedule strength.

A ridge simple-rating-system challenger was added:

\[
	ext{home margin}
=
h+s_	ext{home}-s_	ext{away}+\epsilon
\]

with ridge partial pooling.

Best opponent-adjusted challenger:

- Validation log loss:
  0.627278
- March log loss:
  0.511107

It did not produce a material stable improvement and remains a research
challenger rather than a production feature.

### 5. Feature correlation

The three signals are strongly related.

L2 regularization controls coefficient instability, but it does not create new
information.

Research response:

- Monitor coefficient stability.
- Compare latent-factor models.
- Add genuinely independent player, schedule and market signals.
- Avoid claiming that coefficient magnitudes are causal.

### 6. March calibration

March mean prediction:
54.743%

March actual home-win rate:
60.251%

Gap:
5.508%

The calibration slope above one indicates under-dispersion or conservative
pricing during March.

Research response:

- Continue monitoring over a larger archive.
- Use prospective recalibration only.
- Do not fit a flexible mapping to one month.
- Compare calibration drift with market and lineup changes.

### 7. Evidence-weighted EWMA

Multiplying EWMA margin by games played is unusual.

It deliberately combines current form and evidence, but it is not a standard
posterior mean.

Tested alternatives:

- Pure EWMA.
- Bayesian-shrunken EWMA.
- PCA latent factors.
- Opponent-adjusted SRS.

None earned inclusion.

Research response:

- Dynamic Bayesian state-space strength.
- Hierarchical random walks.
- Kalman-filtered point-margin states.
- Player-driven state transitions.

### 8. No market prices

The model cannot evaluate:

- Closing-line value.
- Vig-adjusted market information.
- Betting return.
- Exposure.
- Limits.
- Trader response.

Research response:

- Remove overround from market prices.
- Blend the fundamental model and market using out-of-sample evidence.
- Evaluate closing-line value before realized P&L.
- Separate fair-value estimation from risk and margin decisions.

## Prioritized production roadmap

### Priority 1: player availability

Expected to add the most independent information.

### Priority 2: market blending

The market encodes injuries, news and collective information unavailable in
the assignment.

### Priority 3: possession-based team and player states

Replace raw scoring summaries with offense, defense and pace components.

### Priority 4: hierarchical dynamic strength

Estimate team/player strength, process noise and uncertainty explicitly.

### Priority 5: larger calibration archive

Fit any calibration layer only on genuinely out-of-sample historical prices.

### Priority 6: production monitoring

Implement:

- Data contracts.
- Feature timestamps.
- Drift alerts.
- Price-band calibration.
- Latency monitoring.
- Shadow challengers.
- Trader override logging.
- Model rollback.

## Final research conclusion

The correct response to these limitations is not to add complexity by default.

Every additional feature or layer must improve forward proper scores
materially, remain stable across time and preserve operational reliability.

The current champion is retained because the available additions do not clear
that threshold.
