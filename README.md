# NBA Home-Win Probability

Leakage-safe NBA home-win probabilities for the bet365 technical task.

The initial champion is an L2-regularized logistic paired-comparison model
using:

- Net-wins differential.
- Cumulative point-margin differential.
- Evidence-weighted recent point-margin differential.

The model is developed chronologically and writes strict March 31 frozen
probabilities for all 96 April games.

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python nba_win_probability.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs
```

The supplied CSV is intentionally excluded from version control.
