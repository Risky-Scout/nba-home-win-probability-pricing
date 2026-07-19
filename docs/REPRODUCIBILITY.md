# Reproducibility

## Source data

The supplied CSV is intentionally excluded from version control.

Pass its local path through `--data`.

## Supported environment

Supported Python versions:

- 3.11
- 3.12
- 3.13

Python 3.12.13 is recommended.

```bash
bash scripts/bootstrap_macos.sh
```

Manual setup:

```bash
brew install python@3.12
rm -rf .venv
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate

python scripts/check_python.py
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

## Official ensemble

```bash
python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Promotion governance

```bash
python model_governance.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs \
  --figure-dir figures
```

## Complete core workflow

```bash
python run_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

## Optional machine-learning challengers

```bash
python -m pip install -r requirements-challengers.txt

python challenger_analysis.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Tests

```bash
python -m pytest -q
```

## Full validator

```bash
python validate_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

## Data-free quality gate

```bash
bash scripts/run_quality_checks.sh
```

## Cross-platform reproduction contract

Wall-clock measurements in `outputs/governance_runtime.csv` are
environment-specific and may change when the governance workflow is rerun.

The official April identifiers, submitted probabilities, fair odds, source
fingerprint, and selected-model metadata reproduce exactly in the supported
locked environment.

Platform-sensitive floating-point summaries may differ only in insignificant
final decimal places across BLAS implementations. PNG bytes may differ across
operating systems because font and rendering backends are platform-specific;
their schemas, dimensions, validity, and underlying numeric artifacts are
validated instead.

The quality gate checks:

- Python support.
- Dependency consistency.
- Source compilation.
- Warning-clean tests.
- Public-repository privacy.
- Prediction and fair-odds identities.
- JSON, CSV, PNG, and documentation integrity.
- Git whitespace.
