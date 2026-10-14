# Credit Card Fraud Detection

Fraud is 0.173% of transactions in this dataset, so a model optimized for accuracy predicts
"not fraud" every time and scores 99.83%. This project trains a SMOTE + RandomForest pipeline
on a chronological split, picks a decision threshold on a validation set by minimizing an
explicit cost function, and reports the result once on a test set that played no part in
choosing anything. It also serves the model over FastAPI with a feedback endpoint, so
detection rate and cost can be recomputed from labels as they arrive.

The headline finding is partly negative and is reported as such below: at these cost
assumptions the cost-minimizing threshold turns out to be 0.50, the same as the naive default.
The measurable win is the model, not the threshold.

## Results

Kaggle [credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud): 284,807
transactions, 492 fraud, 284,315 legitimate (0.172749%, about 1 in 579).

Split chronologically by `Time` into train 60% / validation 20% / test 20%:

| Split | Rows | Fraud | Fraud rate | Used for |
|---|---|---|---|---|
| Train | 170,888 | 360 | 0.2107% | Fitting the pipeline |
| Validation | 56,957 | 57 | 0.1001% | Choosing the threshold |
| Test | 56,962 | 75 | 0.1317% | Final metrics, read once |

Final numbers, test set at threshold 0.50:

| Metric | Value |
|---|---|
| ROC-AUC | 0.9666 |
| Recall (fraud detection rate) | 73.33% (55 of 75) |
| Precision | 93.22% (55 of 59 flags) |
| False positive rate | 0.007031% (4 of 56,887) |
| Cost at $50/FP and $500/FN | $10,200 |
| Logistic regression baseline cost | $11,800 |

The cost gap between the two models is **not statistically reliable on this test set**. The
point estimate is 13.6% cheaper, but resampling the test set 5,000 times with replacement
(`scripts/bootstrap_cost_comparison.py`, results in
[`outputs/bootstrap_cost.json`](outputs/bootstrap_cost.json)) puts the SMOTE pipeline ahead in
84.44% of resamples, behind in 14.78%, and tied in 0.78%. The 95% confidence interval on the
cost difference runs from **-$1,350 to +$4,650, which crosses zero**. With 75 frauds in the
test set, "13.6% cheaper" is a single draw, not a stable result, and it is not quoted as one
anywhere in this repo.

Every figure above is written by `scripts/train.py` to
[`outputs/metrics.json`](outputs/metrics.json), with the run's console output in
[`outputs/train_log.txt`](outputs/train_log.txt). Both are tracked in git, so they can be read
without downloading the dataset or retraining. Nothing in this README is typed by hand.

## Run it locally

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (`numpy`, `pandas`, `scikit-learn` and
`scipy` all require >= 3.11; `uv sync` installs the interpreter itself if you don't have it).

```bash
uv sync --locked   # installs from uv.lock: exact versions, no resolution surprises

# Download creditcard.csv from Kaggle and place it at data/creditcard.csv, then:
uv run scripts/train.py

# Terminal 1: API
PYTHONPATH=. uv run uvicorn app.main:app --port 8000

# Terminal 2: dashboard
uv run streamlit run dashboard/app.py
```

`pyproject.toml` + `uv.lock` are the one source of dependency truth: `uv.lock` pins the full
transitive graph with hashes, not just top-level packages, which is stronger than the old
`requirements.txt` pins were. `uv sync --locked` fails loudly if the lockfile and
`pyproject.toml` ever drift apart, rather than silently installing something else.

Try it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.0, ..., 0.0]}'   # 29 values: V1..V28, Amount
```

### Prove the monitoring is real, not wired-but-dead

```bash
uv run scripts/simulate_traffic.py --n 3000
```

This replays held-out transactions through `/predict`, follows each with `/feedback` carrying
the real label (standing in for a delayed chargeback), and prints `/monitoring` periodically,
so the detection rate moves as real frauds are replayed and flagged.

### Docker

```bash
uv run scripts/train.py   # models/ is gitignored; build it before compose mounts it
docker compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

Verified end to end on Docker 29.7.2 / Compose 5.3.1 (linux/aarch64): both images build from a
clean state, both containers start, `/predict` returns a real score for a held-out fraud row,
and the dashboard container reaches the API over the compose network at `http://fraud-api:8000`.

`models/` is mounted rather than copied into the image, so the container serves exactly the
artifact on disk. Verified by SHA-256: `models/fraud_model.pkl` and `models/metadata.json`
inside the container hash identically to the host copies, and `models/metadata.json` hashes
identically to `outputs/metrics.json`, so the threshold the API serves is the same one recorded
with the published metrics.

Both images are about 1.19 GB, almost entirely the scientific Python stack. The build context
itself is ~0.1 MB thanks to `.dockerignore`; without it, `COPY . /app` would pull in `data/`
(~370 MB) and `.venv/` (~620 MB). `uv sync` runs with `--no-cache` in both Dockerfiles &mdash;
without it, uv's own download cache (~540 MB for this dependency set) gets baked into the image
layer on top of the installed environment, found and fixed while verifying this migration.

### Tests

```bash
PYTHONPATH=. uv run pytest tests/ -v
```

18 tests covering the API endpoints, the SMOTE pipeline (no leakage into held-out data,
reproducible given a fixed seed), the temporal split (no row appears in two splits, splits stay
in chronological order), precision/recall/FPR computation, the cost-threshold sweep and its tie
handling, and the monitoring feedback loop.

## How it works

1. **`scripts/train.py`** — the only source of the shipped model and of every published number.
   Splits chronologically, fits `StandardScaler -> SMOTE -> RandomForestClassifier`, selects the
   threshold on validation, scores test once, and writes `outputs/metrics.json`,
   `outputs/train_log.txt`, `outputs/cost_sensitivity.csv`, plus `models/fraud_model.pkl` and
   `models/metadata.json`.
2. **`app/main.py`** — FastAPI service. Serves `/predict` and `/predict-batch`, reports the
   model card at `/metrics`, accepts ground-truth labels at `/feedback`, and computes live
   metrics at `/monitoring`. The decision threshold is read from the trained artifact and
   nowhere else; if it is missing the API returns 503 rather than guessing.
3. **`monitoring/tracker.py`** — logs every prediction with a `prediction_id`. When `/feedback`
   supplies the real label later, `/monitoring` recomputes detection rate, false positive rate
   and accumulated cost over the most recent labeled window.
4. **`dashboard/app.py`** — Streamlit dashboard. The "Live traffic" tab reads `/monitoring`;
   the "Model card" tab reads `/metrics` and is labeled as fixed training-time results.
5. **`Dockerfile`** / **`Dockerfile.dashboard`** / **`docker-compose.yml`** — API and dashboard
   as two containers.

Two further scripts exist to check the claims above rather than to produce the model:
`scripts/measure_selection_bias.py` and `scripts/bootstrap_cost_comparison.py`.

### One threshold, one place

The threshold lives in `models/metadata.json` under `threshold.value`, written by
`scripts/train.py`. It is deliberately absent from `config.yaml`, from `app/config.py` and from
the compose environment. It is an output of training, not a setting. This repo previously
carried three different values (0.5 in config, 0.35 in compose, 0.41 in the artifact) with no
way to tell which was live.

## Method and analysis

### What the threshold sweep actually found

The threshold is chosen by sweeping `np.linspace(0, 1, 101)` on validation and keeping the
value that minimizes `false_positives * $50 + false_negatives * $500`. Two honest caveats:

- **The minimum is a plateau, not a point.** Six thresholds from 0.50 to 0.55 tie at the same
  validation cost of $7,100. The sweep keeps the lowest. Any value in that range is equally
  justified by the data, so "0.50" should not be read as a precise optimum.
- **The chosen threshold does not move with the cost ratio.** Sweeping FP cost across
  $10/$50/$100 and FN cost across $100/$500/$1,000 selects 0.50 in all nine combinations
  (`outputs/cost_sensitivity.csv`). For this model the cost-sensitive machinery does not shift
  the operating point away from the default. The RandomForest's probabilities are pushed
  toward 0 and 1, so few transactions sit in the range where moving the threshold changes a
  decision.

So the cost function is doing real work as an evaluation tool, but on this data it does not
justify a non-default threshold. Reporting otherwise would require ignoring the sweep.

### Comparison against the baseline

Both models had their threshold selected on validation and were scored once on test.

| | Recall | Precision | FPR | Cost | ROC-AUC |
|---|---|---|---|---|---|
| SMOTE + RandomForest (threshold 0.50) | 73.33% | 93.22% | 0.0070% | $10,200 | 0.9666 |
| Logistic regression, no SMOTE (threshold 0.11) | 72.00% | 67.50% | 0.0457% | $11,800 | 0.9730 |

The RandomForest's advantage is in precision: it raises 59 flags to the baseline's 80 while
catching one more fraud. As noted in Results, the cost difference does not survive bootstrap
resampling of the test set, so this is a direction, not a measured margin. **The baseline has
the higher ROC-AUC** (0.9730 vs 0.9666). Ranking quality is not where the pipeline wins, and a
threshold-free comparison would favour logistic regression.

### Methodology: imbalance handling and threshold selection

**Class imbalance** is handled with SMOTE inside an `imblearn.pipeline.Pipeline`
(`src/models.py::train_smote_pipeline`), never as a preprocessing step on the whole dataset.
imblearn's Pipeline only resamples during `.fit()`, so synthetic minority samples cannot reach
validation or test data even under cross-validation. Oversampling before splitting would copy
information about held-out frauds into training and inflate every number here.

**The split is chronological, not random** (`src/preprocessing.py::train_val_test_split_time`).
Fraud patterns change over time; training on later transactions to score earlier ones leaks
future information and overstates what the model would do on live traffic.

**The threshold is chosen on validation only.** An earlier version of this repo swept
thresholds against `y_test` and then reported metrics at the winning threshold on that same
test set. The threshold is then fit to the test split's specific errors, which is indefensible
regardless of what it does to the number.

`scripts/measure_selection_bias.py` regenerates all three configurations and writes
[`outputs/selection_bias.json`](outputs/selection_bias.json), so the comparison below is
reproducible rather than quoted from a run whose artifact no longer exists:

| | Old: 80/20, threshold on test | Old model, threshold off-test | Now: 60/20/20, threshold on val |
|---|---|---|---|
| Train size | 227,845 rows, 417 fraud | same | 170,888 rows, 360 fraud |
| Threshold | 0.41 | 0.37 | 0.50 |
| Recall | 77.3333% | 77.3333% | 73.3333% |
| Precision | 90.6250% | 87.8788% | 93.2203% |
| ROC-AUC | 0.9774 | 0.9774 | 0.9666 |
| Cost | $8,800 | $8,900 | $10,200 |

The middle column holds the model fixed and changes only where the threshold came from.
**Recall is identical to four decimal places**, and cost moves by $100, one false positive. So
on this data the threshold-on-test mistake inflated the reported recall by essentially nothing.
The entire 77.33% to 73.33% drop comes from training on 57 fewer frauds after carving out the
validation split, not from removing the bias.

That does not make the old protocol acceptable: it was unjustifiable in advance, and the fact
that it happened to cost nothing here is luck, not a defence. But claiming the fix "corrected
an inflated metric" would itself be a false claim, so it is not made.

One caveat on the middle column: the off-test threshold is taken from the last quarter of that
model's own training window, so the model had already seen that data. It isolates the
threshold's provenance, not a fully clean holdout.

## Limitations

- **The test set contains 75 fraud cases.** Recall of 73.33% means 55 of 75. One more catch
  moves recall by 1.3 percentage points, and the 95% confidence interval is roughly 62% to 83%
  (Clopper-Pearson on 55 of 75).
  Any comparison between models that turns on a difference smaller than that is noise. The
  validation set is thinner still at 57 frauds, which is why the cost curve has wide ties.
- **The cost figures are illustrative.** $50 per false positive and $500 per false negative are
  assumptions in `config.yaml`, not real business numbers. The dollar amounts mean nothing on
  their own; only the ratio affects threshold selection, and as shown above even a 10x change
  in that ratio does not move it here.
- **One dataset, one seed.** `random_state=42` throughout, no cross-validation, no repeated
  runs. The reported numbers are a single draw, not a mean with error bars.
- **No hyperparameter tuning.** The RandomForest uses library defaults apart from
  `n_estimators=100` and the seed. There is no grid search or cross-validation anywhere in this
  repo, and the forest is grown unpruned, so it fits the training split perfectly.
- **Features are anonymized PCA components.** V1-V28 carry no business meaning, so nothing here
  transfers to a real fraud system without redoing feature work from scratch.
- Feedback labels update the monitor's in-memory log only; `/monitoring` resets on restart.
  Fixing that properly means the Postgres path scaffolded in `monitoring/db.py`.
- **There is no automated retraining.** Retraining means running `scripts/train.py` by hand.
  A scheduled version would need its own validation split to re-select the threshold, since
  a new model's probabilities will not line up with the old one's operating point.

## Superseded results — do not reuse

Three sets of numbers exist in this repo's git history and all are invalid:

- **`notebooks/04_costs.py`** (deleted) called `predict_proba` on the full dataset, including
  the rows the model was trained on, and wrote the result to the same
  `outputs/cost_sensitivity.csv` path the real pipeline uses, so running it silently replaced
  valid output with in-sample numbers. The thresholds it produced (0.86, 0.99, 1.0) came from
  that leak.
- **`notebooks/01_baseline.py` and `notebooks/03_imbalance.py`** (deleted) both wrote to
  `models/fraud_model.pkl`, the artifact the API serves, without updating
  `models/metadata.json`. Running either replaced the shipped RandomForest with a different
  model while the metadata continued to describe the old one. `03_imbalance.py` additionally
  used a random rather than chronological split, selected its "best" model on the test set, and
  trained on a 34-column frame the API's 29-feature contract would have rejected.
- **`README_CLEAN.md`** (deleted in commit `d6acbd6`) claimed 0.9794 ROC-AUC, ~80% detection,
  ~0.08% false alarm rate and ~73.9% cost savings. No artifact in this repo, at any commit,
  ever backed those figures. They are not reproducible and must not be quoted.

The only numbers this project stands behind are the ones in `outputs/`, regenerated by the
scripts in `scripts/`.

## Stack

Python 3.11, scikit-learn, imbalanced-learn (SMOTE), pandas, FastAPI, Streamlit, Docker.
