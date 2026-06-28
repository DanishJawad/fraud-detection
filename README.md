# Fraud Detection — Cost-Optimized Pipeline

This repository implements an end-to-end fraud detection pipeline focused on minimizing expected business cost (trade-off between false positives and false negatives).

Getting started

1. Create and activate a Python environment (recommended: 3.10+).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Download the Kaggle `creditcard.csv` dataset and place it at `data/creditcard.csv`.

3. Run the baseline experiment (prints metrics and optimal threshold):

```bash
python notebooks/01_baseline.py --data-path data/creditcard.csv
```

4. Run the Streamlit demo locally:

```bash
streamlit run app.py
```

Deployment

- Build the Docker image and push to your preferred cloud (see `Dockerfile`).

Notes

- Default cost parameters are in `config.yaml`. The baseline script performs a threshold sweep to minimize expected cost.
