"""Streamlit demo: score an uploaded CSV and explore the cost/threshold tradeoff.

Separate from dashboard/app.py, which monitors a running API. This one loads
the model directly and scores a file you upload.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import sweep_thresholds
from src.utils import load_config, load_model

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

MODELS_DIR = Path("models")


@st.cache_resource
def load_artifacts():
    """Load the model and its metadata, or (None, None) if not trained yet."""
    try:
        model = load_model(MODELS_DIR / "fraud_model.pkl")
    except Exception:
        return None, None
    metadata = {}
    metadata_path = MODELS_DIR / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return model, metadata


config = load_config("config.yaml")
model, metadata = load_artifacts()

st.title("Fraud Detection — Cost-Aware Demo")

if model is None:
    st.warning(
        "No model found at models/fraud_model.pkl. "
        "Run `python scripts/train.py` first, then reload this page."
    )
    st.stop()

# The model's own feature list, so this demo can't drift from what was trained.
feature_names = metadata.get("features") or []
trained_threshold = (metadata.get("threshold") or {}).get("value")

st.sidebar.header("Cost settings")
cost_fp = st.sidebar.number_input(
    "Cost per false positive", value=float(config.get("cost_fp", 50))
)
cost_fn = st.sidebar.number_input(
    "Cost per false negative", value=float(config.get("cost_fn", 500))
)

if trained_threshold is not None:
    st.sidebar.caption(
        f"Threshold selected on validation during training: {trained_threshold:.2f}"
    )

st.write(
    f"Upload a CSV with the {len(feature_names)} columns the model was trained on "
    "(V1-V28 and Amount). A `Class` column, if present, is used to score the result."
)
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to score transactions.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Rows:", len(df))

missing = [c for c in feature_names if c not in df.columns]
if missing:
    st.error(
        f"CSV is missing {len(missing)} column(s) the model needs: "
        f"{', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}"
    )
    st.stop()

# Select by name in the trained order. Selecting explicitly rather than
# dropping unwanted columns means an extra column in the upload is ignored
# instead of being silently fed to the model in the wrong position.
X = df[feature_names]

with st.spinner("Scoring..."):
    proba = model.predict_proba(X)[:, 1]

scored = df.copy()
scored["fraud_probability"] = proba

default_threshold = float(trained_threshold) if trained_threshold is not None else 0.5
threshold = st.slider("Flag transactions above", 0.0, 1.0, default_threshold, 0.01)
flagged = int((proba >= threshold).sum())
st.metric("Flagged", f"{flagged:,}", f"{100.0 * flagged / len(df):.2f}% of uploaded rows")

st.dataframe(scored.head(20))

if "Class" in df.columns:
    opt_t, opt_cost, _, _ = sweep_thresholds(proba, df["Class"], cost_fp, cost_fn)
    st.write(
        f"Cost-minimizing threshold on this file: {opt_t:.2f} "
        f"(cost ${opt_cost:,.2f} at the sidebar's cost settings)"
    )
    st.caption(
        "Computed on the uploaded file, which is a threshold fit to this data. "
        "It is not a held-out estimate — the shipped threshold comes from the "
        "validation split in scripts/train.py."
    )
else:
    st.info("No `Class` column in the upload, so cost cannot be computed.")

st.download_button(
    "Download scored CSV",
    scored.to_csv(index=False),
    file_name="scored.csv",
    mime="text/csv",
)
