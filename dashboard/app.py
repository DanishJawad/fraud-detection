"""Streamlit monitoring dashboard for the fraud detection API.

Two different kinds of number are shown here and they are deliberately kept on
separate tabs:

  Live traffic — computed by monitoring/tracker.py from predictions that have
    since received a ground-truth label via POST /feedback. Moves as traffic
    arrives. Empty until something is labeled.

  Model card — fixed training-time results on the held-out test set, read from
    the trained artifact. These never change between retrains.

Conflating the two is how a dashboard ends up looking live while actually
displaying a constant.
"""
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="F",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 3rem; max-width: 1100px; }
      [data-testid="stMetricValue"] { font-size: 1.75rem; }
      .caption-muted { color: #6b7280; font-size: 0.875rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

API_URL = st.sidebar.text_input("API URL", os.getenv("API_URL", "http://localhost:8000"))
REQUEST_TIMEOUT = 3


def fetch(path: str):
    """GET an API path. Returns (payload, error_message)."""
    try:
        r = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, f"The API did not respond within {REQUEST_TIMEOUT}s."
    except requests.exceptions.RequestException as e:
        return None, f"Request failed: {e}"

    if r.status_code == 503:
        return None, "The API is up but has no model loaded. Run scripts/train.py."
    if r.status_code != 200:
        return None, f"The API returned HTTP {r.status_code} for {path}."
    return r.json(), None


st.title("Fraud Detection Monitoring")

if st.sidebar.button("Refresh"):
    st.rerun()

live_tab, model_tab, health_tab = st.tabs(["Live traffic", "Model card", "Health"])

with live_tab:
    st.subheader("Live traffic")
    st.caption(
        "Computed from predictions that have received a confirmed label. "
        "Updates as feedback arrives."
    )

    with st.spinner("Loading live metrics..."):
        payload, error = fetch("/monitoring")

    if error:
        st.error(error)
    else:
        summary = payload.get("summary") or {}
        business = payload.get("business_metrics")
        total_predictions = summary.get("total_predictions", 0)

        if not total_predictions:
            st.info(
                "No predictions scored yet. Send some traffic with "
                "`python scripts/simulate_traffic.py --n 2000`."
            )
        elif not business:
            st.warning(
                f"{total_predictions:,} predictions scored, but none have a "
                "confirmed label yet, so detection rate and cost cannot be "
                "computed. Labels arrive via POST /feedback."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Detection rate",
                f"{business.get('fraud_detection_rate', 0):.1%}",
                help="Share of labeled frauds the model flagged.",
            )
            c2.metric(
                "False positive rate",
                f"{business.get('false_positive_rate', 0):.3%}",
                help="Share of labeled legitimate transactions wrongly flagged.",
            )
            c3.metric("Total cost", f"${business.get('total_cost', 0):,}")
            c4.metric("Labeled sample", f"{business.get('sample_size', 0):,}")

            # get_business_metrics only looks at the most recent window of
            # labeled predictions, so the figures above describe that window
            # and not the whole run. Saying "1,000 have a label" when 3,000
            # do would misdescribe what is on screen.
            st.markdown(
                f"<p class='caption-muted'>{total_predictions:,} predictions "
                f"scored in total. The figures above cover the most recent "
                f"{business.get('sample_size', 0):,} labeled predictions, not "
                "the full history.</p>",
                unsafe_allow_html=True,
            )

            cost_fp = business.get("cost_false_positives", 0)
            cost_fn = business.get("cost_false_negatives", 0)
            with st.container(border=True):
                st.markdown("**Where the cost comes from**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Source": "False positives (review cost)", "Cost": f"${cost_fp:,}"},
                            {"Source": "False negatives (missed fraud)", "Cost": f"${cost_fn:,}"},
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

        drift = summary.get("drift_metrics")
        if drift:
            with st.container(border=True):
                st.markdown("**Drift check**")
                d1, d2 = st.columns(2)
                d1.metric(
                    "Mean predicted probability",
                    f"{drift.get('mean_fraud_probability', 0):.4f}",
                )
                d2.metric("Window size", f"{drift.get('sample_size', 0):,}")
                if drift.get("drift_detected"):
                    st.warning(
                        "Mean predicted probability is above the "
                        f"{drift.get('threshold')} alert line. Worth "
                        "investigating; this is a crude check, not a "
                        "calibrated drift test."
                    )
                else:
                    st.success("Mean predicted probability within expected range.")
        elif total_predictions:
            st.caption("Drift check needs at least 100 predictions.")

with model_tab:
    st.subheader("Model card")
    st.caption(
        "Fixed results from training, measured once on the held-out test set. "
        "These do not change with live traffic."
    )

    with st.spinner("Loading model metrics..."):
        metrics, error = fetch("/metrics")

    if error:
        st.error(error)
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        c2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        c3.metric("False positive rate", f"{metrics.get('false_positive_rate', 0):.4%}")
        c4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

        n_test = metrics.get("n_test", 0)
        n_fraud_test = metrics.get("n_fraud_test", 0)
        if n_fraud_test:
            # State the sample size next to the percentages. On 75 positives
            # a single extra catch moves recall by over a point, which the
            # percentage on its own hides.
            step = 100.0 / n_fraud_test
            caption = (
                f"Decision threshold {metrics.get('threshold', 0):.2f}, selected "
                f"on a separate validation split. Test set: {n_test:,} "
                f"transactions containing only {n_fraud_test} frauds, so recall "
                f"moves in steps of roughly {step:.1f} percentage points."
            )
            st.markdown(
                f"<p class='caption-muted'>{caption}</p>", unsafe_allow_html=True
            )

        with st.container(border=True):
            st.markdown("**Cost vs. logistic regression baseline**")
            st.metric(
                "Point estimate, lower cost on test",
                f"{metrics.get('cost_savings_percent', 0):.1f}%",
                help="Both models had their threshold chosen on validation.",
            )
            # Shown without this caveat, the tile reads as a settled result.
            # Bootstrapping the test set puts the 95% interval across zero,
            # so the direction holds but the margin does not.
            st.markdown(
                "<p class='caption-muted'>Single draw on a test set with "
                f"{n_fraud_test} frauds, not a stable margin. Resampling the "
                "test set 5,000 times leaves the baseline cheaper in ~15% of "
                "draws and the 95% interval on the difference crosses zero. "
                "See outputs/bootstrap_cost.json.</p>",
                unsafe_allow_html=True,
            )

with health_tab:
    st.subheader("Health")
    with st.spinner("Checking API..."):
        health, error = fetch("/health")

    if error:
        st.error(error)
    elif health.get("status") == "healthy":
        st.success(f"Model loaded and serving. Version {health.get('model_version')}.")
    else:
        st.warning(
            f"API is running but reports status '{health.get('status')}'. "
            "The model probably failed to load; check that models/ contains "
            "fraud_model.pkl and metadata.json."
        )
