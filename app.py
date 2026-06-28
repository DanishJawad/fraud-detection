import streamlit as st
import yaml
import io
import pandas as pd
from src.utils import load_model, load_config
from src.features import create_features
from src.evaluation import sweep_thresholds

st.set_page_config(page_title="Fraud Detection Demo")

config = load_config('config.yaml')
model = None
try:
    model = load_model('models/fraud_model.pkl')
except Exception:
    model = None

st.title('Fraud Detection — Cost-Aware Demo')

st.sidebar.header('Cost settings')
cost_fp = st.sidebar.number_input('Cost per False Positive', value=config.get('cost_fp',50))
cost_fn = st.sidebar.number_input('Cost per False Negative', value=config.get('cost_fn',500))

st.sidebar.markdown('---')

if model is None:
    st.warning('Model artifact not found. Run the baseline script to create models/fraud_model.pkl')

st.write('Upload a CSV of transactions (with columns similar to the Kaggle dataset).')
uploaded = st.file_uploader('Upload CSV (optional)', type=['csv'])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write('Rows:', len(df))
    df_proc = create_features(df)
    # Prepare features for model
    X = df_proc.drop(columns=[c for c in ['Class'] if c in df_proc.columns], errors='ignore')
    if model is None:
        st.error('No model available to score. Run notebooks/01_baseline.py or notebooks/03_imbalance.py first.')
    else:
        proba = model.predict_proba(X)[:,1]
        df_proc['score'] = proba
        st.write(df_proc.head(10))
        threshold = st.slider('Threshold for positive (flag)', 0.0, 1.0, float(config.get('optimal_threshold',0.5)))
        preds = (proba >= threshold).astype(int)
        st.write('Flagged:', int(preds.sum()))

        # If true labels provided, compute optimal threshold on uploaded data
        if 'Class' in df_proc.columns:
            opt_t, opt_cost, thresholds, costs = sweep_thresholds(proba, df_proc['Class'], cost_fp, cost_fn)
            st.write(f'Optimal threshold for provided cost settings: {opt_t:.3f} (expected cost ${opt_cost:.2f})')
        else:
            st.info('No true labels in uploaded data — cannot compute optimal threshold.')

        st.download_button('Download scored CSV', df_proc.to_csv(index=False), file_name='scored.csv')
else:
    st.info('Upload a CSV to score transactions or run the baseline scripts to generate model artifacts.')
