# Cost-Optimized Fraud Detection System

A production-ready ML system for credit card fraud detection that optimizes for business cost rather than accuracy, featuring SMOTE for imbalanced data, cost-sensitive threshold tuning, FastAPI serving, real-time monitoring, and Docker containerization.

## 🎯 Key Features

- **Cost-Optimized Threshold Tuning**: Finds the optimal decision threshold by minimizing business cost (false alarm vs missed fraud).
- **SMOTE for Class Imbalance**: Handles severe class imbalance (~0.17% fraud) using synthetic oversampling.
- **Production REST API**: FastAPI endpoints for single/batch predictions and metrics.
- **Real-time Monitoring**: Streamlit dashboard + local/PostgreSQL logging for performance tracking.
- **A/B Testing Framework**: Simple framework to safely test new model versions.
- **Automated Retraining**: Pipeline to retrain and promote models when performance improves.
- **Docker & Compose**: Fully containerized for reproducible deployment.

## 📊 Current Results

- **Dataset**: ~285k transactions, 492 frauds (~0.17% fraud rate)
- **Model**: Logistic Regression with SMOTE + StandardScaler
- **ROC-AUC**: 0.9794
- **Fraud Detection Rate**: ~80%
- **False Alarm Rate**: ~0.08%
- **Cost Savings**: ~73.9% vs default threshold (dataset-dependent)

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & docker-compose (optional, for containerized demo)

### Local Setup

```bash
# Create and activate venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Prepare data
unzip -o data/creditcard.csv.zip -d data
PYTHONPATH=. python notebooks/02_features.py --data-path data/creditcard.csv --out-path data/processed.csv

# Train baseline (produces models/fraud_model.pkl, models/metadata.json)
PYTHONPATH=. python notebooks/01_baseline.py --data-path data/processed.csv --cost-fp 50 --cost-fn 500
```

### Run Services

**Option 1: Local Services** (lightweight, no Docker)
```bash
# Terminal 1: Start the lightweight API server
python simple_api.py --host 127.0.0.1 --port 8000

# Terminal 2: Start the Streamlit dashboard
streamlit run dashboard/app.py
```

**Option 2: Docker Compose** (recommended for full demo)
```bash
docker compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.0, 0.5, -1.2, ...]}'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '[{"features":[...]}, {"features":[...]}]'
```

### Get Metrics
```bash
curl http://localhost:8000/metrics
```

## 📁 Project Structure

```
fraud-detection-ml/
├── notebooks/                 # Training scripts (reproducible)
│   ├── 01_baseline.py        # Train baseline, threshold search
│   ├── 02_features.py        # Feature engineering
│   ├── 03_imbalance.py       # SMOTE comparison
│   └── 04_costs.py           # Cost sensitivity analysis
├── src/                       # ML utilities
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py            # SMOTE + baseline training
│   ├── evaluation.py        # Threshold sweep, cost calc
│   └── utils.py
├── app/                       # FastAPI service
│   ├── main.py
│   ├── registry.py          # Model loading
│   ├── models.py            # Pydantic schemas
│   └── config.py
├── dashboard/                 # Streamlit monitoring
│   └── app.py
├── monitoring/                # Logging & drift detection
│   ├── tracker.py
│   └── db.py                # Optional PostgreSQL logging
├── ab_test/                   # A/B testing framework
│   └── framework.py
├── retrain/                   # Automated retraining
│   └── pipeline.py
├── tests/                     # Test suite
│   └── test_api.py
├── models/                    # Trained artifacts (gitignored)
│   ├── fraud_model.pkl      # Trained model
│   ├── scaler.pkl           # Feature scaler
│   └── metadata.json        # Threshold, metrics, confusion matrix
├── Dockerfile                 # API container
├── docker-compose.yml         # Full stack orchestration
├── requirements.txt
└── README.md
```

## 🏗️ How It Works

1. **Data Preparation** (`notebooks/02_features.py`):
   - Load CSV, engineer features, save processed data.

2. **Training** (`notebooks/01_baseline.py`):
   - Train on SMOTE-balanced data, compute predictions on test set.
   - Search thresholds to minimize business cost (false alarms + missed frauds).
   - Save model, scaler, optimal threshold, and metrics.

3. **Serving** (`app/main.py`):
   - Load model and metadata at startup via registry.
   - Handle `/predict`, `/metrics`, `/health` endpoints.
   - Optional: log predictions to PostgreSQL if `DATABASE_URL` is set.

4. **Monitoring** (`dashboard/app.py`):
   - Poll `/metrics` and `/health` to display KPIs.
   - Show fraud detection rate, false alarms, cost savings, and model health.

5. **Improvement** (`retrain/`, `ab_test/`):
   - Retrain on new data and promote if metrics improve.
   - A/B test new threshold/model against production.

## 🧪 Testing

```bash
# Run tests (requires pytest and httpx)
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test
PYTHONPATH=. python -m pytest tests/test_api.py::test_health -v
```

## 🔧 Configuration

### Environment Variables
- `OPTIMAL_THRESHOLD`: Decision threshold (default: 0.35)
- `COST_FALSE_POSITIVE`: Cost of blocking legitimate (default: $50)
- `COST_FALSE_NEGATIVE`: Cost of missing fraud (default: $500)
- `DATABASE_URL`: PostgreSQL connection string (optional)

Example:
```bash
export DATABASE_URL='postgresql://user:pass@localhost:5432/fraud'
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🎓 Key Insights

### Why Threshold Matters
Default threshold 0.5 treats probabilities symmetrically. But fraud costs differ: missing fraud ($500) is worse than blocking a customer ($50). The cost-optimized threshold (~0.05) reflects this asymmetry, catching more fraud with acceptable false alarms.

### Why SMOTE
Training on 99.83% legitimate + 0.17% fraud leads to poor fraud detection. SMOTE synthetically oversamples the minority class, creating balanced training data so the model learns fraud patterns.

### Why REST API
Model serving via HTTP makes the system reusable: dashboard, batch scoring, other services, and future integrations all call the same API. Containerization ensures reproducibility.

## 📊 Results Interpretation

Metrics in `models/metadata.json` are **dataset and cost-dependent**:
- Retrain with different cost values → different optimal threshold.
- Use on a different dataset → different metrics.
- Report as "achieves ~80% fraud detection and ~0.08% false alarm rate on this dataset" rather than as universals.

## 🚢 Deployment

### Docker (Recommended)
```bash
docker compose up --build
```

### Manual (Python)
```bash
source .venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📚 References

- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [SMOTE Oversampling](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 📝 License

MIT

## 👤 Author

Muhammad Danish Jawad  
GitHub: [DanishJawad](https://github.com/DanishJawad)  
Portfolio: [muhammaddanish.com](https://muhammaddanish.com)
