"""Imbalance experiments: compare class_weight vs SMOTE
Saves the best model to models/fraud_model.pkl
"""
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from src.utils import save_model
from src.models import train_baseline


def evaluate_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    return {
        'recall': recall_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'f1': f1_score(y_test, preds)
    }


def main(processed_path):
    df = pd.read_csv(processed_path)
    if 'Class' not in df.columns:
        raise ValueError('Processed data must contain Class column for evaluation')
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    results = {}

    # 1) Class-weighted logistic
    clf1 = train_baseline(X_train, y_train, method='logistic')
    results['logistic'] = evaluate_model(clf1, X_test, y_test)

    # 2) RandomForest class-weight
    clf2 = train_baseline(X_train, y_train, method='rf')
    results['rf'] = evaluate_model(clf2, X_test, y_test)

    # 3) SMOTE + RandomForest
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf3 = train_baseline(X_res, y_res, method='rf')
    results['rf_smote'] = evaluate_model(clf3, X_test, y_test)

    print('Results:', results)

    # Pick best by recall then f1
    scores = {k: (v['recall'], v['f1']) for k,v in results.items()}
    best_key = max(scores.keys(), key=lambda k: (scores[k][0], scores[k][1]))
    best_model = {'logistic': clf1, 'rf': clf2, 'rf_smote': clf3}[best_key]

    os.makedirs('models', exist_ok=True)
    save_model(best_model, 'models/fraud_model.pkl')
    print('Saved best model:', best_key)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-path', default='data/processed.csv')
    args = parser.parse_args()
    main(args.processed_path)
