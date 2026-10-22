"""
Model Training & Benchmarking Pipeline
Trains Black-box Classifiers (XGBoost, Random Forest) alongside baseline Logistic Regression.
Saves models, preprocessed datasets, and evaluation metrics for CDSS inference & auditing.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

from src.data_loader import get_clinical_data

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_bs", "resting_ecg", "max_heart_rate", "exercise_angina",
    "st_depression", "st_slope", "num_major_vessels", "thalassemia"
]
TARGET_COL = "heart_disease"

def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = get_clinical_data()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Preserve sensitive attributes with the split
    sensitive_df = df[["sex", "gender_label", "age_group"]]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_df, test_size=0.25, random_state=42, stratify=y
    )

    # Standard Scaler for models sensitive to scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)

    # 1. XGBoost (Black-box tree ensemble)
    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)

    # 2. Random Forest (Black-box bagging ensemble)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 3. Logistic Regression (Linear baseline)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)

    models = {
        "XGBoost": xgb_model,
        "Random Forest": rf_model,
        "Logistic Regression": lr_model
    }

    metrics_summary = {}

    for name, model in models.items():
        eval_X = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(eval_X)
        y_prob = model.predict_proba(eval_X)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics_summary[name] = {
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Balanced Accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
            "Sensitivity (Recall)": round(float(recall_score(y_test, y_pred)), 4),
            "Specificity": round(float(specificity), 4),
            "Precision": round(float(precision_score(y_test, y_pred)), 4),
            "F1 Score": round(float(f1_score(y_test, y_pred)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4)
        }

    # Save artifacts
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgb_model.pkl"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Save test partition and metadata for app auditing
    test_data_bundle = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "sens_test": sens_test,
        "feature_cols": FEATURE_COLS
    }
    joblib.dump(test_data_bundle, os.path.join(MODELS_DIR, "test_data_bundle.pkl"))

    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("=== Model Benchmarking Completed ===")
    for model_name, m in metrics_summary.items():
        print(f"\n[{model_name}]")
        for k, v in m.items():
            print(f"  {k}: {v}")

    return metrics_summary

if __name__ == "__main__":
    train_and_evaluate()
