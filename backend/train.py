"""
train.py

Train Logistic Regression and Random Forest models on the UCI Heart Disease dataset.
Saves models and preprocessing artifacts to `backend/models/` using joblib.

Run:
    pip install -r requirements.txt
    python train.py

"""
import os
from pprint import pprint
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from preprocessing import load_uc_heart, build_preprocessor, get_feature_list


def main():
    # Load raw data
    df = load_uc_heart()

    # Define features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    feature_cols = numeric_cols + categorical_cols

    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Build preprocessor and fit on entire data (or training split)
    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    # Train-test split to evaluate models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit preprocessor on training data
    preprocessor.fit(X_train)

    # Transform for model training
    X_train_tr = preprocessor.transform(X_train)
    X_test_tr = preprocessor.transform(X_test)

    # Train Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_tr, y_train)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_tr, y_train)

    # Evaluate
    for name, model, X_t in [("LogisticRegression", logreg, X_test_tr), ("RandomForest", rf, X_test_tr)]:
        probs = model.predict_proba(X_t)[:, 1]
        preds = model.predict(X_t)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        print(f"{name}: AUC={auc:.3f}, Acc={acc:.3f}")

    # Save artifacts
    out_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(preprocessor, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(logreg, os.path.join(out_dir, "logistic_regression.pkl"))
    joblib.dump(rf, os.path.join(out_dir, "random_forest.pkl"))

    # Save metadata: feature list (order must match transformed array layout)
    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_list": get_feature_list(numeric_cols, categorical_cols),
    }
    joblib.dump(metadata, os.path.join(out_dir, "metadata.pkl"))

    print("Saved models and metadata to:", out_dir)


if __name__ == "__main__":
    main()
