"""
preprocessing.py

Functions to load the UCI Heart Disease dataset and build a preprocessing
transformer. The transformer handles missing values and scaling.

This module is intentionally beginner-friendly and well-commented.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
import requests


def get_column_names() -> List[str]:
    """Return the column names for the UCI heart disease 'processed.cleveland' data.

    Reference: https://archive.ics.uci.edu/ml/datasets/heart+disease
    """
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]


def download_uc_heart(dest_path: str) -> str:
    """Download the processed Cleveland data if not present.

    Returns local filepath.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Downloading dataset from {url} to {dest_path}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
    return dest_path


def load_uc_heart(filepath: str = None) -> pd.DataFrame:
    """Load the UCI Heart Disease (Cleveland) dataset into a DataFrame.

    - Replaces '?' with NaN
    - Converts all columns to numeric where possible
    - Maps target > 0 to 1 (presence of disease)
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "data", "processed.cleveland.data")
    filepath = download_uc_heart(filepath)

    colnames = get_column_names()
    df = pd.read_csv(filepath, header=None, names=colnames, na_values="?")

    # Convert all columns to numeric (some were read as object due to '?')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # In original dataset target is 0 (no disease) and 1-4 (disease). Convert to binary.
    df["target"] = df["target"].apply(lambda x: 1 if x and x > 0 else 0)

    return df


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and scales numeric columns and imputes categorical columns.

    - Numeric columns: impute with mean, then standard scale
    - Categorical columns: impute with most frequent (mode)

    Returns: fitted ColumnTransformer (fit later on training data)
    """
    # Pipeline for numeric features: impute with mean, then scale
    from sklearn.pipeline import Pipeline

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_list(numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """Return the list of feature names after our simple preprocessing.

    Since we are not one-hot encoding categorical variables, the output feature order
    from ColumnTransformer will be numeric_cols then categorical_cols (as configured).
    """
    return numeric_cols + categorical_cols


if __name__ == "__main__":
    # Quick test: download and load dataset
    df = load_uc_heart()
    print(df.head())
