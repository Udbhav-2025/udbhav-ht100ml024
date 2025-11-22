"""
main.py

FastAPI backend for The Cardio Predictor.

Endpoints:
- GET /health -> simple status
- GET /features -> returns required input fields
- POST /predict -> accepts patient JSON and returns risk score, model used, top factors

Run:
    uvicorn main:app --reload

"""
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib
import os


# Load artifacts on startup if available. If they are missing, endpoints will return a helpful error.
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

preprocessor = None
logreg = None
rf = None
metadata = None

try:
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    logreg = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
    numeric_cols = metadata["numeric_cols"]
    categorical_cols = metadata["categorical_cols"]
    feature_list = metadata["feature_list"]
except Exception:
    # Models not trained yet or files missing. Keep None and provide clear messages from endpoints.
    print("Model artifacts not found in 'models/'. Run 'python train.py' to create them.")
    numeric_cols = []
    categorical_cols = []
    feature_list = []

app = FastAPI(title="Cardio Predictor API")

# Allow CORS for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientData(BaseModel):
    # All fields are optional; preprocessing will impute missing values
    age: Optional[float] = None
    sex: Optional[int] = None
    cp: Optional[int] = None
    trestbps: Optional[float] = None
    chol: Optional[float] = None
    fbs: Optional[int] = None
    restecg: Optional[int] = None
    thalach: Optional[float] = None
    exang: Optional[int] = None
    oldpeak: Optional[float] = None
    slope: Optional[int] = None
    ca: Optional[float] = None
    thal: Optional[float] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/features")
def features():
    """Return the required input fields and their categories."""
    if preprocessor is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run 'python train.py' to train models and create artifacts in the models/ folder.")

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "all_features": feature_list,
        "notes": "All fields are optional on input; missing values are imputed with sensible defaults.",
    }


def _prepare_input(data: Dict) -> np.ndarray:
    """Construct a DataFrame from input dict and apply the preprocessor.

    Any missing features are left as NaN so the preprocessor can impute.
    Returns the transformed numpy array (1 x n_features_tr).
    """
    # Lazy import pandas to avoid hard dependency at module import time
    try:
        import pandas as _pd
    except Exception:
        raise RuntimeError("pandas is required to prepare input. Install pandas in the Python environment.")

    if not feature_list:
        raise ValueError("Feature metadata is not available. Ensure models are trained and metadata.pkl exists.")

    # Build a dictionary for the DataFrame. Missing or invalid values become None (which pandas handles as NaN/None)
    row_dict = {}
    for c in feature_list:
        v = data.get(c, None)
        if v is None:
            row_dict[c] = [None]
            continue
        try:
            row_dict[c] = [float(v)]
        except Exception:
            row_dict[c] = [None]

    # Create DataFrame with correct columns
    X = _pd.DataFrame(row_dict)

    # Ensure columns are in the correct order (though dict preservation usually handles this, explicit is better)
    X = X[feature_list]

    X_tr = preprocessor.transform(X)
    return X_tr


def _get_top_factors(model, X_tr: np.ndarray, k: int = 3) -> List[str]:
    """Return top k contributing feature names for the provided model and a single transformed input row.

    Heuristic contributions used:
    - LogisticRegression: coef * x (signed contributions)
    - RandomForest: feature_importances_ * x (signed)

    We rank by absolute contribution and return original feature names.
    """
    # Lazy import numpy for numeric operations
    try:
        import numpy as _np
    except Exception:
        raise RuntimeError("numpy is required to compute top factors. Install numpy in the Python environment.")

    if hasattr(model, "coef_"):
        # Logistic Regression
        coefs = model.coef_[0]
        contrib = coefs * X_tr[0]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        contrib = importances * X_tr[0]
    else:
        # Fallback: use absolute transformed feature values
        contrib = _np.abs(X_tr[0])

    # Map contributions back to feature names
    contrib_abs = _np.abs(contrib)
    top_idx = _np.argsort(contrib_abs)[::-1][:k]
    top_features = [feature_list[i] for i in top_idx]
    return top_features


@app.post("/predict")
def predict(payload: PatientData, model: Optional[str] = Query(None, description="model=random_forest or model=logreg")):
    """Predict the risk score for a patient.

    - `model` query param: set to `logreg` to use Logistic Regression. Default is Random Forest.
    - Returns probability, risk level, model used, and top contributing factors.
    """
    if preprocessor is None or metadata is None or logreg is None or rf is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run 'python train.py' to train models and create artifacts in the models/ folder.")

    data = payload.dict()

    # Choose model
    model_choice = "RandomForest"
    model_obj = rf
    if model and model.lower() == "logreg":
        model_choice = "LogisticRegression"
        model_obj = logreg

    try:
        X_tr = _prepare_input(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preparing input: {e}")

    # Predict probability of positive class
    try:
        prob = float(model_obj.predict_proba(X_tr)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Risk level thresholds (simple): <0.33 low, 0.33-0.66 medium, >0.66 high
    if prob > 0.66:
        level = "HIGH"
    elif prob > 0.33:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Compute top contributing factors
    try:
        top_factors = _get_top_factors(model_obj, X_tr, k=3)
    except Exception:
        top_factors = []

    return {
        "risk_score": round(prob, 4),
        "risk_level": level,
        "model_used": model_choice,
        "top_factors": top_factors,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
