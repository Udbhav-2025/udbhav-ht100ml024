# Cardio Predictor Backend

This backend implements a simple API to predict heart disease risk using the UCI Heart Disease dataset.

Files:
- `preprocessing.py` - dataset loader and preprocessor builder
- `train.py` - train Logistic Regression and Random Forest models and save artifacts to `models/`
- `main.py` - FastAPI application exposing `/predict`, `/health`, and `/features`
- `requirements.txt` - Python dependencies
- `models/` - directory where trained models and preprocessor are stored (created by `train.py`)

Quick start:

1. Create a virtual environment and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train models (this will download the UCI dataset, train models, and save artifacts):

```pwsh
python train.py
```

3. Run the API:

```pwsh
uvicorn main:app --reload
```

4. Example predict request (using `curl`):

```pwsh
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Notes:
- Missing fields in the JSON are allowed; the backend will impute sensible defaults.
- Default model is Random Forest; switch to Logistic Regression with query param `?model=logreg`.
