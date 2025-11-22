import streamlit as st
import requests

st.title("Heart Disease Risk Predictor")
st.write("Enter the values to get predicted heart disease probability")

# Minimal input fields; the backend will impute any missing values
age = st.number_input("Age", 20, 90, 50)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0], index=0)
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], index=3)
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1], index=0)
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
thalach = st.number_input("Max Heart Rate", 60, 220, 140)
exang = st.selectbox("Exercise induced angina (exang)", [0, 1], index=0)
oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.5, 1.0)
slope = st.selectbox("Slope", [0, 1, 2], index=1)
ca = st.number_input("Number of major vessels (0-3)", 0, 3, 0)
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3], index=0)

BACKEND_URL = st.text_input("Backend URL", value="http://127.0.0.1:8000")

if st.button("Predict Heart Risk"):
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    try:
        res = requests.post(f"{BACKEND_URL}/predict", json=data, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach backend: {e}")
        st.stop()

    if res.status_code == 200:
        body = res.json()
        # backend returns `risk_score`, `risk_level`, `model_used`, `top_factors`
        score = body.get("risk_score")
        level = body.get("risk_level")
        model_used = body.get("model_used")
        top = body.get("top_factors", [])

        st.success(f"Predicted Heart Disease Risk: {score:.2f} ({level})")
        st.info(f"Model used: {model_used}")
        if top:
            st.write("Top contributing factors:", ", ".join(top))
    else:
        # show backend error message when available
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        st.error(f"Backend error ({res.status_code}): {detail}")