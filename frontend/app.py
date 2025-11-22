from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(_name_)

# Load the saved ML model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Heart Disease Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["fbs"],
        data["restecg"],
        data["thalach"],
        data["exang"],
        data["oldpeak"],
        data["slope"],
        data["ca"],
        data["thal"]
    ]

    input_data = np.array(features).reshape(1, -1)
    probability = model.predict_proba(input_data)[0][1]

    # FIXED HERE ⬇⬇⬇ (frontend expects risk_probability)
    return jsonify({
        "risk_probability": float(probability),
        "message": "Prediction successful"
    })

if _name_ == "_main_":
    app.run(debug=True)