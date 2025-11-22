import requests
import json

url = "http://localhost:8000/predict"
data = {
    "age": 60,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 260,
    "fbs": 0,
    "restecg": 1,
    "thalach": 140,
    "exang": 1,
    "oldpeak": 2.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
