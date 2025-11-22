import requests
import time
import statistics

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

latencies = []
for _ in range(10):
    start = time.time()
    try:
        requests.post(url, json=data)
        latencies.append((time.time() - start) * 1000) # ms
    except Exception as e:
        print(f"Error: {e}")

if latencies:
    print(f"Average Latency: {statistics.mean(latencies):.2f} ms")
    print(f"Min Latency: {min(latencies):.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")
else:
    print("No successful requests.")
