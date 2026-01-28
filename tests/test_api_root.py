import requests
import json

base_url = "http://127.0.0.1:8000"
selection = {
    "dataset": "my",
    "driver": "VER",
    "circuit": "Melbourne",
    "weather": "Dry",
    "lap": 1,
    "year": "2018"
}

try:
    res = requests.post(f"{base_url}/demo/pitwindow", json=selection)
    print(f"Status: {res.status_code}")
    if res.ok:
        print(f"Response: {res.json()}")
    else:
        print(f"Error: {res.text}")
except Exception as e:
    print(f"Request failed: {e}")
