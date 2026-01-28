import pandas as pd
import sys
import os
import json

# Set paths
base_dir = r"d:\University\FYP B\FYP_FINAL\pitwall_api"
sys.path.append(base_dir)

from app.demo import run_demo_state

selection = {
    "dataset": "my",
    "driver": "VER",
    "circuit": "Melbourne",
    "weather": "Dry",
    "lap": 1,
    "year": "2018"
}

try:
    state = run_demo_state("my", selection)
    train_payload = state.get("demo", {}).get("train", {}).get("payload", {})
    print(f"Train Payload: {json.dumps(train_payload, indent=2)}")
except Exception as e:
    import traceback
    traceback.print_exc()
