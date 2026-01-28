import sys
from pathlib import Path
import pandas as pd
import traceback

sys.path.append(str(Path.cwd()))
from pitwall_api.app.data import load_dataset
from pitwall_api.app.demo import _prepare_demo, run_demo_state, CIRCUIT_COL_CANDIDATES, _pick_column

def verify():
    print("--- 1. Inspecting Dataset Raw ---")
    try:
        df = load_dataset("my")
        print(f"Columns: {df.columns.tolist()[:10]}... (total {len(df.columns)})")
        if "race_id" in df.columns:
            print(f"Sample race_id: {df['race_id'].unique()[:5]}")
        
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print("\n--- 2. Verifying _prepare_demo Internals ---")
    try:
        # We want to see what _prepare_demo does.
        # Since it's cached, we might need to be careful, but we are running in a fresh process.
        prep = _prepare_demo("my")
        
        print(f"Circuit Col Final: '{prep.get('circuit_col')}'")
        print(f"Drivers: {len(prep.get('drivers', []))}")
        print(f"Circuits: {len(prep.get('circuits', []))}")
        print(f"Years: {len(prep.get('years', []))}")
        
        if prep.get('circuits'):
            print(f"Circuit Samples: {prep['circuits'][:5]}")
        else:
            print("(!) No circuits found.")
            
    except Exception as e:
        traceback.print_exc()

    print("\n--- 3. Verifying Filtering Logic ---")
    try:
        # Simulate a request from UI
        # User screenshot showed: Year 2020, Driver BOT, Circuit (Empty/Select), Weather Wet
        sel = {
            "dataset": "my",
            "year": "2020",
            "driver": "BOT",
            "circuit": None, 
            "weather": "Wet",
            "lap": 1
        }
        print(f"Simulating Selection: {sel}")
        state = run_demo_state("my", sel)
        
        # Check if we got a valid row
        train_row = state.get("train", {}).get("row", {})
        if train_row:
            print("[SUCCESS] Got train row.")
            print(f"Driver: {train_row.get('Driver')}, Lap: {train_row.get('lapno')}")
        else:
            print("[FAILURE] No train row found.")
            
        final_sel = state.get("selection", {})
        print(f"Final Selection State: {final_sel}")
        
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    verify()
