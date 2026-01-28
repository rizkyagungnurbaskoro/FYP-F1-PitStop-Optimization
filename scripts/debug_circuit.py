import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd()))
from pitwall_api.app.data import load_dataset
from pitwall_api.app.demo import _pick_column, CIRCUIT_COL_CANDIDATES

def debug_circuit():
    df = load_dataset("my")
    print(f"Dataset columns: {df.columns.tolist()}")
    
    picked = _pick_column(df, CIRCUIT_COL_CANDIDATES)
    print(f"\nPicked Circuit Column: '{picked}'")
    
    if picked:
        print(f"Values in '{picked}': {df[picked].unique()[:10]}")
        
    print("\n--- All Candidates ---")
    for c in CIRCUIT_COL_CANDIDATES:
        if c in df.columns:
            print(f"Column '{c}': {df[c].dropna().unique()[:10]}")

if __name__ == "__main__":
    debug_circuit()
