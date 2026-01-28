
import pandas as pd
from pitwall_api.app.data import load_dataset

def debug_monaco():
    df = load_dataset("my")
    monaco = df[df["race_id"].astype(str) == "2022_Monaco"]
    
    print("Columns:", monaco.columns.tolist())
    
    # Check LEC and SAI around lap 20-23
    drivers = ["LEC", "SAI"]
    for drv in drivers:
        d = monaco[monaco["Driver"] == drv]
        subset = d[(d["lapno"] >= 19) & (d["lapno"] <= 21)]
        print(f"\n--- {drv} ---")
        cols = ["lapno", "decide_pitstop", "Position_prev", "gap_to_leader_prev", "gap"]
        # Filter cols that actually exist
        exist_cols = [c for c in cols if c in d.columns]
        print(subset[exist_cols].to_string())


if __name__ == "__main__":
    import sys
    # Redirect stdout to a file
    with open("monaco_data.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        debug_monaco()

