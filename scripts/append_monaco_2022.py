import pandas as pd
import fastf1
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
SOURCE_FILE = DATA_DIR / "fastf1_strategy_dataset.csv"
TARGET_FILE = DATA_DIR / "fastf1_demo_dataset.csv"

def append_monaco2022():
    print("Loading source...")
    df_source = pd.read_csv(SOURCE_FILE, low_memory=False)
    monaco2022 = df_source[df_source['race_id'] == '2022_Monaco'].copy()
    
    print(f"Found {len(monaco2022)} rows for 2022_Monaco")
    
    print("Loading target...")
    df_target = pd.read_csv(TARGET_FILE, low_memory=False)
    
    # Ensure columns match (add missing cols to monaco chunk)
    for col in df_target.columns:
        if col not in monaco2022.columns:
            monaco2022[col] = None
            
    # Append
    df_combined = pd.concat([df_target, monaco2022], ignore_index=True)
    
    print(f"Saving combined dataset ({len(df_combined)} rows)...")
    df_combined.to_csv(TARGET_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    append_monaco2022()
