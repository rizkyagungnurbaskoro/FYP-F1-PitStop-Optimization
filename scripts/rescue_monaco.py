import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
SOURCE_FILE = DATA_DIR / "fastf1_strategy_dataset.csv"
TARGET_FILE = DATA_DIR / "fastf1_demo_dataset.csv"

def rescue_scan():
    print("Loading SOURCE dataset (original)...")
    df = pd.read_csv(SOURCE_FILE, low_memory=False)
    
    print("Scanning ALL SAI rows for 1:25.485 (85.485s)...")
    
    mon = df[(df['race_id'] == '2022_Monaco') & (df['Driver'] == 'SAI')]
    cols_of_interest = []
    
    # Check ALL rows
    targets = mon
    
    print("\nTarget Rows (SAI Lap 20-21):")
    # check if any column has value matching string or float
    found = False
    for idx, row in targets.iterrows():
        # print(f"\n--- Row {idx} (Lap {row['lapno_prev']}) ---")
        for col in df.columns:
            val = str(row[col])
            
            # String Search
            if "25.48" in val or "1:25.48" in val:
                 print(f"!!! MATCH FOUND !!! Row {idx} Lap {row['lapno_prev']} Column '{col}': {val}")
                 print(row[['lapno_prev', 'tire_age', 'compound', 'position']].to_dict())
                 found = True

            # Float Search
            try:
                fval = float(row[col])
                if abs(fval - 85.485) < 0.1: # Tolerance
                    print(f"!!! MATCH FOUND (Float) !!! Row {idx} Lap {row['lapno_prev']} Column '{col}': {val}")
                    found = True
            except:
                pass

    if not found:
        print("\nNo exact match for 85.485s found in source dataset columns.")

if __name__ == "__main__":
    rescue_scan()
