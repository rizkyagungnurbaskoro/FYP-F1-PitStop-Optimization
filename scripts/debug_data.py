import pandas as pd
import sys
import os

# Set paths
base_dir = r"d:\University\FYP B\FYP_FINAL"
dataset_path = os.path.join(base_dir, "personal_datasets", "fastf1_strategy_dataset.csv")

if not os.path.exists(dataset_path):
    print(f"Dataset not found: {dataset_path}")
    sys.exit(1)

df = pd.read_csv(dataset_path)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()[:10]} ...")

if "in_pit_window_prev" in df.columns:
    count = (pd.to_numeric(df["in_pit_window_prev"], errors="coerce").fillna(0) > 0).sum()
    print(f"Rows with in_pit_window_prev > 0: {count}")

if "nolaps_prev" in df.columns:
    print(f"Sample nolaps_prev: {df['nolaps_prev'].unique()[:5]}")

# Simulation of _pit_window_bounds
def _test_bounds(df, lap_col="lapno_prev"):
    if df.empty: return "EMPTY DF"
    window_col = None
    for cand in ("in_pit_window_prev", "in_pit_window", "pit_window_prev", "pit_window"):
        if cand in df.columns:
            window_col = cand
            break
    
    current_lap_max = 0
    if lap_col and lap_col in df.columns:
        current_lap_max = int(pd.to_numeric(df[lap_col], errors="coerce").max() or 0)

    if window_col is not None:
        mask = pd.to_numeric(df[window_col], errors="coerce").fillna(0) > 0
        if mask.any() and lap_col and lap_col in df.columns:
            laps = pd.to_numeric(df.loc[mask, lap_col], errors="coerce").dropna()
            if not laps.empty:
                return int(laps.min()), int(laps.max())
    
    total_laps = current_lap_max
    for cand in ("nolaps_prev", "nolaps", "n_laps", "laps"):
        if cand in df.columns:
            total_laps = int(pd.to_numeric(df[cand], errors="coerce").max() or total_laps)
            break
    if total_laps > 10:
        return int(total_laps * 0.22), int(total_laps * 0.48)
    return None

print("\n--- Full Dataset ---")
print(f"Bounds: {_test_bounds(df)}")

print("\n--- Melbourne 2018 VER Detailed ---")
subset = df[(df['Driver'] == 'VER') & (df['race_id'] == '2018_Melbourne')]
window_vals = subset['in_pit_window_prev'].tolist()
print(f"Laps: {subset['lapno_prev'].tolist()}")
print(f"Window Flags: {window_vals}")

print("\n--- Monaco 2022 LEC Detailed ---")
subset_monaco = df[(df['Driver'] == 'LEC') & (df['race_id'] == '2022_Monaco')]
print(f"Rows: {len(subset_monaco)}")
print(f"Window Flags: {subset_monaco['in_pit_window_prev'].tolist()}")

print("\n--- Sample Circuit Check ---")
if 'circuit_name' in df.columns:
    print(f"Circuits: {df['circuit_name'].unique()[:5]}")
else:
    # derive it
    df['circuit_name'] = df['race_id'].astype(str).apply(lambda x: x.split('_', 1)[1] if '_' in x else x)
    print(f"Derived Circuits: {df['circuit_name'].unique()[:5]}")
