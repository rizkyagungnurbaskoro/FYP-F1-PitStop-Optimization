import pandas as pd
import fastf1
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
INPUT_FILE = DATA_DIR / "fastf1_demo_dataset.csv"
CACHE_DIR = BASE_DIR / "f1_cache"

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def diagnose():
    print("Loading simplified dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    race_key = "2022_Monaco"
    driver_code = "LEC" # Test with Leclerc
    
    # Dataset view
    mon = df[(df['race_id'] == race_key) & (df['Driver'] == driver_code)]
    print(f"\nDATASET: Found {len(mon)} rows for {driver_code} @ {race_key}")
    print("Dataset 'lapno_prev' first 5:", mon['lapno_prev'].unique()[:5])
    
    # FastF1 view
    print(f"\nFASTF1: Loading session...")
    session = fastf1.get_session(2022, "Monaco", 'R')
    session.load(telemetry=True, laps=True, weather=False)
    
    drv_laps = session.laps.pick_driver(driver_code)
    print(f"FastF1: Found {len(drv_laps)} laps for {driver_code}")
    print("FastF1 'LapNumber' first 5:", drv_laps['LapNumber'].unique()[:5])
    
    # Check intersection
    ds_laps = set(mon['lapno_prev'].dropna().astype(int))
    ff_laps = set(drv_laps['LapNumber'].astype(int))
    
    common = ds_laps.intersection(ff_laps)
    print(f"\nIntersection Count: {len(common)}")
    print(f"Common Sample: {list(common)[:5]}")

if __name__ == "__main__":
    diagnose()
