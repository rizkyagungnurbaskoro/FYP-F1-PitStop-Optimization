import fastf1
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "f1_cache"
fastf1.Cache.enable_cache(str(CACHE_DIR))

def find_lap_time():
    print("Searching for SAI Lap Time 1:25.485 in FastF1 data...")
    session = fastf1.get_session(2022, "Monaco", 'R')
    session.load(laps=True, telemetry=False, weather=False)
    
    sai_laps = session.laps.pick_driver("SAI")
    
    # Target: 1:25.485 = 85.485s
    target = 85.485
    
    found = False
    for idx, lap in sai_laps.iterlaps():
        lt = lap['LapTime']
        if pd.notna(lt):
            sec = lt.total_seconds()
            if abs(sec - target) < 0.01: # Check exact
                print(f"!!! MATCH FOUND !!! Lap {lap['LapNumber']} Time: {sec:.3f}s (1:25.485)")
                print(f"Propagating Age: {lap.get('TyreLife')}")
                found = True
            elif abs(sec - target) < 0.5: # Check close
                 print(f"Close Match: Lap {lap['LapNumber']} Time: {sec:.3f}s")
    
    if not found:
        print("No exact match found in FastF1.")

import pandas as pd
if __name__ == "__main__":
    find_lap_time()
