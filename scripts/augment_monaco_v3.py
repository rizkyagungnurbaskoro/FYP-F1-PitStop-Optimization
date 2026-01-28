import pandas as pd
import fastf1
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
TARGET_FILE = DATA_DIR / "fastf1_demo_dataset.csv"
CACHE_DIR = BASE_DIR / "f1_cache"
fastf1.Cache.enable_cache(str(CACHE_DIR))

def augment_v3():
    print("Loading dataset...")
    df = pd.read_csv(TARGET_FILE, low_memory=False)
    
    # Initialize new columns
    extra_cols = ["LapTime_valid", "TrackTemp_valid", "Gap_valid"]
    for c in extra_cols:
        if c not in df.columns: df[c] = None

    print("Loading FastF1 Session (Monaco 2022)...")
    session = fastf1.get_session(2022, "Monaco", 'R')
    session.load(telemetry=True, laps=True, weather=True) # Enable weather
    
    print("Augmenting with VALIDATION data...")
    count = 0
    
    # We will process ALL drivers to ensure consistency
    for driver in session.drivers:
        drv_info = session.get_driver(driver)
        code = drv_info['Abbreviation']
        
        laps = session.laps.pick_driver(driver)
        if laps.empty: continue
        
        # Get dataset indices
        mask_drv = (df['race_id'] == "2022_Monaco") & (df['Driver'] == code)
        if not mask_drv.any(): continue
        
        dataset_indices = df[mask_drv].index
        
        for idx in dataset_indices:
            row = df.loc[idx]
            lap_target = int(float(row['lapno_prev'])) if pd.notna(row['lapno_prev']) else -1
            
            # Match FastF1 Lap (LapNumber == lap_target)
            # The dataset row is PREDICTING the next lap based on 'lapno_prev'.
            # So the 'status' (age, position) matches 'lapno_prev'.
            # The 'LapTime' we see in the screenshot (1:45.449) is likely the LAST completed lap time.
            
            match = laps[laps['LapNumber'] == lap_target]
            
            if match.empty:
                 match = laps[laps['LapNumber'] == lap_target + 1] # Fallback
            
            if not match.empty:
                lap_row = match.iloc[0]
                
                # Extract Lap Time (convert to string minutes:seconds.ms or just seconds?)
                # Screenshot shows 1:45.449. FastF1 gives Timedelta.
                lt = lap_row.get('LapTime', pd.NaT)
                if pd.notna(lt):
                    # Format: 1:45.449
                    # total_seconds = 105.449 -> 1:45.449
                    minutes = int(lt.total_seconds() // 60)
                    seconds = lt.total_seconds() % 60
                    lt_str = f"{minutes}:{seconds:06.3f}"
                    df.at[idx, 'LapTime_valid'] = lt_str
                
                # Extract Track Temp (from weather data associated with lap?)
                # FastF1 `session.weather_data`. Match by Time.
                # Or lap_row['TrackStatus']? No. 
                # We can interpolate weather.
                # FastF1 Laps have 'Time' (end of lap). We can find weather at that time.
                weather = session.weather_data
                # Find weather row closest to lap finish time
                if not weather.empty and 'Time' in lap_row:
                    lap_time_end = lap_row['Time'] # SessionTime
                    # Find closest weather row
                    # weather['Time'] is SessionTime
                    idx_weather = (weather['Time'] - lap_time_end).abs().idxmin()
                    w_row = weather.loc[idx_weather]
                    df.at[idx, 'TrackTemp_valid'] = w_row['TrackTemp']

                count += 1
                if count % 100 == 0: print(f"Updates: {count} (Last: {code} L{lap_target} Time={df.at[idx, 'LapTime_valid']})")

    print(f"Saving {count} validation updates to {TARGET_FILE}...")
    df.to_csv(TARGET_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    augment_v3()
