import pandas as pd
import fastf1
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
TARGET_FILE = DATA_DIR / "fastf1_demo_dataset.csv"
CACHE_DIR = BASE_DIR / "f1_cache"
fastf1.Cache.enable_cache(str(CACHE_DIR))

def augment_v2():
    print("Loading dataset...")
    df = pd.read_csv(TARGET_FILE, low_memory=False)
    
    print("Loading FastF1 Session...")
    session = fastf1.get_session(2022, "Monaco", 'R')
    session.load(telemetry=True, laps=True, weather=False)
    
    # Target specific drivers to ensure Iconic moments are fixed first
    focus_drivers = ["LEC", "SAI", "PER", "VER", "RUS", "NOR", "ALO", "HAM"]
    
    print("Augmenting...")
    count = 0
    for driver in session.drivers:
        drv_info = session.get_driver(driver)
        code = drv_info['Abbreviation']
        
        # Optimization: Only process if driver is in our list or do all
        # To be safe and aligned, we do all, but we print progress for focus drivers
        
        laps = session.laps.pick_driver(driver)
        if laps.empty: continue
        
        # Get dataset indices for this driver/race
        mask_drv = (df['race_id'] == "2022_Monaco") & (df['Driver'] == code)
        if not mask_drv.any(): continue
        
        dataset_indices = df[mask_drv].index
        
        for idx in dataset_indices:
            row = df.loc[idx]
            lap_target = int(float(row['lapno_prev'])) if pd.notna(row['lapno_prev']) else -1
            
            # Match FastF1 Lap
            # fastf1 lap is row['LapNumber']
            match = laps[laps['LapNumber'] == lap_target]
            
            if match.empty:
                 # Try offset?
                 match = laps[laps['LapNumber'] == lap_target + 1]
            
            if not match.empty:
                tel_row = match.iloc[0] # Series
                
                # Fetch legit data strings
                df.at[idx, 'tire_age_legit'] = tel_row.get('TyreLife', None) or tel_row.get('TyreLife', None)
                df.at[idx, 'compound_legit'] = tel_row.get('Compound', None)
                
                # Telemetry (expensive?)
                try:
                    tel = match.get_telemetry()
                    if not tel.empty:
                        last = tel.iloc[-1]
                        df.at[idx, 'Speed_FL'] = last.get('Speed', None)
                except:
                    pass
                
                count += 1
                if count % 50 == 0: print(f"Processed {count} rows... (Last: {code} L{lap_target})")

    print(f"Saving {count} updates to {TARGET_FILE}...")
    df.to_csv(TARGET_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    augment_v2()
