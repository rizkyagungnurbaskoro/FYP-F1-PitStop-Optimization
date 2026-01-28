
import pandas as pd
import fastf1
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
INPUT_FILE = DATA_DIR / "fastf1_strategy_dataset.csv"
OUTPUT_FILE = DATA_DIR / "fastf1_demo_dataset.csv"
CACHE_DIR = BASE_DIR / "f1_cache"

# Ensure cache directory exists
if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

def augment_data():
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # New columns to initialize
    new_cols = ["Speed_FL", "RPM_FL", "Gear_FL", "Throttle_FL", "DRS_FL", "tire_age_legit", "compound_legit"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    # Curated Rich Sample for 2024 (Marquee Races)
    targets = [
        "2024_Sakhir", "2024_Jeddah", "2024_Melbourne", "2024_Suzuka",
        "2024_Shanghai", "2024_Miami", "2024_Monaco", "2024_Montreal",
        "2024_Silverstone", "2024_Zandvoort", "2024_Monza", "2024_Baku", "2024_Singapore", "2024_Austin", "2024_Mexico", "2024_São_Paulo", "2024_Las_Vegas", "2024_Qatar", "2024_Abu_Dhabi"
    ]
    
    available_races = df['race_id'].unique()
    matches = [r for r in targets if r in available_races]
    print(f"Targeting augmentation for {len(matches)} marquee 2024 races: {matches}")
    
    for race_key in matches:
        try:
            year, location = race_key.split('_', 1)
            year = int(year)
            location = location.replace('_', ' ')
            
            print(f"\nProcessing {race_key} ({year} {location})...")
            session = fastf1.get_session(year, location, 'R')
            session.load(telemetry=True, laps=True, weather=False)
            
            # Create a lookup for laps
            # We want to match (Driver, LapNumber) -> Telemetry at End of Lap
            
            # Iterate through drivers in the session
            for driver in session.drivers:
                driver_laps = session.laps.pick_driver(driver)
                
                if driver_laps.empty:
                    continue
                    
                # We can iterate laps
                for idx, lap in driver_laps.iterlaps():
                    lap_no = int(lap['LapNumber'])
                    
                    # fastf1 telemetry is usually available
                    # We want values at the 'end' of the lap (approx).
                    # 'telemetry' slice gives us the lap's data.
                    try:
                        tel = lap.get_telemetry()
                        if tel.empty:
                            continue
                            
                        # Take the last row (Finish Line / End of Lap)
                        last_row = tel.iloc[-1]
                        
                        speed = last_row.get('Speed', None)
                        rpm = last_row.get('RPM', None)
                        gear = last_row.get('nGear', None)
                        throttle = last_row.get('Throttle', None)
                        drs = last_row.get('DRS', None)
                        
                        # Added Legit Detail extraction
                        tyre_life = lap.get('TyreLife', None)
                        compound = lap.get('Compound', None)
                        
                        # Update the main dataframe
                        # Match on race_id, Driver, lapno_prev (which is "completed lap" usually)
                        # The dataset strategy is usually predicting for NEXT lap, based on 'lapno_prev'.
                        # If 'lapno_prev' is 5, it means we just finished lap 5.
                        # So we want telemetry from Lap 5.
                        
                        mask = (
                            (df['race_id'] == race_key) & 
                            (df['Driver'] == driver) & 
                            (df['lapno_prev'] == lap_no)
                        )
                        
                        if mask.any():
                            df.loc[mask, 'Speed_FL'] = speed
                            df.loc[mask, 'RPM_FL'] = rpm
                            df.loc[mask, 'Gear_FL'] = gear
                            df.loc[mask, 'Throttle_FL'] = throttle
                            df.loc[mask, 'DRS_FL'] = 10 if (drs_val := drs) and drs_val in [10, 12, 14, 8] else 0 
                            df.loc[mask, 'tire_age_legit'] = tyre_life
                            df.loc[mask, 'compound_legit'] = compound 
                            # DRS in fastf1 is often encoded. 10/12/14 often means open. 
                            # Or check if last_row['DRS'] > 8? fastf1 docs vary, but usually > 8 is enabled.
                            # Usually 0=Off, 1=Off, 8=Detected, 10-14=Open
                            
                    except Exception:
                        # row specific error, skip
                        pass
                        
        except Exception as ex:
            print(f"Failed to process {race_key}: {ex}")
            continue
        
        # Incremental Save after each race
        print(f"\n[INTERNAL] Incremental save after {race_key}...")
        df.to_csv(OUTPUT_FILE.with_suffix('.tmp'), index=False)

    # Filter to 2024 only for a "rich" focused demo
    print("\nFiltering dataset to 2024 season only...")
    # Check if tmp exists, else use df
    if OUTPUT_FILE.with_suffix('.tmp').exists():
        df = pd.read_csv(OUTPUT_FILE.with_suffix('.tmp'), low_memory=False)
    
    df_2024 = df[df['race_id'].astype(str).str.startswith('2024')].copy()
    
    print(f"Saving augmented 2024 dataset to {OUTPUT_FILE} ({len(df_2024)} rows)...")
    df_2024.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    augment_data()
