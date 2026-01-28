import pandas as pd
import fastf1
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "personal_datasets"
INPUT_FILE = DATA_DIR / "fastf1_strategy_dataset.csv"
OUTPUT_FILE = DATA_DIR / "fastf1_demo_dataset.csv" # Overwrite same file for now as test
CACHE_DIR = BASE_DIR / "f1_cache"

if not CACHE_DIR.exists():
    CACHE_DIR.mkdir(parents=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

def augment_silverstone():
    print(f"Loading dataset...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Initialize cols
    new_cols = ["Speed_FL", "RPM_FL", "Gear_FL", "Throttle_FL", "DRS_FL", "tire_age_legit", "compound_legit"]
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    race_key = "2024_Silverstone"
    print(f"Processing {race_key} ONLY...")
    
    try:
        session = fastf1.get_session(2024, "Silverstone", 'R')
        session.load(telemetry=True, laps=True, weather=False)

        for driver in session.drivers:
            # FastF1 returns numbers as strings '44', '1', etc. Need Abbreviation 'HAM', 'VER'.
            drv_info = session.get_driver(driver)
            driver_code = drv_info['Abbreviation']
            
            driver_laps = session.laps.pick_driver(driver)
            if driver_laps.empty: continue
            
            for idx, lap in driver_laps.iterlaps():
                lap_no = int(lap['LapNumber'])
                
                try:
                    # Telemetry
                    tel = lap.get_telemetry()
                    if not tel.empty:
                        last_row = tel.iloc[-1]
                        speed = last_row.get('Speed', None)
                        rpm = last_row.get('RPM', None)
                        gear = last_row.get('nGear', None)
                        throttle = last_row.get('Throttle', None)
                        drs = last_row.get('DRS', None)
                    else:
                        speed = rpm = gear = throttle = drs = None

                    # Legit Detail
                    tyre_life = lap.get('TyreLife', None)
                    compound = lap.get('Compound', None)
                    
                    dataset_laps = df[(df['race_id'] == race_key) & (df['Driver'] == driver_code)]['lapno_prev'].dropna().astype(int).unique()

                    # Match Logic: Try EXACT match first, then OFFSET match
                    # Hypothesis: Dataset 'lapno_prev' = Completed Lap. FastF1 'LapNumber' = Current Lap.
                    # If we want data for the end of the lap we just finished:
                    # LapNumber 5 (Finish Line) matches lapno_prev 5.
                    
                    matched = False
                    
                    # Attempt 1: Exact
                    mask = (
                        (df['race_id'] == race_key) & 
                        (df['Driver'] == driver_code) & 
                        (df['lapno_prev'].fillna(-99).astype(int) == lap_no)
                    )
                    
                    if not mask.any():
                        # Attempt 2: Offset (Maybe lapno_prev is 0-indexed?)
                         mask = (
                            (df['race_id'] == race_key) & 
                            (df['Driver'] == driver_code) & 
                            (df['lapno_prev'].fillna(-99).astype(int) == lap_no - 1)
                        )

                    if mask.any():
                        matched = True
                        # if idx < 5: print(f"MATCH: {driver_code} Lap {lap_no}")
                        df.loc[mask, 'Speed_FL'] = speed
                        df.loc[mask, 'RPM_FL'] = rpm
                        df.loc[mask, 'Gear_FL'] = gear
                        df.loc[mask, 'Throttle_FL'] = throttle
                        df.loc[mask, 'DRS_FL'] = 10 if (drs_val := drs) and drs_val in [10, 12, 14, 8] else 0 
                        df.loc[mask, 'tire_age_legit'] = tyre_life
                        df.loc[mask, 'compound_legit'] = compound
                        
                except Exception:
                    pass
    except Exception as ex:
        print(f"Failed: {ex}")

    # Save just 2024
    df_2024 = df[df['race_id'].astype(str).str.startswith('2024')].copy()
    print(f"Saving {len(df_2024)} rows to {OUTPUT_FILE}")
    df_2024.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    augment_silverstone()
