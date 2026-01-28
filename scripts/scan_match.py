import pandas as pd
from pitwall_api.app.demo import load_dataset

def scan_for_match():
    print("Loading dataset...")
    df = load_dataset('my')
    race_key = "2022_Monaco"
    
    # Target Values from Screenshot
    # LEC: 1:45.449, TireAge ~3
    # SAI: 1:45.486, TireAge ~20
    
    print("\n--- Scanning LEC for 1:45.449 ---")
    lec = df[(df['race_id'] == race_key) & (df['Driver'] == "LEC")]
    # Check columns like 'lap_time_prev' or similar? dataset has 'lap_time'?
    # Usually dataset has 'lapno_prev' and potentially 'lap_time' (if available, often not in strategy dataset directly as 'lap_time' but maybe available)
    # Let's inspect columns first
    print("Columns:", lec.columns.tolist())
    
    # If lap_time isn't there, we can't match on it easily. 
    # But we can match on tire_age_legit (or tire_age) and lap number.
    
    # Check tire age around lap 20-22
    surrounding = lec[(lec['lapno_prev'] >= 18) & (lec['lapno_prev'] <= 24)]
    cols = ['lapno_prev', 'tire_age_legit', 'compound_legit', 'LapTime_valid', 'TrackTemp_valid']
    
    print(surrounding[cols])
    
    print("\n--- Scanning SAI for 1:45.486 ---")
    sai = df[(df['race_id'] == race_key) & (df['Driver'] == "SAI")]
    surrounding_sai = sai[(sai['lapno_prev'] >= 18) & (sai['lapno_prev'] <= 24)]
    print(surrounding_sai[cols])

if __name__ == "__main__":
    scan_for_match()
