import pandas as pd
import sys
import os

# Add the project root to sys.path to import local modules
sys.path.append(os.getcwd())

from pitwall_api.app.demo import _derive_weather_label, _lap_range
from pitwall_api.app.model import detect_crossover_state

data_path = 'd:/University/FYP B/FYP_FINAL/personal_datasets/fastf1_demo_dataset.csv'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    sys.exit(1)

df = pd.read_csv(data_path)
sub = df[df['race_id'] == '2024_Montréal'].copy()
sub['weather_label'] = _derive_weather_label(sub)

wet = sub[sub['weather_label'] == 'Wet']
dry = sub[sub['weather_label'] == 'Dry']

print(f"Montréal Total Laps: {_lap_range(sub, 'lapno_prev')}")
print(f"Montréal Wet Laps:   {_lap_range(wet, 'lapno_prev')}")
print(f"Montréal Dry Laps:   {_lap_range(dry, 'lapno_prev')}")

# Test crossover detection on a drying row if possible
# Usually, Zandvoort 2024 had rain at the start
if not dry.empty:
    sample_dry = dry.iloc[0]
    print(f"Crossover Check (L{sample_dry['lapno_prev']}): {detect_crossover_state(sample_dry)}")
