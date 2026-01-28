from pitwall_api.app.demo import load_dataset, _find_iconic_moments
from pitwall_api.app.config import MYDATA_PATH
import pandas as pd

print(f"Loading from {MYDATA_PATH.parent / 'fastf1_demo_dataset.csv'}")
df = load_dataset('my')
print(f"Loaded {len(df)} rows.")

mon_rows = len(df[df['race_id']=='2022_Monaco'])
print(f"Monaco 22 Rows: {mon_rows}")

moments = _find_iconic_moments(df, 'lapno_prev')
print("Found Moments:")
for m in moments:
    print(f"- {m['Driver']} @ {m['race_id']} | Lap {m['lapno_prev']} | LegitAge: {m.get('tire_age_legit')} | LegitComp: {m.get('compound_legit')}")
