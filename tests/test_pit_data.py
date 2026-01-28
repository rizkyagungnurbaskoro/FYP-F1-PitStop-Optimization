import fastf1
import pandas as pd

fastf1.Cache.enable_cache('d:\\University\\FYP B\\FYP_FINAL\\f1_cache')

print("Loading Bahrain 2024...")
session = fastf1.get_session(2024, 1, 'R')
session.load(laps=True, telemetry=False, weather=False, messages=False)

print(f"Laps: {len(session.laps)}")
print("Columns:", session.laps.columns.tolist())

# Check PitInTime
pit_in_notna = session.laps['PitInTime'].notna().sum()
pit_out_notna = session.laps['PitOutTime'].notna().sum()
print(f"PitInTime count: {pit_in_notna}")
print(f"PitOutTime count: {pit_out_notna}")

# Check calculated duration
# Check calculated duration with shift
print("calculating with shift...")
# Sort just in case
laps = session.laps.sort_values(by=['DriverNumber', 'LapNumber'])
laps['NextPitOut'] = laps.groupby('DriverNumber')['PitOutTime'].shift(-1)

mask = laps['PitInTime'].notna() & laps['NextPitOut'].notna()
print(f"Shifted match count: {mask.sum()}")

if mask.sum() > 0:
    durations = laps.loc[mask, 'NextPitOut'] - laps.loc[mask, 'PitInTime']
    print("Durations head:")
    print(durations.head())
    print("Durations described:")
    print(durations.dt.total_seconds().describe())
