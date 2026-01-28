import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.offsetbox import AnchoredText
import warnings
import sys

# Suppress minor warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
SEASON = 2024
CACHE_DIR = "d:\\University\\FYP B\\FYP_FINAL\\f1_cache"
OUTPUT_PATH = "d:\\University\\FYP B\\FYP_FINAL\\visualization\\pit_stop_distribution_2024.png"

print(f"FastF1 Version: {fastf1.__version__}")

# Style constants matching the reference and F1 theme
BG_COLOR = "#0B0F14"  # Dark background
FG_COLOR = "#F0F0F0"  # Off-white text
ACCENT_COLOR = "#E10600" # F1 Red
GRID_COLOR = "#333333"

# Team Colors (Approximate 2024) - FastF1 usually handles this, but forcing them ensures consistency
TEAM_COLORS = {
    "Red Bull Racing": "#0600EF",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Mercedes": "#00A19B", # PETRONAS Cyan
    "Aston Martin": "#225941",
    "RB": "#6692FF", # VCARB Blue
    "Haas F1 Team": "#B6BABD",
    "Williams": "#64C4FF",
    "Alpine": "#0093CC",
    "Kick Sauber": "#52E252", # Neon Green
}

def setup_style():
    """Configures Matplotlib for a dark, premium F1 look."""
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Roboto', 'DejaVu Sans', 'Arial']
    
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = BG_COLOR
    plt.rcParams['axes.facecolor'] = BG_COLOR
    plt.rcParams['axes.edgecolor'] = BG_COLOR
    plt.rcParams['axes.labelcolor'] = FG_COLOR
    plt.rcParams['xtick.color'] = FG_COLOR
    plt.rcParams['ytick.color'] = FG_COLOR
    plt.rcParams['text.color'] = FG_COLOR
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = GRID_COLOR
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'

def get_pit_stop_data(season):
    """Fetches pit stop data for the entire season."""
    print(f"Fetching schedule for {season}...")
    schedule = fastf1.get_event_schedule(season)
    all_pit_stops = []

    # Use cache if available
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
    except Exception as e:
        print(f"Warning: Could not enable cache at {CACHE_DIR}: {e}")

    # Iterate through completed races
    # We'll limit to actual races (excluding testing) and verify they have happened
    # We filter specifically for conventional races to avoid sprint complications if necessary,
    # but 2024 had sprints. Sprints don't mandate pit stops usually, so 'R' session is key.
    # Note: EventFormat might differ. 'testing' vs 'conventional' vs 'sprint'.
    # We want valid races.
    
    # Ensure dates are datetime and UTC
    schedule['Session5Date'] = pd.to_datetime(schedule['Session5Date'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    
    valid_events = schedule[
        (schedule['EventFormat'].isin(['conventional', 'sprint'])) & 
        (schedule['Session5Date'] < now) 
    ]

    
    print(f"Processing {len(valid_events)} completed events...")

    for idx, (original_idx, row) in enumerate(valid_events.iterrows()):
        round_num = row['RoundNumber']
        event_name = row['EventName']
        print(f"  Loading Round {round_num}: {event_name}")
        
        try:
            # Load the Race session
            session = fastf1.get_session(season, round_num, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            

            # Debug extraction
            # print(f"  [DEBUG] Processing Event {idx}: {event_name}")
            
            if hasattr(session, 'pit_stops') and session.pit_stops is not None and not session.pit_stops.empty:
                stops = session.pit_stops.copy()
                 # Ensure we have teams
                if 'Team' not in stops.columns:
                     # Create mapping from laps if needed (usually robust)
                     driver_team_map = session.laps[['DriverNumber', 'Team']].drop_duplicates().set_index('DriverNumber')['Team'].to_dict()
                     if 'DriverNumber' not in stops.columns and 'Driver' in stops.columns:
                        stops = stops.rename(columns={'Driver': 'DriverNumber'})
                     stops['Team'] = stops['DriverNumber'].map(driver_team_map)
                
            else:
                 # Fallback: Calculate from Laps (Total Pit Lane Time)
                 # Shift logic: PitIn is on Lap N, PitOut is on Lap N+1 (typically)
                 
                 # 1. Sort by Driver and Lap
                 laps = session.laps.sort_values(by=['DriverNumber', 'LapNumber'])
                 
                 # 2. Shift PitOutTime back by 1 (Next lap's PitOut matches Current lap's PitIn)
                 laps['NextPitOut'] = laps.groupby('DriverNumber')['PitOutTime'].shift(-1)
                 
                 # 3. Create Mask
                 mask = laps['PitInTime'].notna() & laps['NextPitOut'].notna()
                 pit_laps = laps[mask].copy()
                 
                 if pit_laps.empty:
                     continue
                     
                 # 4. Calculate
                 pit_laps['Duration'] = pit_laps['NextPitOut'] - pit_laps['PitInTime']
                 
                 # Rename cols to match expectation
                 stops = pit_laps[['Team', 'Duration', 'DriverNumber']].copy()

            # Drop stops where Team is NaN
            stops = stops.dropna(subset=['Team'])
            all_pit_stops.append(stops)
            
            # if not stops.empty:
            #    print(f"  [DEBUG] Appended {len(stops)} stops.")

            sys.stdout.flush()

        except Exception as e:
            print(f"  Skipping {event_name} due to error: {e}")
            sys.stdout.flush()
            continue

    if not all_pit_stops:
        print("No pit stop data found.")
        return pd.DataFrame()

    total_df = pd.concat(all_pit_stops, ignore_index=True)
    return total_df

def create_visualization(df):
    if df.empty:
        print("Empty dataframe, skipping plot.")
        return

    print("Processing visualization data...")
    # Convert duration to float (seconds)
    if 'Duration' not in df.columns:
        print("Error: 'Duration' column missing.")
        return
        
    # Convert duration: '23.456' (str) or timedelta
    # It seems FastF1 returns float seconds or timedelta depending on version/source.
    # We force conversion.
    if pd.api.types.is_timedelta64_dtype(df['Duration']):
        df['Seconds'] = df['Duration'].dt.total_seconds()
    else:
        df['Seconds'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Filter valid stops
    # Pit Lane Time is usually 18s - 35s.
    # We'll filter < 15s (error/virtual?) and > 60s (garage)
    # Adjust as needed.
    df_clean = df[(df['Seconds'] <= 60) & (df['Seconds'] >= 10)].copy()
    
    print(f"Plotting {len(df_clean)} stops from {len(df)} total records.")
    
    # Sort teams by MEAN stop time (fastest on top, to match displayed values)
    team_stats = df_clean.groupby('Team')['Seconds'].agg(['mean', 'median', 'std', 'count'])
    team_order = team_stats.sort_values('mean').index.tolist()
    
    # Setup Figure using GridSpec for layout control if needed, but standard is fine
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Violin Plot
    # inner=None to draw our own elements
    violins = sns.violinplot(
        data=df_clean,
        x="Seconds",
        y="Team",
        order=team_order,
        palette=TEAM_COLORS,
        inner=None,
        linewidth=0,
        scale="width",
        cut=0,
        ax=ax,
        saturation=0.8
    )
    
    # Customize Violin Alpha manually?
    # sns doesn't support alpha directly in args easily for the polycollections.
    for art in ax.collections:
        if isinstance(art, matplotlib.collections.PolyCollection):
            art.set_alpha(0.6)

    # Add Swarmplot (the dots) - cleaner distribution
    # Swarmplot prevents overlap
    sns.swarmplot(
        data=df_clean,
        x="Seconds",
        y="Team",
        order=team_order,
        color="white",
        size=2.5,
        alpha=0.6,
        ax=ax,
        edgecolor=BG_COLOR,
        linewidth=0.5,
        warn_thresh=0.5 # suppress warnings about points not fitting
    )
    
    # Add Inter-Quartile Range (IQR) Bars inside the violins?
    # Or just mean diamonds as requested.
    
    # Add Mean Markers
    means = team_stats.loc[team_order, 'mean']
    
    # We plot scatter points for means
    # y-coordinates are 0, 1, 2... corresponding to team_order
    y_coords = np.arange(len(team_order))
    ax.scatter(
        x=means,
        y=y_coords,
        marker='D',
        color='white',
        edgecolor='black',
        s=50,
        zorder=20,
        label='Mean Time'
    )
    
    # --- Decoration ---
    
    # Title & Subtitle
    # Title & Subtitle
    fig.text(0.13, 0.95, f"{SEASON} F1 Season - Pit Lane Times", fontsize=24, fontweight='bold', color='white', ha='left')
    fig.text(0.13, 0.92, "Total Pit Time (Lane In to Lane Out) | Rounds 1-24", fontsize=14, color=FG_COLOR, alpha=0.7, ha='left')

    ax.set_xlabel("Time (s)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("", fontsize=12) # No Y label needed, team names are enough
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Grid
    ax.grid(visible=True, axis='x', color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    
    # X Limits - focus on the dense area
    # Pit Lane times usually 20-25s.
    ax.set_xlim(16, 35) 
    
    # --- Right Side Stats Column ---
    # We place text on the right side of the graph
    
    # Get current axis limits
    x_min, x_max = ax.get_xlim()
    
    # Position for the stats box
    stats_x_pos = x_max + 0.1 # slightly outside
    
    # Header for stats
    # We can use a trick: annotate on the plot, but using axis fraction for X?
    # No, keep data coordinates for Y, but expand X limit?
    # Let's just draw them at x_max + small_offset and expand xlim later if needed.
    # Actually, simpler to place them at a fixed x-coordinate.
    
    for i, team in enumerate(team_order):
        mean_val = team_stats.loc[team, 'mean']
        color = TEAM_COLORS.get(team, '#555555')
        
        # Label string
        lbl = f"{mean_val:.2f}"
        
        # Add text
        # We align it to the right y-tick
        ax.text(
            x_max + 0.05, i, 
            lbl, 
            ha='left', va='center', 
            fontsize=12, fontweight='bold', color='white',
            bbox=dict(facecolor=color, edgecolor='none', boxstyle='square,pad=0.4')
        )

    # Add column header
    ax.text(x_max + 0.05, -1.0, "Mean", ha='left', va='center', fontsize=11, fontweight='bold', color=FG_COLOR)
    
    # Logo / Footer
    fig.text(0.9, 0.02, "Generated with FastF1", ha='right', fontsize=10, color=FG_COLOR, alpha=0.5)

    # Adjust layout
    plt.tight_layout()
    # Add extra margin on right for the stats, and top for title
    plt.subplots_adjust(right=0.88, left=0.15, top=0.88)
    
    print(f"Saving to {OUTPUT_PATH}...")
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=BG_COLOR, edgecolor='none')
    print("Done.")

def main():
    setup_style()
    df = get_pit_stop_data(SEASON)
    if not df.empty:
        create_visualization(df)
    else:
        print("No data gathered.")

if __name__ == "__main__":
    main()
