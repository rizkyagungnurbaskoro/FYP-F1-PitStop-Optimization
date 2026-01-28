import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path
import sys

# Force non-interactive backend
matplotlib.use('Agg')

# Load project roots
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from experiments.exp_utils import load_csv, add_feature_engineering, safe_numeric
from experiments.exp_config import get_paths

def generate_premium_shap():
    print("🚀 Starting Premium SHAP Generation...")
    paths = get_paths()
    
    # 1. Load and Prepare small subset
    print("📊 Loading data subset...")
    df = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))
    
    # Feature list matching the model's core
    features = [
        "lapno", "race_progress", "pitstops_so_far", "position",
        "gap", "interval", "tireage", "sc_active", "vsc_active"
    ]
    # Add weather if present
    weather_feats = ["AirTemp_prev", "TrackTemp_prev", "Rainfall_prev"]
    for wf in weather_feats:
        if wf in df.columns:
            features.append(wf)
            
    features = [f for f in features if f in df.columns]
    
    # Prune data for speed (SHAP only needs enough to show the trend)
    sample_size = 800
    df_mini = df.sample(min(len(df), sample_size * 10), random_state=42)
    X = df_mini[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df_mini["decide_pitstop"].astype(int).values
    
    # 2. Fast Train optimized for SHAP
    print("⚙️ Training optimized proxy model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=1,
        tree_method='hist',
        random_state=42
    )
    model.fit(X, y)
    
    # 3. Calculate SHAP
    print("🔮 Calculating SHAP values (subset)...")
    X_shap = X.sample(min(len(X), sample_size), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    
    # Handle SHAP output format inconsistencies
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]
    
    # 4. Fancy Plotting
    print("🎨 Formatting 'Best' Visualization...")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # Core F1 Colors: Red (#FF1E00) and White/Cyan
    # We use a custom cmap for the dots
    f1_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#00D2BE", "#FF1E00"]) # Mercedes Cyan to Ferrari Red
    
    # Feature Renaming for Professional Layout
    rename_dict = {
        "lapno": "Current Lap",
        "race_progress": "Race Progress (%)",
        "pitstops_so_far": "Pit Stops Completed",
        "position": "Track Position",
        "gap": "Gap to Leader (s)",
        "interval": "Gap to Front (s)",
        "tireage": "Tyre Age (Laps)",
        "sc_active": "Safety Car Active",
        "vsc_active": "VSC Active",
        "Rainfall_prev": "Rain Intensity",
        "TrackTemp_prev": "Track Temp (°C)"
    }
    X_shap_renamed = X_shap.rename(columns=rename_dict)
    
    shap.summary_plot(
        shap_vals, 
        X_shap_renamed, 
        show=False, 
        plot_type="dot",
        cmap=f1_cmap,
        alpha=0.6,
        plot_size=None # Use existing ax
    )
    
    plt.title("F1 PIT STRATEGY: AI DECISION DRIVERS (SHAP)", fontsize=18, fontweight='bold', pad=30, color='white')
    plt.xlabel("Impact on 'BOX' probability (SHAP Value)", fontsize=14, color='white', labelpad=15)
    
    # Aesthetic tweaks for high visibility
    ax = plt.gca()
    
    # Force Y-axis labels (Feature names) to white and larger
    ax.tick_params(axis='y', colors='white', labelsize=12)
    # Force X-axis labels (SHAP values) to white and larger
    ax.tick_params(axis='x', colors='white', labelsize=12)
    
    # Set axis label colors explicitly
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.grid(axis='x', linestyle='--', alpha=0.2, color='white')
    
    # 5. Save
    out_dir = paths.results_dir / "f1_theme_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_premium_defense.png"
    
    plt.savefig(out_path, bbox_inches='tight', facecolor='#0D0D0D')
    plt.close()
    
    print(f"✅ Success! Premium graph saved to: {out_path}")
    print(f"Summary: Used {len(X_shap)} samples to highlight {len(features)} strategy variables.")

if __name__ == "__main__":
    generate_premium_shap()
