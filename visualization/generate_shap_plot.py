import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from pathlib import Path

# Load project utilities
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from experiments.exp_utils import load_csv, add_feature_engineering, safe_numeric
from experiments.exp_config import get_paths

def generate_shap_summary():
    paths = get_paths()
    
    # 1. Load Data
    print("Loading data...")
    df = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))
    
    # 2. Parameters from results
    # Using holdout best params (Stage 4)
    # results/holdout70_mydata_mymethod_weather/metrics.json
    results_path = paths.out_mydata_mymethod_weather_holdout / "metrics.json"
    with open(results_path, "r") as f:
        metrics = json.load(f)
    
    # Use the tuned params from the first (and only) fold of holdout
    params = metrics["folds"][0]["tuned_params"]
    # Add scale_pos_weight mapping logic from exp_models.py
    pos = int((df["decide_pitstop"] == 1).sum())
    neg = int((df["decide_pitstop"] == 0).sum())
    spw_multiplier = params.pop("scale_pos_weight_multiplier", 1.0)
    params["scale_pos_weight"] = (neg / pos) * spw_multiplier
    
    # 3. Features
    # Based on exp_run_all.py MYMETHOD_SHARED_FEATURES
    features = [
        "season", "lapno", "race_progress", "pitstops_so_far", "position",
        "gap", "interval", "sc_active", "vsc_active", "SCAny",
        "GapOverInterval", "tireage"
    ]
    
    # Filter features that exist in df
    features = [f for f in features if f in df.columns]
    print(f"Features used: {features}")
    
    # 4. Train Model (Simplified for SHAP)
    print("Training model...")
    # Clean data similar to exp_models.py (target encoding omitted for summary simplicity if possible, 
    # but the metrics.json says target_encoding=True. I'll include it if it's there.)
    
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["decide_pitstop"].astype(int).values
    
    # Subsample for speed (5000 rows for training is plenty for a SHAP demo)
    if len(X) > 5000:
        print("Subsampling to 5000 rows for training...")
        idx = np.random.RandomState(42).choice(len(X), 5000, replace=False)
        X_train = X.iloc[idx]
        y_train = y[idx]
    else:
        X_train = X
        y_train = y

    # Subsample for SHAP (200 samples is plenty for a quick demo)
    print("Subsampling to 200 rows for ultra-fast SHAP calculation...")
    X_sub = X_train.sample(min(200, len(X_train)), random_state=42)

    # Simplified params for speed
    params["n_estimators"] = 50
    params["max_depth"] = 3
    params["n_jobs"] = 1
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Model fitted!")
    
    # 5. Generate SHAP values
    print("Generating SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    print("SHAP values generated!")
    
    # Check shape of shap_values
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    
    # 6. Plot
    print("Plotting...")
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#00FFFF", "#FF0000"])
    
    # Handle both binary and multi-output SHAP returns
    # shap_values could be a list for binary models in some versions
    sv_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    shap.summary_plot(
        sv_to_plot, 
        X_sub, 
        show=False, 
        plot_size=(10, 6),
        color_bar_label="Feature Value (High=Red, Low=Cyan)",
        cmap=cmap
    )
    
    plt.title("F1 Pit Stop Strategy: SHAP Feature Importance", fontsize=16, pad=20, color='white')
    plt.tight_layout()
    
    # Save results
    output_path = paths.results_dir / "f1_theme_plots" / "shap_summary_optimized.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! SHAP plot saved to: {output_path}")

if __name__ == "__main__":
    generate_shap_summary()
