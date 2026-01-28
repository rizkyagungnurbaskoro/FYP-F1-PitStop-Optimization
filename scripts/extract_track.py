import fastf1
import numpy as np
import os

def extract_monaco_path():
    # Setup cache (creates folder 'f1_cache' if not exists)
    cache_dir = os.path.join(os.getcwd(), "f1_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    fastf1.Cache.enable_cache(cache_dir)

    print("Loading Monaco 2022 Race Session...", flush=True)
    session = fastf1.get_session(2022, 'Monaco', 'R')
    session.load(telemetry=True, laps=True, weather=False)

    print("Picking fastest lap...", flush=True)
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    
    # Downsample to reduce size (take every 5th point)
    tel = tel.iloc[::5]

    x = tel['X'].values
    y = tel['Y'].values

    # Determine bounds
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    # Aspect ratio preservation
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # Target SVG box width
    width = 1000
    scale = width / max(range_x, range_y)
    height = range_y * scale

    # Normalize coordinates
    # SVG Y maps top-down. FastF1 is Cartesian (Y up).
    # So Y_svg = max_Y_scaled - (y - min_y)*scale
    # Or just invert: (max_y - y) * scale
    
    x_norm = (x - min_x) * scale
    # If using Cartesian Y (bottom-min, top-max), to flip to SVG (top-min, bottom-max):
    y_norm = (max_y - y) * scale

    # Construct Path 'M x0 y0 L x1 y1 ...'
    path_data = [f"M {x_norm[0]:.1f} {y_norm[0]:.1f}"]
    for xi, yi in zip(x_norm[1:], y_norm[1:]):
        path_data.append(f"L {xi:.1f} {yi:.1f}")
    
    # Close loop
    path_data.append("Z")
    path_str = " ".join(path_data)
    
    # Output to file
    with open("monaco_path.txt", "w") as f:
        f.write(path_str)
        
    print(f"Path extracted! Viewbox: 0 0 {width:.0f} {height:.0f}")

if __name__ == "__main__":
    extract_monaco_path()
