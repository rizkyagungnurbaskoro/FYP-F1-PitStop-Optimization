import sys
import traceback
sys.path.insert(0, 'app')

try:
    from app.demo import run_demo_state
    print("Testing run_demo_state...")
    state = run_demo_state("my", {"dataset": "my"})
    print("SUCCESS!")
    opts = state.get("options", {})
    print(f"Drivers: {opts.get('drivers', [])[:5]}")
    print(f"Circuits: {opts.get('circuits', [])[:5]}")
    print(f"Years: {opts.get('years', [])}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
