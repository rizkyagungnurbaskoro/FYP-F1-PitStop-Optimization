import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.demo import run_demo_state

try:
    print("Testing run_demo_state with dataset='my'...")
    state = run_demo_state("my", {"dataset": "my"})
    print("SUCCESS!")
    print(f"Drivers: {state.get('options', {}).get('drivers', [])[:5]}")
except Exception as e:

    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
