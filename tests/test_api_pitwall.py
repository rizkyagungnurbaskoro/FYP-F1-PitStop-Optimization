import requests

try:
    print("=== STANDARD (80/20) SPLIT ===")
    r1 = requests.get('http://127.0.0.1:8001/metrics/summary?mode=standard')
    data1 = r1.json()
    for i, row in enumerate(data1['rows']):
        print(f'S{i+1}: {row["stage"][:35]:35s} | F1={row["mean_f1"]:.4f}')
    
    print("\n=== STRICT (70/30) SPLIT ===")
    r2 = requests.get('http://127.0.0.1:8001/metrics/summary?mode=strict')
    data2 = r2.json()
    for i, row in enumerate(data2['rows']):
        print(f'S{i+1}: {row["stage"][:35]:35s} | F1={row["mean_f1"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
