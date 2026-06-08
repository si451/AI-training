import json
with open('analysis/76.5k/summary.json', 'r') as f:
    d = json.load(f)

for k, v in d['activation_stats'].items():
    if 'alpha' in k or 'beta' in k:
        if '8' in k or '9' in k:
            print(f"{k}: Mean={v.get('mean', 0):.4f}")
