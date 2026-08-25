import json

nb = json.load(open('notebooks/signboard-detection.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    print(src[:500])
    print()
