import json, sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else 'signboard-detection.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    ct = c['cell_type']
    print(f"=== Cell {i} ({ct}) ===")
    print(src[:600])
    print()
