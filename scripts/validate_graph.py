import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "causal_graph.json"

with open(GRAPH_PATH, "r", encoding="utf-8") as f:
    graph = json.load(f)

errors = []
node_ids = {n["id"] for n in graph["nodes"]}

print(f"Loaded {len(graph['nodes'])} nodes, {len(graph['edges'])} edges.")

if not isinstance(graph["edges"], list) or (graph["edges"] and not isinstance(graph["edges"][0], dict)):
    errors.append("edges is not a flat list of edge objects - check for extra nesting brackets")
else:
    for i, e in enumerate(graph["edges"]):
        if e.get("source") not in node_ids:
            errors.append(f"edge {i}: unknown source node '{e.get('source')}'")
        if e.get("target") not in node_ids:
            errors.append(f"edge {i}: unknown target node '{e.get('target')}'")
        if e.get("sign") not in ("+", "-"):
            errors.append(f"edge {i}: invalid sign '{e.get('sign')}'")
        if not (0 < e.get("weight", 0) <= 1):
            errors.append(f"edge {i}: weight out of range")
        if not e.get("mechanism"):
            errors.append(f"edge {i}: missing mechanism")

if errors:
    print(f"\nValidation FAILED with {len(errors)} issue(s):")
    for e in errors:
        print(" -", e)
else:
    print("\nCausal graph is valid.")