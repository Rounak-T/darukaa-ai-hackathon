from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "causal_graph.json"


@lru_cache(maxsize=1)
def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)
    g = nx.DiGraph()
    for node in spec["nodes"]:
        g.add_node(node["id"], label=node["label"], domain=node["domain"])
    for e in spec["edges"]:
        g.add_edge(e["source"], e["target"], sign=e["sign"], weight=e["weight"], mechanism=e["mechanism"])
    return g


def _path_strength(g: nx.DiGraph, path: List[str]) -> Tuple[float, str]:
    """Multiply edge weights for overall strength; multiply signs for net direction."""
    strength = 1.0
    negatives = 0
    for a, b in zip(path, path[1:]):
        data = g[a][b]
        strength *= data["weight"]
        if data["sign"] == "-":
            negatives += 1
    net_sign = "+" if negatives % 2 == 0 else "-"
    return strength, net_sign


def trace_pathways(
    start_nodes: List[str],
    target_nodes: List[str] | None = None,
    max_hops: int = 3,
    top_n: int = 6,
) -> List[Dict[str, Any]]:
    """Find the strongest causal chains from a user's situation to biodiversity outcomes."""
    g = load_graph()
    targets = target_nodes or ["species_richness", "pollinator_abundance", "habitat_connectivity"]
    results: List[Dict[str, Any]] = []

    for s in start_nodes:
        if s not in g:
            continue
        for t in targets:
            if t not in g or s == t:
                continue
            try:
                paths = nx.all_simple_paths(g, s, t, cutoff=max_hops)
            except nx.NetworkXNoPath:
                continue
            for path in paths:
                if len(path) < 3:  # require at least one intermediate variable
                    continue
                strength, sign = _path_strength(g, path)
                hops = [
                    {"source": g.nodes[a]["label"], "target": g.nodes[b]["label"],
                     "sign": g[a][b]["sign"], "mechanism": g[a][b]["mechanism"]}
                    for a, b in zip(path, path[1:])
                ]
                results.append({
                    "path": path,
                    "labels": [g.nodes[n]["label"] for n in path],
                    "hops": hops,
                    "net_sign": sign,
                    "strength": round(strength, 4),
                    "n_hops": len(path) - 1,
                    "domains_crossed": len({g.nodes[n]["domain"] for n in path}),
                })

    results.sort(key=lambda r: -(r["strength"] * (1 + 0.25 * r["domains_crossed"])))
    return results[:top_n]


def describe_pathways(pathways: List[Dict[str, Any]]) -> str:
    """Flatten pathways into plain text the LLM can read and reason over."""
    lines = []
    for p in pathways:
        arrow = " -> ".join(p["labels"])
        lines.append(f"[{p['net_sign']}] {arrow} (strength={p['strength']}, hops={p['n_hops']})")
        for h in p["hops"]:
            lines.append(f"    - {h['source']} {h['sign']} {h['target']}: {h['mechanism']}")
    return "\n".join(lines)