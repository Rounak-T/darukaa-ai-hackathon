import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.graph_reasoner import trace_pathways, describe_pathways

# simulate a user whose land has low soil carbon and is fragmented
pathways = trace_pathways(start_nodes=["soil_organic_carbon", "fragmentation"])

print(describe_pathways(pathways))