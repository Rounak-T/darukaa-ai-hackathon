import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retriever import get_retriever

r = get_retriever()
results = r.search("my soil feels dead and dry, low rainfall, growing only wheat")

for c in results:
    print(f"[{c['id']}] score={c['retrieval_score']} via={c['retrieved_by']} - {c['title']}")