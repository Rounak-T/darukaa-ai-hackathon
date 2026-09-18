from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "data" / "knowledge_cards.json"
INDEX_DIR = ROOT / ".chroma"
COLLECTION_NAME = "biodiversity_kb"

TOP_K_DENSE = 6
TOP_K_SPARSE = 6
TOP_K_FINAL = 5
RRF_K = 60  # standard constant for reciprocal rank fusion


def load_cards() -> List[Dict[str, Any]]:
    with open(CARDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def card_to_text(card: Dict[str, Any]) -> str:
    """The string that gets embedded/searched - mechanism carries most signal."""
    return (
        f"{card['title']}. Domain: {card['domain']}. "
        f"Intervention: {card['intervention']}. "
        f"Mechanism: {card['mechanism']}. "
        f"Effect: {card['quantified_effect']}. "
        f"Metrics: {', '.join(card['metrics_improved'])}. "
        f"Applies when: {', '.join(card['preconditions'])}."
    )


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class HybridRetriever:
    def __init__(self) -> None:
        self.cards = load_cards()
        self.texts = [card_to_text(c) for c in self.cards]
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        self._chroma = None
        self._matrix = None
        self._init_dense()

    def _init_dense(self) -> None:
        model = _embedder()
        embeddings = model.encode(self.texts, normalize_embeddings=True)
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(INDEX_DIR))
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            col.add(
                ids=[c["id"] for c in self.cards],
                documents=self.texts,
                embeddings=[e.tolist() for e in embeddings],
            )
            self._chroma = col
        except Exception as exc:
            print(f"[retriever] Chroma unavailable ({exc}), using numpy fallback.")
            self._matrix = np.asarray(embeddings, dtype="float32")

    def _dense_ranked_ids(self, query: str, k: int) -> List[str]:
        qvec = _embedder().encode([query], normalize_embeddings=True)[0]
        if self._chroma is not None:
            res = self._chroma.query(query_embeddings=[qvec.tolist()], n_results=k)
            return res["ids"][0]
        sims = self._matrix @ qvec
        top = np.argsort(-sims)[:k]
        return [self.cards[i]["id"] for i in top]

    def _sparse_ranked_ids(self, query: str, k: int) -> List[str]:
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(-scores)[:k]
        return [self.cards[i]["id"] for i in top]

    def search(self, query: str, top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
        dense = self._dense_ranked_ids(query, TOP_K_DENSE)
        sparse = self._sparse_ranked_ids(query, TOP_K_SPARSE)

        # Reciprocal Rank Fusion: combine two rankings without needing to
        # normalise their scores onto the same scale.
        fused: Dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        by_id = {c["id"]: c for c in self.cards}
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        out = []
        for cid, score in ordered:
            card = dict(by_id[cid])
            card["retrieval_score"] = round(score, 5)
            card["retrieved_by"] = "both" if cid in dense and cid in sparse else ("dense" if cid in dense else "sparse")
            out.append(card)
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()