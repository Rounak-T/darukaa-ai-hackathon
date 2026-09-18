from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.recommender import advise
from core.retriever import get_retriever
from core.schemas import AdvisoryResponse, SiteProfile
from core.slots import extract_profile
from core.verifier import verify

app = FastAPI(
    title="Darukaa Biodiversity Intelligence API",
    description="Evidence-grounded, multi-variable biodiversity advisory engine.",
    version="1.0.0",
)


class TextQuery(BaseModel):
    message: str
    profile: SiteProfile | None = None


class AnalyseRequest(BaseModel):
    profile: SiteProfile


@app.get("/health")
def health():
    return {"status": "ok", "knowledge_cards": len(get_retriever().cards)}


@app.post("/analyse", response_model=AdvisoryResponse)
def analyse(req: AnalyseRequest):
    """Structured entry point: submit a full site profile, get recommendations."""
    try:
        return verify(advise(req.profile, force_answer=True))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat", response_model=AdvisoryResponse)
def chat(req: TextQuery):
    """Conversational entry point: free text in, slots extracted automatically."""
    try:
        profile = extract_profile(req.message, req.profile)
        return verify(advise(profile))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))