# 🌿 Biodiversity Intelligence System

**An AI environmental advisor, not a chatbot.**

A conversational system that reasons about soil, land, climate, and biodiversity the way a researcher would: grounded in curated evidence, connected across multiple variables through causal reasoning, and independently fact-checked before any answer reaches the user.

**🔗 Live demo:** https://biodiversity-intelligence.streamlit.app/
---

## The problem with most "biodiversity AI" submissions

Ask a general-purpose LLM for biodiversity advice and you get: *"Improve soil health and use sustainable farming practices."* True, and useless — no mechanism, no numbers, no connection between variables, nothing a land manager can act on.

This system is built to make that failure mode structurally impossible. The LLM is never the source of a fact — it only composes from evidence that has already been retrieved, evaluated, and traced through a causal model. If the evidence doesn't support a claim, the claim doesn't ship.

---

## What makes this different from an LLM wrapper

| Layer | What it does | Why it matters |
|---|---|---|
| **Curated knowledge base** — 25 structured cards | Each card holds a real intervention, its scientific mechanism, a quantified effect, the metrics it improves, and a citation to FAO / IPCC / IPBES / peer-reviewed research | Statistics are written and checked by a human, never generated per-query — the LLM cannot hallucinate a number that isn't already in the evidence |
| **Hybrid retrieval** — dense + sparse fusion | ChromaDB semantic embeddings (`all-MiniLM-L6-v2`) combined with BM25 keyword matching via Reciprocal Rank Fusion | Dense search catches paraphrase ("my soil is dead"); BM25 catches exact technical terms ("pH 5.3", "sodic soil") that embeddings blur past. Neither alone is reliable |
| **Causal reasoning graph** — 15 nodes, 30 signed edges | A directed graph of environmental variables, built with `networkx`, traces multi-hop chains like *soil organic carbon → soil moisture retention → drought resilience → species richness* | This is the actual differentiator. Retrieval finds a relevant fact; the graph finds the **pathway** — the indirect, non-obvious effect the user never asked about. Every recommendation connects 3+ variables through a real mechanism, not a coincidence of matching keywords |
| **Self-verification** — independent audit pass | A second, separate LLM call checks each generated claim against its *cited* evidence card and flags anything unsupported, lowering its confidence instead of hiding the failure | Caught in real testing: a recommendation once borrowed a "reduces toxic contaminants" claim that didn't belong to its cited source — the verifier flagged it automatically |
| **Conversational memory + readiness gate** | Structured site profile persists across turns; the system refuses to advise until it has enough information, asking targeted clarifying questions instead | Directly satisfies "ask clarifying questions when inputs are incomplete" — a real consultant doesn't guess |

---

## See it reason, not just retrieve

Given conditions `low soil organic carbon, semi-arid, monoculture, tilled`, the causal graph surfaces chains like:
Soil organic carbon → Soil moisture retention → Drought resilience → Species richness
strength 0.57 | 3 hops | crosses soil → climate → biodiversity domains


This is not three retrieved facts stapled together — it's a traced path through signed, weighted causal edges, explaining *why* raising soil carbon eventually helps species survive dry spells, three steps removed from the obvious answer.

---

## Architecture

                User input (quick-entry form or free-text chat)
                                  │
                                  ▼
                Slot extraction (regex for numbers, keyword rules for
                categorical fields) → structured SiteProfile
                                  │
                                  ▼
             Readiness gate — enough info to answer responsibly?
                      │                        │
                     No                        Yes
                      │                        │
             Clarifying questions      Condition tags derived
                returned                       │
                                ┌───────────────┴────────────────┐
                                ▼                                ▼
                    Hybrid retrieval                  Causal pathway tracing
                    (Chroma dense + BM25               (networkx multi-hop,
                     sparse, RRF fusion)                 signed, cross-domain)
                                │                                │
                                └───────────────┬────────────────┘
                                                 ▼
                                LLM synthesis (Gemini primary,
                                Hugging Face automatic fallback)
                                                 │
                                                 ▼
                                Self-verification audit
                                (generic-advice filter +
                                 independent grounding check)
                                                 │
                                                 ▼
                                Structured, cited, confidence-scored
                                recommendation

## Module map

| File | Responsibility |
|---|---|
| `core/schemas.py` | Pydantic data contracts — `SiteProfile`, `Recommendation`, `AdvisoryResponse` |
| `core/slots.py` | Free-text → structured facts (regex for numbers, keyword rules for categories), readiness gate, clarifying questions |
| `core/retriever.py` | Hybrid dense + sparse retrieval with Reciprocal Rank Fusion; numpy fallback if ChromaDB is unavailable |
| `core/graph_reasoner.py` | Causal graph construction and multi-hop pathway tracing, weighted toward cross-domain chains |
| `core/llm.py` | Provider-agnostic LLM wrapper — Gemini primary, Hugging Face automatic fallback, defensive JSON parsing |
| `core/recommender.py` | Orchestrates the full pipeline: profile → retrieval + causal pathways → LLM synthesis |
| `core/verifier.py` | Deterministic generic-advice filter + independent LLM grounding audit |
| `app.py` | Streamlit UI — themed to match Darukaa.Earth's visual identity, quick-entry form + conversational fallback, pipeline transparency panel |
| `api.py` | FastAPI REST interface — satisfies structured (JSON) input as a first-class entry point, not just a UI convenience |
| `data/knowledge_cards.json` | The curated evidence base |
| `data/causal_graph.json` | The environmental variable graph |
| `scripts/validate_kb.py`, `scripts/validate_graph.py` | Structural integrity checks, run in CI on every push |
| `tests/test_core.py` | 7 offline tests — no API key required — covering schema validity, graph integrity, multi-hop reasoning, the readiness gate, retrieval, and generic-advice rejection |

---

## Knowledge card schema

```json
{
  "id": "KC001",
  "title": "Legume-based cover cropping raises soil organic carbon",
  "domain": "soil_health",
  "intervention": "Legume cover crops sown in fallow windows",
  "mechanism": "Biological nitrogen fixation plus high-lignin root turnover increases carbon input below ground...",
  "quantified_effect": "Soil organic carbon typically rises 15-25% over 2-3 seasons",
  "metrics_improved": ["soil_organic_carbon", "microbial_diversity", "soil_moisture_retention"],
  "preconditions": ["rainfall_low", "monoculture", "low_soc"],
  "time_horizon": "medium",
  "source": "FAO, Soil Organic Carbon: the hidden potential (2017)",
  "source_tier": "tier1_intergovernmental",
  "verified": false
}
```

Every card requires at least 2 `metrics_improved` — enforced structurally by `scripts/validate_kb.py`, not just by convention — because the challenge explicitly bans single-variable answers.

## Causal graph schema

Each edge carries a `sign` (+/−), a confidence `weight` (0–1), and a plain-English `mechanism`. Path strength is the product of edge weights along a chain; net sign is the product of edge signs — so a chain with an odd number of suppressing edges is correctly identified as net-negative, without hardcoding every possible chain by hand.

---

## Tech stack

- **LLM:** Gemini 2.0 Flash (primary, free tier) with automatic Hugging Face fallback — a session doesn't die if one provider's quota runs out mid-demo
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2`, local, zero API cost, zero dependency on LLM availability
- **Vector store:** ChromaDB (with a numpy in-memory fallback if it fails to initialize)
- **Graph reasoning:** networkx
- **UI:** Streamlit, custom-themed to match Darukaa.Earth's forest-green visual identity
- **API:** FastAPI
- **Validation:** Pydantic throughout — every object crossing a module boundary is typed and validated
- **Testing:** pytest, 7 tests, fully offline
- **CI/CD:** GitHub Actions — validates the knowledge base and causal graph structurally, then runs the full test suite on every push to `main`; Streamlit Cloud auto-deploys on green

---

## Local setup

```bash
git clone https://github.com/Rounak-T/darukaa-ai-hackathon
cd darukaa-ai-hackathon

python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY and/or HF_API_KEY
```

Run the chat app:
```bash
streamlit run app.py
```

Run the REST API:
```bash
uvicorn api:app --reload --port 8000
# interactive docs at http://localhost:8000/docs
```

Run the tests (no API key required):
```bash
pytest tests/test_core.py -v
```

---

## Example: structured (JSON) input via the API

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"profile": {
        "soil_organic_carbon_pct": 0.3,
        "rainfall_pattern": "low",
        "land_use": "monoculture",
        "crop": "wheat",
        "region_type": "semi_arid"
      }}'
```

Returns a full `AdvisoryResponse`: summary, diagnosis, and 3+ recommendations, each with its scientific reasoning, quantified expected change, affected metrics, time horizon, confidence score, and cited evidence — already passed through the verification audit.

---

## Conversational behavior

The system will not guess. Below a readiness threshold (3 of 4 core facts: soil organic carbon, rainfall pattern, land use, region type), it returns targeted clarifying questions instead of advice:

> **User:** Biodiversity is declining on my land
>
> **System:** I need a few more site facts before I can give advice specific to your land.
> - What is your soil organic carbon, as a percentage?
> - How would you describe rainfall here — low, moderate, high, or erratic?
> - What is the current land use?

State persists across turns — a fact given in message three combines with facts from message one. Numbers are extracted by regex before the LLM ever sees them, since numeric hallucination is the highest-cost failure mode in an evidence-based system.

---

## Known limitations

- The knowledge base is hand-curated at 25 cards for reviewability and quality control, not exhaustive literature coverage. The retrieval and validation architecture is built to scale directly to full-text corpus ingestion (PDF chunking into the same ChromaDB collection) without any structural change.
- Citations name real institutional sources, but each card's `verified` field should be treated as a checklist — flipped to `true` only once a human has confirmed the specific figure against the original publication.
- Geo-coordinates are captured in the site profile schema but not yet joined to external soil/land-cover raster services — a natural next extension.

---

## What I'd build next

- Full-text ingestion pipeline for FAO/IPCC/IPBES source documents, extending the retrieval layer beyond hand-written cards
- A "why not X" mode — letting users ask why a competing intervention wasn't recommended, forcing the system to explicitly compare causal pathways
- Persistent per-site history, so returning users get trend-aware advice rather than a fresh diagnosis each time
- Multilingual input and response, so a user can describe their land and receive recommendations in their own language - critical for reaching farmers and land managers beyond English speakers
- Voice input for field use - a land manager standing in the field is more likely to describe conditions by speaking than typing, especially in low-literacy or low-connectivity contexts
                            
