"""Orchestrates: profile -> conditions -> (retrieval + causal pathways) -> LLM synthesis."""
from __future__ import annotations

from typing import Any, Dict, List

from core.graph_reasoner import describe_pathways, trace_pathways
from core.llm import complete_json
from core.retriever import get_retriever
from core.schemas import AdvisoryResponse, Evidence, Recommendation, SiteProfile, TimeHorizon
from core.slots import build_questions, derive_conditions, readiness

SYSTEM = """You are an environmental scientist advising a land manager. You are
not a chatbot: you reason like a researcher writing a site-specific note.

Rules:
1. Every recommendation must rest on the EVIDENCE CARDS supplied. Never invent
   a statistic, study, or source not present in the evidence.
2. Every recommendation must connect at least THREE environmental variables.
   A single-variable answer is a failure.
3. Use the CAUSAL PATHWAYS to explain indirect effects the user has not
   considered - this is what makes a recommendation non-obvious.
4. Banned: "use sustainable practices", "improve soil health", "plant more
   trees" with no mechanism, or any advice generic enough to apply to any
   land anywhere.
5. Quantify using ONLY figures present in the evidence cards.
6. Never use emojis, exclamation points, or casual/marketing language. Write
   in plain, neutral scientific prose, matching the tone of a technical report."""

TEMPLATE = """SITE PROFILE
{profile}

DETECTED CONDITIONS
{conditions}

EVIDENCE CARDS (the only permitted basis for claims)
{evidence}

CAUSAL PATHWAYS (multi-hop chains from this site's conditions to biodiversity outcomes)
{pathways}

TASK
Produce 3 recommendations for this specific site. Return JSON:

{{
  "summary": "2-3 sentences on what is actually limiting biodiversity here",
  "diagnosis": "the causal story - which variable is the binding constraint",
  "recommendations": [
    {{
      "action": "specific intervention",
      "scientific_reasoning": "the mechanism, linking at least 3 variables, including one indirect effect from the causal pathways",
      "metrics_improved": ["metric names"],
      "expected_change": "quantified estimate taken from the evidence cards",
      "time_horizon": "short|medium|long",
      "confidence": 0.0-1.0,
      "evidence_card_ids": ["KC0xx"]
    }}
  ]
}}"""


def _evidence_block(cards: List[Dict[str, Any]]) -> str:
    out = []
    for c in cards:
        out.append(
            f"[{c['id']}] {c['title']}\n"
            f"  intervention: {c['intervention']}\n"
            f"  mechanism: {c['mechanism']}\n"
            f"  quantified effect: {c['quantified_effect']}\n"
            f"  metrics: {', '.join(c['metrics_improved'])}\n"
            f"  source: {c['source']} ({c['source_tier']})"
        )
    return "\n\n".join(out)


def advise(profile: SiteProfile, force_answer: bool = False) -> AdvisoryResponse:
    ready, missing = readiness(profile)

    if not ready and not force_answer:
        return AdvisoryResponse(
            mode="clarifying",
            summary="I need a few more site facts before I can give advice specific to your land.",
            clarifying_questions=build_questions(missing),
            missing_slots=missing,
        )

    conditions = derive_conditions(profile)
    query = f"{profile.to_context_string()}. Conditions: {', '.join(conditions)}"
    cards = get_retriever().search(query)

    from core.graph_reasoner import load_graph
    g = load_graph()
    start_nodes = [c for c in conditions if c in g.nodes]
    # map condition tags to graph nodes where names differ
    tag_to_node = {"low_soc": "soil_organic_carbon", "monoculture": "crop_diversity",
                   "rainfall_low": "soil_moisture_retention", "semi_arid": "soil_moisture_retention"}
    for tag in conditions:
        if tag in tag_to_node and tag_to_node[tag] not in start_nodes:
            start_nodes.append(tag_to_node[tag])
    pathways = trace_pathways(start_nodes or ["soil_organic_carbon"])

    prompt = TEMPLATE.format(
        profile=profile.to_context_string(),
        conditions=", ".join(conditions) or "none detected",
        evidence=_evidence_block(cards),
        pathways=describe_pathways(pathways) or "none found",
    )
    data = complete_json(prompt, system=SYSTEM, temperature=0.35)

    by_id = {c["id"]: c for c in cards}
    recs: List[Recommendation] = []
    for r in data.get("recommendations", []):
        ev = []
        for cid in r.get("evidence_card_ids", []):
            c = by_id.get(cid)
            if c:
                ev.append(Evidence(card_id=c["id"], source=c["source"], claim=c["quantified_effect"]))
        try:
            horizon = TimeHorizon(r.get("time_horizon", "medium"))
        except ValueError:
            horizon = TimeHorizon.MEDIUM
        recs.append(Recommendation(
            action=r.get("action", ""),
            scientific_reasoning=r.get("scientific_reasoning", ""),
            metrics_improved=r.get("metrics_improved", []),
            expected_change=r.get("expected_change", ""),
            time_horizon=horizon,
            confidence=float(r.get("confidence", 0.5)),
            evidence=ev,
        ))

    return AdvisoryResponse(
        mode="advisory",
        summary=data.get("summary", ""),
        diagnosis=data.get("diagnosis", ""),
        recommendations=recs,
        missing_slots=missing,
    )