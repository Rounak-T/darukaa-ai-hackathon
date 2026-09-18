from __future__ import annotations

from typing import List

from core.llm import complete_json, LLMError
from core.schemas import AdvisoryResponse, Recommendation

GENERIC_PHRASES = [
    "use sustainable practices", "improve soil health", "adopt best practices",
    "be environmentally friendly", "plant more trees", "raise awareness",
    "follow guidelines", "maintain biodiversity", "reduce environmental impact",
]

AUDIT_SYSTEM = """You audit scientific grounding. For each recommendation you
are given the claim and the evidence excerpt it cites. Decide whether the
evidence genuinely supports the claim - including any causal links made.

Return JSON: {"audits": [{"index": 0, "grounded": true|false, "note": "one sentence"}]}

Mark grounded=false if: the reasoning claims an effect (e.g. on water quality
or contaminants) that the cited evidence does not actually state, the number
is not in the evidence, or the causal claim goes beyond what the evidence says."""


def _is_generic(rec: Recommendation) -> bool:
    blob = f"{rec.action} {rec.scientific_reasoning}".lower()
    if any(p in blob for p in GENERIC_PHRASES) and len(rec.scientific_reasoning) < 220:
        return True
    return len(rec.metrics_improved) < 2


def verify(response: AdvisoryResponse) -> AdvisoryResponse:
    if response.mode != "advisory" or not response.recommendations:
        return response

    for rec in response.recommendations:
        if _is_generic(rec):
            rec.grounded = False
            rec.grounding_note = "Flagged as insufficiently specific or single-variable."
            rec.confidence = min(rec.confidence, 0.35)

    payload_lines: List[str] = []
    for i, rec in enumerate(response.recommendations):
        ev = "\n".join(f"    - [{e.card_id}] {e.claim} (source: {e.source})" for e in rec.evidence) or "    - NO EVIDENCE CITED"
        payload_lines.append(
            f"[{i}] ACTION: {rec.action}\n"
            f"    CLAIM: {rec.expected_change}\n"
            f"    REASONING: {rec.scientific_reasoning}\n"
            f"    EVIDENCE:\n{ev}"
        )

    try:
        result = complete_json(
            "Audit these recommendations:\n\n" + "\n\n".join(payload_lines),
            system=AUDIT_SYSTEM, temperature=0.0,
        )
        for audit in result.get("audits", []):
            idx = int(audit.get("index", -1))
            if 0 <= idx < len(response.recommendations):
                rec = response.recommendations[idx]
                if not audit.get("grounded", True):
                    rec.grounded = False
                    rec.grounding_note = audit.get("note", "Evidence does not fully support the claim.")
                    rec.confidence = round(min(rec.confidence, 0.5), 2)
    except (LLMError, Exception):
        pass  # audit is best-effort; pass 1 still applied

    return response