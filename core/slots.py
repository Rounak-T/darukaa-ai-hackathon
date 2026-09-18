from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from core.schemas import SiteProfile

CORE_SLOTS = ["soil_organic_carbon_pct", "rainfall_pattern", "land_use", "region_type"]
MIN_SLOTS_TO_ANSWER = 3

_NUM = r"(\d+(?:\.\d+)?)"
REGEX_HINTS = [
    ("soil_organic_carbon_pct", rf"(?:soc|organic\s+carbon|organic\s+matter)[^\d%]{{0,20}}{_NUM}\s*%?", float),
    ("soil_ph", rf"\bph\b[^\d]{{0,10}}{_NUM}", float),
    ("rainfall_mm", rf"{_NUM}\s*mm", float),
    ("area_hectares", rf"{_NUM}\s*(?:ha|hectare)", float),
]


def regex_extract(text: str) -> Dict[str, Any]:
    """Deterministic extraction for anything numeric - never guessed, never hallucinated."""
    found: Dict[str, Any] = {}
    low = text.lower()
    for field, pattern, caster in REGEX_HINTS:
        m = re.search(pattern, low)
        if m:
            try:
                found[field] = caster(m.group(1))
            except ValueError:
                pass
    return found


def keyword_extract(text: str) -> Dict[str, Any]:
    """Simple keyword rules for qualitative fields - cheap and doesn't need an
    LLM call for straightforward cases."""
    low = text.lower()
    found: Dict[str, Any] = {}

    if "rainfall" in low or "rain" in low:
        if any(w in low for w in ["low", "scarce", "erratic", "drought", "dry"]):
            found["rainfall_pattern"] = "low"
        elif "moderate" in low:
            found["rainfall_pattern"] = "moderate"
        elif "high" in low or "heavy" in low:
            found["rainfall_pattern"] = "high"
    elif any(w in low for w in ["drought", "dry spell"]):
        found["rainfall_pattern"] = "low"

    if "monoculture" in low:
        found["land_use"] = "monoculture"
    elif "polyculture" in low or "intercrop" in low:
        found["land_use"] = "polyculture"
    elif "grassland" in low or "pasture" in low:
        found["land_use"] = "grassland"
    elif "agroforestry" in low:
        found["land_use"] = "agroforestry"
    elif "fallow" in low:
        found["land_use"] = "fallow"

    if "semi-arid" in low or "semi arid" in low:
        found["region_type"] = "semi_arid"
    elif "arid" in low:
        found["region_type"] = "arid"
    elif "humid" in low:
        found["region_type"] = "humid"

    for crop in ["wheat", "rice", "maize", "corn", "cotton", "sugarcane", "millet", "sorghum"]:
        if crop in low:
            found["crop"] = crop
            break

    pressures = []
    for kw, tag in [("pesticide", "pesticide"), ("fertiliser", "fertiliser"), ("fertilizer", "fertiliser"),
                     ("deforest", "deforestation"), ("graz", "grazing"), ("burn", "burning"), ("till", "tillage")]:
        if kw in low:
            pressures.append(tag)
    if pressures:
        found["human_pressures"] = pressures

    return found


def extract_profile(text: str, existing: SiteProfile | None = None) -> SiteProfile:
    """Merge newly stated facts into the running profile. New facts win, but
    nothing already known gets erased."""
    base = existing.model_dump() if existing else {}

    updates: Dict[str, Any] = {}
    updates.update(keyword_extract(text))
    updates.update(regex_extract(text))  # numeric regex takes priority, applied last

    merged = {**base}
    for k, v in updates.items():
        if k == "human_pressures":
            prior = merged.get("human_pressures") or []
            merged[k] = sorted(set(list(prior) + list(v)))
        elif v not in (None, "", []):
            merged[k] = v

    if not merged.get("observed_problem"):
        merged["observed_problem"] = text.strip()[:300]

    clean = {k: v for k, v in merged.items() if k in SiteProfile.model_fields}
    return SiteProfile(**clean)


SLOT_QUESTIONS = {
    "soil_organic_carbon_pct": "What is your soil organic carbon, as a percentage? A recent soil test figure is ideal.",
    "rainfall_pattern": "How would you describe rainfall here - low, moderate, high, or erratic?",
    "land_use": "What is the current land use - monoculture, polyculture, grassland, agroforestry, or fallow?",
    "region_type": "Which climate zone are you in - arid, semi-arid, sub-humid, or humid?",
}


def missing_slots(profile: SiteProfile) -> List[str]:
    filled = set(profile.filled_fields())
    return [s for s in CORE_SLOTS if s not in filled]


def readiness(profile: SiteProfile) -> Tuple[bool, List[str]]:
    filled = set(profile.filled_fields())
    core_filled = [s for s in CORE_SLOTS if s in filled]
    return len(core_filled) >= MIN_SLOTS_TO_ANSWER, missing_slots(profile)


def build_questions(missing: List[str], limit: int = 3) -> List[str]:
    return [SLOT_QUESTIONS[s] for s in missing[:limit] if s in SLOT_QUESTIONS]


def derive_conditions(profile: SiteProfile) -> List[str]:
    """Translate the profile into tags that match your knowledge card 'preconditions'."""
    c: List[str] = []
    if profile.soil_organic_carbon_pct is not None and profile.soil_organic_carbon_pct < 0.75:
        c.append("low_soc")
    if profile.rainfall_pattern in ("low", "erratic"):
        c.append("rainfall_low")
    if profile.rainfall_mm is not None and profile.rainfall_mm < 700:
        c.append("rainfall_low")
    if profile.region_type in ("arid", "semi_arid"):
        c.append("semi_arid")
    if profile.land_use:
        c.append(profile.land_use)
    for p in profile.human_pressures:
        c.append(p)
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out