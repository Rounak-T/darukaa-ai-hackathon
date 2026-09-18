import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "data" / "knowledge_cards.json"

REQUIRED_FIELDS = {
    "id", "title", "domain", "intervention", "mechanism", "quantified_effect",
    "metrics_improved", "preconditions", "time_horizon", "source", "source_tier",
}
VALID_DOMAINS = {"soil_health", "land_use", "biodiversity", "climate", "human_impact"}
VALID_HORIZONS = {"short", "medium", "long"}

errors = []

with open(CARDS_PATH, "r", encoding="utf-8") as f:
    cards = json.load(f)

print(f"Loaded {len(cards)} cards.")

ids_seen = set()
domains_seen = set()

for card in cards:
    cid = card.get("id", "<missing id>")

    missing = REQUIRED_FIELDS - card.keys()
    if missing:
        errors.append(f"{cid}: missing fields {missing}")

    if cid in ids_seen:
        errors.append(f"{cid}: duplicate ID")
    ids_seen.add(cid)

    if card.get("domain") not in VALID_DOMAINS:
        errors.append(f"{cid}: invalid domain '{card.get('domain')}'")
    domains_seen.add(card.get("domain"))

    if card.get("time_horizon") not in VALID_HORIZONS:
        errors.append(f"{cid}: invalid time_horizon '{card.get('time_horizon')}'")

    if len(card.get("metrics_improved", [])) < 2:
        errors.append(f"{cid}: needs at least 2 metrics_improved (multi-variable rule)")

    if len(card.get("mechanism", "")) < 60:
        errors.append(f"{cid}: mechanism looks too short/generic")

missing_domains = VALID_DOMAINS - domains_seen
if missing_domains:
    errors.append(f"No cards found for required domain(s): {missing_domains}")

if errors:
    print(f"\nValidation FAILED with {len(errors)} issue(s):")
    for e in errors:
        print(" -", e)
else:
    print("\nKnowledge base is valid.")