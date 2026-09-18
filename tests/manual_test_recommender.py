import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schemas import SiteProfile
from core.recommender import advise

profile = SiteProfile(
    soil_organic_carbon_pct=0.3,
    rainfall_pattern="low",
    land_use="monoculture",
    crop="wheat",
    region_type="semi_arid",
)

resp = advise(profile)
print("MODE:", resp.mode)
print("\nSUMMARY:", resp.summary)
print("\nDIAGNOSIS:", resp.diagnosis)
for i, r in enumerate(resp.recommendations, 1):
    print(f"\n--- Recommendation {i} ---")
    print("Action:", r.action)
    print("Reasoning:", r.scientific_reasoning)
    print("Metrics:", r.metrics_improved)
    print("Expected change:", r.expected_change)
    print("Horizon:", r.time_horizon.value, "| Confidence:", r.confidence)
    print("Evidence:", [e.card_id for e in r.evidence])