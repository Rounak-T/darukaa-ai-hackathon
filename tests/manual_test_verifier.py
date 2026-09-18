import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schemas import SiteProfile
from core.recommender import advise
from core.verifier import verify

profile = SiteProfile(
    soil_organic_carbon_pct=0.3,
    rainfall_pattern="low",
    land_use="monoculture",
    crop="wheat",
    region_type="semi_arid",
)

resp = verify(advise(profile))

for i, r in enumerate(resp.recommendations, 1):
    flag = "⚠️ FLAGGED" if not r.grounded else "✅ grounded"
    print(f"\n--- Recommendation {i} [{flag}] ---")
    print("Action:", r.action)
    print("Confidence:", r.confidence)
    if not r.grounded:
        print("Note:", r.grounding_note)