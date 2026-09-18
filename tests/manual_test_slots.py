import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schemas import SiteProfile
from core.slots import extract_profile, readiness, derive_conditions

# turn 1
p = extract_profile("Biodiversity is declining on my land")
ready, missing = readiness(p)
print("Turn 1 - ready:", ready, "| missing:", missing)

# turn 2 - more info added
p = extract_profile("SOC is 0.3%, rainfall is low, semi-arid region, growing monoculture wheat", existing=p)
ready, missing = readiness(p)
print("Turn 2 - ready:", ready, "| missing:", missing)
print("Profile:", p.model_dump(exclude_none=True))
print("Conditions:", derive_conditions(p))