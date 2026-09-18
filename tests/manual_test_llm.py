import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.llm import complete, complete_json

print("--- plain text test ---")
print(complete("Say 'hello, biodiversity system is connected' and nothing else."))

print("\n--- JSON test ---")
result = complete_json("Return a JSON object with one field: status, set to 'ok'.")
print(result)