"""Makes `src` importable when a script is run as `python scripts/xx.py` from any directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
