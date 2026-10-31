"""
Streamlit Community Cloud entrypoint for Q-Catalyst.
"""
import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
src_path = str(ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

app_path = ROOT / "src" / "qcatalyst" / "dashboard" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
