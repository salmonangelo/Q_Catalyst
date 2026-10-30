"""
End-to-End Orchestrator and Live Web Demonstration Runner for Q-Catalyst.
Runs all pipeline stages sequentially if outputs are not cached, then launches the interactive dashboard.
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline():
    """Runs all 6 sequential pipeline scripts."""
    scripts = [
        "prepare_data.py",
        "run_ai.py",
        "select_candidates.py",
        "run_chemistry.py",
        "run_quantum.py",
        "evaluate.py"
    ]
    scripts_dir = PROJECT_ROOT / "scripts"

    print("=" * 70)
    print("Q-CATALYST: Executing Full End-to-End Scientific Pipeline")
    print("=" * 70)

    for s in scripts:
        script_path = scripts_dir / s
        print(f"\n>>> Running {s}...")
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print(f"Error running {s}. Exiting.")
            sys.exit(res.returncode)

    print("\n" + "=" * 70)
    print("All Pipeline Steps Completed Successfully!")
    print("=" * 70)


def launch_dashboard():
    """Launches the Streamlit web dashboard."""
    app_path = PROJECT_ROOT / "src" / "qcatalyst" / "dashboard" / "app.py"
    port = 8501
    url = f"http://localhost:{port}"

    print(f"\nLaunching Q-Catalyst Interactive Web Application...")
    print(f"Direct Link: {url}")
    print("=" * 70)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    run_pipeline()
    launch_dashboard()
