"""
Refresh all figures that depend on hydro XTFT predictions.

Run from the k-diagram repo root AFTER merge_xtft_hydro.py completes:
  python examples/cas/scripts/refresh_after_xtft.py
"""

import subprocess
import sys
from pathlib import Path

scripts = Path(__file__).parent

ORDER = [
    scripts
    / "figure3_rank_stability.py",  # Kendall tau includes hydro XTFT now
    scripts / "figA1_h_sensitivity.py",
    scripts / "figA2_lambda_crossing.py",
    scripts / "figA3_kernel_robustness.py",
    scripts / "figA4_horizon_profile.py",
    scripts / "figA5_directional_cas.py",
]

for script in ORDER:
    print(f"\n{'=' * 60}")
    print(f"Running: {script.name}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=False
    )
    if result.returncode != 0:
        print(f"[WARN] {script.name} exited with code {result.returncode}")
    else:
        print(f"[OK] {script.name} done")

print("\n[Done] All figures refreshed.")
