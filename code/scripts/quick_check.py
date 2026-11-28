"""Quick reproducibility check: recompute one experiment and compare it with the saved result.

Reruns the expected-utility model's (M1) learning curves on Basel--Berlin, every fold, prior and
selection rule, exactly as ``run_curves.py`` does, but writes into a temporary folder. Every array
is then compared with ``results/basel_berlin/curves_M1.npz``; the script fails if any differs by
more than ``TOL``.

    python scripts/quick_check.py      # about a minute, plus the data download on first use
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_curves

TOL = 1e-4  # results differ slightly across machines (about 2e-6 on an Apple-silicon Mac)


def main() -> None:
    """Recompute ``curves_M1`` for Basel--Berlin and compare it array by array."""
    saved_path = run_curves.RESULTS / "basel_berlin" / "curves_M1.npz"
    if not saved_path.exists():
        raise SystemExit(f"missing {saved_path}; the saved results are needed for the comparison")
    with tempfile.TemporaryDirectory() as tmp:
        run_curves.RESULTS = Path(tmp)
        run_curves.run_job("basel_berlin", "curves", "M1")
        new = np.load(Path(tmp) / "basel_berlin" / "curves_M1.npz")
        saved = np.load(saved_path)
        if set(new.files) != set(saved.files):
            raise SystemExit("the recomputed result has different arrays from the saved one")
        worst = max(float(np.nanmax(np.abs(new[k] - saved[k]), initial=0.0)) for k in saved.files)
    if worst > TOL:
        raise SystemExit(f"recomputed curves differ from the saved ones by up to {worst:.2e}")
    print(f"OK: recomputed M1 curves match results/basel_berlin/curves_M1.npz (max difference {worst:.1e})")


if __name__ == "__main__":
    main()
