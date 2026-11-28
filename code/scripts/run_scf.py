"""SCF 2022: what the 0--10 self-rating says about stock ownership (Section 3, H4 paragraph).

Survey-weighted logistic regression of stock ownership on age, age squared, log income, log net
worth and education, with and without the rating; cross-fitted AUC, five implicates combined
with Rubin's rules.

    python scripts/run_scf.py          # writes results/scf.json
"""

import json
from pathlib import Path

from cs329h.data import load_scf
from cs329h.scf import analyze

OUT = Path(__file__).resolve().parents[1] / "results" / "scf.json"


def main() -> None:
    """Run the SCF analysis, write ``results/scf.json`` and print the combined estimates."""
    res = analyze(load_scf())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    c = res["combined"]
    for key in ("auc_without", "auc_with", "d_auc"):
        r = c[key]
        print(f"{key:12s} {r['estimate']:.4f} [{r['lower']:.4f}, {r['upper']:.4f}]")
    o = c["odds_ratio_per_point"]
    print(f"odds ratio of owning stocks per rating point {o['estimate']:.3f} [{o['lower']:.3f}, {o['upper']:.3f}]")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
