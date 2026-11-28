#!/usr/bin/env bash
# Rebuild every result, figure and table in the manuscript from the public data.
#
#   bash run_all.sh            # everything: about 12.5 hours on 4 CPU cores
#   bash run_all.sh --figures  # only the analysis, figures and tables, from the saved results/
#   bash run_all.sh --quick    # tests, one experiment recomputed and checked, then --figures
#
# Every step is deterministic (fixed seeds). run_curves.py, run_robustness.py and
# run_further_checks.py skip outputs already in results/; delete a file there to recompute it.
# Each dataset downloads itself into data/raw/ on first use.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

mode="${1:-}"
case "$mode" in
  ""|--figures|--quick) ;;
  *) echo "usage: bash run_all.sh [--figures | --quick]" >&2; exit 2 ;;
esac

if [[ "$mode" == "--quick" ]]; then
  python -m pytest -q                                     # unit tests
  python scripts/quick_check.py                           # recompute M1 curves on Basel-Berlin, compare
fi
if [[ -z "$mode" ]]; then
  python scripts/run_sim_m1.py                            # simulated investors, M1
  python scripts/run_recovery_grid.py                     # 4 x 4 grid of true by fitted class
  python scripts/run_scf.py                               # SCF 2022 regressions
  for data in basel_berlin twin2k500; do
    python scripts/run_curves.py --data "$data"           # learning curves, every class
    python scripts/run_robustness.py --data "$data"       # lapse rate, prior strength, rank
    python scripts/run_list_coverage.py --data "$data"    # which lists adaptive questions use
    python scripts/run_further_checks.py --data "$data"   # Appendix B: payoff scale, rating, draws
  done
fi
for data in basel_berlin twin2k500; do
  python scripts/analyze.py --data "$data"                # H1-H4 and robustness summaries
done
python scripts/make_figures.py                            # figures/ and tables/
