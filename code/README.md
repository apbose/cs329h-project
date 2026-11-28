# CS329H project code

Code for "Ask About Money, Keep the Questionnaire": how few adaptively chosen lottery questions
predict a person's other risky choices, under four model classes, on simulated investors,
Basel–Berlin, Twin-2K-500 and the SCF.

## Requirements

- Python 3.12 or later (the pinned NumPy and SciPy need it; the results were produced with
  Python 3.13) and the exact package versions in `requirements.txt`. No GPU is needed:
  everything runs on CPUs.
- The full rebuild was run on a machine with 4 CPU cores and 16 GB of RAM.
- About 270 MB of disk for the downloaded data and 35 MB for the results.
- Internet access the first time each dataset is loaded. Every dataset is public and downloads
  itself into `data/raw/` (sources, licenses and links in `data/README.md`).

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

## Quickstart

    bash run_all.sh --quick

This runs the unit tests, recomputes one experiment from scratch (the expected-utility model's
learning curve on Basel–Berlin, all folds and selection rules) into a temporary folder and checks
it against the saved result, and then rebuilds every analysis, figure and table from the saved
results. It takes about 3 minutes, plus the data download on first use.

## Reproducing the paper, step by step

`results/` holds the saved output of every experiment, so the paper's figures and tables can be
rebuilt in minutes without rerunning them (step 3). To recompute everything from the public data:

1. Set up the environment (above) and run `pytest -q` (about a minute; no downloads needed).
2. Run the experiments: `bash run_all.sh`. This runs each script in the table below in order and
   takes about 12.5 hours on 4 cores (about 40 CPU-hours). `run_curves.py`, `run_robustness.py`
   and `run_further_checks.py` skip outputs already in `results/`, so delete those files first to
   recompute them; the other scripts always recompute.
3. Rebuild the analyses, figures and tables from `results/`: `bash run_all.sh --figures`
   (about 2 minutes).
4. Build the paper: `latexmk -pdf main.tex` in `../final-paper/`. It includes the figures from
   `figures/` and the table bodies from `tables/`.

Every step uses fixed random seeds, so a rerun gives the same numbers up to floating-point
differences across machines and library builds.

| Step | Script | Time |
|---|---|---|
| Simulated investors, M1 | `scripts/run_sim_m1.py` | about 1 minute |
| 4 × 4 grid of true by fitted class | `scripts/run_recovery_grid.py` | about 2 hours (4 rows in parallel) |
| SCF regression | `scripts/run_scf.py` | under a minute |
| Learning curves, Basel–Berlin / Twin-2K-500 | `scripts/run_curves.py` | about 2.5 / 3.5 hours (4 jobs in parallel; 7.4 / 9.4 CPU-hours) |
| Robustness checks, Basel–Berlin / Twin-2K-500 | `scripts/run_robustness.py` | about 15 / 30 minutes (4 jobs in parallel) |
| List coverage of adaptive questions, both datasets | `scripts/run_list_coverage.py` | about 3 minutes |
| Further checks (Appendix B), Basel–Berlin / Twin-2K-500 | `scripts/run_further_checks.py` | about 1.5 / 2 hours (4 jobs in parallel; 5.5 / 8 CPU-hours) |
| Hypotheses and summaries | `scripts/analyze.py` | about 1 minute per dataset |
| Figures and tables | `scripts/make_figures.py` | under a minute |

## Where each result in the paper comes from

| In the paper | Built by | From |
|---|---|---|
| Figure 1, left; simulation numbers in "Sanity check" | `make_figures.py` | `results/sim_m1.json` (`run_sim_m1.py`) |
| Figure 1, right; true-by-fitted numbers in "Sanity check" | `make_figures.py` | `results/recovery_grid.json` (`run_recovery_grid.py`) |
| Figure 2 (learning curves) | `make_figures.py` | `results/<data>/analysis.json` (`analyze.py`, from `run_curves.py`) |
| Figure 3 (selection rules) | `make_figures.py` | `results/<data>/analysis.json` (`analyze.py`, from `run_curves.py`) |
| Table 1 and the H1–H4 paragraphs | `make_figures.py` | `results/<data>/analysis.json` (`analyze.py`, from `run_curves.py`) |
| Lottery answers predicting the self-ratings (H4 paragraph) | `analyze.py` | `results/basel_berlin/analysis.json`, key `ratings_from_lotteries` |
| List coverage of adaptive questions (H2 paragraph) | `run_list_coverage.py` | `results/<data>/list_coverage.json` |
| Table 2 and the robustness paragraph | `make_figures.py` | `results/<data>/analysis.json` (`analyze.py`, from `run_robustness.py`) |
| Split-half reliability of the lottery answers (H4 paragraph) | `analyze.py` | `results/basel_berlin/analysis.json`, key `lottery_reliability` |
| Table 3 (accuracy) | `make_figures.py` | `results/<data>/analysis.json`, key `exploratory.accuracy` |
| Table 4 and the simulated-grid sentence in the H2 paragraph | `make_figures.py` | `results/recovery_grid.json`, key `adaptive_minus_best_fixed` (`run_recovery_grid.py`) |
| Payoff scale, financial rating in the prior and more draws of training people (Appendix B) | `analyze.py` | `results/<data>/analysis.json`, key `further_checks` (`run_further_checks.py`) |
| SCF numbers (H4 paragraph) | `run_scf.py` | `results/scf.json` |
| Dataset counts (Data section) | `src/cs329h/data.py` | the downloaded data |

## Scripts

| Script | What it does | Output |
|---|---|---|
| `scripts/make_splits.py` | Five person folds and two held-out price lists per person for Basel–Berlin and Twin-2K-500 | `splits/*.json` (committed) |
| `scripts/run_sim_m1.py` | M1 fitted to Basel–Berlin, simulated investors drawn from it, and the full evaluation on them: k = 0–10 questions under four selection rules, with and without the self-rating, plus parameter recovery | `results/sim_m1.json` |
| `scripts/run_recovery_grid.py` | The 4 × 4 grid: simulated investors from each model class (fitted to Basel–Berlin), each fitted by every class; excess held-out loss over the true model and adaptive minus the best fixed order | `results/recovery_grid.json` |
| `scripts/run_scf.py` | SCF 2022: weighted logistic regression of stock ownership with and without the 0–10 rating; cross-fitted AUC and the rating's odds ratio, five implicates combined with Rubin's rules | `results/scf.json` |
| `scripts/run_curves.py` | Learning curves on real data for every class: held-out loss over k = 0–10 questions under each selection rule, with and without the self-rating, and over N = 50–800 training people at k = 5; runs four jobs at a time | `results/<data>/{curves,ncurve}_<class>.{npz,json}` |
| `scripts/run_robustness.py` | The k = 0–10 curve for M1 and M3 with the lapse rate fixed (0 for M1, 0.05, 0.15), the prior's standard deviation halved or doubled, and M3's rank fixed (1–5) | `results/<data>/robustness/*.{npz,json}` |
| `scripts/run_list_coverage.py` | Exploratory diagnostic for H2: how many distinct price lists the factor model's first five adaptive questions cover, against the best fixed order | `results/<data>/list_coverage.json` |
| `scripts/run_further_checks.py` | Exploratory checks of Appendix B: one payoff scale per dataset instead of one per list (M1, M2), the financial rating in the prior (Basel–Berlin, M1–M3), and four more random draws of training people at N = 50, 100 and 200 | `results/<data>/further/*.{npz,json}` |
| `scripts/analyze.py` | Hypotheses H1–H4 of the pre-analysis plan from those curves, plus the exploratory comparisons (prediction-targeted questions, calibration, accuracy, lottery answers predicting the self-ratings, their split-half reliability) and the summaries of the robustness and Appendix B checks | `results/<data>/analysis.json` |
| `scripts/make_figures.py` | Every figure and table in the manuscript, from the files in `results/` | `figures/*.pdf`, `tables/*.tex` |

Each script is deterministic (fixed seeds). `run_curves.py`, `run_robustness.py`, `run_further_checks.py`,
`run_recovery_grid.py`, `run_list_coverage.py` and `analyze.py` take options; run them with `--help` to list them.

## Layout

- `src/cs329h/` — the package.
  - `data.py` — loaders for Basel–Berlin, Twin-2K-500 and the SCF; lottery datasets share one format (`ChoiceData`).
  - `splits.py` — the fixed folds and held-out price lists.
  - `models/` — the four choice-model classes behind one interface (`base.py`): M1 expected utility (`eu.py`, grid posterior), M2 cumulative prospect theory (`cpt.py`, shared importance samples), M3 low-rank factor model (`factor.py`, Laplace posterior) and M4 neural model (`neural.py`, variational fit and an importance-sampled posterior; optionally pretrained on synthetic M2 choices), sharing the latent-variable machinery in `latent.py`.
  - `select.py` — expected information gain about the person's parameters, the exploratory prediction-targeted rule (expected information gain about the answers to the questions not yet asked), the fixed-order baselines, and the replay that scores held-out choices after each answer.
  - `experiment.py` — the evaluation loop over folds, priors and selection rules, with bootstrap summaries.
  - `simulate.py` — simulated investors with known parameters, from any model class.
  - `scf.py` — the SCF regression and Rubin's rules.
  - `evaluate.py` — calibration and bootstrap intervals.
- `scripts/` — entry points (above).
- `splits/` — the committed evaluation splits.
- `results/` — analysis outputs that the figures are built from.
- `figures/`, `tables/` — generated figures and LaTeX table bodies; regenerate, don't hand-edit.
- `tests/` — `pytest -q` (no downloads needed).

## Code reuse and licenses

All code here was written for this project; none is copied or adapted from other projects. It uses
NumPy, SciPy, pandas and Matplotlib as libraries. The methods it implements are cited in the paper
where they are used: expected information gain for binary answers (Houlsby et al., 2011), its
application to risk preferences (DOSE; Chapman et al., 2018), prediction-targeted information gain
(Bickford Smith et al., 2023) and pretraining on a behavioral model (Bourgin et al., 2019). The
datasets are used under their licenses (CC BY 4.0 for Basel–Berlin and Twin-2K-500, public domain
for the SCF) and are not redistributed here.

