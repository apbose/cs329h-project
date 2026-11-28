# Ask About Money, Keep the Questionnaire

**What an AI adviser should ask about risk.** Before investing a client's savings, an adviser
has to learn how much risk the client accepts. A 0–10 self-rating is cheap but noisy, and long
questionnaires of choices between gambles tire clients. This project asks how few gamble
questions an adviser needs if each one is chosen adaptively, by expected information gain under
a Bayesian posterior, and how the answer depends on the model of choice.

We compare four model classes: expected utility, cumulative prospect theory, a low-rank factor
model and a small neural network. We evaluate them on simulated investors and three public
datasets: the Basel–Berlin Risk Study (1,505 adults), Twin-2K-500 (2,058 US adults) and the
2022 Survey of Consumer Finances (4,595 households). For each class we trace held-out
prediction against the number of questions asked and the number of people trained on.

## Findings

- **Flexible models win, even with little data.** The factor and neural models predict a new
  person's other choices better than expected utility or prospect theory from 100 training
  people on, and tie or beat them with only 50.
- **Adaptive questions don't beat a good fixed list.** On real people, adaptive selection at
  best ties the best fixed list of questions. On Twin-2K-500 it does worse for three of the
  five model classes, its predictions grow overconfident, and after ten questions it is no
  better than a random order. On simulated people, whose model is known, it works as intended.
- **Ask about money directly.** Lottery answers barely predict either self-rating. The
  financial self-rating predicts real stock investment far better than any number of lottery
  answers (AUC 0.88).

The paper is [`final-paper/main.pdf`](final-paper/main.pdf). The hypotheses and analysis plan
were fixed in advance in [`pre-analysis-plan/main.pdf`](pre-analysis-plan/main.pdf).

## Quickstart

    cd code
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt && pip install -e .
    bash run_all.sh --quick

The quickstart takes about 3 minutes, plus a one-time download of the public data (about
270 MB). It runs the unit tests and recomputes one experiment from scratch, checking it against
the saved result. It then rebuilds every figure and table in the paper from the saved results
in `code/results/`.

To rerun every experiment from the public data, use `bash run_all.sh`. This takes about
12.5 hours on 4 CPU cores, and no GPU is needed. [`code/README.md`](code/README.md) gives the
requirements and step-by-step instructions. It also has runtimes for each script and shows
which script produces each figure, table and number in the paper.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | The Python package (`src/cs329h/`), experiment scripts, tests, saved results, figures and tables |
| `final-paper/` | The paper (`main.tex`, `main.pdf`) |
| `pre-analysis-plan/` | The pre-analysis plan (`main.tex`, `main.pdf`) |
| `statements.tex` | Integrity, AI-use and impact statements, included in both documents |
| `refs.bib` | Bibliography for both documents |
| `neurips_2025.sty` | NeurIPS 2025 style file |

## Building the PDFs

This needs a TeX distribution with `latexmk`:

    cd final-paper && latexmk -pdf main.tex
    cd pre-analysis-plan && latexmk -pdf main.tex

The figures and table bodies come from `code/figures/` and `code/tables/`. `latexmk -c`
removes the build files.

## Data

All three datasets are public and are downloaded by the code on first use. They are not
redistributed here. [`code/data/README.md`](code/data/README.md) lists the sources, licenses
and the coding choices the loaders make.

## Citation

    @misc{bose2026askaboutmoney,
      author = {Bose, Apurba},
      title  = {Ask About Money, Keep the Questionnaire: What an {AI} Adviser Should Ask About Risk},
      note   = {CS329H final project, Stanford University},
      year   = {2026}
    }
