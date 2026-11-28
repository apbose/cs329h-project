# Data

Nothing here is committed. Each loader in `src/cs329h/data.py` downloads its files into
`raw/<dataset>/` on first use (about 270 MB in all; no account or API key is needed). To use a copy
downloaded by hand, put the files listed below in `raw/<dataset>/`, named as listed.

| Dataset | Source | License | Files | Loader |
|---|---|---|---|---|
| Basel–Berlin Risk Study (Frey et al. 2017; Pedroni et al. 2017) | OSF project [`rce7g`](https://osf.io/rce7g/) | CC BY 4.0 | `mpl.csv`, `mplProblems.csv`, `quest_scores.csv`, `quest_raw.csv` | `load_basel_berlin()` |
| Twin-2K-500 (Toubia et al. 2025) | Hugging Face [`LLM-Digital-Twin/Twin-2K-500`](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500) | CC BY 4.0 | `question_catalog.json`, `wave1_3_response.csv` | `load_twin()` |
| Survey of Consumer Finances 2022 | Federal Reserve, [SCF 2022](https://www.federalreserve.gov/econres/scfindex.htm) | Public domain | `scfp2022excel.zip`, `scf2022s.zip` | `load_scf()` |

Coding choices made by the loaders:

- **Basel–Berlin:** 1,505 of 1,507 people answered all 66 choices and both self-ratings. The public file codes the SOEP 0–10 ratings as 1–11; the loader subtracts 1. In each question, the "riskier" option is the one with the larger outcome variance; in the one Holt–Laury row where both options are certain, it is the option the study's own `R` flag marks, so `Y` equals `R` throughout.
- **Twin-2K-500:** the seven price lists QID250–252 (gains), QID276–278 (losses) and QID279 (mixed), 14 rows each. The response file codes the lottery (left) as 1 and the sure amount (right) as 2.
- **SCF 2022:** all five implicates (4,595 households each). `X7557` codes a willingness of 0 as −1; the loader recodes it to 0.
