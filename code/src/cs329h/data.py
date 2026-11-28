"""Loaders for the three datasets, in one common format.

Every lottery dataset becomes a :class:`ChoiceData`: an answer matrix ``Y`` with one row per
person and one column per binary question, and for each question the two options written as
two-outcome lotteries. Column ``q`` of ``Y`` is 1 when the person chose the *riskier* option of
question ``q`` (the one with the larger outcome variance; Basel--Berlin's own ``R`` flag on the
one question where both options are certain).

* :func:`load_basel_berlin` -- Basel--Berlin Risk Study (Frey et al. 2017; OSF ``rce7g``,
  CC BY 4.0): 66 choices in seven price lists, the SOEP 0--10 self-ratings, two outcomes.
* :func:`load_twin` -- Twin-2K-500 (Toubia et al. 2025; Hugging Face
  ``LLM-Digital-Twin/Twin-2K-500``, CC BY 4.0): seven 14-row price lists, 98 choices.
* :func:`load_scf` -- Survey of Consumer Finances 2022 (Federal Reserve, public domain): the
  0--10 risk willingness and holdings, all five implicates.

Raw files are downloaded on first use into ``data/raw/<dataset>/`` (not committed).
"""

from __future__ import annotations

import csv
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"

BB_FILES = {
    "mpl.csv": "https://osf.io/download/jtcxq/",
    "mplProblems.csv": "https://osf.io/download/8vmdq/",
    "quest_scores.csv": "https://osf.io/download/tckbj/",
    "quest_raw.csv": "https://osf.io/download/hb2mu/",
}
TWIN_BASE = (
    "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/"
    "question_catalog_and_human_response_csv/"
)
TWIN_FILES = {name: TWIN_BASE + name for name in ("question_catalog.json", "wave1_3_response.csv")}
SCF_BASE = "https://www.federalreserve.gov/econres/files/"
SCF_FILES = {name: SCF_BASE + name for name in ("scfp2022excel.zip", "scf2022s.zip")}

# Twin-2K-500 price lists, in the order the survey shows them: the lottery each list offers
# (two equally likely outcomes, in dollars) and a label. For QID279 the second outcome is the
# row's x and the sure amount is $0. The loader checks these against the question text.
TWIN_LISTS = {
    "QID250": ((6.0, 0.0), "gain 6/0"),
    "QID251": ((8.0, 2.0), "gain 8/2"),
    "QID252": ((10.0, 0.0), "gain 10/0"),
    "QID276": ((-6.0, 0.0), "loss 6/0"),
    "QID277": ((-8.0, -2.0), "loss 8/2"),
    "QID278": ((-10.0, 0.0), "loss 10/0"),
    "QID279": ((-8.0, None), "mixed -8/x"),
}


@dataclass
class ChoiceData:
    """Binary choices between two lotteries, one row per person.

    Attributes:
        name: Dataset name, used for split files and outputs.
        Y: ``(n, Q)`` int array, 1 if the person chose the riskier option of question ``q``.
        x: ``(Q, 2, 2)`` outcomes as ``[option, outcome]`` with option 0 the riskier one and
            option 1 the safer one, divided by ``scale`` so they lie in ``[-1, 1]``.
        p: ``(Q, 2, 2)`` probabilities matching ``x``.
        scale: ``(Q,)`` largest absolute outcome in the question's price list, in the
            dataset's currency; ``x * scale[:, None, None]`` gives the original amounts.
        lists: ``(Q,)`` price-list index of each question, ``0 .. L-1``.
        list_names: Name of each price list.
        person_ids: Identifier of each row of ``Y``.
        rating: ``(n,)`` general 0--10 risk self-rating, or ``None`` if the dataset has none.
        rating_fin: ``(n,)`` financial 0--10 self-rating, or ``None``.
        outcomes: Extra per-person outcomes (``nan`` where missing), keyed by name.
    """

    name: str
    Y: np.ndarray
    x: np.ndarray
    p: np.ndarray
    scale: np.ndarray
    lists: np.ndarray
    list_names: list[str]
    person_ids: list[str]
    rating: np.ndarray | None = None
    rating_fin: np.ndarray | None = None
    outcomes: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_people(self) -> int:
        """Number of people (rows of ``Y``)."""
        return self.Y.shape[0]

    @property
    def n_questions(self) -> int:
        """Number of questions (columns of ``Y``)."""
        return self.Y.shape[1]

    @property
    def n_lists(self) -> int:
        """Number of price lists."""
        return len(self.list_names)

    def subset(self, rows: np.ndarray) -> "ChoiceData":
        """Return the same questions for a subset of people (rows given as indices or a mask)."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        take = lambda a: None if a is None else a[rows]
        return ChoiceData(
            name=self.name,
            Y=self.Y[rows],
            x=self.x,
            p=self.p,
            scale=self.scale,
            lists=self.lists,
            list_names=self.list_names,
            person_ids=[self.person_ids[i] for i in rows],
            rating=take(self.rating),
            rating_fin=take(self.rating_fin),
            outcomes={k: v[rows] for k, v in self.outcomes.items()},
        )


def fetch(files: dict[str, str], folder: Path) -> Path:
    """Download each ``name -> url`` into ``folder`` unless it is already there."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in files.items():
        path = folder / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            # Some hosts (federalreserve.gov) refuse Python's default user agent.
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cs329h data fetch)"})
            try:
                with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            except OSError as err:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"could not download {name} from {url} ({err}). Check the internet connection, "
                    f"or download the file by hand and save it as {path} (see data/README.md)."
                ) from err
            tmp.rename(path)
    return folder


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _variance(out: np.ndarray, pr: np.ndarray) -> np.ndarray:
    mean = (out * pr).sum(-1, keepdims=True)
    return ((out - mean) ** 2 * pr).sum(-1)


def load_basel_berlin(folder: Path | None = None) -> ChoiceData:
    """Load the Basel--Berlin Risk Study lottery choices, self-ratings and outcomes.

    Keeps the people who answered all 66 choices and both SOEP self-ratings (1,505 of 1,507).
    The public file codes the 0--10 self-ratings as 1--11; they are recoded to 0--10 here.

    Outcomes (``nan`` where not answered), as coded in the study's raw questionnaire file:

    * ``stock``: invested in individual stocks in the past year (``DOSPERTjahr_6`` >= 2;
      answer 4, "don't know", is missing).
    * ``fund``: chose the 45% equity retirement fund over the 25% one
      (``RLRQ_Risiko_Altersvorsorge_16`` == 2).
    """
    folder = fetch(BB_FILES, folder or RAW_DIR / "basel_berlin")
    probs = _read_csv(folder / "mplProblems.csv")
    q = len(probs)
    out = np.zeros((q, 2, 2))
    pr = np.zeros((q, 2, 2))
    dp = np.zeros(q, int)
    col = {}
    for i, row in enumerate(probs):
        for j, opt in enumerate("AB"):
            out[i, j] = [float(row[f"{opt}_out1"]), float(row[f"{opt}_out2"])]
            pr[i, j] = [float(row[f"{opt}_p1"]), float(row[f"{opt}_p2"])]
        dp[i] = int(row["dp"])
        col[(row["dp"], row["decision"])] = i

    # The study's flag R marks A when A has the larger variance and B otherwise, including the
    # Holt--Laury row where both options are certain. Reorder so option 0 is the one R marks.
    var = _variance(out, pr)
    risky = np.where(var[:, 0] > var[:, 1], 0, 1)
    rows_q = np.arange(q)
    out = np.stack([out[rows_q, risky], out[rows_q, 1 - risky]], 1)
    pr = np.stack([pr[rows_q, risky], pr[rows_q, 1 - risky]], 1)

    answers = [r for r in _read_csv(folder / "mpl.csv") if r["dp"] != "NA"]
    ids = sorted({r["partid"] for r in answers})
    idx = {pid: k for k, pid in enumerate(ids)}
    Y = np.full((len(ids), q), np.nan)
    for r in answers:
        Y[idx[r["partid"]], col[(r["dp"], r["decision"])]] = float(r["R"])

    quest = {r["partid"]: r for r in _read_csv(folder / "quest_scores.csv")}

    def rating(name: str) -> np.ndarray:
        """0--10 self-rating from column ``name`` (the file's 1--11 code minus one), ``nan`` if missing."""
        vals = [quest.get(pid, {}).get(name, "") for pid in ids]
        return np.array([float(v) - 1 if v not in ("", "NA") else np.nan for v in vals])

    soep, soep_fin = rating("SOEP"), rating("SOEPfin")
    keep = ~np.isnan(Y).any(1) & ~np.isnan(soep) & ~np.isnan(soep_fin)

    raw = {r["partid"]: r for r in _read_csv(folder / "quest_raw.csv")}

    def outcome(name: str, code) -> np.ndarray:
        """Raw questionnaire column ``name`` recoded by ``code``, ``nan`` if missing."""
        vals = [raw.get(pid, {}).get(name, "") for pid in ids]
        return np.array([code(int(v)) if v not in ("", "NA") else np.nan for v in vals])

    outcomes = {
        "stock": outcome("DOSPERTjahr_6", lambda v: np.nan if v == 4 else float(v >= 2)),
        "fund": outcome("RLRQ_Risiko_Altersvorsorge_16", lambda v: float(v == 2)),
    }
    scale = np.array([np.abs(out[dp == d]).max() for d in dp])
    names = ["Holt-Laury"] + [f"list {d}" for d in range(2, 8)]
    data = ChoiceData(
        name="basel_berlin",
        Y=np.nan_to_num(Y, nan=-1).astype(int),
        x=out / scale[:, None, None],
        p=pr,
        scale=scale,
        lists=dp - 1,
        list_names=names,
        person_ids=ids,
        rating=soep,
        rating_fin=soep_fin,
        outcomes=outcomes,
    )
    return data.subset(keep)


def _twin_amount(text: str) -> float:
    m = re.search(r"(-?)\$(\d+(?:\.\d+)?)", text)
    if m is None:
        raise ValueError(f"no dollar amount in {text!r}")
    return float(m.group(2)) * (-1 if m.group(1) else 1)


def load_twin(folder: Path | None = None) -> ChoiceData:
    """Load the seven Twin-2K-500 price lists (98 binary choices per person).

    In each row the left option is the list's lottery and the right option a sure amount; the
    response files code the left option as 1 and the right as 2, so ``Y = 1`` (riskier option)
    means the lottery was chosen. Twin-2K-500 has no 0--10 risk self-rating and no holdings.
    """
    folder = fetch(TWIN_FILES, folder or RAW_DIR / "twin2k500")
    catalog = json.loads((folder / "question_catalog.json").read_text())
    blocks = {
        c["QuestionID"]: c
        for c in catalog
        if c["QuestionType"] == "Matrix" and c["BlockName"].strip() == "Economic preferences"
    }
    xs, ps, lists, cols = [], [], [], []
    for li, (qid, ((o1, o2), _)) in enumerate(TWIN_LISTS.items()):
        entry = blocks[qid]
        stated = [_twin_amount(t) for t in re.findall(r"(?:LOSING|winning) -?\$\d+(?:\.\d+)?", entry["QuestionText"])]
        expected = [abs(o1)] + ([abs(o2)] if o2 is not None else [])
        if [abs(v) for v in stated[: len(expected)]] != expected:
            raise ValueError(f"{qid}: question text does not match the coded lottery")
        for row_label, column in zip(entry["Rows"], entry["csv_columns"]):
            if o2 is None:  # mixed list: lottery -8 or x, sure amount 0
                lottery = (o1, _twin_amount(row_label.split(":")[0]))
                sure = 0.0
            else:
                lottery = (o1, o2)
                sure = _twin_amount(row_label.split(":")[1])
            xs.append([list(lottery), [sure, sure]])
            ps.append([[0.5, 0.5], [0.5, 0.5]])
            lists.append(li)
            cols.append(column)
    out = np.array(xs, float)
    lists = np.array(lists)

    with open(folder / "wave1_3_response.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = [(r["pid"], [r[c] for c in cols]) for r in reader]
    ids = [pid for pid, _ in rows]
    raw = np.array([[float(v) if v != "" else np.nan for v in vals] for _, vals in rows])
    keep = ~np.isnan(raw).any(1)
    if not set(np.unique(raw[keep])) <= {1.0, 2.0}:
        raise ValueError("unexpected answer codes in the Twin-2K-500 price lists")
    scale = np.array([np.abs(out[lists == l]).max() for l in lists])
    data = ChoiceData(
        name="twin2k500",
        Y=(raw == 1).astype(int),
        x=out / scale[:, None, None],
        p=np.array(ps, float),
        scale=scale,
        lists=lists,
        list_names=[label for _, label in TWIN_LISTS.values()],
        person_ids=ids,
    )
    return data.subset(keep)


def load_scf(folder: Path | None = None):
    """Load the 2022 Survey of Consumer Finances: risk willingness, holdings and covariates.

    Returns a pandas DataFrame with one row per household and implicate (``implicate`` 1--5,
    ``household`` the household id). Columns: ``weight`` (WGT), ``age``, ``educ`` (EDCL, 1--4),
    ``income``, ``networth``, ``owns_stock`` (HSTOCKS, directly held stocks), ``risk10`` (X7557,
    willingness to take financial risks on 0--10; the SCF codes 0 as -1, recoded to 0).
    """
    import pandas as pd

    folder = fetch(SCF_FILES, folder or RAW_DIR / "scf2022")
    for name in SCF_FILES:
        with zipfile.ZipFile(folder / name) as z:
            for member in z.namelist():
                if not (folder / member).exists():
                    z.extract(member, folder)
    summary = pd.read_csv(
        folder / "SCFP2022.csv",
        usecols=["YY1", "Y1", "WGT", "AGE", "EDCL", "INCOME", "NETWORTH", "HSTOCKS"],
    )
    full = pd.read_stata(folder / "p22i6.dta", columns=["y1", "x7557"], convert_categoricals=False)
    d = summary.merge(full, left_on="Y1", right_on="y1", validate="one_to_one")
    return pd.DataFrame(
        {
            "household": d.YY1,
            "implicate": d.Y1 % 10,
            "weight": d.WGT,
            "age": d.AGE,
            "educ": d.EDCL,
            "income": d.INCOME,
            "networth": d.NETWORTH,
            "owns_stock": d.HSTOCKS,
            "risk10": d.x7557.clip(lower=0),
        }
    ).reset_index(drop=True)
