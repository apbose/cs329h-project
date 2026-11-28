"""Regenerate every figure and table in the final manuscript from the saved results.

Run from the ``code/`` directory after the analysis scripts::

    python scripts/make_figures.py

Each ``fig_*`` function reads files in ``results/`` and writes one PDF into ``figures/``; each
``tab_*`` function writes one LaTeX table body into ``tables/``, which the manuscript inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#5b5b57",
    "axes.labelcolor": "#2b2b29", "xtick.color": "#5b5b57", "ytick.color": "#5b5b57",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e4e3de", "grid.linewidth": 0.5,
    "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False, "legend.handlelength": 1.8,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PDF_METADATA = {"CreationDate": None}
TAB_DIR = ROOT / "tables"
RESULTS = ROOT / "results"
WIDTH = 5.5
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
MARKERS = ("o", "s", "^", "D", "v")
MODEL_STYLE = {m: dict(color=c, marker=k) for m, c, k in zip(("M1", "M2", "M3", "M4", "M4-pre"), PALETTE, MARKERS)}
RULE_LABELS = {"adaptive": "adaptive (EIG)", "best_fixed": "best fixed order", "random": "random order",
               "original": "questionnaire order", "predictive": "prediction-targeted"}
RULE_STYLE = {r: dict(color=c, marker=k) for r, c, k in zip(RULE_LABELS, PALETTE, MARKERS)}
TITLES = {"basel_berlin": "Basel–Berlin", "twin2k500": "Twin-2K-500"}


def _curve(ax, x, triples, style, label=None, band=True, **kw):
    """Plot the means of ``[mean, lo, hi]`` triples, with the interval as a light band."""
    mean = [t[0] for t in triples]
    ax.plot(x, mean, label=label, markeredgecolor="white", markeredgewidth=0.4, **style, **kw)
    if band:
        ax.fill_between(x, [t[1] for t in triples], [t[2] for t in triples], color=style["color"], alpha=0.14, lw=0)


def _legend_above(fig, handles, labels, ncol, top=0.9):
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 1.0), columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, top), w_pad=1.5, h_pad=1.0)


def fig_simulation() -> None:
    """Checks on simulated people (Section 3, first paragraph).

    Left: held-out loss against questions asked under each rule, M1 people questioned with M1.
    Right: excess held-out loss over the true parameters after ten adaptive questions, for
    every pair of true and fitted class.
    """
    sim_path, grid_path = RESULTS / "sim_m1.json", RESULTS / "recovery_grid.json"
    if not (sim_path.exists() and grid_path.exists()):
        return
    sim, grid = json.loads(sim_path.read_text()), json.loads(grid_path.read_text())
    fig, (ax_k, ax_g) = plt.subplots(1, 2, figsize=(WIDTH, 0.42 * WIDTH), gridspec_kw={"width_ratios": (1.25, 1)})
    for rule in ("adaptive", "best_fixed", "random", "original"):
        curve = sim["loss"]["none"][rule]
        _curve(ax_k, range(len(curve)), curve, RULE_STYLE[rule], RULE_LABELS[rule])
    ax_k.axhline(sim["oracle_loss"][0], color="#2b2b29", ls=(0, (4, 3)), lw=0.8, label="true parameters")
    ax_k.set_xlabel("questions asked, $k$")
    ax_k.set_ylabel("held-out log loss (nats per choice)")
    ax_k.set_xticks(range(0, 11, 2))
    ax_k.set_title("M1 people questioned with M1", loc="left")
    classes = list(grid)
    excess = np.array([[grid[t]["fitted"][f]["adaptive"]["excess_loss"]["10"][0] for f in classes] for t in classes])
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec5f4", "#3987e5", "#184f95"])
    ax_g.imshow(excess, cmap=cmap, vmin=0, vmax=excess.max())
    ax_g.grid(False)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    ax_g.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax_g.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax_g.grid(which="minor", color="white", lw=1.5)
    ax_g.tick_params(which="both", length=0)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = excess[i, j]
            ax_g.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                      color="white" if v > 0.55 * excess.max() else "#2b2b29",
                      fontweight="bold" if v == excess[i].min() else "normal")
    ax_g.set_xticks(range(len(classes)), classes)
    ax_g.set_yticks(range(len(classes)), classes)
    ax_g.set_xlabel("fitted class")
    ax_g.set_ylabel("true class")
    ax_g.set_title("excess loss, $k = 10$", loc="left")
    _legend_above(fig, *ax_k.get_legend_handles_labels(), ncol=5, top=0.88)
    fig.savefig(FIG_DIR / "fig_simulation.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_rules() -> None:
    """Held-out loss against questions asked under each selection rule, M3 on both datasets (H2)."""
    paths = {d: RESULTS / d / "analysis.json" for d in TITLES}
    if not all(p.exists() for p in paths.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 0.4 * WIDTH))
    for ax, (data_name, path) in zip(axes, paths.items()):
        loss = json.loads(path.read_text())["exploratory"]["loss"]["M3"]
        prior = "rating" if "rating" in loss else "none"
        for rule, label in RULE_LABELS.items():
            curve = loss[prior][rule]
            _curve(ax, np.arange(len(curve)), curve, RULE_STYLE[rule], label,
                   band=rule in ("adaptive", "best_fixed"), ls=(0, (3, 1.5)) if rule == "predictive" else "-")
        ax.set_title(TITLES[data_name], loc="left")
        ax.set_xlabel("questions asked, $k$")
        ax.set_xticks(range(0, 11, 2))
    axes[0].set_ylabel("held-out log loss (nats per choice)")
    _legend_above(fig, *axes[0].get_legend_handles_labels(), ncol=5, top=0.87)
    fig.savefig(FIG_DIR / "fig_rules.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def fig_learning_curves() -> None:
    """Both learning curves for every class: rows are datasets; columns are held-out loss over
    questions asked (all training people) and over training people at k = 5.

    Adaptive selection, prior as in deployment (with the self-rating when the data has one).
    """
    paths = {d: RESULTS / d / "analysis.json" for d in TITLES}
    if not all(p.exists() for p in paths.values()):
        return
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 0.72 * WIDTH), sharey="row")
    for row, (data_name, path) in zip(axes, paths.items()):
        res = json.loads(path.read_text())
        loss = res["exploratory"]["loss"]
        prior = "rating" if "rating" in next(iter(loss.values())) else "none"
        n_all = res["H1"]["n_train_all"]
        ax_k, ax_n = row
        for m in loss:
            curve = loss[m][prior]["adaptive"]
            _curve(ax_k, np.arange(len(curve)), curve, MODEL_STYLE[m], m)
            table = res["H1"]["loss_k5"][m]
            sizes = [int(N) if N != "all" else n_all for N in table]
            _curve(ax_n, sizes, list(table.values()), MODEL_STYLE[m])
        ax_k.set_title(f"{TITLES[data_name]}: questions asked", loc="left")
        ax_n.set_title(f"{TITLES[data_name]}: training people ($k = 5$)", loc="left")
        ax_k.set_ylabel("held-out log loss")
        ax_k.set_xticks(range(0, 11, 2))
        ax_n.set_xscale("log")
        ax_n.set_xticks(sizes, [str(N) for N in sizes[:-1]] + ["all"])
        ax_n.minorticks_off()
    axes[1, 0].set_xlabel("questions asked, $k$")
    axes[1, 1].set_xlabel("training people, $N$")
    _legend_above(fig, *axes[0, 0].get_legend_handles_labels(), ncol=5, top=0.93)
    fig.savefig(FIG_DIR / "fig_curves.pdf", metadata=PDF_METADATA)
    plt.close(fig)


def _ci(v, digits=3, sign=True) -> str:
    """``est [lo, hi]`` with a true minus sign."""
    f = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return " ".join([f.format(v[0]), "[" + f.format(v[1]) + ", " + f.format(v[2]) + "]"]).replace("-", "$-$")


def _analysis(data_name: str) -> dict | None:
    path = RESULTS / data_name / "analysis.json"
    return json.loads(path.read_text()) if path.exists() else None


def tab_hypotheses() -> None:
    """H1--H4 with their decision rules, one column per dataset."""
    bb, tw = _analysis("basel_berlin"), _analysis("twin2k500")
    if bb is None or tw is None:
        return
    rows = []
    for label, key in (("$N = 50$", "50"), ("all training people", "all")):
        cells = []
        for a in (bb, tw):
            gaps = a["H1"]["flexible_minus_best_simple"]
            best = min(gaps, key=lambda m: gaps[m][key][0])
            cells.append(f"{_ci(gaps[best][key])} ({best})")
        rows.append(f"H1: best flexible minus best simple, {label} & " + " & ".join(cells) + r" \\")
    cells = []
    for a in (bb, tw):
        h2 = a["H2"]
        pr = "rating" if "rating" in h2["adaptive_minus_best_fixed"][h2["class"]] else "none"
        cells.append(f"{_ci(h2['adaptive_minus_best_fixed'][h2['class']][pr])} ({h2['class']})")
    rows.append("H2: adaptive minus best fixed order, $k = 5$ & " + " & ".join(cells) + r" \\")
    h3 = bb["H3"]
    rows.append(f"H3: loss reduction from the self-rating, $k = 0$ & {_ci(h3['rating_gain'][h3['class']][0], 4)} ({h3['class']}) & --- \\\\")
    best = bb["H4"]["class"]
    for name, label in (("stock", "stock investment"), ("fund", "equity fund choice")):
        e = bb["H4"]["outcomes"][name]
        d = e["classes"][best]["10"]["posterior_minus_rating_alone_auc"]
        rows.append(f"H4: AUC, posterior ($k = 10$) minus self-rating, {label} & {_ci(d, 2)} & --- \\\\")
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / "tab_hypotheses.tex").write_text("\n".join(rows) + "\n")


def tab_robustness() -> None:
    """Adaptive minus best fixed order at k = 5 and 10, and calibration, under each check."""
    bb, tw = _analysis("basel_berlin"), _analysis("twin2k500")
    if bb is None or tw is None or not bb.get("robustness"):
        return
    names = {"main": "as fitted", "eps0": r"$\varepsilon = 0$", "eps0.05": r"$\varepsilon = 0.05$",
             "eps0.15": r"$\varepsilon = 0.15$", "prior0.5": r"prior s.d.\ $\times 0.5$", "prior2": r"prior s.d.\ $\times 2$",
             **{f"K{k}": f"$K = {k}$" for k in range(1, 6)}}
    rows = []
    keys = [f"{m}_{c}" for m in ("M1", "M3") for c in names if f"{m}_{c}" in bb["robustness"]]
    for key in keys:
        model, check = key.split("_", 1)
        if key not in tw.get("robustness", {}):
            continue
        if rows and model == "M3" and check == "main":
            rows.append(r"\addlinespace")
        cells = []
        for a in (bb, tw):
            r = a["robustness"][key]
            k5 = r["adaptive_minus_best_fixed_k5"][0]
            cells += [f"{k5:+.3f}".replace("-", "$-$"), _ci(r["adaptive_minus_best_fixed_k10"]),
                      f"{r['ece_adaptive_k10']:.3f}"]
        rows.append(f"{model} & {names[check]} & " + " & ".join(cells) + r" \\")
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / "tab_robustness.tex").write_text("\n".join(rows) + "\n")


def tab_accuracy() -> None:
    """Held-out accuracy at k = 5 and 10, adaptive and best fixed order, prior as in deployment."""
    bb, tw = _analysis("basel_berlin"), _analysis("twin2k500")
    if bb is None or tw is None:
        return
    rows = []
    for m in bb["exploratory"]["accuracy"]:
        cells = []
        for a, pr in ((bb, "rating"), (tw, "none")):
            acc = a["exploratory"]["accuracy"][m][pr]
            cells += [f"{acc[rule][k][0]:.3f}" for k in ("5", "10") for rule in ("adaptive", "best_fixed")]
        rows.append(f"{m} & " + " & ".join(cells) + r" \\")
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / "tab_accuracy.tex").write_text("\n".join(rows) + "\n")


def tab_grid_rules() -> None:
    """Adaptive minus best fixed order on the simulated grid at k = 5 and 10 (rows: true class)."""
    path = RESULTS / "recovery_grid.json"
    if not path.exists():
        return
    grid = json.loads(path.read_text())
    classes = list(grid)
    rows = []
    for k in ("5", "10"):
        if rows:
            rows.append(r"\addlinespace")
        for i, t in enumerate(classes):
            cells = [_ci(grid[t]["fitted"][f]["adaptive_minus_best_fixed"][k]) for f in classes]
            rows.append((f"$k = {k}$" if i == 0 else "") + f" & {t} & " + " & ".join(cells) + r" \\")
    TAB_DIR.mkdir(exist_ok=True)
    (TAB_DIR / "tab_grid_rules.tex").write_text("\n".join(rows) + "\n")


def main() -> None:
    """Draw every figure and write every LaTeX table."""
    FIG_DIR.mkdir(exist_ok=True)
    fig_simulation()
    fig_learning_curves()
    fig_rules()
    tab_hypotheses()
    tab_robustness()
    tab_accuracy()
    tab_grid_rules()
    print(f"figures written to {FIG_DIR}, tables to {TAB_DIR}")


if __name__ == "__main__":
    main()
