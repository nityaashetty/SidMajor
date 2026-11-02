"""Paper figures (PDF, IEEE single-column width) from analysis/oof and analysis/results.

    python figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score                        # noqa: E402

HERE = Path(__file__).resolve().parent
OOF, RES, FIG = HERE / "oof", HERE / "results", HERE.parent / "figures"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
W = 3.45

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})


def alignment():
    a = pd.read_csv(RES / "alignment.csv")
    a = a[a.n_pauses > 0]
    fig, ax = plt.subplots(figsize=(W, 1.7))
    bins = np.logspace(0, np.log10(1000), 31)
    clip = lambda v: np.clip(np.abs(v), 1, None)
    ax.hist(clip(a.offset_raw_s), bins, color=ORANGE, alpha=0.85, label="raw event index", edgecolor="white", lw=0.5)
    ax.hist(clip(a.offset_corr_s), bins, color=BLUE, alpha=0.85, label="pause-corrected", edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("|final event time $-$ signal end| (s)")
    ax.set_ylabel("recordings")
    ax.legend(loc="upper right")
    ax.set_title(f"Recordings with at least one pause (n={len(a)})", loc="left", color=MUTED)
    fig.savefig(FIG / "alignment.pdf")


def protocol():
    st = json.loads((RES / "dataset_stats.json").read_text())["by_group"]
    order = ["Normal: community", "Normal: SMI", "MCI", "Dementia"]
    g = {r["group"]: r for r in st}
    labels = ["Comm.\nnormal", "SMI", "MCI", "Dem."]
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.6), gridspec_kw={"wspace": 0.45})
    x = np.arange(4)
    for ax, key, title, fmt in ((axes[0], "has_photic", "With photic stimulation", "{:.0%}"),
                                (axes[1], "age_mean", "Mean age (years)", "{:.0f}")):
        v = [g[o][key] for o in order]
        colors = [ORANGE] + [GRAY] * 3
        ax.bar(x, v, width=0.62, color=colors)
        for xi, vi in zip(x, v):
            ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x, labels, fontsize=6)
        ax.set_title(title, loc="left", color=MUTED)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.12)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].set_ylim(0, 88)
    fig.savefig(FIG / "protocol.pdf")


def _oof(name):
    d = pd.read_csv(OOF / f"{name}.csv")
    return d[d.fold >= 0]


def roc_internal_external():
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    for name, lab, c in (("sanf_net_cv5", "SANF-Net", BLUE), ("base_bioage_norm_lr", "Biomarkers+age LR", ORANGE),
                         ("base_age_lr", "Age only", GRAY)):
        d = _oof(name)
        d = d[d.y < 2]
        s = 1 - d.p0
        f, t, _ = roc_curve(d.y, s)
        ax.plot(f, t, color=c, label=f"{lab} ({roc_auc_score(d.y, s):.2f})")
    ax.set_title("CAUEEG CV: MCI vs Normal", loc="left", color=MUTED)
    ax2 = axes[1]
    for ds, cohort, lab, c in (("isfahan", None, "Isfahan", BLUE), ("geneeg", None, "GENEEG", ORANGE)):
        f = OOF / f"external_cv_{ds}.csv"            # CV-fold ensemble (paper's main models) when available
        d = pd.read_csv(f if f.exists() else OOF / f"external_{ds}.csv")
        for setting, ls in (("none", "--"), ("site", "-")):
            e = d[d.setting == setting]
            s = 1 - e.p0
            f, t, _ = roc_curve(e.y, s)
            ax2.plot(f, t, color=c, ls=ls, lw=1.3,
                     label=f"{lab}{' +calib' if setting == 'site' else ''} ({roc_auc_score(e.y, s):.2f})")
    ax2.set_title("External: MCI vs Normal", loc="left", color=MUTED)
    for a in axes:
        a.plot([0, 1], [0, 1], color="#c9c8c3", lw=0.8, zorder=0)
        a.set_xlabel("False positive rate")
        a.set_aspect("equal")
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=5.8, handlelength=1.6)
    axes[0].set_ylabel("True positive rate")
    fig.savefig(FIG / "roc.pdf")


def confusion():
    d = _oof("sanf_net_cv5")
    y, p = d.y.to_numpy(), d[["p0", "p1", "p2"]].to_numpy().argmax(1)
    cm = pd.crosstab(y, p).reindex(index=range(3), columns=range(3), fill_value=0).to_numpy()
    rn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(1.75, 1.6))
    ax.imshow(rn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f7fc", BLUE]), vmin=0, vmax=1)
    names = ["Normal", "MCI", "Dementia"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                    color="white" if rn[i, j] > 0.55 else INK)
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names, rotation=90, va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(FIG / "confusion.pdf")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    alignment()
    protocol()
    roc_internal_external()
    confusion()
    print("figures written to", FIG)
