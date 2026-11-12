"""Figures for the README (dark style)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import RUNS, VIEWS_NPZ
from .evaluate import RESULTS

FIG = Path(__file__).resolve().parents[1] / "figures"
BG, FG, GRID = "#0b0f1a", "#e6e9ef", "#2a3142"
COLORS = {"astronet": "#7aa2f7", "global": "#565f89", "local": "#9aa5ce", "extended_noscalars": "#e0af68",
          "extended": "#f7768e", "scalars_gbm": "#9ece6a"}
NAMES = {"astronet": "AstroNet (global + local)", "global": "AstroNet, global only",
         "local": "AstroNet, local only", "extended_noscalars": "+ odd/even + secondary views",
         "extended": "+ views + DV scalars", "scalars_gbm": "DV scalars only (gradient boosting)"}


def style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
        "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": GRID,
        "grid.color": GRID, "axes.grid": True, "grid.linewidth": 0.6, "font.size": 11,
        "axes.titleweight": "bold", "legend.frameon": False,
    })


def _runs(split="test"):
    out = {}
    for d in sorted(RUNS.iterdir()):
        f = d / f"pred_{split}.csv"
        if f.exists():
            out[d.name] = pd.read_csv(f)
    return out


def roc_pr(split="test"):
    runs = _runs(split)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2))
    for name, df in runs.items():
        fpr, tpr, _ = roc_curve(df.label, df.p)
        prec, rec, _ = precision_recall_curve(df.label, df.p)
        c = COLORS.get(name, FG)
        from sklearn.metrics import roc_auc_score, average_precision_score
        a.plot(fpr, tpr, color=c, lw=2, label=f"{NAMES.get(name, name)}  AUC {roc_auc_score(df.label, df.p):.3f}")
        b.plot(rec, prec, color=c, lw=2, label=f"{NAMES.get(name, name)}  AP {average_precision_score(df.label, df.p):.3f}")
    a.plot([0, 1], [0, 1], ls=":", color=GRID)
    a.set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC, held-out test stars")
    b.set(xlabel="recall (completeness)", ylabel="precision (reliability)", title="Precision-recall",
          ylim=(0.3, 1.02))
    a.legend(fontsize=8.5, loc="lower right")
    b.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "roc_pr.png", dpi=150)
    plt.close(fig)


def ablation(split="test"):
    m = pd.read_csv(RESULTS / f"metrics_{split}.csv")
    order = [k for k in ["global", "local", "astronet", "extended_noscalars", "extended", "scalars_gbm"]
             if k in set(m.model)]
    m = m.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(m))
    ax.barh(y, m.auc, color=[COLORS.get(k, FG) for k in m.index], height=0.6)
    ax.errorbar(m.auc, y, xerr=[m.auc - m.auc_lo, m.auc_hi - m.auc], fmt="none", ecolor=FG, capsize=4, lw=1)
    lo = max(0.5, float(m.auc_lo.min()) - 0.03)
    for yi, (auc, r99) in enumerate(zip(m.auc, m.recall_at_p99)):
        ax.text(lo + 0.002, yi, f"AUC {auc:.3f}   recall at 99 % precision {r99:.2f}", va="center",
                fontsize=9, color=BG, fontweight="bold")
    ax.set_yticks(y, [NAMES.get(k, k) for k in m.index])
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("ROC AUC on held-out stars (bars: 95 % bootstrap over stars)")
    ax.set_title("Ablation: what each input adds")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=150)
    plt.close(fig)


def calibration(split="test", bins=8):
    runs = _runs(split)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.plot([0, 1], [0, 1], ls=":", color=GRID)
    for name, df in runs.items():
        if name in ("global", "local"):
            continue
        # equal-count bins: the test set is small and most scores sit near 0
        order = np.argsort(df.p.values)
        chunks = np.array_split(order, bins)
        xs = [df.p.values[c].mean() for c in chunks]
        ys = [df.label.values[c].mean() for c in chunks]
        ax.plot(xs, ys, "o-", color=COLORS.get(name, FG), label=NAMES.get(name, name), ms=4)
    ax.set(xlabel="mean predicted probability in the bin", ylabel="observed fraction of PCs",
           title=f"Reliability ({bins} equal-count bins, test set)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)


def recall_vs_mes(split="test", model="extended"):
    runs = _runs(split)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    edges = np.array([7.1, 9, 12, 16, 25, 50, 1e9])
    for name in ("astronet", "extended"):
        if name not in runs:
            continue
        df = runs[name][runs[name].label == 1]
        mid, rec, n = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (df.mes >= lo) & (df.mes < hi)
            if sel.sum() >= 3:
                mid.append(np.sqrt(lo * min(hi, 100))); rec.append((df.p[sel] >= 0.5).mean()); n.append(sel.sum())
        ax.plot(mid, rec, "o-", color=COLORS[name], label=NAMES[name])
        for x, r, k in zip(mid, rec, n):
            ax.text(x, r + 0.02, str(k), color=COLORS[name], fontsize=8, ha="center")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([8, 10, 15, 20, 30, 50, 70]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="multiple event statistic, MES (signal strength)", ylabel="recall of PCs at p > 0.5",
           title="Recall of planet candidates vs signal strength (numbers: PCs per bin)", ylim=(0, 1.08))
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "recall_vs_mes.png", dpi=150)
    plt.close(fig)


def example_views(path=VIEWS_NPZ):
    """A planet candidate and an eclipsing binary (AFP with a clear secondary), all five views."""
    z = np.load(path)
    y, av = z["label"], z["av_training_set"]
    sec_depth = z["secondary"].min(1)
    oe = np.abs(z["odd"].min(1) - z["even"].min(1))
    loc_noise = np.std(z["local"][:, :40], 1)
    pc = np.flatnonzero((av == "PC") & (loc_noise < 0.05) & (z["period"] > 5))
    eb = np.flatnonzero((av == "AFP") & (sec_depth < -0.3) & (loc_noise < 0.05))
    picks = [("Planet candidate", pc[0] if len(pc) else 0), ("Eclipsing binary (AFP)", eb[0] if len(eb) else 1)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), gridspec_kw={"width_ratios": [2.2, 1, 1, 1]})
    for row, (title, i) in enumerate(picks):
        a0, a1, a2, a3 = axes[row]
        a0.plot(z["global"][i], ".", ms=1.5, color="#7aa2f7")
        a0.set_title(f"{title}: KIC {z['kepid'][i]}-{z['plnt'][i]}, P = {z['period'][i]:.2f} d\nglobal view (2001 bins)",
                     fontsize=10)
        a1.plot(z["local"][i], ".", ms=2.5, color="#9aa5ce"); a1.set_title("local view (201)", fontsize=10)
        a2.plot(z["odd"][i], ".", ms=2.5, color="#e0af68", label="odd")
        a2.plot(z["even"][i], ".", ms=2.5, color="#bb9af7", label="even")
        a2.set_title("odd vs even", fontsize=10); a2.legend(fontsize=8, loc="lower left")
        a3.plot(z["secondary"][i], ".", ms=2.5, color="#f7768e"); a3.set_title("at the secondary phase", fontsize=10)
        lo = min(-1.3, float(z["secondary"][i].min()) - 0.1, float(min(z["odd"][i].min(), z["even"][i].min())) - 0.1)
        for a in (a1, a2, a3):
            a.set_ylim(lo, 0.5)
        a0.set_ylabel("normalised flux")
    fig.tight_layout()
    fig.savefig(FIG / "example_views.png", dpi=150)
    plt.close(fig)


def training_curves():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name in ("astronet", "extended"):
        d = RUNS / name
        for k, f in enumerate(sorted(d.glob("history*.csv"))):
            h = pd.read_csv(f)
            ax.plot(h.epoch + 1, h.val_auc, color=COLORS[name], alpha=0.6, lw=1,
                    label=NAMES[name] if k == 0 else None)
    ax.set(xlabel="epoch", ylabel="validation ROC AUC", title="Training, one line per seed")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=150)
    plt.close(fig)


def occlusion(window=50, step=10):
    """Shallue & Vanderburg (2018, Fig. 8): slide a window of zeros over the global view and record the mean
    planet probability whenever each bin was hidden. Uses the AstroNet seed-0 model on test TCEs."""
    import torch

    from .dataset import Views
    from .model import build
    from .train import to_batch

    v = Views()
    test = v.index("test")
    df = pd.read_csv(RUNS / "astronet" / "pred_test.csv")
    model = build("astronet")
    model.load_state_dict(torch.load(RUNS / "astronet" / "seed0.pt", map_location="cpu"))
    model.eval()
    quiet = np.std(v["local"][test][:, :40], 1) < 0.05
    pc = test[(v["label"][test] == 1) & (df.p.values > 0.9) & quiet]
    eb = test[(v["av_training_set"][test] == "AFP") & (df.p.values < 0.2) & quiet
              & (v["secondary"][test].min(1) < -0.1) & (v["secondary"][test].min(1) > -0.9)]
    picks = [(i, "planet candidate") for i in pc[:1]] + [(i, "eclipsing binary (AFP)") for i in eb[:1]]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.2 * len(picks)), squeeze=False)
    for ax, (i, title) in zip(axes[:, 0], picks):
        g = v["global"][i]
        starts = np.arange(0, len(g) - window + 1, step)
        batch = np.repeat(g[None], len(starts), 0)
        for k, s0 in enumerate(starts):
            batch[k, s0:s0 + window] = 0.0
        b = to_batch(v, np.repeat(i, len(starts)), "cpu")
        b["global"] = torch.from_numpy(batch)
        with torch.no_grad():
            p = torch.sigmoid(model(b)).numpy()
        heat, cnt = np.zeros(len(g)), np.zeros(len(g))
        for k, s0 in enumerate(starts):
            heat[s0:s0 + window] += p[k]
            cnt[s0:s0 + window] += 1
        heat /= np.maximum(cnt, 1)
        sc = ax.scatter(np.arange(len(g)), g, c=heat, cmap="plasma", s=3, vmin=0, vmax=1)
        ax.set_title(f"{title}: KIC {v['kepid'][i]}-{v['plnt'][i]}, unoccluded p = {df.p.values[test == i][0]:.2f}",
                     fontsize=10)
        ax.set_ylabel("global view")
        fig.colorbar(sc, ax=ax, pad=0.01).set_label("mean p when this bin is hidden", fontsize=8)
    axes[-1, 0].set_xlabel("bin (transit at the centre)")
    fig.tight_layout()
    fig.savefig(FIG / "occlusion.png", dpi=150)
    plt.close(fig)


def make_all():
    style()
    FIG.mkdir(exist_ok=True)
    for f in (example_views, roc_pr, ablation, calibration, recall_vs_mes, training_curves, occlusion):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("skipped", f.__name__, repr(e))
