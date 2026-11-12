"""Metrics on held-out predictions: ROC AUC (bootstrap CI over stars), average precision, accuracy,
precision and recall at 0.5, recall at fixed precision, Brier score and expected calibration error."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from .config import RUNS

RESULTS = Path(__file__).resolve().parents[1] / "results"


def recall_at_precision(y, p, target):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def bootstrap_auc(df, n=2000, seed=0, col="p"):
    """95 % interval of the AUC, resampling stars (TCEs of one star stay together)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("kepid")]
    ys = [g.label.values for g in groups]
    ps = [g[col].values for g in groups]
    aucs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([ys[i] for i in pick])
        p = np.concatenate([ps[i] for i in pick])
        if 0 < y.sum() < len(y):
            aucs.append(roc_auc_score(y, p))
    return np.percentile(aucs, [2.5, 97.5])


def metrics(df: pd.DataFrame, col="p") -> dict:
    y, p = df.label.values, df[col].values
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    lo, hi = bootstrap_auc(df, col=col)
    seeds = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
    seed_auc = [roc_auc_score(y, df[c]) for c in seeds]
    return {
        "n": len(y), "n_pc": int(y.sum()),
        "auc": roc_auc_score(y, p), "auc_lo": lo, "auc_hi": hi,
        "seed_auc_mean": float(np.mean(seed_auc)) if seed_auc else np.nan,
        "seed_auc_std": float(np.std(seed_auc)) if seed_auc else np.nan,
        "avg_precision": average_precision_score(y, p),
        "accuracy": (tp + tn) / len(y), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "recall_at_p90": recall_at_precision(y, p, 0.90), "recall_at_p95": recall_at_precision(y, p, 0.95),
        "recall_at_p99": recall_at_precision(y, p, 0.99),
        "brier": brier_score_loss(y, p), "ece": ece(y, p), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate_all(split="test", runs=RUNS, out=RESULTS) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(runs).iterdir()):
        f = d / f"pred_{split}.csv"
        if f.exists():
            rows.append({"model": d.name, **metrics(pd.read_csv(f))})
    res = pd.DataFrame(rows)
    out.mkdir(exist_ok=True)
    res.to_csv(out / f"metrics_{split}.csv", index=False)
    show = res[["model", "n", "n_pc", "auc", "auc_lo", "auc_hi", "seed_auc_mean", "avg_precision", "accuracy",
                "precision", "recall", "recall_at_p90", "recall_at_p95", "recall_at_p99", "brier", "ece"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return res
