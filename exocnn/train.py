"""Training: Adam, binary cross-entropy, random horizontal reflection, batch 64 (S&V 2018, Sec. 4.2 and 5.2).

Each seed keeps the weights of the epoch with the best validation ROC AUC (the paper tuned on validation
AUC). An ensemble is the mean probability of its seeds (S&V used 10 copies, Sec. 4.3).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import RUNS, VIEWS_NPZ
from .dataset import VIEW_KEYS, Views
from .model import build

LR = {"astronet": 1e-5, "global": 1e-5, "local": 1e-5, "extended": 1e-5, "extended_noscalars": 1e-5}


def to_batch(v: Views, idx, device, flip=None) -> dict:
    b = {k: torch.from_numpy(v[k][idx]).to(device) for k in VIEW_KEYS}
    if flip is not None and flip.any():
        f = torch.from_numpy(flip).to(device)
        for k in VIEW_KEYS:
            b[k] = torch.where(f[:, None], b[k].flip(1), b[k])
    b["scalars"] = torch.from_numpy(v["scalars_std"][idx]).to(device)
    return b


@torch.no_grad()
def predict(model, v: Views, idx, device, bs=512) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(to_batch(v, idx[i:i + bs], device))).cpu().numpy()
           for i in range(0, len(idx), bs)]
    return np.concatenate(out) if out else np.array([])


def train_one(name, v: Views, seed, epochs=50, lr=None, bs=64, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(name, n_scalars=v["scalars"].shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr or LR[name], betas=(0.9, 0.999), eps=1e-8)
    lossf = torch.nn.BCEWithLogitsLoss()
    tr, va = v.index("train"), v.index("val")
    y = torch.from_numpy(v["label"].astype(np.float32))
    best = (-1.0, None, -1)
    history = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            b = to_batch(v, idx, device, flip=rng.random(len(idx)) < 0.5)
            loss = lossf(model(b), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        auc = roc_auc_score(v["label"][va], predict(model, v, va, device))
        history.append((ep, tot / len(tr), auc))
        if auc > best[0]:
            best = (auc, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, ep)
        if verbose and (ep % 10 == 9 or ep == epochs - 1):
            print(f"    [{name} s{seed}] epoch {ep + 1}: loss {tot / len(tr):.4f} val AUC {auc:.4f}", flush=True)
    model.load_state_dict(best[1])
    return model, best[0], best[2], history


def train_ensemble(name, n_seeds=5, epochs=50, lr=None, path=VIEWS_NPZ, tag=None):
    v = Views(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = RUNS / (tag or name)
    out.mkdir(parents=True, exist_ok=True)
    preds = {"val": [], "test": []}
    info = []
    for seed in range(n_seeds):
        t0 = time.time()
        model, auc, ep, hist = train_one(name, v, seed, epochs, lr, device=device)
        torch.save(model.state_dict(), out / f"seed{seed}.pt")
        pd.DataFrame(hist, columns=["epoch", "train_loss", "val_auc"]).to_csv(out / f"history{seed}.csv", index=False)
        for s in preds:
            preds[s].append(predict(model, v, v.index(s), device))
        info.append({"seed": seed, "best_val_auc": auc, "best_epoch": ep + 1, "minutes": (time.time() - t0) / 60})
        print(f"  seed {seed}: best val AUC {auc:.4f} at epoch {ep + 1} ({info[-1]['minutes']:.1f} min)")
    for s, p in preds.items():
        idx = v.index(s)
        df = pd.DataFrame({"kepid": v["kepid"][idx], "plnt": v["plnt"][idx], "label": v["label"][idx],
                           "av_training_set": v["av_training_set"][idx], "mes": v["mes"][idx]})
        for k, arr in enumerate(p):
            df[f"p{k}"] = arr
        df["p"] = np.mean(p, axis=0)
        df.to_csv(out / f"pred_{s}.csv", index=False)
    (out / "train_info.json").write_text(json.dumps(
        {"model": name, "lr": lr or LR[name], "epochs": epochs, "seeds": info}, indent=1))
    return out


def train_scalars_gbm(n_seeds=5, path=VIEWS_NPZ):
    """Baseline without any light-curve view: gradient-boosted trees on the DV scalar features only."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    v = Views(path)
    out = RUNS / "scalars_gbm"
    out.mkdir(parents=True, exist_ok=True)
    X, y = v["scalars_std"], v["label"]
    tr = np.concatenate([v.index("train")])
    preds = {"val": [], "test": []}
    for seed in range(n_seeds):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, early_stopping=True,
                                             validation_fraction=0.1, random_state=seed)
        clf.fit(X[tr], y[tr])
        for s in preds:
            preds[s].append(clf.predict_proba(X[v.index(s)])[:, 1])
    for s, p in preds.items():
        idx = v.index(s)
        df = pd.DataFrame({"kepid": v["kepid"][idx], "plnt": v["plnt"][idx], "label": v["label"][idx],
                           "av_training_set": v["av_training_set"][idx], "mes": v["mes"][idx]})
        for k, arr in enumerate(p):
            df[f"p{k}"] = arr
        df["p"] = np.mean(p, axis=0)
        df.to_csv(out / f"pred_{s}.csv", index=False)
    return out
