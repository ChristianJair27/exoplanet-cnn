"""The full experiment behind the README: learning-rate choice on validation, then every ensemble.

``python -m exocnn.experiments`` (after ``download`` and ``preprocess``).
"""
from __future__ import annotations

import json

from .config import RUNS
from .dataset import Views
from .evaluate import evaluate_all
from .train import train_ensemble, train_one, train_scalars_gbm

LRS = (1e-5, 1e-4)


def choose_lr(name, v, epochs=30):
    """Seed-0 validation AUC for each candidate learning rate (the paper used 1e-5 after a Vizier search)."""
    scores = {}
    for lr in LRS:
        _, auc, ep, _ = train_one(name, v, seed=0, epochs=epochs, lr=lr, verbose=False)
        scores[lr] = auc
        print(f"  lr choice {name}: lr={lr:g} best val AUC {auc:.4f} (epoch {ep + 1})", flush=True)
    return max(scores, key=scores.get), scores


def run_all(seeds=5, epochs=60, small_seeds=3):
    v = Views()
    RUNS.mkdir(parents=True, exist_ok=True)
    lr_astro, s1 = choose_lr("astronet", v)
    lr_ext, s2 = choose_lr("extended", v)
    (RUNS / "lr_choice.json").write_text(json.dumps(
        {"astronet": {str(k): x for k, x in s1.items()}, "extended": {str(k): x for k, x in s2.items()},
         "chosen": {"astronet": lr_astro, "extended": lr_ext}}, indent=1))
    plan = [("astronet", lr_astro, seeds), ("extended", lr_ext, seeds), ("extended_noscalars", lr_ext, small_seeds),
            ("global", lr_astro, small_seeds), ("local", lr_astro, small_seeds)]
    for name, lr, n in plan:
        print(f"== {name} (lr {lr:g}, {n} seeds, {epochs} epochs)", flush=True)
        train_ensemble(name, n_seeds=n, epochs=epochs, lr=lr)
    print("== scalars_gbm", flush=True)
    train_scalars_gbm(n_seeds=seeds)
    for split in ("val", "test"):
        print(f"== metrics on {split}")
        evaluate_all(split=split)


if __name__ == "__main__":
    run_all()
