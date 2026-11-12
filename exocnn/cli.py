"""Command line: ``python -m exocnn <step>`` (or the ``exocnn`` script after ``pip install -e .``)."""
from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="exocnn", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="TCE table, star subset and Kepler light curves from MAST")
    d.add_argument("--stars", type=int, default=1900, help="number of host stars (0 = all 9,865)")
    d.add_argument("--workers", type=int, default=6)

    pp = sub.add_parser("preprocess", help="detrend, fold and bin every TCE into its views")
    pp.add_argument("--workers", type=int, default=6)

    t = sub.add_parser("train", help="train an ensemble")
    t.add_argument("--model", choices=["astronet", "extended", "extended_noscalars", "global", "local", "scalars_gbm"],
                   default="astronet")
    t.add_argument("--seeds", type=int, default=5)
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--lr", type=float, default=None)

    e = sub.add_parser("evaluate", help="metrics of every trained ensemble on the test split")
    e.add_argument("--split", default="test")

    sub.add_parser("figures", help="write the figures to figures/")

    a = p.parse_args(argv)
    if a.cmd == "download":
        from .download import download_many
        from .tce import make_subset
        sub_df = make_subset(None if a.stars == 0 else a.stars)
        print(f"{len(sub_df)} labelled TCEs on {sub_df.kepid.nunique()} stars")
        download_many(sub_df.kepid.unique(), workers=a.workers)
    elif a.cmd == "preprocess":
        from .dataset import build_views
        build_views(workers=a.workers)
    elif a.cmd == "train":
        from .train import train_ensemble, train_scalars_gbm
        if a.model == "scalars_gbm":
            train_scalars_gbm(n_seeds=a.seeds)
        else:
            train_ensemble(a.model, n_seeds=a.seeds, epochs=a.epochs, lr=a.lr)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate_all
        evaluate_all(split=a.split)
    elif a.cmd == "figures":
        from .figures import make_all
        make_all()


if __name__ == "__main__":
    main()
