"""The DR24 TCE table: fetch it, pick a reproducible star subset, split train/val/test by star."""
from __future__ import annotations

import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .config import LABELS, SUBSET_CSV, TCE_COLUMNS, TCE_CSV

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch_table(path=TCE_CSV) -> pd.DataFrame:
    """Download the Q1-Q17 DR24 TCE table (about 20,000 rows, 1.5 MB) from the NASA Exoplanet Archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        query = f"select {','.join(TCE_COLUMNS)} from q1_q17_dr24_tce"
        url = TAP + "?" + urllib.parse.urlencode({"query": query, "format": "csv"})
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    return pd.read_csv(path)


def labelled(tces: pd.DataFrame) -> pd.DataFrame:
    """Keep TCEs with a PC/AFP/NTP Autovetter training label; add the binary target (PC = 1)."""
    d = tces[tces.av_training_set.isin(LABELS)].copy()
    d["label"] = (d.av_training_set == "PC").astype(int)
    return d.reset_index(drop=True)


def choose_stars(tces: pd.DataFrame, n_stars: int | None, seed: int = 42) -> np.ndarray:
    """A uniform random sample of host stars (all their labelled TCEs come along).

    Sampling stars, not TCEs, keeps the natural class mix of the catalogue and keeps every TCE of a
    star in the same split. ``n_stars=None`` keeps every star.
    """
    stars = np.sort(tces.kepid.unique())
    if n_stars is None or n_stars >= len(stars):
        return stars
    # prefix of one fixed permutation: a larger n_stars extends a smaller sample instead of replacing it
    return np.sort(np.random.default_rng(seed).permutation(stars)[:n_stars])


def split_by_star(tces: pd.DataFrame, frac=(0.8, 0.1, 0.1), seed: int = 0) -> pd.Series:
    """80/10/10 split as in Shallue & Vanderburg (2018), but grouped by star.

    The paper split TCEs at random; grouping by KIC prevents two TCEs of one eclipsing binary (often the
    primary and the secondary eclipse, or period aliases) from landing on both sides of the split.
    """
    stars = np.sort(tces.kepid.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(stars)
    n_train = int(round(frac[0] * len(stars)))
    n_val = int(round(frac[1] * len(stars)))
    which = {s: "train" for s in stars[:n_train]}
    which.update({s: "val" for s in stars[n_train:n_train + n_val]})
    which.update({s: "test" for s in stars[n_train + n_val:]})
    return tces.kepid.map(which)


def make_subset(n_stars: int | None, seed: int = 42, path=SUBSET_CSV) -> pd.DataFrame:
    """Labelled TCEs of ``n_stars`` random stars, with a ``split`` column; written to ``path``."""
    tces = labelled(fetch_table())
    keep = choose_stars(tces, n_stars, seed)
    sub = tces[tces.kepid.isin(keep)].copy()
    sub["split"] = split_by_star(sub)
    # other TCEs on the same star (any label, including UNK) are masked before detrending
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub
