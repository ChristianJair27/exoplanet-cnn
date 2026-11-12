"""Turn the downloaded light curves into model inputs (views + scalar features), and load them for training."""
from __future__ import annotations

import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import CACHE_DIR, LC_DIR, SUBSET_CSV, TCE_CSV, VIEWS_NPZ
from .lightcurve import flatten, read_kepler, transit_mask
from .views import all_views

VIEW_KEYS = ("global", "local", "odd", "even", "secondary")

#: Scalar features (DR24 DV quantities). Names map to a transform in ``scalar_features``. The DR24 table
#: has no values for wst_depth, tce_albedo_stat or tce_ptemp_stat (DR25 quantities), so they are not used.
SCALAR_NAMES = [
    "log_period", "log_duration_h", "log_depth_ppm", "log_prad", "impact", "log_model_snr", "log_mes",
    "log_ntransits", "log_oddeven_stat", "log_secondary_mes", "kic_offset_sigma", "oot_offset_sigma", "log_boot_fap",
    "star_teff", "star_logg", "log_star_radius",
]


def scalar_features(t: pd.DataFrame) -> np.ndarray:
    """Raw (unstandardised) scalar features; NaN where the catalogue has no value."""
    lg = lambda x: np.log10(np.clip(np.asarray(x, float), 1e-12, None))
    f = {
        "log_period": lg(t.tce_period),
        "log_duration_h": lg(t.tce_duration),
        "log_depth_ppm": lg(t.tce_depth),
        "log_prad": lg(t.tce_prad),
        "impact": np.clip(t.tce_impact, 0, 2),
        "log_model_snr": lg(t.tce_model_snr),
        "log_mes": lg(t.tce_max_mult_ev),
        "log_ntransits": lg(t.tce_num_transits),
        "log_oddeven_stat": np.log10(1 + np.clip(t.tce_bin_oedp_stat, 0, None)),
        "log_secondary_mes": np.sign(t.tce_maxmes) * np.log10(1 + np.abs(t.tce_maxmes)),
        "kic_offset_sigma": np.clip(t.tce_dikco_msky / t.tce_dikco_msky_err, 0, 50),
        "oot_offset_sigma": np.clip(t.tce_dicco_msky / t.tce_dicco_msky_err, 0, 50),
        "log_boot_fap": np.clip(lg(t.boot_fap), -40, 0),
        "star_teff": t.tce_steff / 1000.0,
        "star_logg": t.tce_slogg,
        "log_star_radius": lg(t.tce_sradius),
    }
    return np.column_stack([np.asarray(f[k], float) for k in SCALAR_NAMES]).astype(np.float32)


def commensurate(p1: float, p2: float, max_ratio: int = 10, tol: float = 0.01) -> bool:
    """True if the periods are in an integer or half-integer ratio (aliases, EB primary/secondary pairs)."""
    r = max(p1, p2) / min(p1, p2)
    for k in range(1, 2 * max_ratio + 1):
        if abs(r - k / 2) < tol * r:
            return True
    return False


def process_star(kepid: int, star_tces: pd.DataFrame, all_tces: pd.DataFrame) -> list[dict]:
    """Views for every labelled TCE of one star. ``all_tces`` holds every TCE on the star (any label)."""
    folder = LC_DIR / f"{int(kepid):09d}"
    cache = CACHE_DIR / f"{int(kepid):09d}.npz"
    time = None
    if cache.exists():
        try:
            z = np.load(cache)
            time, flat = z["time"], z["flat"]
        except Exception:      # truncated by an interrupted run: recompute
            time = None
    if time is None:
        time, flux = read_kepler(folder)
        if len(time) < 100:
            return []
        mask = np.zeros(len(time), bool)
        for r in all_tces.itertuples():
            mask |= transit_mask(time, r.tce_period, r.tce_time0bk, r.tce_duration / 24.0, width=1.0)
        flat, _ = flatten(time, flux, mask)
        ok = np.isfinite(flat)
        time, flat = time[ok], flat[ok]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez_compressed(tmp, time=time.astype(np.float64), flat=flat.astype(np.float32))
        tmp.replace(cache)     # atomic: an interrupted run never leaves a truncated cache

    out = []
    for r in star_tces.itertuples():
        keep = np.ones(len(time), bool)
        # remove the transits of the other TCEs on the star (S&V 2018, Sec. 3.2 removed other confirmed
        # planets), except those whose period is commensurate with this one: for an eclipsing binary they
        # are the secondary eclipse or a period alias of the same object, i.e. the evidence itself
        for o in all_tces.itertuples():
            if o.tce_plnt_num != r.tce_plnt_num and not commensurate(o.tce_period, r.tce_period):
                keep &= ~transit_mask(time, o.tce_period, o.tce_time0bk, o.tce_duration / 24.0)
        v = all_views(time[keep], flat[keep], r.tce_period, r.tce_time0bk, r.tce_duration / 24.0,
                      secondary_phase=r.tce_maxmesd)
        v["kepid"], v["plnt"] = int(r.kepid), int(r.tce_plnt_num)
        out.append(v)
    return out


def _star_job(args):
    kepid, star_tces, all_tces = args
    try:
        return kepid, process_star(kepid, star_tces, all_tces), None
    except Exception as e:  # report, do not kill the pool
        return kepid, [], repr(e)


def build_views(workers: int = 6, path=VIEWS_NPZ) -> None:
    sub = pd.read_csv(SUBSET_CSV)
    full = pd.read_csv(TCE_CSV)
    stars = [k for k in sub.kepid.unique() if (LC_DIR / f"{int(k):09d}" / "DONE").exists()]
    print(f"{len(stars)} of {sub.kepid.nunique()} stars downloaded")
    jobs = [(k, sub[sub.kepid == k], full[full.kepid == k]) for k in stars]
    rows, t0 = [], _time.time()
    with ProcessPoolExecutor(workers) as pool:
        futs = [pool.submit(_star_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            kepid, res, err = f.result()
            if err:
                print(f"  KIC {kepid}: {err}")
            rows.extend(res)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} stars, {len(rows)} TCEs, {_time.time() - t0:.0f} s", flush=True)

    key = {(v["kepid"], v["plnt"]): v for v in rows}
    sub = sub[[(k, p) in key for k, p in zip(sub.kepid, sub.tce_plnt_num)]].reset_index(drop=True)
    arrays = {k: np.stack([key[(a, b)][k] for a, b in zip(sub.kepid, sub.tce_plnt_num)]) for k in VIEW_KEYS}
    np.savez_compressed(
        path, **arrays, scalars=scalar_features(sub), label=sub.label.values.astype(np.int64),
        split=sub.split.values.astype(str), kepid=sub.kepid.values, plnt=sub.tce_plnt_num.values,
        av_training_set=sub.av_training_set.values.astype(str), mes=sub.tce_max_mult_ev.values,
        period=sub.tce_period.values, scalar_names=np.array(SCALAR_NAMES),
    )
    print(f"wrote {path}: {len(sub)} TCEs")


class Views:
    """The npz loaded in memory, with scalar features standardised on the training split."""

    def __init__(self, path=VIEWS_NPZ):
        z = np.load(path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        s = self.z["scalars"].astype(np.float64)
        train = self.z["split"] == "train"
        med = np.nan_to_num(np.nanmedian(np.where(np.isfinite(s[train]), s[train], np.nan), axis=0))
        s = np.where(np.isfinite(s), s, med)
        mu, sd = s[train].mean(0), s[train].std(0) + 1e-6
        self.z["scalars_std"] = ((s - mu) / sd).astype(np.float32)

    def index(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.z["split"] == split)

    def __getitem__(self, key):
        return self.z[key]
