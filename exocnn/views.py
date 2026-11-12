"""Phase folding and median binning into fixed-length views.

Base views as in Shallue & Vanderburg (2018, Sec. 3.3 and 5.2):
  * global: whole orbit, 2001 bins, lambda = delta = P/2001;
  * local: +-4 transit durations, 201 bins, lambda = 8d/201, delta = 0.16 d;
  both normalised to median 0 and minimum -1.

Diagnostic views added after ExoMiner (Valizadegan et al. 2022, Sec. 3.3):
  * odd and even local views (transits 1, 3, 5... and 2, 4, 6...);
  * a local view centred on the strongest secondary event (DR24 ``tce_maxmesd``).
  These are scaled with the *primary* local view's median and depth, so a depth difference between odd and
  even transits, or a secondary eclipse, survives the normalisation instead of being stretched to -1.
"""
from __future__ import annotations

import numpy as np

from .config import GLOBAL_BINS, LOCAL_BINS, LOCAL_DELTA_FRAC, LOCAL_K


def fold(time, period, t0):
    """Phase in days, transit at 0, range [-P/2, P/2)."""
    return np.mod(time - t0 + 0.5 * period, period) - 0.5 * period


def median_bin(x, y, nbins, xmin, xmax, width):
    """Median of y in ``nbins`` windows of ``width`` whose centres span [xmin + width/2, xmax - width/2].

    Empty windows are linearly interpolated from their neighbours (NaN if the whole view is empty).
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    if nbins == 1:
        centres = np.array([(xmin + xmax) / 2])
    else:
        centres = np.linspace(xmin + width / 2, xmax - width / 2, nbins)
    lo = np.searchsorted(x, centres - width / 2, side="left")
    hi = np.searchsorted(x, centres + width / 2, side="right")
    out = np.full(nbins, np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = np.median(y[a:b])
    good = np.isfinite(out)
    if good.sum() == 0:
        return out
    if not good.all():
        out[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), out[good])
    return out


def global_view(phase, flux, period, nbins=GLOBAL_BINS):
    w = period / nbins
    return median_bin(phase, flux, nbins, -period / 2, period / 2, w)


def local_view(phase, flux, duration, nbins=LOCAL_BINS, k=LOCAL_K, delta_frac=LOCAL_DELTA_FRAC,
               centre=0.0, period=None):
    """Window of +-k durations around ``centre`` (days of phase). ``period`` wraps the phase if given."""
    x = phase - centre
    y = flux
    half = k * duration
    if period is not None:
        x = np.mod(x + 0.5 * period, period) - 0.5 * period
        if half + delta_frac * duration > 0.5 * period:     # window wider than the orbit: tile the phase
            n = int(np.ceil((half + delta_frac * duration) / period))
            x = np.concatenate([x + j * period for j in range(-n, n + 1)])
            y = np.tile(flux, 2 * n + 1)
    sel = np.abs(x) <= half + delta_frac * duration
    return median_bin(x[sel], y[sel], nbins, -half, half, delta_frac * duration)


def normalise(view, ref=None):
    """Median 0, minimum -1 (S&V 2018). With ``ref`` = (median, depth), reuse another view's scale."""
    v = np.asarray(view, float)
    if ref is None:
        med = np.nanmedian(v)
        depth = med - np.nanmin(v)
    else:
        med, depth = ref
    if not np.isfinite(depth) or depth <= 0:
        depth = 1.0
    return (v - med) / depth, (med, depth)


def all_views(time, flux, period, t0, duration, secondary_phase=np.nan):
    """Every view for one TCE. ``duration`` and ``secondary_phase`` in days. Returns a dict of arrays."""
    phase = fold(time, period, t0)
    g, _ = normalise(global_view(phase, flux, period))
    loc_raw = local_view(phase, flux, duration, period=period)
    loc, ref = normalise(loc_raw)

    n = np.round((time - t0) / period).astype(int)
    odd = (n % 2) == 1
    odd_v, _ = normalise(local_view(phase[odd], flux[odd], duration, period=period), ref)
    even_v, _ = normalise(local_view(phase[~odd], flux[~odd], duration, period=period), ref)

    sec_c = secondary_phase if np.isfinite(secondary_phase) else period / 2
    sec_v, _ = normalise(local_view(phase, flux, duration, centre=sec_c, period=period), ref)
    # the secondary window shares the out-of-transit level of the primary; centre it on its own median
    sec_v = sec_v - np.nanmedian(sec_v)

    out = {"global": g, "local": loc, "odd": odd_v, "even": even_v, "secondary": sec_v}
    for key, v in out.items():
        out[key] = np.nan_to_num(v, nan=0.0).astype(np.float32)
    return out
