"""Read Kepler long-cadence FITS files and flatten the light curve with a BIC-chosen spline.

Follows Shallue & Vanderburg (2018, Sec. 3.2): PDCSAP flux; a basis spline divided out, fitted with the
transits masked and 3-sigma outliers iteratively removed; the break-point spacing chosen per star by the
Bayesian Information Criterion. One deliberate simplification: the spline is fitted once per star with
the transits of *every* TCE on that star masked, instead of once per TCE.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

GAP_DAYS = 0.75                                  # split segments at gaps longer than this
BKSPACES = np.logspace(np.log10(0.5), np.log10(20.0), 20)   # candidate break-point spacings [d]


def read_kepler(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all quarters: (time [BKJD], flux normalised to median 1 per quarter)."""
    from astropy.io import fits

    ts, fs = [], []
    for f in sorted(Path(folder).glob("kplr*_llc.fits")):
        with fits.open(f, memmap=False) as h:
            d = h[1].data
            t = np.asarray(d["TIME"], float)
            y = np.asarray(d["PDCSAP_FLUX"], float)
            q = np.asarray(d["SAP_QUALITY"], int)
        ok = np.isfinite(t) & np.isfinite(y) & (q == 0)
        if ok.sum() < 10:
            continue
        ts.append(t[ok])
        fs.append(y[ok] / np.median(y[ok]))
    if not ts:
        return np.array([]), np.array([])
    t = np.concatenate(ts)
    order = np.argsort(t)
    return t[order], np.concatenate(fs)[order]


def transit_mask(time, period, t0, duration_days, width=1.5):
    """True for points within ``width`` x duration of a transit centre (total window 2 x width x d)."""
    phase = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
    return np.abs(phase) < width * duration_days


def segments(time, gap=GAP_DAYS):
    """Index slices of contiguous stretches of data."""
    cut = np.where(np.diff(time) > gap)[0] + 1
    edges = np.concatenate([[0], cut, [len(time)]])
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 0]


def _fit_segment(t, y, use, bkspace, maxiter=5, nsigma=3.0):
    """Iterative sigma-clipped cubic spline on one segment. Returns (model, n_params) or (None, 0)."""
    use = use.copy()
    model = None
    for _ in range(maxiter):
        tt = t[use]
        if len(tt) < 8:
            return None, 0
        knots = np.arange(tt[0] + bkspace, tt[-1] - bkspace / 2, bkspace)
        # every knot interval needs data (Schoenberg-Whitney); drop knots in gaps
        if len(knots):
            counts = np.histogram(tt, bins=np.concatenate([[tt[0]], knots, [tt[-1]]]))[0]
            if (counts < 4).any():
                keep = np.ones(len(knots), bool)
                for i in np.where(counts < 4)[0]:
                    keep[min(i, len(knots) - 1)] = False
                knots = knots[keep]
        try:
            spl = LSQUnivariateSpline(tt, y[use], knots, k=3)
        except Exception:
            return None, 0
        model = spl(t)
        resid = y - model
        sig = 1.4826 * np.median(np.abs(resid[use] - np.median(resid[use])))
        new = use & (np.abs(resid) < nsigma * sig) if sig > 0 else use
        if new.sum() == use.sum():
            break
        use = new
    return model, len(knots) + 4


def flatten(time, flux, mask_in_transit=None, bkspaces=BKSPACES):
    """Divide out the BIC-best spline. Returns (flat flux, chosen spacing [d]).

    ``mask_in_transit`` marks points that must not pull the spline (they are interpolated over).
    """
    if mask_in_transit is None:
        mask_in_transit = np.zeros(len(time), bool)
    segs = segments(time)
    # white-noise level from point-to-point scatter, used for the likelihood in the BIC
    sigma = 1.4826 * np.median(np.abs(np.diff(flux))) / np.sqrt(2)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.std(flux)
    best = (np.inf, None, np.nan)
    for bk in bkspaces:
        model = np.full(len(time), np.nan)
        npar = 0
        for s in segs:
            m, k = _fit_segment(time[s], flux[s], ~mask_in_transit[s], bk)
            if m is None:   # very short segment: fall back to its median
                m, k = np.full(s.stop - s.start, np.median(flux[s])), 1
            model[s] = m
            npar += k
        ok = ~mask_in_transit & np.isfinite(model)
        resid = (flux[ok] - model[ok]) / sigma
        resid = resid[np.abs(resid) < 3 * 1.4826 * np.median(np.abs(resid)) + 1e-12]  # ignore outliers
        bic = np.sum(resid ** 2) + npar * np.log(ok.sum())
        if bic < best[0]:
            best = (bic, model, bk)
    _, model, bk = best
    return flux / model, bk
