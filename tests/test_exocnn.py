"""Offline tests on synthetic light curves (no downloads)."""
import numpy as np
import pandas as pd
import pytest
import torch

from exocnn import views
from exocnn.dataset import SCALAR_NAMES, VIEW_KEYS, Views, commensurate, scalar_features
from exocnn.evaluate import ece, metrics, recall_at_precision
from exocnn.lightcurve import flatten, segments, transit_mask
from exocnn.model import build
from exocnn.tce import choose_stars, split_by_star


def synthetic(period=3.7, t0=1.3, dur=0.15, depth=0.01, depth_odd=None, secondary=0.0, days=200,
              noise=2e-4, trend=0.003, seed=0):
    """Kepler-like cadence (29.4 min) with a box transit, optional odd/even difference and secondary."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 29.4 / 1440)
    t = t[(t % 90) > 1.0]                                  # quarterly gaps
    f = 1 + trend * np.sin(2 * np.pi * t / 23.0) + rng.normal(0, noise, len(t))
    phase = views.fold(t, period, t0)
    n = np.round((t - t0) / period).astype(int)
    intr = np.abs(phase) < dur / 2
    d = np.where((n % 2 == 1) & (depth_odd is not None), depth_odd or depth, depth)
    f[intr] -= d[intr]
    if secondary:
        sec = np.abs(views.fold(t, period, t0 + period / 2)) < dur / 2
        f[sec] -= secondary
    return t, f


def test_fold_range_and_centre():
    t = np.linspace(0, 100, 10001)
    ph = views.fold(t, 7.0, 3.0)
    assert ph.min() >= -3.5 and ph.max() < 3.5
    assert abs(views.fold(np.array([3.0, 10.0, 17.0]), 7.0, 3.0)).max() < 1e-9


def test_median_bin_partition_and_fill():
    x = np.linspace(-1, 1, 1000)
    y = x ** 2
    b = views.median_bin(x, y, 21, -1, 1, 2 / 21)
    assert b.shape == (21,) and np.isfinite(b).all()
    assert b[10] < 0.01 and b[0] > 0.8
    gap = views.median_bin(x[np.abs(x) > 0.3], y[np.abs(x) > 0.3], 21, -1, 1, 2 / 21)
    assert np.isfinite(gap).all()                        # empty bins interpolated


def test_segments_split_on_gaps():
    t = np.concatenate([np.arange(0, 10, 0.02), np.arange(12, 20, 0.02)])
    s = segments(t)
    assert len(s) == 2 and s[0].stop == 500


def test_flatten_removes_trend_keeps_depth():
    t, f = synthetic()
    mask = transit_mask(t, 3.7, 1.3, 0.15)
    flat, bk = flatten(t, f, mask)
    out = ~mask
    assert np.std(flat[out]) < 4e-4                      # trend of 3e-3 removed down to the noise
    intr = np.abs(views.fold(t, 3.7, 1.3)) < 0.05
    assert abs((1 - np.median(flat[intr])) - 0.01) < 1e-3
    assert 0.5 <= bk <= 20


def test_views_shapes_and_normalisation():
    t, f = synthetic(trend=0)
    v = views.all_views(t, f, 3.7, 1.3, 0.15)
    assert v["global"].shape == (2001,) and v["local"].shape == (201,)
    for k in ("odd", "even", "secondary"):
        assert v[k].shape == (201,)
    assert v["local"].min() == pytest.approx(-1, abs=1e-6)
    assert abs(np.median(v["local"])) < 1e-6
    assert abs(v["secondary"].min()) < 0.2               # no secondary eclipse injected


def test_odd_even_and_secondary_are_preserved():
    t, f = synthetic(depth=0.02, depth_odd=0.01, secondary=0.005, trend=0)
    v = views.all_views(t, f, 3.7, 1.3, 0.15)
    assert v["odd"].min() / v["even"].min() == pytest.approx(0.5, abs=0.1)   # shared scale keeps the ratio
    t2, f2 = synthetic(depth=0.02, secondary=0.005, trend=0)
    v2 = views.all_views(t2, f2, 3.7, 1.3, 0.15, secondary_phase=3.7 / 2)
    assert v2["secondary"].min() < -0.2                   # secondary ~ 0.005/0.02 of the primary scale


@pytest.mark.parametrize("name", ["astronet", "global", "local", "extended", "extended_noscalars"])
def test_models_forward(name):
    m = build(name, n_scalars=len(SCALAR_NAMES))
    b = {k: torch.zeros(3, 2001 if k == "global" else 201) for k in VIEW_KEYS}
    b["scalars"] = torch.zeros(3, len(SCALAR_NAMES))
    assert m(b).shape == (3,)


def test_astronet_matches_paper_columns():
    m = build("astronet")
    convs = [l for l in m.g if isinstance(l, torch.nn.Conv1d)]
    assert [c.out_channels for c in convs] == [16, 16, 32, 32, 64, 64, 128, 128, 256, 256]
    fcs = [l for l in m.head if isinstance(l, torch.nn.Linear)]
    assert [l.out_features for l in fcs] == [512, 512, 512, 512, 1]


def test_split_by_star_is_disjoint():
    df = pd.DataFrame({"kepid": np.repeat(np.arange(200), 2), "label": 0})
    df["split"] = split_by_star(df)
    assert df.groupby("kepid").split.nunique().max() == 1
    assert set(df.split) == {"train", "val", "test"}


def test_choose_stars_is_nested():
    df = pd.DataFrame({"kepid": np.arange(1000)})
    small, big = choose_stars(df, 100), choose_stars(df, 300)
    assert set(small) <= set(big)


def test_metrics_basics():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    df = pd.DataFrame({"kepid": np.arange(6), "label": y, "p": p, "p0": p})
    m = metrics(df)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0
    assert recall_at_precision(y, p, 0.99) == 1.0
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0


def test_scalar_features_shape():
    row = {c: 1.0 for c in ["tce_period", "tce_duration", "tce_depth", "tce_prad", "tce_impact",
                            "tce_model_snr", "tce_max_mult_ev", "tce_num_transits", "tce_bin_oedp_stat",
                            "wst_depth", "tce_maxmes", "tce_albedo_stat", "tce_ptemp_stat",
                            "tce_dikco_msky", "tce_dikco_msky_err", "tce_dicco_msky", "tce_dicco_msky_err",
                            "boot_fap", "tce_steff", "tce_slogg", "tce_sradius"]}
    assert scalar_features(pd.DataFrame([row, row])).shape == (2, len(SCALAR_NAMES))


def test_training_runs_on_tiny_data(tmp_path):
    from exocnn.train import train_one
    rng = np.random.default_rng(0)
    n = 40
    label = np.tile([0, 1], n // 2)
    arr = {k: rng.normal(0, 0.1, (n, 2001 if k == "global" else 201)).astype(np.float32) for k in VIEW_KEYS}
    arr["local"][label == 1, 95:106] -= 1                 # planets have a dip
    split = np.array(["train"] * 24 + ["val"] * 8 + ["test"] * 8)
    p = tmp_path / "v.npz"
    np.savez(p, **arr, scalars=rng.normal(size=(n, len(SCALAR_NAMES))).astype(np.float32), label=label,
             split=split, kepid=np.arange(n), plnt=np.ones(n, int), av_training_set=np.array(["PC"] * n),
             mes=np.ones(n), period=np.ones(n), scalar_names=np.array(SCALAR_NAMES))
    model, auc, ep, hist = train_one("local", Views(p), seed=0, epochs=2, lr=1e-3, device="cpu", verbose=False)
    assert len(hist) == 2 and 0 <= auc <= 1


def test_commensurate_periods():
    assert commensurate(3.0, 6.0) and commensurate(3.0, 3.01) and commensurate(2.0, 3.0)
    assert not commensurate(3.0, 4.37)


def test_local_view_wider_than_orbit_has_no_empty_edges():
    t, f = synthetic(period=0.6, t0=0.1, dur=0.2, depth=0.01, trend=0)
    v = views.local_view(views.fold(t, 0.6, 0.1), f, 0.2, period=0.6)
    assert np.isfinite(v).all() and np.std(v[:20]) > 0      # edges filled with real (tiled) data
