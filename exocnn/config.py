"""Paths and constants shared by every step."""
import os
from pathlib import Path

#: Root for downloaded and derived data (never committed). Override with $EXOCNN_DATA.
DATA = Path(os.environ.get("EXOCNN_DATA", Path.home() / ".exocnn"))

TCE_CSV = DATA / "dr24_tce.csv"          # Kepler Q1-Q17 DR24 TCE table (NASA Exoplanet Archive)
SUBSET_CSV = DATA / "subset_tces.csv"    # TCEs of the downloaded stars, with split column
LC_DIR = DATA / "lightcurves"            # raw Kepler long-cadence FITS, one folder per KIC
CACHE_DIR = DATA / "cache"               # per-star detrended light curves (npz)
VIEWS_NPZ = DATA / "views.npz"           # all model inputs
RUNS = DATA / "runs"                     # trained weights and predictions

# Input representations (Shallue & Vanderburg 2018, Sec. 5.2)
GLOBAL_BINS = 2001                       # lambda = delta = P / 2001
LOCAL_BINS = 201                         # k = 4 durations each side, lambda = 2kd/201, delta = 0.16 d
LOCAL_K = 4.0
LOCAL_DELTA_FRAC = 0.16

LABELS = ("PC", "AFP", "NTP")            # av_training_set values used (UNK ignored)

TCE_COLUMNS = [
    "kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration", "tce_depth",
    "tce_model_snr", "tce_prad", "tce_impact", "tce_max_mult_ev", "tce_num_transits",
    "tce_bin_oedp_stat", "tce_maxmes", "tce_maxmesd", "wst_depth", "tce_albedo_stat",
    "tce_ptemp_stat", "tce_dikco_msky", "tce_dikco_msky_err", "tce_dicco_msky",
    "tce_dicco_msky_err", "boot_fap", "tce_steff", "tce_slogg", "tce_sradius", "av_training_set",
]
