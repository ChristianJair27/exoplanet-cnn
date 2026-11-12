"""Resumable, parallel, polite download of Kepler long-cadence light curves from MAST.

Files come from the public directory tree
``https://archive.stsci.edu/missions/kepler/lightcurves/<kic[:4]>/<kic>/kplr<kic>-<stamp>_llc.fits``.
A star is complete when ``<kic>/DONE`` exists; partial files are written as ``.part`` and renamed.
"""
from __future__ import annotations

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import LC_DIR

BASE = "https://archive.stsci.edu/missions/kepler/lightcurves"
UA = {"User-Agent": "exoplanet-cnn/0.1 (research; python urllib)"}


def _get(url: str, tries: int = 6, timeout: int = 60) -> bytes:
    delay = 2.0
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


def star_url(kic: int) -> str:
    k = f"{int(kic):09d}"
    return f"{BASE}/{k[:4]}/{k}/"


def download_star(kic: int, root: Path = LC_DIR) -> int:
    """Download every long-cadence quarter of one star; returns bytes fetched (0 if already done)."""
    folder = root / f"{int(kic):09d}"
    if (folder / "DONE").exists():
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    listing = _get(star_url(kic)).decode("latin-1")
    names = sorted(set(re.findall(r"kplr\d{9}-\d{13}_llc\.fits", listing)))
    total = 0
    for name in names:
        dst = folder / name
        if dst.exists():
            continue
        data = _get(star_url(kic) + name)
        part = dst.with_suffix(".part")
        part.write_bytes(data)
        part.replace(dst)
        total += len(data)
    (folder / "DONE").write_text(f"{len(names)} files\n")
    return total


def download_many(kics, workers: int = 6, root: Path = LC_DIR, log_every: int = 50) -> None:
    kics = [int(k) for k in kics]
    todo = [k for k in kics if not (root / f"{k:09d}" / "DONE").exists()]
    print(f"{len(kics)} stars, {len(kics) - len(todo)} already complete, {len(todo)} to fetch")
    t0, done, nbytes, failed = time.time(), 0, 0, []
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(download_star, k, root): k for k in todo}
        for f in as_completed(futs):
            done += 1
            try:
                nbytes += f.result()
            except Exception as e:  # keep going; a rerun resumes
                failed.append(futs[f])
                print(f"  KIC {futs[f]} failed: {e}")
            if done % log_every == 0 or done == len(todo):
                dt = time.time() - t0
                print(f"  {done}/{len(todo)} stars, {nbytes / 1e9:.2f} GB, {nbytes / 1e6 / dt:.1f} MB/s", flush=True)
    if failed:
        print(f"{len(failed)} stars failed; rerun the same command to resume")
