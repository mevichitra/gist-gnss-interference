"""Download one day of adsb.lol globe_history (ODbL) and extract it.

Each day is a GitHub release with a split tar (``.tar.aa``, ``.tar.ab``...). We pick the
larger of prod/staging (the README says to ignore the much smaller one), stream the parts
through ``tar`` so the tarball never sits on disk, and return the extracted directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from datetime import date
from pathlib import Path

import httpx

RAW = Path("data_store/raw")
API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
MIN_FREE_GB = 40  # PLAN §5.1 hard disk cap


def _release_assets(day: date, flavour: str) -> list[dict]:
    tag = f"v{day:%Y.%m.%d}-planes-readsb-{flavour}-0"
    r = httpx.get(API.format(year=day.year, tag=tag), timeout=30, follow_redirects=True)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return sorted(r.json()["assets"], key=lambda a: a["name"])


def pick_release(day: date) -> list[dict]:
    """Return the assets of the larger of the prod/staging releases for ``day``."""
    options = [_release_assets(day, f) for f in ("prod", "staging")]
    options = [o for o in options if o]
    if not options:
        raise FileNotFoundError(f"no adsb.lol release for {day}")
    return max(options, key=lambda assets: sum(a["size"] for a in assets))


def free_gb(path: Path = Path(".")) -> float:
    return shutil.disk_usage(path).free / 1e9


def download_day(day: date, out_root: Path = RAW) -> Path:
    out = out_root / f"{day:%Y.%m.%d}"
    if (out / ".done").exists():
        return out
    if free_gb() < MIN_FREE_GB:
        raise RuntimeError(f"free disk {free_gb():.0f} GB < {MIN_FREE_GB} GB cap")
    assets = pick_release(day)
    out.mkdir(parents=True, exist_ok=True)
    urls = " ".join(f"'{a['browser_download_url']}'" for a in assets)
    # curl concatenates the parts in order into tar's stdin.
    cmd = f"curl -sSfL --retry 5 {urls} | tar -xf - -C '{out}'"
    subprocess.run(cmd, shell=True, check=True)
    (out / ".done").touch()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day", type=date.fromisoformat)
    print(download_day(ap.parse_args().day))


if __name__ == "__main__":
    main()
