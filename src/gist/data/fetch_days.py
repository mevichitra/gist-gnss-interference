"""Download → parse → delete raw, for a list of days. Resumable; logs to data_store/fetch.log."""

from __future__ import annotations

import shutil
import sys
import time
import traceback
from datetime import date
from pathlib import Path

from gist.data.download import download_day, free_gb
from gist.data.parse import parse_day

POINTS = Path("data_store/points")
LOG = Path("data_store/fetch.log")

DEV_DAYS = ["2026-10-25", "2026-01-01", "2025-07-01", "2025-01-01", "2024-07-01", "2024-01-01"]


def all_days() -> list[str]:
    days = []
    for y in (2024, 2025, 2026):
        for m in range(1, 13):
            for d in (1, 15):
                day = date(y, m, d)
                if day <= date(2026, 10, 25):
                    days.append(day.isoformat())
    return DEV_DAYS + [d for d in days if d not in DEV_DAYS]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def fetch(day_iso: str) -> None:
    out = POINTS / day_iso
    if (out / "_SUCCESS").exists():
        return
    day = date.fromisoformat(day_iso)
    t0 = time.time()
    raw = download_day(day)
    n = parse_day(raw, out)
    (out / "_SUCCESS").write_text(str(n))
    shutil.rmtree(raw)
    log(f"{day_iso} ok rows={n} {time.time() - t0:.0f}s free={free_gb():.0f}GB")


def main(days: list[str]) -> None:
    for d in days:
        try:
            fetch(d)
        except Exception as e:  # keep going; failures are listed in the log
            log(f"{d} FAILED {type(e).__name__}: {e}")
            traceback.print_exc()
            if "disk" in str(e):
                break


if __name__ == "__main__":
    main(sys.argv[1:] or all_days())
