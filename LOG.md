# Execution log

## 2026-10-27 — Phase 0
- Env: uv + Python 3.12, torch 2.14.1 (MPS available), polars 2.1, h3 4.5. Machine: Apple M3, 8 GB RAM.
- No PyTorch Geometric; graph attention will be plain PyTorch.

## 2026-10-27 — Phase 1
- **Data source switch:** ADSBx samples are evaluation-only and their policy forbids derivative works → not used. adsb.lol globe_history is ODbL-1.0 (2023–2026), daily, ~3.6 GB/day split tar, readsb trace format + `mlatonly` release. Primary source now.
- Consequence: no 1st-of-month restriction, so real events on any day ≥ 2023 are usable. KDEN 2022 still needs OpenSky.
- Gap check: searched 2025–2026 literature; found classical localization (Liu 2022/2025, Schäfer "Jammer on the Horizon"), DLR space-ADS-B detection (SHaRC 2026), Aireon white paper, ERAU/UFSM ML features. No work combining DL jam-vs-spoof classification + cross-aircraft GNN + learned/physics localization with weak supervision → gap holds.
- Simulator constants from Liu 2022: tolerable jamming power at aircraft ≈ −115 dBW (C/N0 threshold 25 dB-Hz), Friis at L1 with unit gains as baseline.
- Primary venue: JOAS (rolling). See paper/VENUES.md.

## 2026-10-27 — Phase 2
- Trace files: 67,333 aircraft/day (2026-10-25). Point sources: adsb_icao 95 %, adsr 2 %, **mlat 2 %**, tisb <1 %. Median point spacing 4.3 s → keep raw points, bin to 30 s steps in features.
- Parse: 114 s/day with 6 workers, 340 MB RSS, 359 MB Parquet (region+control boxes, airborne only). Gate 2 passes (NIC present > 99.8 % everywhere; BAL 99.97 %).
- First signal check (share of points with NIC ≤ 6): BAL 15.8 %, MEA 9.1 %, BLK 8.5 %, EMED 4.7 %, INPK 4.7 %, CEU 4.1 %, CUS 5.4 %, AUS 1.7 %. BAL alt_geom availability drops to 88 % (vs 97–99 % elsewhere).
- Day list: 1st & 15th of each month 2024-01 → 2026-10 (68 days), dev days first. Fetch runs in background (`gist.data.fetch_days`), resumable, raw deleted after parse.
- Traces carry no receiver location (rr_lat/lon) → LF_horizon dropped; independent-position check will use adsb.lol `mlatonly` releases instead (Phase 3).
