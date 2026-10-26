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
