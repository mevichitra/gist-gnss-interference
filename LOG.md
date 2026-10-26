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

## 2026-10-27 — Phase 3
- Wind: naive wind triangle from forward-filled tas/heading is useless (p99 several hundred kt in controls). Fixed by using readsb ws/wd only on points carrying a *fresh* details object, excluding turning steps (track range ≥ 5°) and taking consensus per H3-res3 cell × 5 min × 6000 ft band (≥ 3 trusted aircraft). Median deviation 4–9 kt; still heavy-tailed over CUS → wind is a model feature, **not** an LF.
- Spoof-sensitive features validated on 2026-10-25 (rate per 10k steps with NIC ≥ 7):
  - |alt_geom − alt_baro − neighbour median| > 1000 ft: INPK 23.2, MEA 7.7, EMED 1.4, BLK 1.1 vs CUS 0.06, CEU 0.02, AUS 0.
  - Position cluster (≥ 3 aircraft in same 0.005° cell within 60 s, ≥ 5000 ft altitude spread): MEA 11.2, INPK 1.8 vs controls ≈ 0. (0.05° grid gave false clusters at airports — stacked arrivals.)
  - Position jumps > 20 km with NIC ≥ 7: 0 in all controls.
- LFs: 10 defined, `mlat` held out for evaluation. Region names never used inside LFs.
- Gate 3: coverage 77–92 % in all regions ✓; CUS 99.9 % / AUS 99.6 % normal ✓; **CEU 98.7 % normal** — 1.2 % jammed traced to (a) 14 long-haul widebodies with all-day degraded NIC (post-interference carry-over) and (b) a cluster at 48–50°N 13–14°E. Not tuned away. **Decision: CEU is reported as "low-interference Europe", clean controls = CUS + AUS.**
- Label-model LF accuracy estimates: nic_zero .998, cell_jam .995, nic_drop .94, gdiff .80, teleport .79, pos_loss .68, cluster .30 (tiny coverage, no co-firing LF).

## 2026-10-27 — Phase 4
- Simulator injects at step-feature level and recomputes neighbour/cluster features afterwards. Jammer: Friis + 4/3-Earth LOS + elevation-dependent antenna gain; per-receiver tolerance N(−115 dBW, 3 dB); NIC 0 + probabilistic position loss when above tolerance, NIC 1–6 within 6 dB below. Spoofer: capture at margin > −5 dB, sticky capture, static/circle/drift fake trajectories, NIC forced ≥ 8, alt_geom offset, wind/gs consistent with the fake track.
- First EIRP prior (1–100 W) affected a median 0.4 % of aircraft per scene → unrealistic. Raised to 20–50 dBW jammers / 10–40 dBW spoofers (Liu: ~1 kW for KDEN): median 41 % affected (q10 5 %, q90 82 %).
- NLS bugs found & fixed: EIRP grid too low (local minima), and per-window median positions (aircraft move ~140 km / window). Per-step observations fixed it.
- Results on 37 simulated jammer scenes (realistic noise): NLS median 16.1 km (p95 46.9), centroid 97.6 km, horizon 114.3 km.
- **Gate 4 ✓:** noise-free ground jammers (σ_tol = 0, h = 10 m): NLS median 2.4 km (11/12 < 6 km).
- Infra: GitHub release downloads stall → per-part resumable curl with stall timeout. Machine has 8 GB RAM and ~5 GB swap in use when parse + analysis overlap → run heavy jobs sequentially, parse workers 4.

## 2026-10-27 — Phase 5 (in progress)
- **Label-model bug:** windows where all LFs abstain got the Dawid–Skene posterior of the all-abstain pattern, which favours SPOOFED (highest abstention class) → 12.8 % "spoofed" in AUS on 2026-01-01. Fix: uncovered windows are unlabelled (label −1, NaN probs). After fix, controls ≥ 99.6 % normal on both days.
- Download throttling: GitHub release assets dropped to bytes/s after ~7 GB (general bandwidth 25 MB/s). Fetcher backs off up to 10 min per stalled attempt; 1st-of-month days prioritised over 15ths.
- Localization head memory: 41×41 grid × stride 2 needed 2.3 GB autograd and took 17 min/step under swap. 21×21 grid (40 km cells, soft-argmax refinement), stride 4, ≤ 256 nodes: 0.36 GB, 0.05 s/step.
