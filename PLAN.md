# GIST — GNSS Interference Spatio-Temporal Network
### Detecting, classifying and localizing GPS jamming & spoofing from crowdsourced ADS-B

**Owner:** Vichitra · **Executor:** autonomous coding agent · **Purpose:** final-year Deep Learning project + paper submission
**Plan date:** 2026-10-26 · **Hardware:** Apple M3, 8 GB RAM, ~148 GB free disk, macOS

This document is written so an agent can execute it end to end without asking anything. Every phase has inputs, steps, outputs, and an **acceptance gate**. If a gate fails, follow the listed fallback; do not stop to ask unless the "Stop and ask" rules (§12) trigger.

---

## 1. One-paragraph pitch (use in abstract / report intro)

GNSS interference has become a daily hazard for civil aviation (Baltic, Eastern Mediterranean, Black Sea, Middle East, India–Pakistan border). Public monitors (e.g. gpsjam.org) threshold a single integrity field (NIC) per hexagon per day: they cannot separate **jamming** (signal denied) from **spoofing** (fake signal, aircraft *reports high integrity while lying*), they react at day granularity, and they do not locate the emitter. **GIST** treats every aircraft as a remote GNSS sensor. It (1) builds **physics-consistency features** that a spoofed receiver cannot fake (barometric vs geometric altitude, air-data vs GNSS kinematics, neighbor-consistent wind, MLAT cross-checks, receiver radio-horizon plausibility), (2) learns per-aircraft temporal representations with self-supervised pretraining, (3) shares evidence across nearby aircraft with a spatial graph-attention network, and (4) localizes the emitter with a **differentiable radio-horizon likelihood head** trained on synthetic jammers/spoofers injected into real traffic. Labels come from programmatic weak supervision, so no manual annotation is needed.

## 2. Contributions (paper claims — each must be backed by an experiment in §8)

1. **Weakly-supervised dataset** of jamming/spoofing labels over global ADS-B, built with a label model over physics-based labeling functions (released with code).
2. **Spoofing-aware features** that are invariant to the integrity fields a spoofer controls; show NIC-threshold methods miss spoofing and GIST does not.
3. **GIST architecture**: trajectory Transformer + spatial graph attention + 3-way classifier (normal / jammed / spoofed).
4. **Differentiable localization head** (radio-horizon + Friis likelihood) trained sim-to-real; evaluated on real events with known/approximate emitter locations.
5. **Minute-level detection latency** vs day-level public maps.

## 3. Prior work to position against (cite; agent re-verifies and expands in Phase 1)

- Liu, Lo, Blanch, Chen, Walter — *Locating GNSS Interference Sources using ADS-B with Non-linear Least Squares*, NAVIGATION 72(3), 2025 (Denver KDEN Jan-2022, ~4 km error). https://navi.ion.org/content/72/3/navi.716 — **classical baseline for localization; reimplement.**
- Liu et al., ION ITM 2022 (ADS-B jamming simulator + localization). https://web.stanford.edu/group/scpnt/gpslab/pubs/papers/Liu_ION_ITM_2022_ADSB.pdf
- Schäfer et al., *Jammer on the Horizon* (radio-horizon localization). https://disco.cs.rptu.de/downloads/publicationfiles/jammer_on_the_horizon.pdf
- Eichstaedt, Budroweit, Stehle (DLR), *Detecting and Measuring Global GNSS Signal Interference via Space-Based ADS-B*, IEEE SHaRC 2026. https://elib.dlr.de/226012/
- Aireon / PNT white paper, *GNSS Spoofing Detection in Aviation*, Apr 2026. https://aireon.com/wp-content/uploads/2026/04/White-Paper-GNSS-Spoofing-Detection-in-Aviation-1.0-1-1.pdf
- ML framework for GNSS interference from ADS-B with neighbor features (ERAU/UFSM). https://static.even3.com/anais/1239773.pdf
- GPS spoofing/jamming ML/DL survey-type work, arXiv:2501.02352.
- gpsjam.org methodology (NIC-threshold daily hex map) — **classical baseline for detection; reimplement.**

**Gap we fill:** no prior work jointly (a) separates jamming vs spoofing with physics invariants, (b) learns across aircraft with a GNN, (c) localizes with a learned + physics hybrid, (d) needs zero manual labels. Phase 1 must confirm this gap still holds; if a near-identical paper exists, shift emphasis to contribution 2 + 4 (§11 R1).

---

## 4. Environment setup (Phase 0)

- Use `uv` with **Python 3.12** (system Python is 3.14; PyTorch/geo wheels are safest on 3.12).
- Core deps: `torch` (MPS), `polars`, `pyarrow`, `duckdb`, `numpy`, `scipy`, `h3`, `pyproj`, `scikit-learn`, `httpx`, `tqdm`, `hydra-core` (configs), `wandb` **offline** or `tensorboard`, `matplotlib`, `pydeck` or `folium` (maps), `pytest`, `ruff`, `pyopensky` (later).
- **Do not** depend on PyTorch Geometric; implement kNN graph attention in plain PyTorch (MPS-safe, fewer install issues).
- Repo layout:
```
gist/
  data/        # download.py, parse.py, regions.py, opensky.py
  features/    # physics.py, neighbors.py, windows.py
  labels/      # lfs.py (labeling functions), label_model.py, events.yaml
  sim/         # jammer.py, spoofer.py, propagation.py
  models/      # baselines.py, traj_encoder.py, graph_attn.py, heads.py, gist.py
  train/       # pretrain.py, train.py, eval.py
  viz/         # maps.py, figures.py, demo/ (static HTML map)
configs/  notebooks/  tests/  paper/ (LaTeX)  report/ (college)  data_store/ (gitignored)
```
- `git init`, commit after every phase with message `phase N: <summary>`.
- **Gate 0:** `python -c "import torch; print(torch.backends.mps.is_available())"` → True; `pytest` runs.

## 5. Data (Phase 1–2)

> **Revised 2026-10-27 (Phase 1):** primary source is now **adsb.lol globe_history (ODbL-1.0)** — daily releases (every day, not just the 1st), full per-aircraft readsb traces, plus a separate MLAT-only release. ADS-B Exchange samples are "for testing and evaluation" and their policy bars derivative works, so they are **not used for any result**. See `DATA_LICENSE.md`. The field notes below still apply (same readsb format).

### 5.1 (superseded) ADS-B Exchange free samples
- URL pattern (verified 2026-10-26): `https://samples.adsbexchange.com/readsb-hist/YYYY/MM/01/HHMMSSZ.json.gz` — global snapshot every 5 s, **1st day of each month only**. Server returns plain JSON (~4.6 MB/snapshot, ~10k aircraft).
- Fields present (verified): `hex, flight, lat, lon, alt_baro, alt_geom, gs, ias, tas, mach, track, mag_heading, true_heading, roll, baro_rate, geom_rate, nic, nac_p, nac_v, sil, sil_type, gva, sda, nic_baro, rc, version, mlat, tisb, rssi, seen, seen_pos, gpsOkBefore, gpsOkLat, gpsOkLon, lastPosition, rr_lat, rr_lon, ws, wd, oat, squawk, category, t, dbFlags`.
  - `gpsOkBefore/gpsOkLat/gpsOkLon` = readsb's own GPS-loss heuristic → weak label + baseline.
  - `mlat` lists fields derived by multilateration → **independent position truth** for spoofing labels.
  - `alt_geom` (GNSS) vs `alt_baro` (pressure) → spoof-sensitive.
  - `rr_lat/rr_lon` = rough receiver location → radio-horizon plausibility check.
- **Before downloading at scale:** read ADS-B Exchange terms of use / samples page; confirm research use and attribution requirement. Record the outcome in `DATA_LICENSE.md`. If research use is disallowed → switch to OpenSky only (§5.3) and the public OpenSky datasets.
- **Budget (8 GB RAM, 148 GB disk):** never store raw JSON. Stream → filter → Parquet (zstd) → delete.
  - Cadence **30 s** by default (2,880 files/day, ~13 GB transfer/day, ≈1–2 GB Parquet/day).
  - **10 s** cadence only for “focus days” used in localization eval.
  - Concurrency ≤ 4, retry with backoff, resumable manifest (`data_store/manifest.parquet`). Be polite.
- **Date set:** 1st of every month, **2024-01 → 2026-10** (34 days). Start with 6 days for development; expand after Gate 3.
- **Hard disk cap:** stop downloading if free disk < 40 GB.

### 5.2 Regions of interest (bounding boxes in `regions.py`)
| ID | Region | Expected phenomenon |
|---|---|---|
| BAL | Baltic / Kaliningrad / Gulf of Finland | persistent jamming |
| EMED | Eastern Mediterranean (Cyprus–Lebanon–Israel–Egypt) | spoofing + jamming |
| BLK | Black Sea / Romania / Turkey | jamming |
| MEA | Iraq / Iran / Persian Gulf | spoofing (circle patterns) |
| INPK | India–Pakistan border (Punjab, Rajasthan) | jamming/spoofing |
| MYN | Myanmar | jamming |
| CTRL | Central Europe, CONUS mid-west, Australia | **negative control** (clean) |

Keep global rows too (for pretraining), but downsample CTRL/global to 20 %.

### 5.3 Secondary source — OpenSky Network (once Vichitra's Trino access is approved)
- Credentials will be placed by Vichitra in `~/.config/pyopensky/settings.conf` (agent never creates accounts or types passwords).
- Use `pyopensky` Trino: `state_vectors_data4` + `position_data4` (has NIC/NACp in raw `rawdata` if needed).
- Pull **specific event windows** ADS-B Exchange samples cannot cover (non-1st-of-month): esp. **Denver KDEN 2022-01-21/22** (FAA-documented jammer location → real localization ground truth; Liu et al. reported ~4 km).
- If access is not granted by the time Phase 6 starts: skip Denver, note it as future work, use approximate-location events only (§6.3).

### 5.4 Processing (Phase 2)
1. `parse.py`: JSON → rows; cast types; drop `tisb`/ground vehicles (`category` C*/ground), drop `seen_pos > 30 s`.
2. Build per-aircraft tracks: group by `hex`, sort by time, split at gaps > 5 min.
3. Windows: **10-minute windows, 30 s step** (20 steps) for 30 s cadence data; 60 steps for 10 s focus days.
4. Store `tracks.parquet` partitioned by `date/region`.
- **Gate 2:** for 1 dev day — row count sane (≥ 20 M rows global at 30 s), ≥ 95 % of BAL windows have NIC present, memory peak < 5 GB (process per-hour chunks).

## 6. Features & labels (Phase 3)

### 6.1 Physics-consistency features (`features/physics.py`) — per time step
| Feature | Intuition | Spoofer can fake? |
|---|---|---|
| `nic, nac_p, sil, gva` (+ deltas) | integrity collapse = jamming | **Yes** (spoofed receivers report high integrity) |
| kinematic residual: ‖Δpos − gs·track·Δt‖ | position jumps | partially |
| `alt_geom − alt_baro` minus local neighbor median | GNSS altitude drift | no (baro is independent) |
| implied wind (GNSS gs/track vs tas/heading) minus neighbor-consensus wind | spoofed positions break the wind triangle | **no** |
| ADS-B pos vs MLAT pos (when `mlat` contains lat/lon) | direct truth | **no** |
| distance(rr_lat/lon, claimed pos) vs radio horizon of claimed altitude | claimed pos unreceivable | **no** |
| `gpsOkBefore` age, distance to `gpsOkLat/Lon` | readsb's loss flag | — |
| position-loss: has track/alt but no lat/lon | jamming | — |
| `rssi` | weak signal context | — |
| altitude, phase of flight, time of day | context | — |
Normalize with robust (median/IQR) scaling fitted on CTRL train days.

> Caveat to document: readsb `ws/wd` are *derived* from gs/tas/heading, so they are not independent; we use them only relative to neighbor consensus.

### 6.2 Labeling functions (`labels/lfs.py`) → label model
Each LF votes `NORMAL / JAMMED / SPOOFED / ABSTAIN` per window:
- LF_nic_zero: ≥ 30 % steps with NIC ≤ 1 at cruise → JAMMED
- LF_pos_loss: lat/lon missing while alt/track present ≥ 2 min → JAMMED
- LF_gpsok: readsb gpsOk flag active → JAMMED
- LF_mlat_mismatch: MLAT vs ADS-B > 2 NM with NIC ≥ 7 → SPOOFED
- LF_horizon: claimed position beyond receiver radio horizon + 50 km → SPOOFED
- LF_cluster: ≥ 3 distinct aircraft reporting positions within 1 km of each other while their baro/air-data imply separation (classic "spoof point") → SPOOFED
- LF_teleport: kinematic residual > 10 km with NIC ≥ 7 → SPOOFED
- LF_wind: wind-triangle inconsistency > 4σ vs neighbors for ≥ 3 min → SPOOFED
- LF_clean: CTRL region + all integrity nominal + all residuals < 1σ → NORMAL
- Label model: implement a small **generative label model** (Dawid–Skene / Snorkel-style EM over LF accuracies & correlations) in NumPy — no Snorkel dependency. Output probabilistic labels; train with soft cross-entropy.
- **Hold out LF_mlat_mismatch and LF_horizon from training labels** for one evaluation variant (they become *independent* eval signals → avoids circularity, see §8.2).

### 6.3 Real evaluation events (`labels/events.yaml`)
Build automatically from public reporting (gpsjam.org history, EASA SIBs, OPSGROUP / FAA notices, news): for each event record `date, region, type (jam/spoof), approx_source (lat, lon, uncertainty_km), source_url`. Only include events whose dates are 1st-of-month (ADSBx) or retrievable via OpenSky. Targets: ≥ 15 events, ≥ 5 spoofing. Known-location anchors: KDEN 2022 (exact, via OpenSky), Kaliningrad (region-level, ~100 km), Eastern Med spoof points (reported fake coordinates, e.g. near Beirut airport).
**Manual sanity check replacement:** the agent renders a map PNG per event and inspects it visually; discard events whose map shows no anomaly.

- **Gate 3:** label coverage ≥ 60 % of windows in interference regions; LF conflict rate reported; per-region label histograms look plausible (BAL mostly JAMMED, EMED/MEA contains SPOOFED, CTRL > 99 % NORMAL). If CTRL has > 1 % non-normal → tighten thresholds and repeat.

## 7. Simulation for localization (Phase 4) — `sim/`

Ground-truth emitter locations are rare, so we inject synthetic interference into **clean real traffic** (CTRL + non-event days):
- **Jammer:** sample location (land/sea), height 2–50 m (ground) or 1–10 km (airborne), EIRP 1–100 W. For each aircraft: Friis path loss + 4/3-earth radio-horizon line-of-sight + RTCA-like antenna pattern (gain depends on elevation angle) → J/N → map to NIC degradation & position loss per Liu et al. 2022/2025 thresholds (NIC=0 above −115 dBW-equivalent, NIC 1–6 moderate band; add stochastic receiver tolerance per aircraft type).
- **Spoofer:** sample location, range by power + LOS; affected aircraft get positions replaced by (a) a static fake point, (b) a fake circle, or (c) a slow drift offset; `alt_baro`, `tas`, `mag_heading` stay true; `alt_geom` follows the fake; NIC set high (realistic).
- 50k simulated scenes (10-min windows, all aircraft in a 600 km box). Store scene → emitter params.
- **Gate 4:** reimplemented Liu NLS localizer reaches < 10 km median error on clean-noise sims (sanity check of the simulator + baseline).

## 8. Models & experiments (Phase 5–6)

### 8.1 Baselines (`models/baselines.py`)
1. **gpsjam-style:** per H3 res-4 hex per hour, flag if share of aircraft with NIC ≤ 6 > 10 %. (detection only)
2. **readsb gpsOk flag** as a detector.
3. **Isolation Forest** on window-aggregated features.
4. **LSTM autoencoder** (reconstruction-error anomaly score).
5. **Per-aircraft Transformer only** (no graph) — ablation-baseline.
6. **Localization:** Liu NLS (Gauss-Newton), radio-horizon intersection (Schäfer-style), centroid of affected aircraft.

### 8.2 GIST (`models/gist.py`) — sized for M3 / 8 GB
- **Trajectory encoder:** 4-layer Transformer, d=128, 4 heads, sinusoidal time encoding of irregular Δt, input = §6.1 features + learned type/phase embeddings. ~1.5 M params.
- **Self-supervised pretraining:** masked-step reconstruction (mask 30 % of steps / feature groups) + next-position prediction, on all clean+unlabeled windows. ≤ 3 h on MPS.
- **Spatial graph attention:** at each window, nodes = aircraft; edges = k=16 nearest within 400 km (great-circle, computed with H3 + haversine); edge features = distance, bearing, Δalt, LOS-visible-to-common-point flag. 3 layers GATv2-style, implemented in plain PyTorch with padded neighbor tensors. ~0.8 M params.
- **Heads:**
  - Classification: per-aircraft-window {NORMAL, JAMMED, SPOOFED} with soft labels; focal loss for imbalance.
  - **Localization head:** predicts a heatmap on a 10 km grid over the scene + emitter power; refined by a **differentiable likelihood layer**: for candidate source s, expected affected-probability p_i(s) = σ((J/N_i(s) − τ)/T) using Friis + horizon geometry; combine with network affectedness predictions via BCE log-likelihood → soft-argmax over grid. Trained on §7 sims, fine-tuned (unsupervised consistency loss) on real events.
- Training: AdamW, lr 3e-4 cosine, batch = 8 scenes (gradient accumulation), bf16 off (MPS fp32), early stopping on val macro-F1. Seeds ×3 for all headline numbers.

### 8.3 Splits (prevent leakage)
- **Temporal:** train 2024-01 → 2025-06, val 2025-07 → 2025-12, test 2026-01 → 2026-10.
- **Regional generalization:** leave-one-region-out (train without EMED, test on EMED, etc.).
- Never put windows from the same day in both train and test.

### 8.4 Metrics
- Detection: window-level PR-AUC, macro-F1 (3-class), **spoofing recall at fixed 1 % FPR**, event-level recall (event detected if ≥ 1 alarm within event region-hour), **detection latency** (minutes from first affected aircraft to alarm), false alarm rate per 10k flight-hours on CTRL.
- Localization: median & 95th-percentile error (km) on sims; error on real events vs approximate sources (report with source uncertainty); KDEN error if OpenSky available.
- Calibration: ECE.
- Agreement with independent signals: precision of SPOOFED predictions against held-out MLAT/horizon LFs (circularity-free check).

### 8.5 Ablations (each = 1 row in paper table)
No physics features (integrity fields only) · no graph · no pretraining · no simulation (localization) · label model → majority vote · 30 s vs 10 s cadence · k neighbors ∈ {4, 16, 32}.

### 8.6 Compute plan
All runs ≤ 2 h each on M3; ~35 runs total ≈ 50–70 machine-hours. Use `caffeinate -i` for long runs; checkpoint every epoch; log to `runs/`. If memory > 6 GB: cut scene size (cap 400 aircraft/scene) and batch.

- **Gate 5:** GIST beats best baseline on val macro-F1 by ≥ 5 points AND spoofing recall@1%FPR by ≥ 15 points. If not → §11 R3.
- **Gate 6:** sim localization median error < 15 km; real-event errors within source uncertainty for ≥ 60 % of events.

## 9. Deliverables (Phase 7–8)

1. **GitHub-ready repo** with README, `make data / make train / make eval / make paper`, tests, MIT license (code) + data license notes.
2. **Released dataset card:** derived labels + features only (no raw redistribution unless terms allow), with LF definitions.
3. **Paper** (`paper/`): LaTeX, IEEE conference template, 6–8 pages + references. Sections: Intro, Related Work, Data & Weak Supervision, Method, Simulation, Experiments, Case Studies (BAL, EMED, INPK, KDEN), Limitations & Ethics, Conclusion. Figures: pipeline diagram, per-region maps, PR curves, latency CDF, localization error CDF, case-study maps. All numbers auto-generated from `runs/` into `paper/tables/*.tex`.
4. **College report** (`report/`): longer version (~40 pages) with background on GNSS/ADS-B, full methodology, extra experiments, screenshots; follows a generic final-year structure (abstract, acknowledgements placeholder, chapters, references, appendix). Placeholders: `<<COLLEGE>>`, `<<GUIDE>>`, `<<ROLL_NO>>`.
5. **Demo:** static HTML map (deck.gl / folium) — pick a date, see per-hex jamming/spoofing probability + estimated emitters; host-ready as a single folder.
6. **Slides:** 12–15 slide viva deck.

**Candidate venues** (agent verifies dates in Phase 1 and writes `paper/VENUES.md`; submission itself is Vichitra's action): OpenSky Symposium (best topical fit), ION GNSS+ / ION ITM, IEEE/AIAA DASC, IEEE ICNS, ML workshops at NeurIPS/ICLR on ML for physical sciences or safety. Format the paper for the earliest fitting deadline; keep an IEEE double-column default.

## 10. Execution phases (agent runbook)

| Phase | Work | Output | Gate |
|---|---|---|---|
| 0 | Env + repo scaffold | repo, CI-less tests | G0 |
| 1 | Lit review refresh, license check, venue check, event catalog draft | `paper/related.bib`, `DATA_LICENSE.md`, `VENUES.md`, `events.yaml` | gap still holds |
| 2 | Download 6 dev days, parse, tracks, windows | Parquet store | G2 |
| 3 | Physics features, LFs, label model | labels Parquet, label report | G3 |
| 4 | Jammer/spoofer simulator + NLS baseline | `sim/` + 50k scenes | G4 |
| 5 | Baselines + GIST classification (pretrain → train) | runs, metrics | G5 |
| 6 | Localization head, real-event eval, expand data to all 34 days, final 3-seed runs, ablations | final tables | G6 |
| 7 | Figures, demo map, paper | `paper/main.pdf`, demo | compiles, all numbers sourced |
| 8 | College report, slides, README polish, final self-review | `report/main.pdf`, slides | §13 checklist |

Write a running `LOG.md` (decisions, thresholds changed, failed attempts) — this becomes the report's "challenges" chapter.

## 11. Risks & fallbacks

- **R1 Prior work overlap** → emphasize spoof-invariant features + weak-supervision dataset as primary contribution.
- **R2 ADSBx terms disallow research use / samples disappear** → OpenSky only; reduce scope to regions/time windows pulled via Trino.
- **R3 GIST doesn't beat baselines** → report honestly; pivot paper to "spoof-invariant features + label model" (features alone in gradient boosting), keep the GNN as an ablation. A negative-but-rigorous result is still a valid college project.
- **R4 Weak labels too noisy** → raise LF thresholds, add LF diagnostics, evaluate only on high-confidence + independent-signal subsets.
- **R5 Too few spoofing examples** → oversample EMED/MEA, add simulated spoofing to training (mark as sim), report real-only metrics separately.
- **R6 Memory / disk limits** → per-hour chunking, 30 s cadence, cap aircraft per scene, delete intermediates.
- **R7 OpenSky approval delayed** → KDEN case study becomes future work.

## 12. Agent operating rules — "stop and ask" only if:
1. A step requires creating an account, entering credentials, payment, or accepting terms on Vichitra's behalf.
2. Data terms are ambiguous about academic publication.
3. Free disk < 40 GB and nothing safe to delete.
4. Two consecutive fallbacks for the same gate fail.
Otherwise proceed, log the decision in `LOG.md`, and continue.

Ethics note for paper: data is public broadcast; no individual tracking is published (aggregate to hexes/events; no per-tail-number tables); the work is defensive (detection/attribution of interference).

## 13. Final checklist
- [ ] Every number in paper/report generated from a script, 3 seeds, mean ± std
- [ ] Temporal + regional splits, no same-day leakage
- [ ] Spoofing evaluated against independent signals (MLAT / horizon) not used in training labels
- [ ] Baselines reimplemented and cited
- [ ] Data license & attribution stated
- [ ] Demo opens offline
- [ ] Placeholders listed in `TODO_FOR_VICHITRA.md`

## 14. What Vichitra must do (the only human steps)
1. **Apply for OpenSky Network research / Trino access** now; when approved, put credentials into `~/.config/pyopensky/settings.conf` yourself.
2. At the end: fill placeholders (college name, guide, roll no.), choose the venue, and submit.
