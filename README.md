# GIST — GNSS Interference Spatio-Temporal network

Detect, classify (jamming vs spoofing) and localize GNSS interference from crowdsourced ADS-B,
with no manual labels. Final-year deep learning project + paper. Full design: [PLAN.md](PLAN.md);
decisions and dead ends: [LOG.md](LOG.md).

## Pipeline

| Step | Command | Output |
|---|---|---|
| Download + parse days (adsb.lol, ODbL) | `uv run python -m gist.data.fetch_days [YYYY-MM-DD ...]` | `data_store/points/<day>/` |
| Features + weak labels | `uv run python -m gist.labels.build [days]` | `data_store/steps/`, `data_store/windows/` |
| Scene datasets (real + simulated emitters) | `uv run python -m gist.train.dataset <days> --real 300 --sim 300` | `data_store/scenes/<split>/` |
| Train GIST | `uv run python -m gist.train.train --train train --val val --name gist` | `runs/gist/` |
| Baselines | `uv run python -m gist.train.eval_baselines --train train --test test --out runs/baselines.json` | JSON metrics |
| Tests | `uv run pytest` | |

Splits are by day: train ≤ 2025-06, val 2025-07..2025-12, test ≥ 2026-01.

## Layout

```
src/gist/data      download (resumable, throttle-aware), trace parser, regions
src/gist/features  physics-consistency step features, 10-min windows
src/gist/labels    labeling functions, Dawid–Skene label model, build script, event catalogue
src/gist/sim       RF propagation (NumPy/PyTorch), jammer/spoofer injection, scene sampling
src/gist/models    GIST (Transformer + spatial attention + physics localizer), baselines, NLS
src/gist/train     scene tensors, datasets, training, evaluation
paper/             LaTeX (IEEE), bibliography, venues
```

## Data and licences
- adsb.lol globe_history — ODbL-1.0. Derived datasets we release are ODbL; cite "ADSB.lol and contributors".
- ADS-B Exchange samples are **not** used (evaluation-only terms). See [DATA_LICENSE.md](DATA_LICENSE.md).
- OpenSky Network (Trino) — requested 2026-10-27; used only for events before 2023 (e.g. Denver KDEN 2022).

Code: MIT.
