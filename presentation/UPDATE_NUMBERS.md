# Updating the deck numbers

The result slides are generated from JSON. Nothing on slides 14 and 17–19 or 23 is typed in by hand.

## Rebuild after the final pipeline

```bash
presentation/sync_results.sh "/path/to/main/project"   # copies runs/*.json into presentation/data/runs/
NODE_PATH=<node_modules with pptxgenjs, react, react-dom, react-icons, sharp> \
PPTX_SKILL=<pptx skill dir, for apply_theme.js> node presentation/build_deck.js
```

`build_deck.js` checks for `data/runs/gist_final_s{0,1,2}/eval_test.json`.
- **Final files present:** GIST cells show mean ± std over the seeds found, and the status pill reads `FINAL · N SEEDS · TEST`.
- **Final files absent (current):** GIST cells come from `gist_s0/best_val.json`, and the pill reads `INTERIM · SEED 0 · VALIDATION`. The footnotes say the baselines are on test days, so the comparison is not like-for-like yet.

`.gitignore` ignores `runs/`, so commit the synced JSONs with `git add -f presentation/data/runs`.

Re-export the PDF afterwards (LibreOffice: `soffice --headless --convert-to pdf GIST_DL_Project.pptx`).

## Which slide uses which field

| Slide | Element | File (under `data/runs/`) | Field |
|---|---|---|---|
| 14 Localization head | coarse vs refined median | GIST file* | `loc.median_km`, `loc_refined.median_km` |
| 17 Detection | baseline rows | `baselines_test.json` | `{gpsjam,iforest,lstm_ae}.real.{macro_f1,jam_ap,spoof_ap,spoof_recall_at_1fpr}`, `.sim.{macro_f1,spoof_recall_at_1fpr}` |
| 17 Detection | GIST row + chart | GIST file* | `real.{macro_f1,jam_ap,spoof_ap,spoof_recall_at_1fpr}`, `sim.{macro_f1,spoof_recall_at_1fpr}` |
| 18 Localization | baseline rows, chart, “+64 %” | `baselines_test.json`, `baselines_test_distract.json` | `localization.{centroid,horizon,nls}.{median_km,p95_km}` |
| 18 Localization | GIST clean | GIST file* | `loc_refined.{median_km,p95_km}` |
| 18 Localization | GIST distractor (now “pending”) | `gist_final_s*/eval_test_distract.json` | `loc_refined.{median_km,p95_km}` |
| 19 Ablations (now “pending”) | full / no-graph / integrity-only / no-sim | `gist_final_s0`, `gist_nograph_s0`, `gist_integ_s0`, `gist_nosim_s0` → `eval_test.json` | `real.macro_f1`, `real.spoof_recall_at_1fpr`, `sim.macro_f1`, `loc_refined.median_km` |
| 23 Conclusion | spoof recall, localization median | GIST file*, `baselines_test.json` | `real.spoof_recall_at_1fpr`, `loc_refined.median_km`, `gpsjam.real.spoof_recall_at_1fpr` |

\* GIST file = `gist_final_s*/eval_test.json` (mean over seeds) when present, else `gist_s0/best_val.json`.

**Assumption to check:** the final `eval_test.json` files use the same keys as `best_val.json` (`real`, `sim`, `loc`, `loc_refined`). If the keys differ, change `loadResults()` in `build_deck.js`.

## Numbers typed in by hand (edit `build_deck.js` if they change)

| Slide | Value | Source |
|---|---|---|
| 4, 9 | NIC ≤ 6 shares (BAL 15.8 % … AUS 1.7 %) | LOG.md Phase 2 |
| 9 | 67 M points, 1.22 M windows, 5 days | `paper/tables/macros.tex` (`\datasetsentence`). **Will change** when all days are parsed |
| 9 | 67,333 aircraft/day, 114 s/day, 340 MB, point-source mix | LOG.md Phase 2 |
| 11 | altitude-mismatch rates per 10k steps | LOG.md Phase 3 |
| 12 | LF accuracies, coverage 77–92 %, CUS 99.9 % / AUS 99.6 %, AUS 12.8 % bug | LOG.md Phases 3 and 5 |
| 14 | 2.3 GB vs 0.36 GB memory | LOG.md Phase 5 |
| 15 | 0.4 % → 41 % affected, NLS 2.4 km noise-free | LOG.md Phase 4 |
| 17 | n = 18,348 windows, 132 spoofed (interim footnote) | `gist_s0/best_val.json` `real.n`, `real.n_spoof`. Remove when final |
| 18 | n = 167 / 334 / 160 (interim footnote) | `loc.n` fields. Remove when final |
| 18 | oracle 17.2 km | LOG.md Phase 5 dev diagnostics |
| 21 | 7.6 GB, epochs 2 → 36 min | LOG.md Phase 5 |
| 22 | 132 spoofed in validation | `gist_s0/best_val.json`; update for the test split |
| 23 | 22.1 vs 23.9 km (same dev scenes) | LOG.md Phase 5 dev results. Replace with the final test comparison |

## Slides to revisit by hand once final results arrive
- **20 Case study:** the figure is from the dry-run checkpoint (`runs/dryrun/fig/case_BAL_2026-01-01.png`, cropped into `assets/`). Replace it with the final `paper/figures/` version and drop the “DRY-RUN” pill.
- **19 Ablations:** the three cards are hypotheses. Rewrite them as findings when the numbers are in.
- **22 Limitations, item 5:** delete it once the numbers are final.
- **23 Conclusion:** the paragraph cites dev-scene localization (22.1 vs 23.9 km). Replace it with the final test numbers, including the distractor result.
