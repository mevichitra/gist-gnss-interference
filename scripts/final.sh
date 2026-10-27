#!/usr/bin/env bash
# End-to-end final pipeline (PLAN §10 phases 6–8). Idempotent per step; logs to runs/final.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/final.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
PY="nice -n 5 uv run python -m"

DAYS=()
for y in 2024 2025 2026; do for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  d="$y-$m-01"; [[ "$d" > "2026-10-25" ]] || DAYS+=("$d"); done; done
DAYS+=("2026-10-25")
TEST_DAYS=(); for d in "${DAYS[@]}"; do [[ "$d" < "2026-01-01" ]] || TEST_DAYS+=("$d"); done

# 1. wait for every 1st-of-month day to be parsed, then stop the fetcher (it competes for RAM)
while true; do
  missing=0; for d in "${DAYS[@]}"; do [[ -f data_store/points/$d/_SUCCESS ]] || missing=$((missing+1)); done
  [[ $missing -eq 0 ]] && break
  log "waiting for $missing days"; sleep 300
done
pkill -f gist.data.fetch_days || true
log "all ${#DAYS[@]} days parsed"

# 2. features + weak labels (label model refit across all days)
$PY gist.labels.build "${DAYS[@]}" 2>&1 | tail -1 | tee -a "$LOG"

# 3. scene datasets (day-based splits)
if [[ ! -f data_store/scenes/.final_done ]]; then
  rm -rf data_store/scenes/train data_store/scenes/val data_store/scenes/test data_store/scenes/test_distract
  for d in "${DAYS[@]}"; do
    if [[ "$d" < "2025-07-01" ]]; then n=120; elif [[ "$d" < "2026-01-01" ]]; then n=100; else n=100; fi
    $PY gist.train.dataset "$d" --real $n --sim $n 2>&1 | tail -1 | tee -a "$LOG"
  done
  for d in "${TEST_DAYS[@]}"; do
    $PY gist.train.dataset "$d" --split test_distract --real 0 --sim 40 --kind jam --distractors 0.05 2>&1 | tail -1 | tee -a "$LOG"
  done
  touch data_store/scenes/.final_done
fi

# 4. training + evaluation
train_eval() {  # name, extra args...
  local name=$1; shift
  if [[ ! -f runs/$name/history.json ]]; then
    log "train $name $*"
    $PY gist.train.train --train train --val val --name "$name" --epochs 8 "$@" 2>&1 | grep -v Warn | tail -1 >> "$LOG"
  fi
  for split in test test_distract; do
    [[ -f runs/$name/eval_$split.json ]] || $PY gist.train.train --train train --val $split --name "$name" "$@" \
      --eval-only runs/$name/best.pt > /dev/null 2>&1
  done
  log "done $name"
}
train_eval gist_final_s0 --seed 0
train_eval gist_final_s1 --seed 1
train_eval gist_final_s2 --seed 2
train_eval gist_nograph_s0 --seed 0 --no-graph
train_eval gist_integ_s0 --seed 0 --integrity-only
train_eval gist_nosim_s0 --seed 0 --no-sim

# 5. baselines
[[ -f runs/baselines_test.json && -f runs/.baselines_final ]] || {
  $PY gist.train.eval_baselines --train train --test test --out runs/baselines_test.json > /dev/null 2>&1
  $PY gist.train.eval_baselines --test test_distract --loc-only --out runs/baselines_test_distract.json > /dev/null 2>&1
  touch runs/.baselines_final; log "baselines done"; }

# 6. localization error CDF inputs
uv run python - <<'EOF'
import json
from pathlib import Path
for split in ("test", "test_distract"):
    g = json.loads(Path(f"runs/gist_final_s0/locerr_{split}.json").read_text())
    b = json.loads(Path(f"runs/baselines_{split}.json").read_text()).get("localization_errors", {})
    names = {"nls": "NLS (Liu et al.)", "centroid": "Centroid", "horizon": "Radio horizon"}
    out = {"GIST": g["GIST"], **{names[k]: v for k, v in b.items()}}
    Path(f"runs/gist_final_s0/locerr_{split}_all.json").write_text(json.dumps(out))
for f in Path("runs/gist_final_s0").glob("locerr_test.json"):
    f.rename(f.with_suffix(".raw"))
for f in Path("runs/gist_final_s0").glob("locerr_test_distract.json"):
    f.rename(f.with_suffix(".raw"))
EOF

# 7. whole-day inference for the demo and case studies
for d in 2026-10-01 2026-01-01 2024-07-01; do
  [[ -f runs/gist_final_s0/pred_$d.parquet ]] || $PY gist.train.predict --ckpt runs/gist_final_s0/best.pt --days $d --every 2 2>&1 | tail -1 | tee -a "$LOG"
done

# 8. figures, tables, demo page, PDFs
$PY gist.viz.figures --run runs/gist_final_s0 --out paper/figures \
  --cases 2026-01-01:BAL 2026-10-01:BAL 2024-07-01:EMED 2026-10-01:MEA 2026-01-01:BLK 2>&1 | tail -3 | tee -a "$LOG"
$PY gist.viz.tables --prefix gist_final 2>&1 | tail -1 | tee -a "$LOG"
$PY gist.viz.build_demo --run runs/gist_final_s0 --days 2024-07-01 2026-01-01 2026-10-01 2>&1 | tail -2 | tee -a "$LOG"
(cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null && bibtex main > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null) || true
(cd report && pdflatex -interaction=nonstopmode main.tex > /dev/null && bibtex main > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null; pdflatex -interaction=nonstopmode main.tex > /dev/null) || true
log "FINAL PIPELINE COMPLETE"
