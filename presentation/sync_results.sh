#!/usr/bin/env bash
# Copy the result JSONs the deck reads from the main project's runs/ into presentation/data/runs/,
# then rebuild the deck. Usage: presentation/sync_results.sh /path/to/main/project
set -euo pipefail
SRC=${1:?path to the main project folder}
DST="$(cd "$(dirname "$0")" && pwd)/data/runs"
mkdir -p "$DST"
cd "$SRC/runs"
for f in baselines_test.json baselines_test_distract.json gist_s0/best_val.json \
         gist_final_s*/eval_test.json gist_final_s*/eval_test_distract.json \
         gist_nograph_s0/eval_test.json gist_integ_s0/eval_test.json gist_nosim_s0/eval_test.json; do
  [[ -f $f ]] && { mkdir -p "$DST/$(dirname "$f")"; cp "$f" "$DST/$f"; echo "copied $f"; }
done
