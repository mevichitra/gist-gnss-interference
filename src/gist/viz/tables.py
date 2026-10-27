"""Generate every LaTeX table / macro in the paper from run JSONs (no hand-typed numbers).

Expected run layout (missing runs are reported as "--"):
  runs/<prefix>_s{0,1,2}/eval_test.json, eval_test_distract.json   GIST seeds
  runs/<ablation>_s0/eval_test.json                                 ablations
  runs/baselines_test.json, runs/baselines_test_distract.json       baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

RUNS = Path("runs")
ABLATIONS = [("gist_nograph", "w/o spatial graph attention"), ("gist_integ", "integrity fields only"),
             ("gist_nosim", "w/o simulated training scenes")]


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def get(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def ms(vals, pct=False, digits=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    f = 100 if pct else 1
    m = np.mean(vals) * f
    if len(vals) == 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}$\\pm${np.std(vals) * f:.{digits}f}"


def seeds(prefix: str, file: str) -> list[dict]:
    return [d for d in (load(RUNS / f"{prefix}_s{i}" / file) for i in range(3)) if d]


def results_table(prefix: str) -> str:
    g = seeds(prefix, "eval_test.json")
    b = load(RUNS / "baselines_test.json") or {}
    gd = seeds(prefix, "eval_test_distract.json")
    bd = load(RUNS / "baselines_test_distract.json") or {}
    rows = []
    for name, src in (("gpsjam-style threshold", b.get("gpsjam")), ("Isolation Forest", b.get("iforest")),
                      ("LSTM autoencoder", b.get("lstm_ae"))):
        rows.append([name] + [ms([get(src, k, m)]) for k in ("real", "sim")
                              for m in ("macro_f1", "jam_ap", "spoof_ap", "spoof_recall_at_1fpr")])
    rows.append([f"\\textbf{{GIST}} ({len(g)} seeds)"] + [ms([get(x, k, m) for x in g]) for k in ("real", "sim")
                                                          for m in ("macro_f1", "jam_ap", "spoof_ap", "spoof_recall_at_1fpr")])
    det = "\n".join(" & ".join(r) + r" \\" for r in rows)
    loc_rows = []
    for name, key in (("Centroid of affected", "centroid"), ("Radio-horizon score", "horizon"), ("NLS (Liu et al.)", "nls")):
        loc_rows.append([name, ms([get(b, "localization", key, "median_km")], digits=1),
                         ms([get(b, "localization", key, "p95_km")], digits=1),
                         ms([get(bd, "localization", key, "median_km")], digits=1),
                         ms([get(bd, "localization", key, "p95_km")], digits=1)])
    loc_rows.append(["\\textbf{GIST}", ms([get(x, "loc_refined", "median_km") for x in g], digits=1),
                     ms([get(x, "loc_refined", "p95_km") for x in g], digits=1),
                     ms([get(x, "loc_refined", "median_km") for x in gd], digits=1),
                     ms([get(x, "loc_refined", "p95_km") for x in gd], digits=1)])
    loc = "\n".join(" & ".join(r) + r" \\" for r in loc_rows)
    return rf"""\begin{{table*}}[t]\centering\small
\caption{{Test split (2026 days). Detection metrics per aircraft-window: macro-F1 over 3 classes, average precision (AP) and spoofing recall at 1\% false-positive rate (R@1\%). Real = label-model targets; Sim = exact simulated truth. Single-score baselines cannot separate jamming from spoofing; their anomaly score is used for both. Mean$\pm$std over seeds.}}
\label{{tab:results}}
\begin{{tabular}}{{@{{}}lcccc|cccc@{{}}}}\toprule
 & \multicolumn{{4}}{{c|}}{{Real traffic}} & \multicolumn{{4}}{{c}}{{Simulated emitters}}\\
Method & F1 & Jam AP & Spoof AP & Spoof R@1\% & F1 & Jam AP & Spoof AP & Spoof R@1\% \\\midrule
{det}
\bottomrule\end{{tabular}}\vspace{{6pt}}

\begin{{tabular}}{{@{{}}lcc|cc@{{}}}}\toprule
 & \multicolumn{{2}}{{c|}}{{Clean scenes}} & \multicolumn{{2}}{{c}}{{+5\% benign low-NIC aircraft}}\\
Jammer localization [km] & median & p95 & median & p95 \\\midrule
{loc}
\bottomrule\end{{tabular}}
\end{{table*}}
"""


def ablation_table(prefix: str) -> str:
    rows = []
    full = seeds(prefix, "eval_test.json")
    entries = [("GIST (full)", full)] + [(label, seeds(name, "eval_test.json")) for name, label in ABLATIONS]
    for label, rs in entries:
        rows.append([label, ms([get(r, "real", "macro_f1") for r in rs]), ms([get(r, "real", "spoof_recall_at_1fpr") for r in rs]),
                     ms([get(r, "sim", "macro_f1") for r in rs]), ms([get(r, "sim", "spoof_recall_at_1fpr") for r in rs]),
                     ms([get(r, "loc_refined", "median_km") for r in rs], digits=1)])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return rf"""\begin{{table}}[t]\centering\small
\caption{{Ablations on the test split (seed 0 unless noted).}}
\label{{tab:ablations}}
\begin{{tabular}}{{@{{}}lccccc@{{}}}}\toprule
 & \multicolumn{{2}}{{c}}{{Real}} & \multicolumn{{2}}{{c}}{{Sim}} & Loc. \\
Variant & F1 & Spoof R@1\% & F1 & Spoof R@1\% & med. km \\\midrule
{body}
\bottomrule\end{{tabular}}
\end{{table}}
"""


def dataset_numbers() -> dict:
    days = sorted(p.stem for p in Path("data_store/windows").glob("*.parquet"))
    n_windows = 0
    n_points = 0
    for d in days:
        n_windows += pl.scan_parquet(f"data_store/windows/{d}.parquet").select(pl.len()).collect().item()
        s = Path(f"data_store/points/{d}/_SUCCESS")
        n_points += int(s.read_text()) if s.exists() else 0
    split = {"train": [d for d in days if d < "2025-07-01"], "val": [d for d in days if "2025-07-01" <= d < "2026-01-01"],
             "test": [d for d in days if d >= "2026-01-01"]}
    return {"days": len(days), "windows": n_windows, "points": n_points, **{f"n_{k}": len(v) for k, v in split.items()}}


def macros(prefix: str) -> str:
    g = seeds(prefix, "eval_test.json")
    b = load(RUNS / "baselines_test.json") or {}
    gd = seeds(prefix, "eval_test_distract.json")
    bd = load(RUNS / "baselines_test_distract.json") or {}
    dn = dataset_numbers()
    f = lambda v, d=2: "--" if v is None else f"{v:.{d}f}"  # noqa: E731
    mean = lambda xs: None if not [x for x in xs if x is not None] else float(np.mean([x for x in xs if x is not None]))  # noqa: E731
    spoof_g = mean([get(x, "real", "spoof_recall_at_1fpr") for x in g])
    spoof_b = get(b, "gpsjam", "real", "spoof_recall_at_1fpr")
    loc_g = mean([get(x, "loc_refined", "median_km") for x in gd])
    loc_n = get(bd, "localization", "nls", "median_km")
    abstract = (f"GIST reaches a spoofing recall of {f(spoof_g)} at 1\\% false-positive rate on real 2026 traffic, "
                f"where an integrity-threshold monitor reaches {f(spoof_b)}, and localizes simulated jammers among "
                f"benign low-integrity aircraft with a median error of {f(loc_g, 1)}\\,km versus {f(loc_n, 1)}\\,km for "
                f"non-linear least squares.")
    lines = [
        rf"\newcommand{{\resultsabstract}}{{{abstract}}}",
        rf"\newcommand{{\datasetsentence}}{{We use {dn['days']} days (1st of each month, January 2024 to October 2026), "
        rf"{dn['points'] / 1e6:.0f}\,M airborne trace points inside the study regions and {dn['windows'] / 1e6:.2f}\,M aircraft-windows.}}",
        rf"\newcommand{{\splitsentence}}{{This gives {dn['n_train']} training, {dn['n_val']} validation and {dn['n_test']} test days.}}",
        r"\newcommand{\simsentence}{Each training day contributes simulated scenes alongside real scenes.}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gist_final")
    ap.add_argument("--out", type=Path, default=Path("paper/tables"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.tex").write_text(results_table(a.prefix))
    (a.out / "ablations.tex").write_text(ablation_table(a.prefix))
    (a.out / "macros.tex").write_text(macros(a.prefix))
    print("wrote", *sorted(p.name for p in a.out.glob("*.tex")))


if __name__ == "__main__":
    main()
