# Data sources and licensing

## Primary: adsb.lol globe_history (ODbL-1.0) — USED
- Repos: github.com/adsblol/globe_history_{2023,2024,2025,2026}; licence ODbL-1.0 (checked via GitHub API, 2026-10-27).
- One release per day (prod/staging, split tar ~3.6 GB) + a separate `mlatonly` release (independent MLAT positions).
- Format: readsb trace JSONs (gzip), see wiedehopf/readsb README-json.md#trace-jsons.
- Obligations: attribute "ADSB.lol and contributors (ODbL)"; any *derived database* we publicly release (labels, features) must be released under ODbL with attribution. Code may be MIT. Published figures/papers ("produced works") need attribution only.

## ADS-B Exchange samples — NOT USED for results
- samples.adsbexchange.com offers 1st-of-month data "for testing and evaluation".
- ADSBx data-use policy prohibits sharing/derivative works without authorisation; publication use is unclear.
- Decision (PLAN §12 rule 2 avoided by switching sources): do not use ADSBx for any reported result. It was only used for a one-off field inspection.

## OpenSky Network (pending Trino approval; request submitted 2026-10-27)
- Non-commercial research use; cite Schäfer et al. 2014 (IPSN) per OpenSky terms.
- Used only for event windows not covered by adsb.lol (e.g. Denver KDEN Jan 2022; adsb.lol archives start 2023).
