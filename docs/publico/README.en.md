# PRC Data Challenge 2026 — taxi-out time

Team **`outgoing-boat`** (solo team). Solution for the EUROCONTROL Performance
Review Commission / OpenSky Network [PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/):
predict the taxi-out time of every departure at 10 major European airports.

```
TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt      (PHASE_mvt == "DEP")
```

Metric: RMSE in seconds on the ranking set (January and July 2026 movements).

| | |
|---|---|
| Best official score | **244.89 s** (v33) |
| Airports | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH |
| Stack | Python 3, pandas/pyarrow, LightGBM, CatBoost (XGBoost optional) |
| Hardware | 6 physical cores, 15 GB RAM, no GPU required |
| License | GNU GPL v3 (see `LICENSE`) |

Official score history (RMSE on the public leaderboard):

| Version | Change | Official RMSE |
|---|---|---|
| v1 | first model, outliers dropped from training | 514.5 |
| v2 | outliers kept, unit bug in congestion windows fixed | 384.7 |
| v3 | two-stage model (copy classifier + regressor) | 338.7 |
| v5 | per-airport lines for flights without a Network Manager record | 331.0 |
| v6 | ADS-B features from adsb.lol | 314.76 |
| v9 | cross-fitted corrector + LOBT window | 275.90 |
| v11 | arrival context and neighbour features in the corrector | 266.81 |
| v13 | Rome post-rule | 264.35 |
| v17–v30 | weather (METAR), EUROCONTROL daily series, OPDI, airport map, per-airport regressor | 247.76 |
| v32 | mixture of two correctors over the same base | 247.11 |
| v33 | base with weather/rotation/NM-consistency features + both correctors rebuilt on it | **244.89** |

## Approach

The prediction is produced in two stages plus a deterministic post-rule. The exact
configuration that produced the current submission is stored in `champion.json`
(fields `base`, `membros`, `pos_regras`) and is rebuilt end-to-end by
`src/train.py submit`.

```
organiser data (2025 + 2026 ranking)
ADS-B, METAR, OPDI, EUROCONTROL series, X-Plane apt.dat
            │
            ▼
      feature cache  ──►  BASE model (shared)
                              two-stage: copy classifier × normal regressor
                              + lines for flights without an NM record
                              + LOBT window projection
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        corrector member 1          corrector member 2      (cross-fitted, out-of-block)
                 └────────────┬────────────┘
                              ▼
                     simple average of members
                              ▼
                 LOBT window projection (± 3606 s)
                              ▼
                 post-rule "Rome"  ──►  submission parquet
```

### 1. Base model (`src/models.py`, `src/experiment.py`)

The target has a heavy tail: a fraction of the official off-block timestamps is a
*copy* of a planned time, which produces true taxi-out values of several hours.
Those outliers are kept in training and in validation — they are ~37 % of the
squared error, and removing them makes the model blind to them.

* **Stage 1** — LightGBM classifier of `eq = |BLOCK − SCHED| ≤ 60 s`, i.e. "the
  off-block timestamp is a copy of the scheduled time".
* **Stage 2** — L2 LightGBM regressor trained only on the normal flights (`~eq`),
  optionally one model per airport (`--base-por-apt`) and with the target clipped
  at 2 h (`--reg-corte 7200`).
* **Expectation mixture** — `ŷ = p·(MVT − SCHED) + (1 − p)·ŷ_normal`, floored at 0.
  Never `argmax`: a wrong class costs hours².
* **Flights without a Network Manager record** (`nm_missing`) — the prediction is
  replaced by a per-airport line `y = a + b·(MVT − SCHED)`, fitted on the training
  fold, and only where `MVT − SCHED` exceeds `--nm-min-ms` (6 h in the champion).
* **LOBT window** (`--janela-lobt`) — in 100 % of the 2025 departures that have a
  LOBT, `|BLOCK − LOBT| ≤ 3606 s`. Every prediction is projected onto
  `MVT − LOBT ± 3606 s`, and the copy probability `p` is zeroed when SCHED falls
  outside that window. This rule was verified to hold in 2026 as well.
* **Features** (`src/features.py`, `src/contexto.py`, `src/plano13.py`): P10
  reference per (airport, stand, runway) computed inside the training fold only;
  time between each planned/NM timestamp (SCHED, LOBT, IOBT, EOBT, AOBT_3) and the
  take-off; differences between those timestamps and rounding flags; airport and
  runway congestion in 10/20/30/60 min windows; hour, weekday, categoricals;
  `nm_missing`; ADS-B derived columns (`adsb_*`); METAR (`met_*`); stand rotation
  and airline statistics.

### 2. Corrector (`src/stack.py`, `src/crossfit.py`)

A second family of models learns the residual `y − pred_base` from **out-of-block**
base predictions: the year is split into 2-month blocks and the base is retrained
without the block it predicts (`src/crossfit.py`), mirroring the way the ranking set
is built. Inputs are the base prediction, airport, `nm_missing`, hour,
`to_takeoff_from_*`, `adsb_*`, external-data columns (`ext_*`, `met_*`, `map_*`,
`sup_*`, `cel_*`) and the distances to both edges of the LOBT window
(`dist_lo`, `dist_hi`). The output is projected back onto the window.

Each corrector run (`--conjunto`) is itself the average of a global LightGBM, a
per-airport LightGBM and a CatBoost model. The champion averages **two** corrector
members over the same base; the member list lives in `champion.json`.

### 3. Post-rule (`src/pos_regras.py`)

`roma`: LIRF departures without an NM record that take off 15–30 h after the
scheduled time are a mixture of "copy of SCHED" and "off-block recorded on the SCHED
date (24 h + taxi)". The prediction for those flights is replaced by the expectation
of the two cases, with the mixing weight fitted outside January and July.
It affects 4 flights in the ranking set.

### Validation

`src/experiment.py` simulates the ranking set (January + July 2025, target blanked
exactly as the organiser does) and appends one line per run to `experiments.jsonl`.
`src/compare.py` compares two runs with a paired day-level bootstrap and only
promotes a challenger if the gain is ≥ 10 s with a positive 95 % CI, the gain
survives removing the 10 most-improved flights, and the CI is positive in January
and July separately. Slice metrics (`normais_nm`, `alarmes_falsos`, `cauda_copia`,
`sem_loteria`) are reported alongside the full RMSE, because the full RMSE is
dominated by a handful of unexplainable multi-hour flights.

## External datasets

All external data is openly licensed and downloadable with the commands below.
None of the 2026 organiser ground truth is used; 2026 months enter only as model
input (ADS-B traces and weather), like any other feature.

| Source | What we use | License | How to download |
|---|---|---|---|
| **adsb.lol** global history: [`globe_history_2025`](https://github.com/adsblol/globe_history_2025), [`globe_history_2026`](https://github.com/adsblol/globe_history_2026) | daily worldwide ADS-B/MLAT traces (one release per day, 2–4 GB); we keep only points on the ground or ≤ 3000 ft within ±0.10° of the 10 airports | **ODbL 1.0** | `bin/run src/adsb.py baixar --dias 2025-01,…,2026-07 --procs 10` — picks a mirror from `PREFERRED_RELEASES.txt`, streams the tar and writes `$PRC_ADSB_RAIZ/cut/YYYY-MM-DD.parquet` (~7–19 MB/day; 427 days ≈ 5.7 GB) |
| **EUROCONTROL daily series** — <https://ansperformance.eu/csv/>: `atfm_slot_adherence`, `all_pre_departure_delays`, `atc_pre_departure_delays` (2025, 2026) | per airport and day: regulated flights, departures off slot, total and ATC pre-departure delay per flight | public EUROCONTROL data, free use with attribution | `bin/run src/externos.py baixar` → `data/externo/*.csv` (~33 MB) |
| **OPDI v0.0.2** (EUROCONTROL / OpenSky) — <https://www.opdi.aero/> — flight lists 2025-01…2025-12, 2026-01, 2026-07 | one flight per row: `icao24`, `adep`, `ades`, `first_seen`, `last_seen` (processed ADS-B); used for aircraft ground time since the previous landing | open data, "freely used … provided that the data source is attributed" | `bin/run src/externos.py baixar` → `data/externo/opdi/flight_list_YYYYMM.parquet` (~30–55 MB/month) |
| **METAR** from IEM ASOS, Iowa State University — <https://mesonet.agron.iastate.edu/request/download.phtml> — 2025-01-01…2026-08-01, 10 airports | half-hourly surface observations: temperature, dew point, wind, gusts, visibility, `wxcodes`, ceiling; de-icing proxy | public IEM/NOAA data, free use with attribution | `bin/run src/plano13.py baixar` → `data/externo/metar/<ICAO>.csv` (~1.5 MB/airport) |
| **X-Plane `apt.dat`** from the Airport Scenery Gateway — <https://gateway.x-plane.com/api> — the 10 airports | stand, taxiway and runway-threshold geometry → stand-to-runway distances | **GNU GPL** (per the [official apt.dat format specification](https://developer.x-plane.com/article/airport-data-apt-dat-file-format-specification/)); keep the copyright notice when redistributing | `bin/run src/mapa.py` → `data/mapa/distancias.parquet` (`map_*` columns) |

The challenge data itself (12 monthly training files, `ranking.parquet`,
`submitting.parquet`) is distributed by the organisers through an OpenSky MinIO
bucket and requires the per-team credentials they issue; it is **not** redistributed
here. See the organisers' [data page](https://prc-data-challenge-2026.netlify.app/data.html).

## Reproducing the submission

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # BUCKET_ACCESS_KEY/SECRET from the organisers, TEAM_NAME
export PRC_ADSB_RAIZ=/path/with/6GB        # ADS-B cut-outs and events.parquet

# 1. Raw data
.venv/bin/python src/s3.py download        # organiser parquet files → data/
bin/run src/adsb.py baixar --dias 2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-07 --procs 10
bin/run src/externos.py baixar             # EUROCONTROL daily series + OPDI flight lists
bin/run src/plano13.py baixar              # METAR of the 10 airports
bin/run src/mapa.py                        # X-Plane apt.dat → data/mapa/distancias.parquet

# 2. Derived inputs
bin/run src/adsb_events.py                 # per-flight ADS-B events → $PRC_ADSB_RAIZ/events.parquet
bin/run src/cache.py                       # feature cache for the 5 splits

# 3. Base model (configuration of the champion)
bin/run src/experiment.py base --model two_stage_nm --seed 0 \
    --nm-min-ms 21600 --janela-lobt --reg-corte 7200 \
    --base-ctx --base-p13 --base-por-apt

# 4. Corrector members, cross-fitted on the base (ids printed by the previous step)
bin/run src/stack.py m0 --crossfit --base <base-id> --corretor-rounds 500 \
    --conjunto --externos --plano13 --dist-plano --mapa --corretor-ref
bin/run src/stack.py m1 --crossfit --base <base-id> --corretor-rounds 500 \
    --conjunto --externos --plano13 --dist-plano --mapa --corretor-ref \
    --fila --superficie --corretor-params '{"num_leaves": 127}' --reusar-oof <m0-id>

# 5. Decide against the current champion (writes champion.json when it wins)
bin/run src/compare.py <m1-id> --promover

# 6. Rebuild the champion on the full year and write the submission file
bin/run src/train.py submit 33             # → submissions/<TEAM_NAME>_v33.parquet
.venv/bin/python src/s3.py submit submissions/<TEAM_NAME>_v33.parquet
```

`champion.json` is versioned, so steps 3–5 can be skipped: `src/train.py submit N`
reads it, retrains the base and every corrector member on the full year, averages
the members and applies the post-rules. `src/train.py submit N --corrida <id>`
rebuilds a single run instead of the champion ensemble.

Useful extras:

```bash
bin/run src/teto.py <base.parquet> <new.parquet> --oficial-base <RMSE> [--min-ms 21600] [--salvar out.parquet]
bin/run src/compare.py <id> --promover --aceitar-fragil   # promote a non-significant gain explicitly
.venv/bin/python -m pytest -q
```

### Hardware and runtime

Measured on a Xeon E5-2670 v3 (6 physical cores used via `bin/run`, which pins the
job with `taskset` and lowers its priority) with 15 GB of RAM, CPU only:

| Step | Time | Peak RAM | Disk |
|---|---|---|---|
| ADS-B download + cut (427 days) | ~1 day with 10 processes | low | 5.7 GB |
| `adsb_events.py` (427 days → ~1.26 M take-offs) | not benchmarked | — | 26 MB (`events.parquet`) |
| `cache.py` (5 splits) | ~3 min | 4.8 GB | ~0.9 GB (`data/cache`) |
| `experiment.py` (base) | ~9 min | 5.25 GB | — |
| `stack.py --crossfit` (first member, computes the out-of-block base) | ~50 min | 6.2 GB | out-of-block cache in `data/cache/oof_base/` |
| `stack.py --crossfit --reusar-oof` (further members) | ~10 min | 6.2 GB | — |
| `train.py submit N` (full year, 2 members) | ~38 min | 7.10 GB | — |

LightGBM runs with `num_threads=12`, `deterministic` and `force_row_wise`, so the
same seed reproduces the same numbers on the same machine; only the CatBoost model
inside `--conjunto` may differ in the last digits. A GPU is optional
(`PRC_DEVICE=gpu`, `--motor xgb`, `--corretor-xgb`) and was measured to be neither
faster nor better on this problem.

## Repository layout

```
bin/run              # hardware budget (taskset + nice) for every heavy command
src/s3.py            # organiser bucket: list, download, submit (MinIO)
src/cache.py         # feature cache for the splits (train2025, holdout2025, full2025, ranking)
src/features.py      # features and the P10 reference
src/models.py        # SingleLGBM, TwoStage, TwoStageNM and the expectation mixture
src/experiment.py    # one run on the calibrated simulation of the ranking set
src/compare.py       # paired day-level bootstrap, promotion rule, champion.json
src/teto.py          # upper bound of the official gain before submitting
src/train.py         # rebuilds the champion on the full year and writes the submission
src/adsb.py          # daily adsb.lol cut-out around the 10 airports (ODbL)
src/adsb_events.py   # take-off and off-block events from ADS-B, matching, adsb_* features
src/stack.py         # corrector on top of a base run (out-of-fold, or --crossfit)
src/crossfit.py      # out-of-block base predictions (2-month blocks)
src/contexto.py      # arrival taxi-in and neighbour features (ctx_*)
src/externos.py      # EUROCONTROL daily series and OPDI (ext_*)
src/plano13.py       # METAR, stand rotation, NM consistency, airline (met_*)
src/mapa.py          # X-Plane apt.dat geometry (map_*)
src/refcel.py        # airport × stand × runway cell statistics (cel_*)
src/superficie.py    # aircraft on the surface at the estimated push-back (sup_*)
src/campeao.py       # champion v2: members, averaged prediction, run registry
src/pos_regras.py    # post-rules applied to the final prediction (Rome)
src/esteira.py       # 24/7 experiment queue; src/regua.py: its acceptance rule
tests/               # pytest
experiments.jsonl    # one line per run (versioned)
champion.json        # current champion: base, members, post-rules (versioned)
```

## License

This repository is released under the **GNU General Public License v3.0** — see
[`LICENSE`](../../LICENSE). Copyright © 2026 team `outgoing-boat`.

```
This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
```

## Attributions

* Challenge data: EUROCONTROL Performance Review Commission and the OpenSky Network,
  PRC Data Challenge 2026.
* ADS-B traces: [adsb.lol](https://adsb.lol/) global history, ODbL 1.0 — contains
  information from adsb.lol, made available under the Open Database License.
* OPDI: Open Performance Data Initiative (EUROCONTROL / OpenSky Network),
  <https://www.opdi.aero/>.
* Daily ATFM and pre-departure delay series: EUROCONTROL,
  <https://ansperformance.eu/csv/>.
* METAR observations: Iowa Environmental Mesonet, Iowa State University
  (ASOS/AWOS/METAR archive), built on NOAA data.
* Airport geometry: X-Plane Airport Scenery Gateway `apt.dat` data, released under
  the GNU GPL by the X-Plane Scenery Gateway contributors.
* Third-party libraries: pandas, PyArrow, NumPy, LightGBM, CatBoost, XGBoost,
  MinIO Python SDK, python-dotenv, psutil, pytest.

### Originality

All modelling code here is original work by the team. Published competitor
repositories and the challenge Discord were read for *ideas* (for instance, the
existence of a hard LOBT window was reported by another team and then re-derived and
re-implemented from scratch on our own data); no third-party competition code was
copied or adapted.
