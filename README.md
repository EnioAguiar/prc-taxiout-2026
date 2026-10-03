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
| Best official score | **243.9532 s** (v37) |
| Simulation of the same model (Jan+Jul 2025) | 294.86 s full · 227.74 s without the lotteries |
| Airports | EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH |
| Stack | Python 3, pandas/pyarrow, LightGBM, CatBoost (XGBoost optional) |
| Hardware | 6 physical cores, 15 GB RAM, no GPU required |
| License | GNU GPL v3 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) |

The work diary, in Portuguese and in chronological order, is [`docs/diario.md`](docs/diario.md);
the individual studies are in [`docs/research/`](docs/research); the design documents and
the plans that were executed are in [`docs/superpowers/`](docs/superpowers).

Official score history (RMSE on the public leaderboard). Every submission is one line
of `submissions.jsonl`, with the date, the reason and the simulated score it was chosen by:

| Version | Change | Official RMSE |
|---|---|---|
| v1 | first model, outliers dropped from training | 514.53 |
| v2 | outliers kept, unit bug in congestion windows fixed | 384.74 |
| v3 | two-stage model (copy classifier + regressor) | 338.67 |
| v5 | per-airport lines for flights without a Network Manager record | 330.95 |
| v6 | ADS-B features from adsb.lol | 314.76 |
| v9 | cross-fitted corrector + LOBT window | 275.90 |
| v11 | arrival context and neighbour features in the corrector | 266.81 |
| v13 | Rome post-rule | 264.35 |
| v17–v30 | weather (METAR), EUROCONTROL daily series, OPDI, airport map, per-airport regressor | 247.76 |
| v32 | mixture of two correctors over the same base | 247.11 |
| v33 | base with weather/rotation/NM-consistency features, both correctors rebuilt on it | 244.89 |
| v34 | gate-retention block (`--retencao`) in corrector 1 | 244.18 |
| v35 | corrector 1 no longer trains on the rule-served rows (`--corretor-sem-regra`) | 244.13 |
| v36 | third corrector member (`--pista --retencao --corretor-sem-regra`) | 243.9755 |
| v37 | fourth member over a second base (`--cls-peso quad`) | **243.9532** |

## 1. Model design

The design is driven by one property of the target: **the label is not a single
population**. A fraction of the official off-block timestamps is a *copy* of a planned
time, which produces true taxi-out values of several hours. In the 2025 holdout, 84.7 % of
the flights with `y > 3 h` satisfy `|BLOCK − SCHED| ≤ 60 s`; those outliers are ~37 % of the
squared error. They are kept in training and in validation — removing them (version v1)
makes the model blind to them and makes validation optimistic.

So the model is a mixture of two regimes plus a learned residual, not one regressor:

```
organiser data (2025 + 2026 ranking)
ADS-B, METAR, OPDI, EUROCONTROL series, X-Plane apt.dat
            │
            ▼
      feature cache  ──►  BASE model (shared by every member)
                              stage 1: copy classifier  p = P(|BLOCK − SCHED| ≤ 60 s)
                              stage 2: L2 regressor on the normal flights only
                              mixture by expectation, never argmax
                              + per-airport lines for flights without an NM record
                              + LOBT window projection
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        member e122      member e76_m1     member e140      (cross-fitted, out-of-block)
             └────────────────┼────────────────┘
                              ▼
                     simple average of members
                              ▼
                 LOBT window projection (± 3606 s)
                              ▼
                 post-rule "Rome"  ──►  submission parquet
```

The exact configuration that produced the current submission is `champion.json`
(fields `base`, `oof`, `membros`, `pos_regras`) and is rebuilt end-to-end by
`src/train.py submit`.

### 1.1 Why two stages, and why an expectation mixture

A single regressor on this target has to interpolate between "22 minutes of taxi" and
"9 hours because the off-block field was filled with the schedule". It cannot: the loss is
squared, so it answers with the conditional mean of a bimodal distribution and is wrong in
both regimes at once.

* **Stage 1** — LightGBM classifier of `eq = |BLOCK − SCHED| ≤ 60 s` ("the off-block
  timestamp is a copy of the scheduled time").
* **Stage 2** — L2 LightGBM regressor trained **only** on the normal flights (`~eq`), with
  the target clipped at 2 h (`--reg-corte 7200`) so the tail does not distort the body, and
  optionally one model per airport added to the global one (`--base-por-apt`).
* **Mixture by expectation** — `ŷ = p·(MVT − SCHED) + (1 − p)·ŷ_normal`, floored at 0.

The last point is the design decision that matters most, and it is deliberate:
**never `argmax`**. Choosing the most likely class and committing to it minimises
classification error, not RMSE. A missed copy at `p = 0.45` costs `(MVT − SCHED)²` —
hours squared. The expectation spends `p · (MVT − SCHED)` seconds of prediction on a
possibility, which is exactly what a squared loss asks for. The flip side is false alarms
(normal flights predicted above 1 h); they are tracked as a dedicated slice
(`alarmes_falsos`), and the two attempts to reduce them with a sharper decision rule both
failed (§3).

Two more pieces are not learned but derived from the data and verified:

* **Flights without a Network Manager record** (`nm_missing`) — the prediction is replaced
  by a per-airport line `y = a + b·(MVT − SCHED)`, fitted inside the training fold, and only
  where `MVT − SCHED` exceeds `--nm-min-ms` (6 h in the champion). Applying it to all
  `nm_missing` flights was measured and is worse (it was version v4: simulated −42 s,
  official −1.5 s; diagnosis in
  [`docs/research/2026-09-24-diagnostico-v4.md`](docs/research/2026-09-24-diagnostico-v4.md)).
* **LOBT window** (`--janela-lobt`) — in **100 %** of the 2,062,577 departures of 2025 that
  have a LOBT, `|BLOCK − LOBT| ≤ 3606 s`. Every prediction is projected onto
  `MVT − LOBT ± 3606 s`, and the copy probability `p` is zeroed when SCHED falls outside the
  window. This is a hard property of the data, not a model: measurement in
  [`docs/research/2026-09-27-janela-lobt.md`](docs/research/2026-09-27-janela-lobt.md).
  It was confirmed to hold in 2026 by a single diagnostic submission (v10, §2.3).

### 1.2 Why the corrector is cross-fitted, and against what

A second family of models learns the residual `y − pred_base`. The naive version — train the
corrector on the base's own predictions — learns the base's *training* error, which is not
the error it will have on the ranking set. The correct input is an **out-of-block** base
prediction: the year is split into 2-month blocks and the base is retrained without the
block it predicts (`src/crossfit.py`), mirroring exactly how the ranking set is built
(January and July held out of a year of training).

This is the single change that forced the pipeline to be honest, and it moved the official
score from 314.76 (v6) to 275.90 (v9) together with the LOBT window.

Corrector inputs: the base prediction, airport, `nm_missing`, hour,
`to_takeoff_from_*`, `adsb_*`, the external-data columns (`ext_*`, `met_*`, `map_*`,
`sup_*`, `cel_*`, `pista_*`, `ret_*`) and the distances to both edges of the LOBT window
(`dist_lo`, `dist_hi`). The output is projected back onto the window.

Each corrector run (`--conjunto`) is itself the average of a global LightGBM, a per-airport
LightGBM and a CatBoost model — the gain comes from the average, not from any single one of
them (CatBoost alone: 309.53 against 309.27 for the LightGBM).

### 1.3 The current champion (`champion.json`)

Four corrector members averaged with equal weights, **over two different bases**. Each
member carries its own base in `config.base`/`config.base_config`; the submission groups the
members by base, trains one final base per group in sequence and averages everything at the
end (`src/train.py`, `grupos_por_base`).

Base A — `20260930-130008-e76_base`, shared by three members:

```
--model two_stage_nm --seed 0 --nm-min-ms 21600 --janela-lobt --reg-corte 7200
--base-ctx --base-p13 --base-por-apt          (400 + 400 rounds)
```

Base B — `20261001-150429-e113_base`, the same recipe plus `--cls-peso quad`.

Out-of-block base predictions are computed once per base (`20260930-130921-e76_m0` and
`20261001-151412-e113_m0`) and reused by every member of that base (`--reusar-oof`).

| Member | Base | Blocks on top of the common recipe |
|---|---|---|
| `20261001-185827-e122` | A | `--corretor-sem-regra --retencao` |
| `20260930-140011-e76_m1` | A | `--fila --superficie --corretor-params '{"num_leaves": 127}'` |
| `20261002-015703-e140` | A | `--pista --retencao --corretor-sem-regra` |
| `20261001-160427-e113_m1` | B | `--fila --superficie --corretor-params '{"num_leaves": 127}'` |

Common recipe of every member: `--crossfit --conjunto --corretor-rounds 500 --externos
--plano13 --dist-plano --mapa --corretor-ref --seed 0`.
Post-rules: `roma`.

Equal weights are a decision, not an oversight: fitted blend weights were tested and the
members are strongly correlated, so a fitted weight buys noise. What the ensemble is
actually exploiting is *decorrelation of the blocks* — each member reads a different part of
the airport state. Adding a member over a **different base** is the same idea one level
down: the fourth member above only changes the base (`--cls-peso quad`) and took the
holdout simulation from 295.63 to 294.86 (no-lottery 228.69 → 227.74). **It did not
transfer**: v37 scored 243.9532 against 243.9755 for v36 (−0.02 s). 129 % of the simulated
gain came from 20 flights of the tail, and the member was picked among 266 saved runs on the
same holdout. That failure is what produced the transfer gate of §2.4.

### 1.4 The feature blocks

| Module | Columns | What it reads |
|---|---|---|
| `src/features.py` | base features | P10 reference per (airport, stand, runway) computed inside the training fold only; time from each planned/NM timestamp (SCHED, LOBT, IOBT, EOBT, AOBT_3) to the take-off; differences between those timestamps and rounding flags; airport and runway congestion in 10/20/30/60 min windows; hour, weekday, categoricals; `nm_missing` |
| `src/adsb_events.py` | `adsb_*` | off-block and take-off observed in the adsb.lol trace, ground speed at the first point, points and gaps on the ground |
| `src/contexto.py` | `ctx_*` | taxi-in of the arrivals per airport and runway, last arrival at the same stand, neighbour departures' `MVT − AOBT_3` |
| `src/plano13.py` | `met_*`, queue counts | METAR up to 2 h before the movement, de-icing proxy, stand rotation, NM consistency, airline |
| `src/externos.py` | `ext_*` | EUROCONTROL daily series (regulated flights, off-slot departures, pre-departure delay), OPDI ground time since the previous landing, SCHED-copy rate per airline |
| `src/mapa.py` | `map_*` | stand → runway-threshold taxi distance from the X-Plane `apt.dat` geometry |
| `src/refcel.py` | `cel_*` | median, P90, spread and size of the (airport × stand × runway) cell |
| `src/superficie.py` | `sup_*` | aircraft on the surface at the *estimated* push-back (`MVT − pred`) |
| `src/pista.py` | `pista_*` (`--pista`) | runway configuration of the airport in a ±60 min window and whether it changed in the last hour, dominant runway, same-runway queue between `AOBT_3` and take-off weighted by wake category, cadence of the last 20 departures |
| `src/pista.py` | `ret_*` (`--retencao`) | gate retention at the *real* push clock (`AOBT_3`): departures already past their EOBT that have not pushed, departures active on the same runway, mean `AOBT_3 − SCHED` of the previous 15 min, EWMA of departures |
| `src/roma.py` | `roma_*` (`--roma-tdg`) | `T = D − G` decomposition at LIRF — **measured and switched off**, see §3 |
| `src/fe_auto.py` | `fe_*` (`--fe-auto`) | automatically generated features (OpenFE-style operators) that survived out-of-month screening, a placebo band and the 2025×2026 adversarial gate — **one** column out of 755 candidates, see [`docs/research/2026-10-03-fe-auto.md`](docs/research/2026-10-03-fe-auto.md) |

Two flags change *what is trained on* rather than what is read:

* `--corretor-sem-regra` — the corrector does not train on the rows that the base hands to a
  fixed rule (the `nm_missing` lines and the Rome rule, `models.linhas_de_regra`) and leaves
  the base prediction untouched there. Without it the corrector learns to "correct" the
  output of a deterministic rule, which is noise. Worth −0.05 s official (v35) and it is in
  two of the three members.
* `--treino-sem-regra` — the same idea one level down: the classifier and the regressor of
  the base skip those rows (207 of the 2.085 M departures of 2025). The lines themselves stay
  fitted on all `nm_missing` rows.

## 2. Validation: how a candidate is accepted

The public leaderboard is a single scalar on a hidden set, with 5 submissions per UTC day.
It cannot be used as a validation signal without overfitting it, and the organisers monitor
attempts to do so. So the entire decision procedure is local, and it is the part of this
repository that took the most work.

### 2.1 The simulation

`src/experiment.py` builds the ranking set *as the organisers do*: January and July of 2025
are taken out of the training year and their target is blanked, with the same column
treatment and the same per-month composition. One run appends one line to
`experiments.jsonl` (predictions in `runs/<id>.parquet`) with the full RMSE and four slices:

* `normais_nm` — normal flights (`y ≤ 1 h`) that have an NM record: 43 % of the squared
  error, and the only slice that is ~5× more precise than the full RMSE;
* `alarmes_falsos` — normal flights predicted above 1 h: count and share of the squared error;
* `cauda_copia` — tail flights (`y > 1 h`) within 5 min of some planned timestamp;
* `sem_loteria` — RMSE excluding the flights with `y > 3 h` that no planned timestamp
  explains (11 flights in the holdout, 32 % of the squared error).

The full RMSE is reported but is **not** the decision metric, because a handful of
unexplainable multi-hour flights dominate it and move it by tens of seconds for reasons no
feature can see. `sem_loteria` is the metric the acceptance rule uses.

### 2.2 The A/B ruler: halves of days, bootstrap, and a gate

`src/compare.py` compares two runs with a **paired day-level bootstrap** (the pairing is
what gives the resolution: the same flights, two models). Its promotion rule — the one used
for every manual decision — accepts a challenger only if **all** of the following hold:

1. gain ≥ 10 s with a 95 % CI strictly above zero;
2. the gain survives removing the 10 most-improved flights, and is ≥ 10 % of the full gain;
3. the CI is positive in January **and** in July separately.

Anything else is `FRÁGIL` and does not promote; `--aceitar-fragil` promotes explicitly and is
only used after `src/teto.py` has bounded the achievable official gain and the human has
agreed. Both exceptions are recorded in the diary.

Criterion 2 exists because of a measured failure: version v4's simulated gain of 42 s came
from 10 LIRF flights (120.9 % of the gain; without them v4 was 9.7 s *worse*), and the
official gain was 1.5 s. Criterion 3 exists because the ranking set is two very different
months.

`src/regua.py` is the automated version used by the 24/7 experiment queue
(`src/esteira.py`). It splits the days of the holdout into two halves:

* in half **A** a candidate is only *selected* with `sem_loteria` gain ≥ **0.3 s** and the
  lower CI bound > 0;
* the proposal selected in A is confirmed **blind** in half **B** with gain > 0 — B never
  selects, it only confirms;
* on all days together, the `sem_loteria` gain needs a lower CI bound > 0 and the full RMSE
  must not drop more than 0.5 s.

B is a budget on multiple comparisons: the queue runs ~75 candidates per day, and without a
held-back half the best of 75 is a winner by selection alone. Queries to half B are counted
and reported in `docs/esteira.md`; the split is reseeded (`esteira.py semente N`) when the
count gets high.

### 2.3 Submissions were decided by the simulation, not by the score

No model choice in this repository was made by reading the leaderboard. `champion.json` is
updated by `compare.py`/`regua.py` from the 2025 simulation; `train.py submit N` then rebuilds
exactly that champion on the full year. The official score is used afterwards, as a check on
the simulation ratio (official/simulated has been 0.82–0.87 since v9), and every submission
is recorded in `submissions.jsonl` with its simulated score and the reason it was sent.

Two submissions are documented as deliberate diagnostics rather than improvements, and are
labelled as such in `submissions.jsonl`:

* **v10** — version v6 with *only* the 117 rows that the LOBT window would move. It returned
  284.17, below the guaranteed bound of 288.01, which proved that the window rule derived on
  2025 also holds on the 2026 ranking set. One submission bought a fact the holdout could not
  give.
* **v14** — version v12 without `adsb_lat0`/`adsb_lon0`, to test a 2025 → 2026 drift that a
  2025 holdout cannot see by construction. It came back 1.54 s **worse** and was discarded.

Everything else was promoted locally first.

### 2.4 The transfer check: which regime does the gain come from?

The ruler of §2.2 answers "does the candidate win on the 2025 holdout?". It does not answer
"will that win survive the move to 2026?", and version v37 showed the difference: simulated
−0.77 s of full RMSE, official −0.02 s. The post-mortem is that the simulated gain was in
the wrong place — **+0.586 s of it came from flights with `y > 1 h`** (lottery, not signal),
20 flights were worth 129 % of it, and the member had been picked among 266 runs scored on
the same holdout.

`src/transferencia.py` measures what the ruler cannot. The RMSE gain decomposes **exactly**
into a per-flight sum — `RMSE(b) − RMSE(n) = Σᵢ wᵢ·dᵢ / (Σw · (RMSE_b + RMSE_n))` with
`dᵢ = (y−b)² − (y−n)²` — so the same sum restricted to a subset *is* the share of the gain
that comes from it, in seconds of RMSE, and the parts add up. Three readings follow: the
**body** (`y ≤ 3600 s`), the **tail** (`y > 3600 s`), and the share held by the 20
most-improved flights. A fourth reading reweights the holdout to the 2026 covariate
distribution with importance weights `w = p/(1−p)` from an adversarial 2025-vs-2026
classifier (out-of-fold AUC 0.866), clipped at the 95th percentile (IWCV, Sugiyama 2007).

Backtested on the six consecutive submissions with a known official delta (v30 → v37), the
**body share alone predicts the official delta with a ratio of 1.03 and a mean absolute
error of 0.15 s** (Pearson 0.985), against 0.44 s for the full RMSE and 0.36 s for
`sem_loteria`; the tail share is *anti*-correlated with the official delta (−0.86). So the
gate is:

1. the body share of the gain is positive, with a day-bootstrap 95 % CI strictly above zero;
2. the 20 most-improved flights hold less than 200 % of the body gain;
3. the body share stays positive under the 2026 importance weights.

It accepts 3 of 3 submissions that were worth ≥ 0.65 s officially and refuses 3 of 3 that
were worth ≤ 0.16 s, **v37 included**. The 200 % limit is not a typo: with ~1 s of gain
spread over 344 k flights, concentration is the norm — v33, worth +2.21 s officially, has
125 % of its body gain in 20 flights, and a 50 % limit would have refused it.

`esteira.talvez_enviar` calls the gate before building a submission: the ruler still decides
what enters `champion.json`, but a submission — 2.6 h of `train.py submit` and one of the 5
daily slots — is only produced when the transfer check passes. Measurement, calibration and
the honest limits are in
[`docs/research/2026-10-03-transferencia.md`](docs/research/2026-10-03-transferencia.md).

## 3. What did not work

Negative results, with the measurement that killed each one. All of them are reproducible
from this repository; the code of the discarded mechanisms was removed, but
`experiments.jsonl` keeps the runs.

| Idea | Measured result | Where |
|---|---|---|
| **`T = D − G` decomposition at LIRF** (`src/roma.py`, `--roma-tdg`): predict the gate delay `G = BLOCK − SCHED` and serve `T̂ = max(D − Ĝ, 0)` | The reparametrisation itself is real (643.5 vs 1220.3 s at LIRF, same columns), and the cheap 5-fold screen gained +0.46 s `sem_loteria`. The decision run (`--crossfit`, against champion member `e122`) **lost 1.479 s** of `sem_loteria`, CI −2.580…−0.395, entirely below zero. Cause: `T̂` is a *rival prediction*, not context, so a corrector trained against the weaker out-of-block base over-trusts it. Flag kept, off by default, out of the queue's search space | [`docs/research/2026-10-02-roma-t-d-g.md`](docs/research/2026-10-02-roma-t-d-g.md) |
| **Arrival mirror**: score a taxi-in model trained on 2025 against the ARR rows of `ranking.parquet` (the only true 2026 label) and feed the residual to the corrector | The drift measurement works (EHAM January 2026 residual RMSE 436.5 vs 129.6 in July, and it is a handful of disruption days, not a new level). As a feature it is worthless: \|r\| ≤ 0.055 against our own error at every airport, signs disagreeing per airport at day level, and an **in-sample** linear ceiling of +0.28 s `sem_loteria` — below the 0.3 s gate. It predicts the *magnitude* of our error, not its *direction*, and a squared loss only pays for direction | [`docs/research/2026-10-02-espelho-chegadas.md`](docs/research/2026-10-02-espelho-chegadas.md) |
| **Seed averaging** (`--seeds N`) | Base with 5 seeds: 323.29 vs 323.50, gain 0.2 s (CI −1.9…1.7). Full pipeline `v8_cf` 320.36 vs `v7_cf` 320.67: 0.3 s (CI −1.8…1.8) for 5× the compute. Not adopted | `docs/diario.md`, plan 6 |
| **Map columns in the base** (`--base-mapa`) | 298.97 against 298.05 for the same `map_*` columns only in the corrector — 0.9 s worse. `--mapa` stayed in the corrector, `--base-mapa` is off | `experiments.jsonl` (`v30_base_mapa`) |
| **XGBoost** | As the base engine (`--motor xgb`, GPU): 318.70 against LightGBM; a 0.8/0.2 blend ties at 311.11. As a 4th corrector (`--corretor-xgb`): −0.2 s. LightGBM on GPU: same error (310.84 vs 311.18) and only 15 % faster. Default stays CPU LightGBM, which is bit-for-bit deterministic | `docs/diario.md`, "GPU" |
| **Multiclass copy mixture** (`copy_mix`): classify *which* timestamp the off-block copied | −4.0 s (CI −9.4…0.9). The multiclass part works (false alarms 10.0 % → 8.7 % of the squared error with normal flights untouched); what kills it is the fixed 0.164 rate of the "24 h + taxi" class, which adds ~14,170 s to *every* flight of that cell. Code removed | `docs/diario.md`, plan 3b item 3 |
| **Residual target on `MVT − AOBT_3`** | −0.2 s as a target reparametrisation, −0.6 s as an extra feature. `ref` is already a feature (`to_takeoff_from_AOBT_3_flt`) and the model already uses it; rewriting the target only removes the freedom to ignore it. Code removed | `docs/diario.md`, plan 3b item 4 |
| **Hybrid `p·line + (1−p)·regressor` for `nm_missing`** | −46.5 s. It does reduce false alarms (204 → 173 flights) but destroys the tail: on the 16 true copies in the changed rows the RMSE goes from 816 to 15,101 s. Code removed | `docs/diario.md`, plan 3b item 2b |
| **Per-cell isotonic calibration of `p`** | +0.1 s (CI −0.1…0.3). The classifier is already calibrated (mean out-of-fold `p` 0.0949 against a true copy rate of 0.0972); false alarms come from isolated flights with large `p·ms`, not from a biased cell mean. Code removed | `docs/diario.md`, plan 3b item 2a |
| **Bias correction from the official published monthly taxi-out means** | 311.09 → 311.56. The published mean excludes flights without a reference and its valid fraction moved between 2025 and 2026 (EDDM 0.77 → 0.67) | `docs/diario.md` |
| **ADS-B stand-position detector** and **ADS-B queue counts** | +0.8 s and +0.2 s in the cheap screen, against a 3 s gate. The corrector already learns the "seen moving" delay from `adsb_gs0`/`adsb_lat0`/`adsb_lon0`, and NM congestion windows already carry the queue | `docs/diario.md` |
| **CatBoost alone as the corrector** | 309.53 against 309.27 for LightGBM. It only pays inside the `--conjunto` average | `docs/diario.md` |
| **Dropping `adsb_lat0`/`adsb_lon0`** (anti-drift) | Simulation said −2.1 s, the official score said +1.54 s (v14 = 266.28 against 264.74). Documented as a failed bet, not a measurement | `submissions.jsonl`, v14 |

Two open datasets were surveyed in depth and **not built** — they are hypotheses in the
research notes, never measured, and nothing in the solution depends on them: OPDI
`flight_events` v0.0.2 (ground milestones from the OpenSky network: the same network has
almost no ground coverage at LFPG, EGLL, LIRF, EDDM and LTFM, which are exactly the airports
we are blind at, for ~24 GB of download) and the EUROCONTROL APT-DLY daily cause codes
(de-icing, aerodrome capacity, strikes — day × airport granularity only). See
[`docs/research/2026-10-01-dados-abertos-aeroportos.md`](docs/research/2026-10-01-dados-abertos-aeroportos.md)
and [`docs/research/2026-10-01-regras-e-precedentes.md`](docs/research/2026-10-01-regras-e-precedentes.md).

## 4. External datasets

All external data is openly licensed and downloadable with the commands below.
None of the 2026 organiser ground truth is used; 2026 months enter only as model
input (ADS-B traces and weather), like any other feature. Attributions: [`NOTICE`](NOTICE).

| Source | What we use | License | How to download |
|---|---|---|---|
| **adsb.lol** global history: [`globe_history_2025`](https://github.com/adsblol/globe_history_2025), [`globe_history_2026`](https://github.com/adsblol/globe_history_2026) | daily worldwide ADS-B/MLAT traces (one release per day, 2–4 GB); we keep only points on the ground or ≤ 3000 ft within ±0.10° of the 10 airports | **ODbL 1.0** | `bin/run src/adsb.py baixar --dias 2025-01,…,2026-07 --procs 10` — picks a mirror from `PREFERRED_RELEASES.txt`, streams the tar and writes `data/adsb/cut/YYYY-MM-DD.parquet` (~7–19 MB/day; 427 days ≈ 5.7 GB) |
| **EUROCONTROL daily series** — <https://ansperformance.eu/csv/>: `atfm_slot_adherence`, `all_pre_departure_delays`, `atc_pre_departure_delays` (2025, 2026) | per airport and day: regulated flights, departures off slot, total and ATC pre-departure delay per flight | public EUROCONTROL data, free use with attribution | `bin/run src/externos.py baixar` → `data/externo/*.csv` (~33 MB) |
| **OPDI v0.0.2** (EUROCONTROL / OpenSky) — <https://www.opdi.aero/> — flight lists 2025-01…2025-12, 2026-01, 2026-07 | one flight per row: `icao24`, `adep`, `ades`, `first_seen`, `last_seen` (processed ADS-B); used for aircraft ground time since the previous landing | open data, "freely used … provided that the data source is attributed" | `bin/run src/externos.py baixar` → `data/externo/opdi/flight_list_YYYYMM.parquet` (~30–55 MB/month) |
| **METAR** from IEM ASOS, Iowa State University — <https://mesonet.agron.iastate.edu/request/download.phtml> — 2025-01-01…2026-08-01, 10 airports | half-hourly surface observations: temperature, dew point, wind, gusts, visibility, `wxcodes`, ceiling; de-icing proxy | public IEM/NOAA data, free use with attribution | `bin/run src/plano13.py baixar` → `data/externo/metar/<ICAO>.csv` (~1.5 MB/airport) |
| **X-Plane `apt.dat`** from the Airport Scenery Gateway — <https://gateway.x-plane.com/api> — the 10 airports | stand, taxiway and runway-threshold geometry → stand-to-runway distances | **GNU GPL** (per the [official apt.dat format specification](https://developer.x-plane.com/article/airport-data-apt-dat-file-format-specification/)); the copyright notice is kept in [`NOTICE`](NOTICE) | `bin/run src/mapa.py` → `data/mapa/distancias.parquet` (`map_*` columns) |

The challenge data itself (12 monthly training files, `ranking.parquet`,
`submitting.parquet`) is distributed by the organisers through an OpenSky MinIO
bucket and requires the per-team credentials they issue; it is **not** redistributed
here. See the organisers' [data page](https://prc-data-challenge-2026.netlify.app/data.html).

## 5. Reproducing the submission

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # BUCKET_ACCESS_KEY/SECRET from the organisers, TEAM_NAME

# 1. Raw data
.venv/bin/python src/s3.py download        # organiser parquet files → data/
bin/run src/adsb.py baixar --dias 2025-01,2025-02,2025-03,2025-04,2025-05,2025-06,2025-07,2025-08,2025-09,2025-10,2025-11,2025-12,2026-01,2026-07 --procs 10
bin/run src/externos.py baixar             # EUROCONTROL daily series + OPDI flight lists
bin/run src/plano13.py baixar              # METAR of the 10 airports
bin/run src/mapa.py                        # X-Plane apt.dat → data/mapa/distancias.parquet

# 2. Derived inputs
bin/run src/adsb_events.py                 # per-flight ADS-B events → data/adsb/events.parquet
bin/run src/cache.py                       # feature cache for the splits

# 3. Base model (configuration of the champion)
bin/run src/experiment.py base --model two_stage_nm --seed 0 \
    --nm-min-ms 21600 --janela-lobt --reg-corte 7200 \
    --base-ctx --base-p13 --base-por-apt

# 4. Corrector members, cross-fitted on that base (ids printed by the previous step)
bin/run src/stack.py m0 --crossfit --base <base-id> --corretor-rounds 500 \
    --conjunto --externos --plano13 --dist-plano --mapa --corretor-ref \
    --corretor-sem-regra --retencao
bin/run src/stack.py m1 --crossfit --base <base-id> --corretor-rounds 500 \
    --conjunto --externos --plano13 --dist-plano --mapa --corretor-ref \
    --fila --superficie --corretor-params '{"num_leaves": 127}' --reusar-oof <m0-id>
bin/run src/stack.py m2 --crossfit --base <base-id> --corretor-rounds 500 \
    --conjunto --externos --plano13 --dist-plano --mapa --corretor-ref \
    --pista --retencao --corretor-sem-regra --reusar-oof <m0-id>

# 5. Decide against the current champion (writes champion.json when it wins)
bin/run src/compare.py <run-id> --promover

# 6. Rebuild the champion on the full year and write the submission file
bin/run src/train.py submit 37             # → submissions/<TEAM_NAME>_v37.parquet
.venv/bin/python src/s3.py submit submissions/<TEAM_NAME>_v37.parquet
```

`champion.json` is versioned, so steps 3–5 can be skipped: `src/train.py submit N`
reads it, retrains the base and every corrector member on the full year, averages
the members and applies the post-rules. `src/train.py submit N --corrida <id>`
rebuilds a single run instead of the champion ensemble.

The ADS-B cut-outs and `events.parquet` (~6 GB) default to `data/adsb`, which may be a
symlink to another disk; set `PRC_ADSB_RAIZ` to put them elsewhere.

Useful extras:

```bash
bin/run src/teto.py <base.parquet> <new.parquet> --oficial-base <RMSE> [--min-ms 21600] [--salvar out.parquet]
bin/run src/compare.py <id> --promover --aceitar-fragil   # promote a non-significant gain explicitly
bin/run src/esteira.py add --tipo corretor --receita '{"--pista": true}'   # queue a candidate
bin/run src/esteira.py status                             # queue throughput, failures, half-B queries
bin/run -m pytest tests/ -q
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
| `train.py submit N` (full year, 3 members) | ~45 min | 7.10 GB | — |

LightGBM runs with `num_threads=12`, `deterministic` and `force_row_wise`, so the
same seed reproduces the same numbers on the same machine; only the CatBoost model
inside `--conjunto` may differ in the last digits. A GPU is optional
(`PRC_DEVICE=gpu`, `--motor xgb`, `--corretor-xgb`) and was measured to be neither
faster nor better on this problem (§3).

## 6. Repository layout

```
bin/run              # hardware budget (taskset + nice) for every heavy command
src/s3.py            # organiser bucket: ls, download, submit (MinIO)
src/cache.py         # feature cache for the 5 splits (train2025, holdout2025, blind2025, full2025, ranking2026)
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
src/externos.py      # EUROCONTROL daily series, OPDI and airline copy rate (ext_*)
src/plano13.py       # METAR, stand rotation, NM consistency, airline (met_*)
src/mapa.py          # X-Plane apt.dat geometry (map_*)
src/refcel.py        # airport × stand × runway cell statistics (cel_*)
src/superficie.py    # aircraft on the surface at the estimated push-back (sup_*)
src/pista.py         # runway state (pista_*) and gate retention (ret_*)
src/roma.py          # T = D − G decomposition at LIRF (roma_*); measured, off by default
src/campeao.py       # champion v2: members, averaged prediction, run registry
src/pos_regras.py    # post-rules applied to the final prediction (Rome)
src/esteira.py       # 24/7 experiment queue; src/regua.py: its acceptance rule
src/runlog.py        # per-phase progress, ETA, RAM/CPU/GPU and the run registry
src/memoria.py       # soltar(): gc.collect + malloc_trim after each large del
src/dispositivo.py   # CPU/GPU selection (PRC_DEVICE)
ferramentas/auditoria.py  # local daily self-audit of the working machine (output not versioned)
ferramentas/projecao.py   # local deadline tracking (output not versioned)
ferramentas/prc-esteira.service   # systemd user unit for the experiment queue
tests/               # pytest
experiments.jsonl    # one line per run (versioned)
submissions.jsonl    # one line per submission: date, reason, simulated and official score
champion.json        # current champion: base, oof, members, post-rules (versioned)
docs/diario.md       # work diary (pt-BR), chronological
docs/esteira.md      # current state of the experiment queue
docs/research/       # individual studies, including the negative results of §3
docs/superpowers/    # design documents and executed plans
```

## 7. License

This repository is released under the **GNU General Public License v3.0** — see
[`LICENSE`](LICENSE). Copyright © 2026 team `outgoing-boat`. Copyright notice and
third-party data attributions: [`NOTICE`](NOTICE).

```
This program is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
```

### Originality

All modelling code here is original work by the team. Published competitor
repositories and the challenge Discord were read for *ideas* — for instance, the
existence of a hard LOBT window was reported by another team and then re-derived and
re-implemented from scratch on our own data, and two ideas read that way were
implemented, measured and **rejected** (§3) — but no third-party competition code was
copied or adapted.
