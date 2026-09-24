"""Features para prever taxi-out (TAXITIME_SEC_mvt) das decolagens."""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "TAXITIME_SEC_mvt"
ID = "MVT_ID_mvt"
AIRPORT = "AIRPORT"  # aeroporto que reporta: ADEP nas decolagens, ADES nos pousos

TIME_COLS = [
    "MVT_TIME_UTC_mvt",
    "BLOCK_TIME_UTC_mvt",
    "SCHED_TIME_UTC_mvt",
    "LOBT_flt",
    "IOBT_flt",
    "EOBT_1_flt",
    "AOBT_3_flt",
]

CATEGORICAL = [
    AIRPORT,
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_RULE_mvt",
    "FLIGHT_TYPE_flt",
]

# Horários planejados/NM comparados à decolagem: taxi-out + atraso de saída.
PLAN_REFS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]

WINDOWS_MIN = [10, 20, 30, 60]

# Referência "sem impedimento" (P10), da chave mais específica à mais geral.
REF_KEYS = [
    [AIRPORT, "STAND_mvt", "RUNWAY_mvt"],
    [AIRPORT, "RUNWAY_mvt"],
    [AIRPORT],
]
REF_MIN_FLIGHTS = 10


def load(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    for col in TIME_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df[AIRPORT] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    return df


def _key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    key = df[cols[0]].astype("string").fillna("?")
    for col in cols[1:]:
        key = key + "|" + df[col].astype("string").fillna("?")
    return key


def _counts_around(
    events: pd.DataFrame, queries: pd.DataFrame, cols: list[str], prefix: str
) -> pd.DataFrame:
    """Para cada consulta, quantos eventos do mesmo grupo ocorrem antes/depois dela."""
    ev_key = _key(events, cols)
    q_key = _key(queries, cols)
    ev_t = events["MVT_TIME_UTC_mvt"].dt.as_unit("s").astype("int64").to_numpy()
    q_t = queries["MVT_TIME_UTC_mvt"].dt.as_unit("s").astype("int64").to_numpy()

    out = np.zeros((len(queries), 2 * len(WINDOWS_MIN)), dtype=np.int32)
    ev_by_key = {k: np.sort(ev_t[idx]) for k, idx in ev_key.groupby(ev_key).indices.items()}
    for k, idx in q_key.groupby(q_key).indices.items():
        ev = ev_by_key.get(k)
        if ev is None:
            continue
        q = q_t[idx]
        at = np.searchsorted(ev, q, side="left")
        after = np.searchsorted(ev, q, side="right")
        for j, w in enumerate(WINDOWS_MIN):
            s = w * 60
            out[idx, 2 * j] = at - np.searchsorted(ev, q - s, side="left")
            out[idx, 2 * j + 1] = np.searchsorted(ev, q + s, side="right") - after

    names = [f"{prefix}_{d}_{w}m" for w in WINDOWS_MIN for d in ("prev", "next")]
    return pd.DataFrame(out, index=queries.index, columns=names)


def build(movements: pd.DataFrame) -> pd.DataFrame:
    """Recebe todos os movimentos de um período (DEP + ARR) e devolve as DEP com features.

    As contagens usam MVT_TIME (decolagem/pouso), que existe no ranking; nunca usam
    BLOCK_TIME nem o alvo.
    """
    movements = movements[movements["MVT_TIME_UTC_mvt"].notna()]
    dep = movements[movements["PHASE_mvt"] == "DEP"].copy()
    arr = movements[movements["PHASE_mvt"] == "ARR"]
    t = dep["MVT_TIME_UTC_mvt"]

    dep["hour"] = t.dt.hour + t.dt.minute / 60
    dep["dow"] = t.dt.dayofweek
    for col in PLAN_REFS:
        if col in dep:
            dep[f"to_takeoff_from_{col}"] = (t - dep[col]).dt.total_seconds()
    # O off-block oficial costuma ser cópia de um destes horários (AOBT_3 em ~38% dos
    # voos, EOBT/LOBT ~20%, SCHED ~17%). Diferenças entre eles e o arredondamento
    # (segundos zerados, minuto múltiplo de 5) ajudam a descobrir qual foi copiado.
    for i, a in enumerate(PLAN_REFS):
        for b in PLAN_REFS[i + 1 :]:
            if a in dep and b in dep:
                dep[f"gap_{a}_{b}"] = (dep[a] - dep[b]).dt.total_seconds()
        if a in dep:
            dep[f"round_{a}"] = (dep[a].dt.second == 0).astype("float") + (
                dep[a].dt.minute % 5 == 0
            ).where(dep[a].notna()).astype("float")

    parts = [
        _counts_around(dep, dep, [AIRPORT], "apt_dep"),
        _counts_around(arr, dep, [AIRPORT], "apt_arr"),
        _counts_around(dep, dep, [AIRPORT, "RUNWAY_mvt"], "rwy_dep"),
    ]
    return pd.concat([dep, *parts], axis=1)


def fit_reference(train_dep: pd.DataFrame) -> list[pd.DataFrame]:
    """P10 do taxi-out por chave, só onde há ao menos REF_MIN_FLIGHTS voos ≤ P10."""
    tables = []
    for cols in REF_KEYS:
        g = train_dep.groupby(cols, observed=True)[TARGET]
        p10 = g.quantile(0.10).rename("p10")
        n_below = (
            train_dep.join(p10, on=cols)
            .assign(below=lambda d: d[TARGET] <= d["p10"])
            .groupby(cols, observed=True)["below"]
            .sum()
        )
        tables.append(p10[n_below >= REF_MIN_FLIGHTS].reset_index())
    return tables


def apply_reference(dep: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.Series:
    ref = pd.Series(np.nan, index=dep.index)
    for cols, table in zip(REF_KEYS, tables):
        vals = dep[cols].merge(table, on=cols, how="left")["p10"].to_numpy()
        ref = ref.fillna(pd.Series(vals, index=dep.index))
    return ref


def feature_columns(df: pd.DataFrame) -> list[str]:
    derived = [
        c for c in df.columns if c.startswith(("to_takeoff_from_", "apt_", "rwy_", "gap_", "round_"))
    ]
    return [*CATEGORICAL, "hour", "dow", "ref_p10", *derived]


def as_categories(frames: list[pd.DataFrame]) -> None:
    """Mesmo vocabulário de categorias em todos os frames (treino, validação, ranking)."""
    for col in CATEGORICAL:
        cats = pd.Index(sorted(set().union(*(f[col].dropna().astype(str) for f in frames))))
        for f in frames:
            f[col] = pd.Categorical(f[col].astype("string"), categories=cats)
