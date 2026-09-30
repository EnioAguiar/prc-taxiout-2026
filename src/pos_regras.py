"""Regras aplicadas à previsão final, depois do modelo (lista `pos_regras` da campeã).

roma: voos de LIRF sem NM que decolam 15–30 h depois do horário programado. Parte deles tem o
off-block gravado na data do SCHED ("24 h + táxi"), parte é cópia do SCHED; a previsão é a
esperança das duas, ajustada fora de jan/jul (q = 0,62; ver README "Pós-regra de Roma").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

MS = "to_takeoff_from_SCHED_TIME_UTC_mvt"
ROMA_A, ROMA_B = 54310.76, 0.38
ROMA_MIN_S, ROMA_MAX_S = 15 * 3600, 30 * 3600


def roma(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    ms = df[MS].to_numpy(float)
    sel = ((df[F.AIRPORT].astype(str) == "LIRF").to_numpy() & df["FLIGHT_ID_mvt"].isna().to_numpy()
           & (ms > ROMA_MIN_S) & (ms <= ROMA_MAX_S))
    out = np.asarray(pred, float).copy()
    out[sel] = ROMA_A + ROMA_B * ms[sel]
    return out


REGRAS = {"roma": roma}


def aplicar(df: pd.DataFrame, pred: np.ndarray, regras: list[str]) -> np.ndarray:
    for nome in regras:
        if nome not in REGRAS:
            raise ValueError(f"regra desconhecida: {nome}")
        pred = REGRAS[nome](df, pred)
    return pred
