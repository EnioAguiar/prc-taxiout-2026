import argparse

import numpy as np
import pandas as pd

import features as F
from cache import TRUTH
from experiment import config, lottery_mask, metrics

GAPS = [f"to_takeoff_from_{c}" for c in F.PLAN_REFS]
MS = GAPS[0]  # to_takeoff_from_SCHED_TIME_UTC_mvt = MVT − SCHED, em segundos


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Voos com y, ms (MVT − SCHED) e os demais horários planejados nulos por padrão."""
    df = pd.DataFrame(rows)
    for col in [*GAPS, TRUTH, "FLIGHT_ID_mvt"]:
        if col not in df:
            df[col] = np.nan
    return df


def test_loteria_so_pega_cauda_sem_copia():
    df = _frame([
        {TRUTH: 84_240.0, MS: 1_740.0},   # 23 h de taxi sem nenhum horário perto
        {TRUTH: 30_000.0, MS: 29_900.0},  # cauda, mas o SCHED explica (100 s)
        {TRUTH: 900.0, MS: 600.0},        # voo normal
    ])
    assert list(lottery_mask(df, df[TRUTH])) == [True, False, False]


def test_horario_nulo_nao_conta_como_perto():
    df = _frame([{TRUTH: 84_240.0, MS: 84_240.0}])
    df[GAPS[1]] = np.nan
    assert not lottery_mask(df, df[TRUTH])[0]  # o SCHED explica
    df[MS] = np.nan  # sem nenhum horário: continua loteria
    assert lottery_mask(df, df[TRUTH])[0]


def test_metricas_por_fatia():
    df = _frame([
        {TRUTH: 600.0, MS: 500.0, "FLIGHT_ID_mvt": 1.0},   # normal com NM
        {TRUTH: 600.0, MS: 500.0},                         # normal sem NM, alarme falso
        {TRUTH: 7_200.0, MS: 7_100.0, "FLIGHT_ID_mvt": 2.0},  # cauda que é cópia
        {TRUTH: 84_240.0, MS: 1_740.0, "FLIGHT_ID_mvt": 3.0},  # loteria
    ])
    df[F.AIRPORT] = "LIRF"
    pred = np.array([700.0, 5_000.0, 7_000.0, 1_000.0])
    m = metrics(df, pred)

    assert m["normais_nm"] == 100.0  # só o primeiro voo
    assert m["cauda_copia"] == 200.0  # só o terceiro
    assert m["alarmes_falsos"]["n"] == 1
    err2 = (df[TRUTH].to_numpy() - pred) ** 2
    assert m["alarmes_falsos"]["parte_erro2"] == round(err2[1] / err2.sum(), 4)
    sem_lot = np.sqrt(np.mean(err2[:3]))
    assert m["sem_loteria"] == round(sem_lot, 2)


def _args(**kw) -> argparse.Namespace:
    padrao = dict(model="two_stage_nm", rounds=400, cls_rounds=400, reg_rounds=400,
                  nm_split_ms=False, nm_min_ms=21600.0, seed=0, seeds=1, janela_lobt=False,
                  sem_feature=[], reg_corte=0.0, reg_sem_lirf_nm=False, base_ctx=False,
                  base_p13=False, cat_max=0, motor="lgb")
    return argparse.Namespace(**{**padrao, **kw})


def test_config_so_registra_seeds_quando_tem_media():
    assert "seeds" not in config(_args(seeds=1))
    assert "seeds" not in config(_args(model="single", seeds=1))
    assert config(_args(seeds=5))["seeds"] == 5
    assert config(_args(model="single", seeds=5))["seeds"] == 5


def test_config_de_uma_seed_continua_a_de_hoje():
    assert config(_args(model="single")) == {"model": "single", "rounds": 400, "seed": 0}
    assert config(_args()) == {
        "model": "two_stage_nm", "cls_rounds": 400, "reg_rounds": 400, "seed": 0,
        "nm_split_ms": False, "nm_min_ms": 21600.0,
    }


def test_config_so_registra_a_janela_quando_ligada():
    assert "janela_lobt" not in config(_args())
    assert config(_args(janela_lobt=True))["janela_lobt"] is True
    assert config(_args(model="single", janela_lobt=True))["janela_lobt"] is True


def test_config_so_registra_sem_features_com_a_flag():
    assert "sem_features" not in config(_args())
    assert config(_args(sem_feature=["adsb_lat0", "adsb_lon0"]))["sem_features"] == [
        "adsb_lat0", "adsb_lon0"]


def test_config_so_registra_o_regressor_limpo_quando_ligado():
    assert "reg_corte" not in config(_args())
    assert "reg_sem_lirf_nm" not in config(_args())
    assert config(_args(reg_corte=7200.0))["reg_corte"] == 7200.0
    assert config(_args(reg_sem_lirf_nm=True))["reg_sem_lirf_nm"] is True
