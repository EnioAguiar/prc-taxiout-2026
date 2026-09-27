import numpy as np
import pandas as pd
import pytest

import contexto
import features as F
import stack

T0 = pd.Timestamp("2025-03-10 12:00", tz="UTC")
M = pd.Timedelta(minutes=1)


def _mvt(mid, fase, apt, mvt, *, ades="EDDF", rwy="25", stand="A1", block=None,
         aobt=None, taxi=np.nan):
    return {
        F.ID: float(mid),
        "PHASE_mvt": fase,
        "ADEP_mvt": apt if fase == "DEP" else "EGLL",
        "ADES_mvt": ades if fase == "DEP" else apt,
        "RUNWAY_mvt": rwy,
        "STAND_mvt": stand,
        "MVT_TIME_UTC_mvt": mvt,
        "BLOCK_TIME_UTC_mvt": block,
        "AOBT_3_flt": aobt,
        F.TARGET: taxi,
    }


def _raw(linhas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(linhas)
    df[F.AIRPORT] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    return df


def _ctx(linhas) -> pd.DataFrame:
    return contexto.contexto(_raw(linhas)).set_index(F.ID)


def test_taxi_in_das_chegadas_por_janela_e_aeroporto():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 10 * M),
        _mvt(2, "ARR", "LIRF", T0 - 10 * M, block=T0 - 5 * M, taxi=300.0),
        _mvt(3, "ARR", "LIRF", T0 - 20 * M, block=T0 - 10 * M, taxi=600.0),
        _mvt(4, "ARR", "LIRF", T0 - 70 * M, block=T0 - 60 * M, taxi=900.0),
        _mvt(5, "ARR", "EDDM", T0 - 5 * M, block=T0, taxi=1200.0),
    ])
    assert ctx.loc[1.0, "ctx_arr_tin_apt_15"] == 300.0
    assert ctx.loc[1.0, "ctx_arr_n_apt_15"] == 1
    assert ctx.loc[1.0, "ctx_arr_tin_apt_60"] == 450.0
    assert ctx.loc[1.0, "ctx_arr_n_apt_60"] == 2
    assert ctx.loc[1.0, "ctx_arr_tin_rwy_60"] == 450.0  # todas na mesma pista


def test_sem_chegadas_na_janela_media_nan_e_contagem_zero():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 10 * M),
        _mvt(2, "ARR", "EDDM", T0 - 5 * M, block=T0, taxi=1200.0),
    ])
    assert np.isnan(ctx.loc[1.0, "ctx_arr_tin_apt_60"])
    assert ctx.loc[1.0, "ctx_arr_n_apt_60"] == 0


def test_stand_pega_a_ultima_chegada_ate_o_mvt():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, stand="A1", aobt=T0 - 10 * M),
        _mvt(2, "ARR", "LIRF", T0 - 90 * M, stand="A1", block=T0 - 80 * M, taxi=600.0),
        _mvt(3, "ARR", "LIRF", T0 - 40 * M, stand="A1", block=T0 - 30 * M, taxi=420.0),
        _mvt(4, "ARR", "LIRF", T0 + 5 * M, stand="A1", block=T0 + 10 * M, taxi=999.0),
        _mvt(5, "ARR", "LIRF", T0 - 5 * M, stand="B9", block=T0 - 1 * M, taxi=111.0),
    ])
    assert ctx.loc[1.0, "ctx_stand_ult_tin"] == 420.0
    assert ctx.loc[1.0, "ctx_stand_ult_idade"] == 30 * 60


def test_stand_ignora_chegada_com_mais_de_um_dia():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, stand="A1", aobt=T0 - 10 * M),
        _mvt(2, "ARR", "LIRF", T0 - 3000 * M, stand="A1", block=T0 - 2000 * M, taxi=600.0),
    ])
    assert np.isnan(ctx.loc[1.0, "ctx_stand_ult_idade"])


def test_vizinhos_separam_passado_e_futuro_e_excluem_a_propria_dep():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 20 * M),  # proxy próprio 1200
        _mvt(2, "DEP", "LIRF", T0 - 5 * M, aobt=T0 - 15 * M),  # passado, 600
        _mvt(3, "DEP", "LIRF", T0 + 5 * M, aobt=T0 - 5 * M),  # futuro, 600
        _mvt(4, "DEP", "LIRF", T0 + 10 * M, aobt=T0 - 20 * M),  # futuro, 1800
    ])
    assert ctx.loc[1.0, "ctx_viz_pas_apt_15"] == 600.0
    assert ctx.loc[1.0, "ctx_viz_fut_apt_15"] == 1200.0
    assert ctx.loc[1.0, "ctx_viz_proprio_menos_pas"] == 600.0
    assert ctx.loc[1.0, "ctx_viz_proprio_menos_fut"] == 0.0
    assert np.isnan(ctx.loc[2.0, "ctx_viz_pas_apt_60"])  # nada antes dela


def test_vizinhos_ignoram_proxy_fora_da_faixa():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 20 * M),
        _mvt(2, "DEP", "LIRF", T0 - 5 * M, aobt=T0 + 5 * M),  # proxy negativo
        _mvt(3, "DEP", "LIRF", T0 - 6 * M, aobt=T0 - 300 * M),  # proxy > 7200
        _mvt(4, "DEP", "LIRF", T0 - 7 * M, aobt=None),  # sem AOBT
    ])
    assert np.isnan(ctx.loc[1.0, "ctx_viz_pas_apt_60"])
    assert np.isnan(ctx.loc[2.0, "ctx_viz_proprio_menos_fut"])


def test_pista_separa_os_vizinhos():
    ctx = _ctx([
        _mvt(1, "DEP", "LIRF", T0, rwy="25", aobt=T0 - 20 * M),
        _mvt(2, "DEP", "LIRF", T0 - 5 * M, rwy="16L", aobt=T0 - 15 * M),
    ])
    assert ctx.loc[1.0, "ctx_viz_pas_apt_15"] == 600.0
    assert np.isnan(ctx.loc[1.0, "ctx_viz_pas_rwy_15"])


def test_nao_usa_o_alvo_nem_o_in_block_das_decolagens():
    linhas = [
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 20 * M, block=T0 - 20 * M, taxi=1200.0),
        _mvt(2, "DEP", "LIRF", T0 - 5 * M, aobt=T0 - 15 * M, block=T0 - 15 * M, taxi=600.0),
        _mvt(3, "ARR", "LIRF", T0 - 10 * M, block=T0 - 5 * M, taxi=300.0),
    ]
    antes = _ctx(linhas)
    sujas = [dict(ln) for ln in linhas]
    for ln in sujas:
        if ln["PHASE_mvt"] == "DEP":
            ln[F.TARGET] = np.nan
            ln["BLOCK_TIME_UTC_mvt"] = None
    depois = _ctx(sujas)
    pd.testing.assert_frame_equal(antes, depois)


def test_uma_linha_por_dep_com_as_vinte_colunas():
    ctx = contexto.contexto(_raw([
        _mvt(1, "DEP", "LIRF", T0, aobt=T0 - 10 * M),
        _mvt(2, "DEP", "LIRF", T0 + 3 * M, aobt=T0 - 10 * M),
        _mvt(3, "ARR", "LIRF", T0 - 10 * M, block=T0 - 5 * M, taxi=300.0),
    ]))
    assert list(ctx.columns) == [F.ID, *contexto.COLS]
    assert len(contexto.COLS) == 20
    assert ctx[F.ID].tolist() == [1.0, 2.0]


@pytest.fixture
def frame_com_ctx(raw_movements) -> pd.DataFrame:
    df = F.build(raw_movements)
    ctx = contexto.contexto(raw_movements)
    return df.merge(ctx, on=F.ID, how="left")


def test_base_ignora_as_colunas_ctx(frame_com_ctx):
    cols = F.feature_columns(frame_com_ctx)
    assert not [c for c in cols if c.startswith("ctx_")]


def test_corretor_usa_as_colunas_ctx(frame_com_ctx):
    X = stack.corrector_frame(frame_com_ctx, np.zeros(len(frame_com_ctx)), adsb=False)
    assert [c for c in X.columns if c.startswith("ctx_")] == list(contexto.COLS)
