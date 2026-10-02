import numpy as np
import pandas as pd
import pytest

import fe_auto
import features as F


def _voos(n_por_grupo=(2, 3), valores=None) -> pd.DataFrame:
    """Dois aeroportos num mesmo dia, com uma numérica `x` e um stand por linha."""
    t = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    apt = ["LIRF"] * n_por_grupo[0] + ["EDDF"] * n_por_grupo[1]
    x = valores if valores is not None else list(range(len(apt)))
    return pd.DataFrame({
        F.AIRPORT: apt,
        "STAND_mvt": ["A1"] * len(apt),
        "RUNWAY_mvt": ["25"] * len(apt),
        "AIRCRAFT_OPERATOR_flt": ["XX"] * len(apt),
        "AIRCRAFT_TYPE_mvt": ["A320"] * len(apt),
        "hour": [10] * len(apt),
        "MVT_TIME_UTC_mvt": [t] * len(apt),
        "x": np.asarray(x, float),
    })


def test_media_e_desvio_do_grupo_saem_por_linha_e_nao_misturam_aeroportos():
    df = _voos((2, 3), [10.0, 20.0, 1.0, 2.0, 3.0])
    media = fe_auto.valores(df, ("gmed", "x", "dia_apt"))
    desvio = fe_auto.valores(df, ("gdes", "x", "dia_apt"))
    assert media.tolist() == [15.0, 15.0, 2.0, 2.0, 2.0]
    assert desvio[0] == pytest.approx(5.0)
    assert desvio[2] == pytest.approx(np.sqrt(2 / 3))


def test_a_diferenca_e_o_z_usam_a_media_do_proprio_grupo():
    df = _voos((2, 3), [10.0, 20.0, 1.0, 2.0, 3.0])
    assert fe_auto.valores(df, ("gdif", "x", "dia_apt")).tolist() == [-5.0, 5.0, -1.0, 0.0, 1.0]
    z = fe_auto.valores(df, ("gz", "x", "dia_apt"))
    assert z[0] == pytest.approx(-1.0)
    assert z[4] == pytest.approx(1 / np.sqrt(2 / 3))


def test_grupo_de_uma_linha_so_tem_desvio_nulo_e_z_vazio():
    df = _voos((1, 1), [7.0, 9.0])
    assert np.isnan(fe_auto.valores(df, ("gdes", "x", "dia_apt"))).all()
    assert np.isnan(fe_auto.valores(df, ("gz", "x", "dia_apt"))).all()
    assert fe_auto.valores(df, ("gdif", "x", "dia_apt")).tolist() == [0.0, 0.0]


def test_nan_na_numerica_nao_entra_na_media_e_sai_nan_na_propria_linha():
    df = _voos((3, 1), [10.0, np.nan, 20.0, 5.0])
    media = fe_auto.valores(df, ("gmed", "x", "dia_apt"))
    assert media[:3].tolist() == [15.0, 15.0, 15.0]  # o NaN não puxa a média do grupo
    assert np.isnan(fe_auto.valores(df, ("gdif", "x", "dia_apt"))[1])


def test_posto_vai_de_zero_a_um_dentro_do_grupo():
    df = _voos((2, 3), [10.0, 20.0, 3.0, 1.0, 2.0])
    pos = fe_auto.valores(df, ("gpos", "x", "dia_apt"))
    assert pos[:2].tolist() == [0.5, 1.0]
    assert pos[2:].tolist() == [1.0, pytest.approx(1 / 3), pytest.approx(2 / 3)]


def test_contagem_e_o_tamanho_do_grupo_do_dia():
    df = _voos((2, 3))
    assert fe_auto.valores(df, ("gcont", "dia_apt")).tolist() == [2.0, 2.0, 3.0, 3.0, 3.0]


def test_dias_diferentes_sao_grupos_diferentes():
    df = _voos((2, 2), [1.0, 3.0, 10.0, 30.0])
    df[F.AIRPORT] = "LIRF"
    df.loc[2:, "MVT_TIME_UTC_mvt"] = pd.Timestamp("2025-01-16 10:00", tz="UTC")
    assert fe_auto.valores(df, ("gmed", "x", "dia_apt")).tolist() == [2.0, 2.0, 20.0, 20.0]


def test_razao_por_zero_sai_nan_e_diferenca_e_produto_sao_os_obvios():
    df = _voos((2, 0), [4.0, 0.0])
    df["y"] = [2.0, 0.0]
    assert fe_auto.valores(df, ("raz", "x", "y")).tolist()[0] == 2.0
    assert np.isnan(fe_auto.valores(df, ("raz", "x", "y"))[1])
    assert fe_auto.valores(df, ("dif", "x", "y")).tolist() == [2.0, 0.0]
    assert fe_auto.valores(df, ("pro", "x", "y")).tolist() == [8.0, 0.0]


def test_chave_sem_valor_vira_categoria_propria_e_nao_quebra():
    df = _voos((2, 2), [1.0, 3.0, 10.0, 30.0])
    df[F.AIRPORT] = "LIRF"
    df.loc[2:, "STAND_mvt"] = None
    assert fe_auto.valores(df, ("gmed", "x", "apt_stand")).tolist() == [2.0, 2.0, 20.0, 20.0]


def test_quadro_vazio_devolve_colunas_vazias():
    df = _voos((0, 0))
    for spec in (("gmed", "x", "dia_apt"), ("gcont", "dia_apt"), ("gpos", "x", "apt_stand")):
        assert len(fe_auto.valores(df, spec)) == 0


def test_contagem_so_existe_em_chave_com_dia():
    specs = fe_auto.candidatas(["x"], fe_auto.CHAVES, ["x", "y"])
    contagens = [s for s in specs if s[0] == "gcont"]
    assert contagens and all(s[-1].startswith("dia_") for s in contagens)
    assert len(contagens) == sum(1 for k in fe_auto.CHAVES if k.startswith("dia_"))


def test_candidatas_cobrem_os_pares_e_as_agregacoes():
    specs = set(fe_auto.candidatas(["x"], ["apt_stand"], ["x", "y"]))
    assert ("raz", "x", "y") in specs and ("dif", "x", "y") in specs and ("pro", "x", "y") in specs
    assert ("raz", "y", "x") not in specs  # par uma vez só, na ordem do pool
    assert {("gmed", "x", "apt_stand"), ("gpos", "x", "apt_stand")} <= specs


def test_cache_de_codigos_e_reusado_entre_candidatas_da_mesma_chave():
    df = _voos((2, 3))
    cache: dict = {}
    fe_auto.valores(df, ("gmed", "x", "dia_apt"), cache)
    cache["dia_apt"] = np.zeros(len(df), int)  # todo mundo no mesmo grupo
    assert fe_auto.valores(df, ("gcont", "dia_apt"), cache).tolist() == [5.0] * 5


def test_colunas_devolve_exatamente_as_escolhidas_com_uma_linha_por_voo():
    df = _voos((3, 2))
    df = df.assign(**{c: np.arange(len(df), dtype=float)
                      for c in _fontes(fe_auto.ESCOLHIDAS) if c not in df.columns})
    out = fe_auto.colunas(df)
    assert list(out) == [fe_auto.nome(s) for s in fe_auto.ESCOLHIDAS]
    assert all(len(v) == len(df) and np.isfinite(v).all() for v in out.values())


def test_as_escolhidas_so_usam_chaves_e_colunas_que_o_quadro_do_corretor_tem():
    """As fontes precisam existir no holdout e no ranking — senão o corretor quebra no envio."""
    colunas_do_quadro = set(_voos((1, 1)).columns) | {"to_takeoff_from_AOBT_3_flt"}
    for spec in fe_auto.ESCOLHIDAS:
        assert spec[0] in fe_auto.PARES + fe_auto.GRUPO + ("gcont",), spec
        assert spec[-1] in fe_auto.CHAVES or spec[0] in fe_auto.PARES, spec
        for col in spec[1:]:
            assert col in fe_auto.CHAVES or col in colunas_do_quadro, col


def _fontes(specs) -> set[str]:
    """Colunas de origem das candidatas (tudo que não é operador nem chave de grupo)."""
    return {p for s in specs for p in s[1:] if p not in fe_auto.CHAVES}
