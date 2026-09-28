import numpy as np
import pandas as pd
import pytest

import features as F
import plano13 as P

M = pd.Timedelta(minutes=1)
H = pd.Timedelta(hours=1)


def _dep(horas, stands, cias=None, tipos=None, apt="LIRF") -> pd.DataFrame:
    """Decolagens com o que o plano 13 lê: aeroporto, stand, voo, tipo, hora e plano NM."""
    n = len(horas)
    t = [pd.Timestamp(h, tz="UTC") for h in horas]
    return pd.DataFrame({
        F.AIRPORT: [apt] * n,
        "STAND_mvt": stands,
        "FLIGHT_mvt": cias or [f"AZA{i}" for i in range(n)],
        "AIRCRAFT_TYPE_mvt": tipos or ["A320"] * n,
        P.TIME: t,
        "ADEP_mvt": [apt] * n,
        "ADES_mvt": ["EDDF"] * n,
        "ADEP_flt": [apt] * n,
        "ADES_flt": ["EDDF"] * n,
        "ADES_FILED_flt": ["EDDF"] * n,
        "AIRCRAFT_TYPE_flt": tipos or ["A320"] * n,
        "EOBT_1_flt": t,
        "ARVT_1_flt": [h + H for h in t],
        "AOBT_3_flt": t,
        "ARVT_3_flt": [h + 2 * H for h in t],
    })


# ---------------------------------------------------------------- rotação no stand


@pytest.fixture
def dados(tmp_path):
    """`data/` sintético: um pouso em 31/01, um em 01/02 e uma decolagem no mesmo stand."""
    def bruto(fase, hora, stand, voo, tipo):
        return {"PHASE_mvt": fase, "ADES_mvt": "LIRF", "STAND_mvt": stand,
                P.BLOCK: pd.Timestamp(hora, tz="UTC"), "FLIGHT_mvt": voo,
                "AIRCRAFT_TYPE_mvt": tipo}

    janeiro = pd.DataFrame([
        bruto("ARR", "2025-01-31 23:00", "A1", "AZA100", "A320"),
        bruto("ARR", "2025-01-10 08:00", "B9", "DLH900", "B738"),  # velho demais em fevereiro
    ])
    fevereiro = pd.DataFrame([
        bruto("ARR", "2025-02-01 10:00", "A1", "DLH400", "B738"),
        bruto("DEP", "2025-02-01 08:00", "A1", "AZA700", "A320"),  # decolagem: não é rotação
    ])
    janeiro.to_parquet(tmp_path / "training_2025-01-01_2025-02-01.parquet", index=False)
    fevereiro.to_parquet(tmp_path / "training_2025-02-01_2025-03-01.parquet", index=False)
    return tmp_path


def test_a_rotacao_e_o_pouso_anterior_mesmo_no_parquet_do_mes_passado(dados):
    rot = P.rotacao(_dep(["2025-02-01 01:00"], ["A1"], ["AZA200"], ["A320"]), dados)

    assert rot["rot_idade"].tolist() == [7200.0]  # pouso das 23:00 de 31/01
    assert rot["rot_mesma_cia"].tolist() == [1.0] and rot["rot_mesmo_tipo"].tolist() == [1.0]


def test_a_rotacao_ignora_o_pouso_futuro_e_a_decolagem_no_mesmo_stand(dados):
    # às 09:00 o pouso das 10:00 ainda não aconteceu e a decolagem das 08:00 não conta
    rot = P.rotacao(_dep(["2025-02-01 09:00"], ["A1"], ["DLH300"], ["B738"]), dados)

    assert rot["rot_idade"].tolist() == [10 * 3600.0]  # o pouso das 23:00 de 31/01
    assert rot["rot_mesma_cia"].tolist() == [0.0] and rot["rot_mesmo_tipo"].tolist() == [0.0]


def test_pouso_de_mais_de_24_h_ou_stand_sem_pouso_ficam_sem_rotacao(dados):
    rot = P.rotacao(_dep(["2025-02-03 12:00", "2025-02-01 09:00"], ["A1", "C7"]), dados)

    assert rot.notna().to_numpy().sum() == 0


def test_a_rotacao_de_um_voo_nao_muda_com_as_outras_linhas_do_quadro(dados):
    horas = ["2025-02-01 01:00", "2025-02-01 09:00", "2025-02-03 12:00"]
    juntas = P.rotacao(_dep(horas, ["A1"] * 3), dados)
    sozinha = P.rotacao(_dep([horas[1]], ["A1"]), dados)

    np.testing.assert_allclose(sozinha["rot_idade"], juntas["rot_idade"].to_numpy()[1:2])


# ---------------------------------------------------------------- METAR


@pytest.fixture
def raiz(tmp_path):
    """`data/externo` sintético: dois METAR no LIRF, com 3 h entre eles."""
    linhas = pd.DataFrame({
        "station": ["LIRF", "LIRF"],
        "valid": ["2025-01-15 09:50", "2025-01-15 12:50"],
        "tmpf": [35.0, 50.0],
        "dwpf": [34.0, 30.0],
        "sknt": [5.0, 12.0],
        "gust": ["null", 20.0],
        "vsby": [2.0, 10.0],
        "wxcodes": ["-SN", "null"],
        "skyc1": ["OVC", "FEW"],
        "skyl1": [400.0, 5000.0],
    })
    (tmp_path / "metar").mkdir()
    for apt in P.APTS:
        alvo = tmp_path / "metar" / f"{apt}.csv"
        (linhas if apt == "LIRF" else linhas.iloc[:0]).to_csv(alvo, index=False)
    return tmp_path


def test_o_metar_usa_a_observacao_anterior_de_ate_duas_horas(raiz):
    met = P.metar(_dep(["2025-01-15 10:00", "2025-01-15 12:20", "2025-01-15 12:50"], ["A1"] * 3),
                  raiz)

    np.testing.assert_allclose(met["met_vis"], [2.0, np.nan, 10.0])  # 12:20 está a 2 h 30
    np.testing.assert_allclose(met["met_temp"], [(35 - 32) / 1.8, np.nan, 10.0])


def test_o_metar_nunca_vem_do_futuro(raiz):
    met = P.metar(_dep(["2025-01-15 08:00"], ["A1"]), raiz)

    assert met.notna().to_numpy().sum() == 0


def test_o_degelo_pede_frio_com_ar_umido_ou_precipitacao(raiz):
    met = P.metar(_dep(["2025-01-15 10:00", "2025-01-15 12:50"], ["A1", "A1"]), raiz)

    np.testing.assert_allclose(met["met_degelo"], [1.0, 0.0])
    np.testing.assert_allclose(met["met_neve"], [1.0, 0.0])


def test_aeroporto_sem_metar_fica_sem_colunas_do_tempo(raiz):
    met = P.metar(_dep(["2025-01-15 10:00"], ["A1"], apt="EDDF"), raiz)

    assert met.notna().to_numpy().sum() == 0 and len(met.columns) == len(P.MET_COLS) + 1


# ---------------------------------------------------------------- companhia


def _com_cias(cias: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"FLIGHT_mvt": [f"{c}{i}" for i, c in enumerate(cias)]})


def test_o_vocabulario_do_treino_vale_igual_no_holdout_e_no_ranking():
    treino = _com_cias(["AZA", "AZA", "AZA", "DLH", "DLH", "RYR"])
    cias = P.vocabulario(treino, n=2)

    treino_cat = P.coluna_cia(treino, cias)
    ranking_cat = P.coluna_cia(_com_cias(["DLH", "RYR", "XQ"]), cias)

    assert cias == ["AZA", "DLH"]
    assert list(treino_cat.categories) == list(ranking_cat.categories) == ["AZA", "DLH", "outro"]
    assert list(ranking_cat) == ["DLH", "outro", "outro"]  # fora do vocabulário vira "outro"


def test_o_quadro_do_plano_13_sai_na_ordem_das_linhas(raiz, dados):
    df = _dep(["2025-01-15 10:00", "2025-01-15 12:20"], ["A1", "C7"], ["AZA1", "XQ2"])

    p13 = P.colunas_p13(df, ["AZA"], raiz, dados)

    assert len(p13) == len(df) and list(p13.index) == [0, 1]
    assert list(p13["cia"]) == ["AZA", "outro"]
    assert isinstance(p13["cia"].dtype, pd.CategoricalDtype)
    assert [c for c in p13.columns if c != "cia"] == [*P.MET_COLS, "met_degelo", "rot_mesma_cia",
                                                      "rot_mesmo_tipo", "rot_idade",
                                                      "nm_dur_plan", "nm_dur_real", "nm_dur_dif",
                                                      "nm_dur_razao", "nm_ades_igual",
                                                      "nm_ades_filed_igual", "nm_tipo_igual",
                                                      "nm_adep_igual"]
