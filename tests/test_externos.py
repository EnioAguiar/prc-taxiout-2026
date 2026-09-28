import numpy as np
import pandas as pd
import pytest

import features as F
from externos import CopiaCia, colunas_ext, diarias, opdi

# ---------------------------------------------------------------- CopiaCia


def _brutos() -> pd.DataFrame:
    """DEP brutas de 3 meses no LIRF, todas com MVT − SCHED = 2 h.

    RYR sem NM copia o SCHED em jan e mar e nunca em fev; AZ com NM nunca copia.
    """
    linhas = []
    for mes, cia, nm, n, copias in [
        (1, "RYR", 1, 4, 4),
        (2, "RYR", 1, 4, 0),
        (3, "RYR", 1, 4, 4),
        (1, "AZ", 0, 2, 0),
        (3, "AZ", 0, 2, 0),
    ]:
        for i in range(n):
            sched = pd.Timestamp(f"2025-{mes:02d}-05 06:00", tz="UTC") + pd.Timedelta(minutes=i)
            linhas.append({
                "FLIGHT_ID_mvt": np.nan if nm else 100.0 + i,
                "FLIGHT_mvt": f"{cia}{100 + i}",
                F.AIRPORT: "LIRF",
                "PHASE_mvt": "DEP",
                "SCHED_TIME_UTC_mvt": sched,
                "MVT_TIME_UTC_mvt": sched + pd.Timedelta(hours=2),
                "BLOCK_TIME_UTC_mvt": sched + pd.Timedelta(seconds=30 if i < copias else 3600),
            })
    return pd.DataFrame(linhas)


def _linhas(mes: int = 2) -> pd.DataFrame:
    """Uma linha de cada caso: grupo com histórico, companhia nova e aeroporto novo."""
    t = pd.Timestamp(f"2025-{mes:02d}-10 08:00", tz="UTC")
    return pd.DataFrame({
        F.AIRPORT: ["LIRF", "LIRF", "EDDF"],
        "FLIGHT_mvt": ["RYR900", "XX900", "RYR901"],
        "nm_missing": [1, 1, 1],
        "MVT_TIME_UTC_mvt": [t, t, t],
        "SCHED_TIME_UTC_mvt": [t - pd.Timedelta(hours=2)] * 3,
    })


def test_a_taxa_da_companhia_ignora_o_mes_da_propria_linha():
    copia = CopiaCia().fit(_brutos())
    df = _linhas(mes=2)

    ext = copia.transform(df, meses_treino=(1, 2, 3))

    # meses 1 e 3: RYR sem NM 8 de 8; LIRF 8 de 12 → (8 + 20 × 2/3) / (8 + 20)
    assert ext["ext_taxa_cia"][0] == pytest.approx((8 + 20 * 2 / 3) / 28)
    # o mês 2 (0 de 4) não entra: tirar o mês da linha ou não ter o mês dá o mesmo
    np.testing.assert_allclose(ext["ext_taxa_cia"], copia.transform(df, (1, 3))["ext_taxa_cia"])


def test_os_outros_meses_de_treino_entram_na_taxa():
    copia = CopiaCia().fit(_brutos())

    ext = copia.transform(_linhas(mes=1), meses_treino=(1, 2, 3))

    # meses 2 e 3: RYR sem NM 4 de 8; LIRF 4 de 10 → (4 + 20 × 0,4) / (4 + 20 + 4)
    assert ext["ext_taxa_cia"][0] == pytest.approx((4 + 20 * 0.4) / 28)


def test_grupo_sem_historico_cai_na_taxa_do_aeroporto_e_aeroporto_novo_fica_sem_taxa():
    ext = CopiaCia().fit(_brutos()).transform(_linhas(mes=2), meses_treino=(1, 2, 3))

    assert ext["ext_taxa_cia"][1] == pytest.approx(2 / 3)  # companhia nunca vista no LIRF
    assert np.isnan(ext["ext_taxa_cia"][2])  # EDDF não tem histórico nenhum


def test_a_taxa_em_segundos_multiplica_o_atraso_sobre_o_sched():
    ext = CopiaCia().fit(_brutos()).transform(_linhas(mes=2), meses_treino=(1, 2, 3))

    np.testing.assert_allclose(ext["ext_taxa_cia_ms"], ext["ext_taxa_cia"] * 7200)


def test_o_fit_pode_ser_chamado_mes_a_mes():
    brutos = _brutos()
    mes = brutos["MVT_TIME_UTC_mvt"].dt.month
    inteiro = CopiaCia().fit(brutos)
    parcelado = CopiaCia()
    for m in (1, 2, 3):
        parcelado.fit(brutos[mes == m])

    np.testing.assert_allclose(
        inteiro.transform(_linhas(), (1, 2, 3))["ext_taxa_cia"],
        parcelado.transform(_linhas(), (1, 2, 3))["ext_taxa_cia"],
    )


# ---------------------------------------------------------------- séries diárias


@pytest.fixture
def raiz(tmp_path):
    """`data/externo` sintético: um dia (02/01) em LIRF e EDDF nas três séries."""
    (tmp_path / "atfm_slot_adherence_2025.csv").write_text(
        "YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,"
        "FLT_DEP_1,FLT_DEP_REG_1,FLT_DEP_OUT_EARLY_1,FLT_DEP_IN_1,FLT_DEP_OUT_LATE_1\n"
        "2025,1,JAN,2025-01-02,LIRF,Roma,Italy,200,50,1,45,4\n"
        "2025,1,JAN,2025-01-02,EDDF,Frankfurt,Germany,300,60,0,55,5\n",
        encoding="utf-8",
    )
    (tmp_path / "all_pre_departure_delays_2025.csv").write_text(
        "YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,"
        "FLT_DEP_1,FLT_DEP_IFR_2,DLY_ALL_PRE_2\n"
        "2025,01,JAN,2025-01-02,LIRF,Roma,Italy,200,180,540\n"
        "2025,01,JAN,2025-01-02,EDDF,Frankfurt,Germany,300,300,150\n",
        encoding="utf-8",
    )
    (tmp_path / "atc_pre_departure_delays_2025.csv").write_text(
        "YEAR,MONTH_NUM,MONTH_MON,FLT_DATE,APT_ICAO,APT_NAME,STATE_NAME,"
        "FLT_DEP_1,FLT_DEP_IFR_2,DLY_ATC_PRE_2,FLT_DEP_3,DLY_ATC_PRE_3\n"
        "2025,01,JAN,2025-01-02,LIRF,Roma,Italy,200,180,90,100,50\n"
        "2025,01,JAN,2025-01-02,EDDF,Frankfurt,Germany,300,300,30,150,15\n",
        encoding="utf-8",
    )
    return tmp_path


def _voos_dia() -> pd.DataFrame:
    """Três decolagens: duas no dia 02/01 (uma quase à meia-noite) e uma já no dia 03."""
    return pd.DataFrame({
        F.AIRPORT: ["LIRF", "EDDF", "LIRF"],
        "MVT_TIME_UTC_mvt": pd.to_datetime(
            ["2025-01-02 10:00", "2025-01-02 23:30", "2025-01-03 00:30"], utc=True
        ),
    })


def test_as_series_diarias_casam_por_dia_utc_e_aeroporto(raiz):
    ext = diarias(_voos_dia(), raiz)

    np.testing.assert_allclose(ext["ext_reg_frac"], [50 / 200, 60 / 300, np.nan])
    np.testing.assert_allclose(ext["ext_late_frac"], [4 / 50, 5 / 60, np.nan])
    np.testing.assert_allclose(ext["ext_reg_n"], [50, 60, np.nan])
    np.testing.assert_allclose(ext["ext_pre_min"], [540 / 180, 150 / 300, np.nan])
    np.testing.assert_allclose(ext["ext_atc_min"], [50 / 100, 15 / 150, np.nan])


def test_dia_sem_dado_na_serie_fica_sem_valor(raiz):
    ext = diarias(_voos_dia(), raiz)

    assert ext.iloc[2].isna().all()  # 03/01 não está nos CSV
    assert len(ext) == 3


# ---------------------------------------------------------------- OPDI


@pytest.fixture
def raiz_opdi(tmp_path):
    """Flight list de jan/2025: dois aviões, cada um com uma chegada e uma decolagem."""
    def t(hhmm: str) -> pd.Timestamp:
        return pd.Timestamp(f"2025-01-02 {hhmm}")

    voos = pd.DataFrame({
        "icao24": ["aaa111", "aaa111", "bbb222", "bbb222"],
        "flt_id": ["RYR100 ", "RYR123  ", "DLH50   ", "DLH900  "],
        "adep": ["EDDF", "LIRF", "EGLL", "LIRF"],
        "ades": ["LIRF", None, "EDDF", None],
        "first_seen": [t("08:00"), t("10:00"), t("09:30"), t("12:00")],
        "last_seen": [t("09:00"), t("11:00"), t("10:30"), t("13:00")],
    })
    (tmp_path / "opdi").mkdir()
    voos.to_parquet(tmp_path / "opdi" / "flight_list_202501.parquet", index=False)
    return tmp_path


def _voos_opdi() -> pd.DataFrame:
    return pd.DataFrame({
        F.AIRPORT: ["LIRF"] * 4,
        "CALLSIGN_flt": ["RYR123 ", "ZZZ999", "ZZZ888", "DLH900"],
        "MVT_TIME_UTC_mvt": pd.to_datetime(
            ["2025-01-02 10:08:00", "2025-01-02 10:00:30",
             "2025-01-02 10:05:00", "2025-01-02 12:09:00"], utc=True,
        ),
    })


def test_o_opdi_casa_pelo_callsign_em_ate_600_s(raiz_opdi):
    ext = opdi(_voos_opdi(), raiz_opdi)

    # RYR123 decolou 10:00 (8 min do MVT) e o mesmo avião pousou 09:00: 1 h no solo
    assert ext["ext_solo_s"][0] == pytest.approx(3600)
    assert ext["ext_mesmo_apt"][0] == 1  # a chegada anterior foi no LIRF


def test_sem_callsign_o_opdi_casa_so_pelo_aeroporto_em_ate_90_s(raiz_opdi):
    ext = opdi(_voos_opdi(), raiz_opdi)

    assert ext["ext_solo_s"][1] == pytest.approx(3600)  # 30 s da decolagem do RYR123
    assert np.isnan(ext["ext_solo_s"][2])  # 5 min: fora dos 90 s e sem callsign


def test_o_solo_conta_do_pouso_anterior_da_mesma_aeronave(raiz_opdi):
    ext = opdi(_voos_opdi(), raiz_opdi)

    assert ext["ext_solo_s"][3] == pytest.approx(5400)  # 12:00 − 10:30
    assert ext["ext_mesmo_apt"][3] == 0  # o voo anterior do bbb222 pousou em EDDF
    assert len(ext) == 4


# ---------------------------------------------------------------- montagem


def test_as_colunas_externas_saem_na_ordem_do_frame(raiz, raiz_opdi, tmp_path):
    df = _voos_opdi()
    df["FLIGHT_mvt"] = ["RYR900", "XX900", "RYR901", "LH900"]
    df["nm_missing"] = [1, 1, 1, 0]
    df["SCHED_TIME_UTC_mvt"] = df["MVT_TIME_UTC_mvt"] - pd.Timedelta(hours=2)

    ext = colunas_ext(df, CopiaCia().fit(_brutos()), (1, 2, 3), tmp_path)

    assert set(ext) == {
        "ext_taxa_cia", "ext_taxa_cia_ms", "ext_reg_frac", "ext_late_frac", "ext_reg_n",
        "ext_pre_min", "ext_atc_min", "ext_solo_s", "ext_mesmo_apt",
    }
    assert all(len(v) == len(df) for v in ext.values())
    assert ext["ext_solo_s"][0] == pytest.approx(3600)
