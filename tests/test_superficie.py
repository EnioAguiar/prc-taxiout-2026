import numpy as np
import pandas as pd
import pytest

import features as F
import superficie as S

T = pd.Timestamp("2025-01-15 10:00", tz="UTC")
M = 60.0  # um minuto em segundos


def _dep(minutos, taxi, apt="LIRF", rwy="25") -> pd.DataFrame:
    """Decolagens com a hora de decolar em minutos depois de T (o `taxi` vira o `pred`)."""
    n = len(minutos)
    return pd.DataFrame({
        F.AIRPORT: [apt] * n if isinstance(apt, str) else list(apt),
        "RUNWAY_mvt": [rwy] * n if isinstance(rwy, str) else list(rwy),
        S.TIME: [T + pd.Timedelta(minutes=m) for m in minutos],
        S.BLOCK: [T + pd.Timedelta(minutes=m - t / M) for m, t in zip(minutos, taxi)],
    })


@pytest.fixture
def dados(tmp_path):
    """`data/` sintético: pousos no LIRF e no EDDF, e uma decolagem com BLOCK absurdo."""
    def bruto(fase, apt, mvt, block):
        return {"PHASE_mvt": fase, "ADES_mvt": apt,
                S.TIME: T + pd.Timedelta(minutes=mvt), S.BLOCK: T + pd.Timedelta(minutes=block)}

    pd.DataFrame([
        bruto("ARR", "LIRF", 0, 10),    # taxiando de 10:00 a 10:10
        bruto("ARR", "LIRF", 20, 25),   # taxiando de 10:20 a 10:25
        bruto("ARR", "EDDF", 0, 30),    # outro aeroporto: nunca conta no LIRF
        bruto("DEP", "LIRF", 40, -600),  # off-block de decolagem: nunca pode ser lido
    ]).to_parquet(tmp_path / "training_2025-01-01_2025-02-01.parquet", index=False)
    return tmp_path


# ---------------------------------------------------------------- nada de BLOCK das DEP


def test_o_bloco_nao_muda_quando_o_off_block_das_decolagens_some(dados):
    df = _dep([0, 5, 9], [900.0, 600.0, 300.0])
    pred = np.array([900.0, 600.0, 300.0])
    com = S.contagens(df, pred, dados)

    sem = S.contagens(df.drop(columns=[S.BLOCK]), pred, dados)
    nan = S.contagens(df.assign(**{S.BLOCK: pd.NaT}), pred, dados)

    pd.testing.assert_frame_equal(com, sem)
    pd.testing.assert_frame_equal(com, nan)


def test_o_off_block_de_decolagem_no_parquet_bruto_nao_vira_pouso(dados):
    # a linha DEP do parquet tem in-block 10 h antes do movimento: se fosse lida como
    # chegada, ela apareceria taxiando em todo o período.
    df = _dep([40], [600.0])

    out = S.contagens(df, np.array([600.0]), dados)

    assert out["sup_arr_taxiando_push"].tolist() == [0.0]


# ---------------------------------------------------------------- bordas dos intervalos


def test_a_decolagem_vizinha_conta_do_push_ate_o_instante_antes_de_decolar(dados):
    # i decola às 10:30 com 10 min de táxi (push 10:20); as vizinhas decolam às 10:20
    # (push 10:10), 10:25 (push 10:15) e 10:20 em ponto com táxi zero.
    df = _dep([30, 20, 25, 20], [600.0, 600.0, 600.0, 0.0])

    out = S.contagens(df, np.array([600.0, 600.0, 600.0, 0.0]), dados)

    # push de i = 10:20: a vizinha que decola exatamente às 10:20 já saiu (MVT ≤ o_i) e a
    # de táxi zero está no mesmo instante do push, mas não está taxiando (o_j == MVT_j).
    assert out["sup_dep_taxiando_push"].iloc[0] == 1.0     # só a das 10:25
    assert out["sup_dep_decolam_durante"].iloc[0] == 1.0   # só a das 10:25, em (10:20, 10:30)


def test_o_pouso_conta_do_toque_ate_o_instante_antes_do_in_block(dados):
    # push exatamente no toque (10:00) e push exatamente no in-block (10:10)
    df = _dep([10, 20], [600.0, 600.0])

    out = S.contagens(df, np.array([600.0, 600.0]), dados)

    assert out["sup_arr_taxiando_push"].tolist() == [1.0, 0.0]
    # o voo das 10:20 taxia em (10:10, 10:20): nenhum pouso nesse intervalo aberto
    assert out["sup_arr_pousam_durante"].tolist() == [0.0, 0.0]


def test_o_pouso_durante_o_taxi_e_o_intervalo_aberto(dados):
    # táxi de 10:15 a 10:25: o pouso das 10:20 entra; o das 10:25 seria a própria borda
    df = _dep([25], [600.0])

    out = S.contagens(df, np.array([600.0]), dados)

    assert out["sup_arr_pousam_durante"].tolist() == [1.0]
    np.testing.assert_allclose(out["sup_arr_pousam_durante_min"], [1.0 / 10])


# ---------------------------------------------------------------- isolamento


def test_o_proprio_voo_nunca_se_conta(dados):
    df = _dep([30], [1800.0])

    out = S.contagens(df, np.array([1800.0]), dados)

    assert out["sup_dep_taxiando_push"].tolist() == [0.0]
    assert out["sup_dep_decolam_durante"].tolist() == [0.0]


def test_cada_aeroporto_e_cada_pista_contam_so_os_seus(dados):
    # quatro voos simultâneos: dois no LIRF (pistas 25 e 16L) e dois no EDDF
    df = _dep([30] * 4, [600.0] * 4, apt=["LIRF", "LIRF", "EDDF", "EDDF"],
              rwy=["25", "16L", "25", "25"])

    out = S.contagens(df, np.full(4, 600.0), dados)

    assert out["sup_dep_taxiando_push"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert out["sup_dep_decolam_durante_rwy"].tolist() == [0.0, 0.0, 0.0, 0.0]
    # os pousos do LIRF não aparecem no EDDF e vice-versa (push 10:20)
    assert out["sup_arr_taxiando_push"].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_a_contagem_de_pista_so_ve_a_mesma_pista(dados):
    # i decola às 10:30 (push 10:20); duas vizinhas decolam às 10:25, uma em cada pista
    df = _dep([30, 25, 25], [600.0] * 3, rwy=["25", "25", "16L"])

    out = S.contagens(df, np.full(3, 600.0), dados)

    assert out["sup_dep_decolam_durante"].iloc[0] == 2.0
    assert out["sup_dep_decolam_durante_rwy"].iloc[0] == 1.0


def test_as_razoes_saem_por_minuto_de_taxi_previsto(dados):
    df = _dep([30, 25], [1200.0, 600.0])

    out = S.contagens(df, np.array([1200.0, 600.0]), dados)

    # i taxia de 10:10 a 10:30 e a vizinha decola às 10:25: 1 voo em 20 min
    np.testing.assert_allclose(out["sup_dep_decolam_durante_min"].iloc[0], 1 / 20)


def test_o_quadro_sai_na_ordem_e_no_indice_das_linhas(dados):
    df = _dep([30, 5, 20], [600.0] * 3).set_index(pd.Index([7, 8, 9]))

    out = S.contagens(df, np.full(3, 600.0), dados)

    assert list(out.columns) == S.COLS and out.index.tolist() == [7, 8, 9]
    sozinho = S.contagens(df.iloc[[0]], np.array([600.0]), dados)
    assert sozinho["sup_arr_taxiando_push"].iloc[0] == out["sup_arr_taxiando_push"].iloc[0]


# ---------------------------------------------------------------- contra a definição


def _forca_bruta(df: pd.DataFrame, pred: np.ndarray, pousos: pd.DataFrame) -> dict:
    """As mesmas contagens escritas voo a voo, direto da definição (O(n²))."""
    mvt = S._segundos(df[S.TIME])
    o = mvt - np.clip(pred, 0, None)
    apt, rwy = df[F.AIRPORT].to_numpy(), df["RUNWAY_mvt"].to_numpy()
    pouso, bloco, pa = (S._segundos(pousos[S.TIME]), S._segundos(pousos[S.BLOCK]),
                        pousos["ADES_mvt"].to_numpy())
    out = {c: np.zeros(len(df)) for c in S.COLS[:5]}
    for i in range(len(df)):
        j = np.setdiff1d(np.where(apt == apt[i])[0], [i])
        p = np.setdiff1d(np.where((apt == apt[i]) & (rwy == rwy[i]))[0], [i])
        k = pa == apt[i]
        out["sup_dep_taxiando_push"][i] = ((o[j] <= o[i]) & (o[i] < mvt[j])).sum()
        out["sup_dep_decolam_durante"][i] = ((o[i] < mvt[j]) & (mvt[j] < mvt[i])).sum()
        out["sup_dep_decolam_durante_rwy"][i] = ((o[i] < mvt[p]) & (mvt[p] < mvt[i])).sum()
        out["sup_arr_taxiando_push"][i] = ((pouso[k] <= o[i]) & (o[i] < bloco[k])).sum()
        out["sup_arr_pousam_durante"][i] = ((o[i] < pouso[k]) & (pouso[k] < mvt[i])).sum()
    return out


def test_as_contagens_batem_com_a_definicao_voo_a_voo(tmp_path):
    """300 voos com empates de horário e táxi previsto zero: o atalho das buscas binárias
    tem que dar exatamente o mesmo que contar um a um."""
    rng = np.random.default_rng(0)
    n = 300
    df = _dep(rng.integers(0, 24 * 60, n), np.zeros(n),
              apt=rng.choice(["LIRF", "EDDF"], n), rwy=rng.choice(["25", "16L"], n))
    pred = rng.integers(0, 40, n) * M   # inclui táxi zero (intervalo vazio)
    pousos = pd.DataFrame({
        "PHASE_mvt": ["ARR"] * 150,
        "ADES_mvt": rng.choice(["LIRF", "EDDF"], 150),
        S.TIME: [T + pd.Timedelta(minutes=int(x)) for x in rng.integers(0, 24 * 60, 150)],
    })
    pousos[S.BLOCK] = pousos[S.TIME] + pd.to_timedelta(rng.integers(0, 20, 150), unit="m")
    pousos.to_parquet(tmp_path / "training_2025-01-01_2025-02-01.parquet", index=False)

    out = S.contagens(df, pred, tmp_path)

    for nome, esperado in _forca_bruta(df, pred, pousos).items():
        np.testing.assert_array_equal(out[nome].to_numpy(), esperado, err_msg=nome)
