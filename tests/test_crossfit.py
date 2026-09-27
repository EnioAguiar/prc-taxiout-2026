import numpy as np
import pandas as pd
import pytest

import features as F
import models
from cache import TRUTH
from crossfit import month_blocks, oof_base

AEROPORTO, STAND, PISTA = "LIRF", "A1", "25"
POR_MES = 12  # ≥ features.REF_MIN_FLIGHTS: a P10 do grupo só existe com voos suficientes


def _frame(meses, alvo, id0: int, cega: bool = False) -> pd.DataFrame:
    """Voos sintéticos com as colunas que prepare() exige, alvo constante por mês."""
    linhas = []
    ident = id0
    for mes in meses:
        for i in range(POR_MES):
            t = pd.Timestamp(f"2025-{mes:02d}-05 08:00", tz="UTC") + pd.Timedelta(minutes=i)
            linhas.append({F.ID: float(ident), "MVT_TIME_UTC_mvt": t, F.TARGET: float(alvo(mes))})
            ident += 1
    df = pd.DataFrame(linhas)
    if cega:
        df[TRUTH] = df[F.TARGET]
        df[F.TARGET] = np.nan
    fixas = {F.AIRPORT: AEROPORTO, "RUNWAY_mvt": PISTA, "STAND_mvt": STAND}
    for col in F.CATEGORICAL:
        df[col] = fixas.get(col, "X")
    df["hour"], df["dow"], df["nm_missing"] = 8.0, 1, 0
    return df


def _falso(registra, previsao):
    class Falso:
        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, train, cols, run=None, valid=None):
            self.meses_vistos = sorted(set(train["MVT_TIME_UTC_mvt"].dt.month))
            self.media = float(train[F.TARGET].mean())
            return self

        def predict(self, df):
            registra(self.meses_vistos, df[F.ID].tolist())
            return previsao(self, df)

    return Falso


@pytest.fixture
def modelo_falso(monkeypatch):
    """Registra em MODELS um modelo falso; devolve (nome, chamadas, escolhe_previsao)."""
    chamadas = []
    escolha = {"fn": lambda m, df: np.full(len(df), m.media)}

    def registra(meses, ids):
        chamadas.append({"meses_treino": meses, "ids": ids})

    monkeypatch.setitem(
        models.MODELS, "falso", _falso(registra, lambda m, df: escolha["fn"](m, df))
    )
    return chamadas, escolha


def test_blocos_de_dois_meses_consecutivos():
    assert month_blocks([2, 3, 4, 5, 6, 8, 9, 10, 11, 12]) == {
        2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 8: 2, 9: 3, 10: 3, 11: 4, 12: 4
    }
    ano = month_blocks(range(1, 13))
    assert len(set(ano.values())) == 6
    assert ano[1] == ano[2] and ano[3] == ano[4] and ano[11] == ano[12]


def test_blocos_ignoram_ordem_e_repeticao():
    assert month_blocks([5, 2, 5, 3, 2, 4]) == month_blocks([2, 3, 4, 5])


def test_numero_impar_de_meses_deixa_o_ultimo_bloco_com_um():
    blocos = month_blocks([3, 4, 9])
    assert blocos == {3: 0, 4: 0, 9: 1}


def test_nenhuma_linha_prevista_por_modelo_que_viu_o_mes_dela(modelo_falso):
    chamadas, _ = modelo_falso
    train = _frame([1, 2, 3, 4], lambda m: 100 * m, id0=1)
    cega = _frame([1, 2, 3, 4], lambda m: 50 * m, id0=1000, cega=True)
    sem_verdade = _frame([2], lambda m: 700, id0=9000, cega=True)
    sem_verdade[TRUTH] = np.nan
    fora = _frame([7], lambda m: 700, id0=8000, cega=True)  # mês ausente do treino
    blind = pd.concat([cega, sem_verdade, fora], ignore_index=True)

    out = oof_base({"model": "falso"}, train, blind, blind.copy())

    mes_por_id = dict(zip(cega[F.ID], cega["MVT_TIME_UTC_mvt"].dt.month))
    for chamada in chamadas:
        meses_previstos = {mes_por_id[i] for i in chamada["ids"]}
        assert not meses_previstos & set(chamada["meses_treino"])
    assert sorted(out[F.ID]) == sorted(cega[F.ID])  # todo DEP cego com verdade, uma vez só
    assert out[F.ID].is_monotonic_increasing
    assert list(out.columns) == [F.ID, "dia", "mes", TRUTH, "pred"]
    esperado = cega.set_index(F.ID)[TRUTH].to_dict()
    assert out.set_index(F.ID)[TRUTH].to_dict() == esperado
    assert out["mes"].tolist() == out["dia"].str.slice(5, 7).astype(int).tolist()
    assert out["pred"].notna().all()


def test_p10_de_cada_bloco_ignora_o_alvo_dos_meses_do_bloco(modelo_falso):
    chamadas, escolha = modelo_falso
    escolha["fn"] = lambda m, df: df["ref_p10"].to_numpy(float)
    train = _frame([1, 2, 3, 4], lambda m: 100 * m, id0=1)
    cega = _frame([1, 2, 3, 4], lambda m: 50 * m, id0=1000, cega=True)
    cfg = {"model": "falso"}

    antes = oof_base(cfg, train, cega, cega.copy())
    mexido = train.copy()
    do_bloco = mexido["MVT_TIME_UTC_mvt"].dt.month.isin([1, 2])
    mexido.loc[do_bloco, F.TARGET] *= 10
    depois = oof_base(cfg, mexido, cega, cega.copy())

    mes = antes["mes"].to_numpy()
    assert antes["pred"].notna().all()
    bloco0 = np.isin(mes, [1, 2])
    assert antes.loc[bloco0, "pred"].tolist() == depois.loc[bloco0, "pred"].tolist()
    assert antes.loc[~bloco0, "pred"].tolist() != depois.loc[~bloco0, "pred"].tolist()


def test_nao_altera_os_frames_recebidos(modelo_falso):
    train = _frame([1, 2, 3, 4], lambda m: 100 * m, id0=1)
    cega = _frame([1, 2, 3, 4], lambda m: 50 * m, id0=1000, cega=True)
    copias = train.copy(), cega.copy()

    oof_base({"model": "falso"}, train, cega, cega.copy())

    assert "ref_p10" not in train and "ref_p10" not in cega
    pd.testing.assert_frame_equal(train, copias[0])
    pd.testing.assert_frame_equal(cega, copias[1])
