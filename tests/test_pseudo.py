import numpy as np
import pandas as pd
import pytest

import features as F
import models
import pseudo
import stack


def _rk(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({F.ID: np.arange(1.0, n + 1), "nm_missing": np.zeros(n)})


def test_corpo_fica_so_com_voo_de_plano_nm_ate_o_corte_e_com_alvo_finito():
    alvo = np.array([900.0, 3600.0, 3600.1, 1200.0, np.nan])
    nm_missing = np.array([0, 0, 0, 1, 0])

    assert pseudo.corpo(alvo, nm_missing).tolist() == [True, True, False, False, False]


def test_corpo_tira_as_linhas_que_a_base_entrega_a_regra():
    alvo = np.array([900.0, 1000.0, 1100.0])
    regra = np.array([False, True, False])

    assert pseudo.corpo(alvo, np.zeros(3), regra).tolist() == [True, False, True]
    assert pseudo.corpo(alvo, np.zeros(3), regra, corte=1000).tolist() == [True, False, False]


def test_corpo_exige_previsao_da_base_no_voo_de_2026():
    """Sem base finita o corretor aprenderia `alvo − NaN` e o ajuste inteiro iria junto."""
    alvo = np.array([900.0, 1000.0])
    base = np.array([800.0, np.nan])

    assert pseudo.corpo(alvo, np.zeros(2), base=base).tolist() == [True, False]


def test_juntar_poe_2026_no_fim_com_peso_menor_e_mantem_2025_com_peso_1():
    X = pd.DataFrame({"pred": [100.0, 200.0]})
    X_ps = pd.DataFrame({"pred": [300.0]})

    juntos, y, base, pesos = pseudo.juntar(X, [10.0, 20.0], [1.0, 2.0], X_ps, [30.0], [3.0], 0.3)

    assert juntos["pred"].tolist() == [100.0, 200.0, 300.0]
    assert y.tolist() == [10.0, 20.0, 30.0]
    assert base.tolist() == [1.0, 2.0, 3.0]
    assert pesos.tolist() == [1.0, 1.0, 0.3]


def test_juntar_preserva_os_codigos_das_categoricas_de_2025():
    """Concatenar dois vocabulários trocaria os códigos — e o LightGBM lê o código."""
    X = pd.DataFrame({"apt": pd.Categorical(["LIRF", "EDDF"]), "pred": [1.0, 2.0]})
    X_ps = pd.DataFrame({"apt": pd.Categorical(["EDDF", "LFPG"]), "pred": [3.0, 4.0]})

    juntos, *_ = pseudo.juntar(X, [1.0, 2.0], [0.0, 0.0], X_ps, [3.0, 4.0], [0.0, 0.0])

    assert list(juntos["apt"].cat.categories) == list(X["apt"].cat.categories)
    assert juntos["apt"].cat.codes.tolist()[:2] == X["apt"].cat.codes.tolist()
    # LFPG não existe em 2025: a linha de 2026 entra sem aeroporto, não com o código de outro
    assert juntos["apt"].tolist()[:3] == ["LIRF", "EDDF", "EDDF"]
    assert pd.isna(juntos["apt"].iloc[3])


def test_juntar_reordena_as_colunas_de_2026_na_ordem_de_2025():
    X = pd.DataFrame({"a": [1.0], "b": [2.0]})
    X_ps = pd.DataFrame({"b": [20.0], "a": [10.0]})

    juntos, *_ = pseudo.juntar(X, [0.0], [0.0], X_ps, [0.0], [0.0])

    assert list(juntos.columns) == ["a", "b"]
    assert juntos["a"].tolist() == [1.0, 10.0]


def test_so_a_fonte_campea_viu_o_holdout():
    assert pseudo.meses_do_mestre("propria") == (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
    assert pseudo.meses_do_mestre("campea") == tuple(range(1, 13))
    assert pseudo.vazado("campea") is True
    assert pseudo.vazado("propria") is False


def test_ultima_submissao_e_a_de_maior_versao_e_ignora_o_oof(tmp_path):
    for nome in ("time_v9.parquet", "time_v37.parquet", "time_v38_oof.parquet"):
        (tmp_path / nome).touch()

    assert pseudo.ultima_submissao(tmp_path).name == "time_v37.parquet"


def test_sem_envio_nenhum_a_fonte_campea_para_a_corrida(tmp_path):
    with pytest.raises(SystemExit):
        pseudo.ultima_submissao(tmp_path)


def test_alvo_da_submissao_casa_por_id_e_deixa_nan_em_quem_nao_esta_no_arquivo(tmp_path):
    caminho = tmp_path / "time_v1.parquet"
    pd.DataFrame({F.ID: [2.0, 1.0], F.TARGET: [222.0, 111.0]}).to_parquet(caminho, index=False)

    alvo = pseudo.alvo_da_submissao([1.0, 2.0, 3.0], caminho)

    assert alvo[:2].tolist() == [111.0, 222.0]
    assert np.isnan(alvo[2])


def test_a_chave_separa_bases_diferentes_e_repete_para_a_mesma():
    cfg = {"model": "two_stage_nm", "cls_rounds": 400}

    assert pseudo.chave(cfg) == pseudo.chave(dict(cfg))
    assert pseudo.chave(cfg) != pseudo.chave({**cfg, "cls_rounds": 800})


def test_base_2026_reaproveita_o_arquivo_na_ordem_das_linhas_do_ranking(tmp_path):
    cfg = {"model": "two_stage_nm"}
    destino = pseudo.caminho_base(cfg, tmp_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({F.ID: [3.0, 1.0, 2.0], "pred": [300.0, 100.0, 200.0]}).to_parquet(
        destino, index=False)
    rk = _rk(3)

    pred = pseudo.base_2026(cfg, rk, pasta=tmp_path)

    assert pred.tolist() == [100.0, 200.0, 300.0]


def _voos(meses, ano: int, id0: int, com_alvo: bool = True) -> pd.DataFrame:
    """Voos sintéticos com as colunas que `prepare` exige (padrão do test_crossfit)."""
    linhas = []
    for mes in meses:
        for i in range(12):  # ≥ features.REF_MIN_FLIGHTS
            t = pd.Timestamp(f"{ano}-{mes:02d}-05 08:00", tz="UTC") + pd.Timedelta(minutes=i)
            linhas.append({F.ID: float(id0), "MVT_TIME_UTC_mvt": t,
                           F.TARGET: float(100 * mes) if com_alvo else np.nan})
            id0 += 1
    df = pd.DataFrame(linhas)
    fixas = {F.AIRPORT: "LIRF", "RUNWAY_mvt": "25", "STAND_mvt": "A1"}
    for col in F.CATEGORICAL:
        df[col] = fixas.get(col, "X")
    df["hour"], df["dow"], df["nm_missing"] = 8.0, 1, 0
    return df


def test_a_base_de_2026_so_e_treinada_nos_dez_meses_sem_jan_jul(tmp_path, monkeypatch):
    """O pseudo-rótulo não pode nascer de um modelo que viu o holdout."""
    vistos = {}

    class Falso:
        def __init__(self, cfg):
            pass

        def fit(self, train, cols, run=None, valid=None):
            vistos["meses"] = sorted(set(train["MVT_TIME_UTC_mvt"].dt.month))
            vistos["n"] = vistos.get("n", 0) + 1
            return self

        def predict(self, df):
            return np.full(len(df), 777.0)

    monkeypatch.setitem(models.MODELS, "falso", Falso)
    cfg = {"model": "falso"}
    treino = _voos([2, 3, 4, 5, 6, 8, 9, 10, 11, 12], 2025, id0=1)
    rk = _voos([1, 7], 2026, id0=10_000, com_alvo=False)

    pred = pseudo.base_2026(cfg, rk, train=treino, pasta=tmp_path)

    assert vistos["meses"] == [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
    assert pred.tolist() == [777.0] * len(rk)
    assert pseudo.caminho_base(cfg, tmp_path).exists()

    pseudo.base_2026(cfg, rk, train=treino, pasta=tmp_path)
    assert vistos["n"] == 1  # a segunda corrida lê o arquivo em vez de reajustar


class _Corretor:
    """Corretor falso: devolve sempre a mesma correção."""

    def __init__(self, correcao: float) -> None:
        self.correcao = correcao

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.correcao)


def _rk_pseudo() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """4 partidas de 2026: normal, sem plano NM, cauda e normal de novo."""
    rk = pd.DataFrame({F.ID: [1.0, 2.0, 3.0, 4.0], "nm_missing": [0, 1, 0, 0]})
    pred_rk = np.array([900.0, 1000.0, 3500.0, 1100.0])
    return rk, pred_rk, pd.DataFrame({"pred": pred_rk})


def test_pseudo_propria_rotula_com_o_corretor_da_corrida_e_so_guarda_o_corpo():
    rk, pred_rk, X_rk = _rk_pseudo()

    X_ps, alvo, base = stack.pseudo_2026(_Corretor(200.0), rk, pred_rk, X_rk, "propria",
                                         corte=3600.0, regra_rk=None, janela=False)

    # o voo 2 não tem plano NM e o voo 3 passa do corte com a correção (3.500 + 200)
    assert base.tolist() == [900.0, 1100.0]
    assert alvo.tolist() == [1100.0, 1300.0]
    assert X_ps["pred"].tolist() == [900.0, 1100.0]


def test_pseudo_propria_nao_corrige_as_linhas_de_regra_e_elas_ficam_fora_do_treino():
    rk, pred_rk, X_rk = _rk_pseudo()
    regra = np.array([True, False, False, False])

    X_ps, alvo, base = stack.pseudo_2026(_Corretor(200.0), rk, pred_rk, X_rk, "propria",
                                         corte=3600.0, regra_rk=regra, janela=False)

    assert base.tolist() == [1100.0]
    assert alvo.tolist() == [1300.0]


def test_pseudo_campea_rotula_com_o_ultimo_envio_e_ignora_o_corretor(monkeypatch):
    rk, pred_rk, X_rk = _rk_pseudo()
    monkeypatch.setattr(pseudo, "alvo_da_submissao",
                        lambda ids: np.array([800.0, 800.0, 7000.0, 1200.0]))

    X_ps, alvo, base = stack.pseudo_2026(None, rk, pred_rk, X_rk, "campea",
                                         corte=3600.0, regra_rk=None, janela=False)

    assert alvo.tolist() == [800.0, 1200.0]
    assert base.tolist() == [900.0, 1100.0]


def test_o_peso_das_linhas_de_2026_chega_ao_lightgbm():
    """Sem peso o corretor aprende a média das duas metades; com peso 0, só a primeira."""
    X = pd.DataFrame({"pred": np.zeros(40)})
    base = np.zeros(40)
    y = np.r_[np.full(20, 100.0), np.full(20, -100.0)]
    pesos = np.r_[np.ones(20), np.zeros(20)]

    sem_peso = stack.fit_corrector(X, y, base, rounds=200).predict(X)
    com_peso = stack.fit_corrector(X, y, base, rounds=200, peso=pesos).predict(X)

    assert abs(sem_peso.mean()) < 1.0
    assert com_peso.mean() > 90.0
