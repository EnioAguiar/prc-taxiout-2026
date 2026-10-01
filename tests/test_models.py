from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import models
from models import (
    JANELA_LOBT_S, apply_lines, build_model, combine, copied_from_sched, fit_lines,
    janela_lobt, limitar_janela, line_rows, nm_groups,
)


def test_combina_pela_esperanca_da_mistura():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    ms = np.array([5000.0, 5000.0, 5000.0, np.nan])  # sem SCHED: só o regressor
    reg = np.array([900.0, 900.0, 900.0, 900.0])
    assert combine(p, ms, reg).tolist() == [900.0, 5000.0, 2950.0, 900.0]


def test_previsao_nunca_negativa():
    assert combine(np.array([0.0]), np.array([0.0]), np.array([-50.0])).tolist() == [0.0]


def test_rotulo_de_copia_usa_60_s_e_trata_nulos_como_normais():
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    s = pd.Timedelta(seconds=1)
    df = pd.DataFrame({
        "SCHED_TIME_UTC_mvt": [t, t, t, t, pd.NaT],
        "BLOCK_TIME_UTC_mvt": [t + 60 * s, t - 60 * s, t + 61 * s, pd.NaT, t],
    })
    assert copied_from_sched(df).tolist() == [True, True, False, False, False]


def test_retas_por_grupo_recuperam_a_relacao_e_usam_fallback():
    ms = np.array([0.0, 1000.0, 2000.0, 3000.0] * 20 + [0.0, 1000.0])
    keys = np.array(["LIRF"] * 80 + ["EDDF"] * 2)
    y = np.where(keys == "LIRF", -2818 + 1.062 * ms, 900.0)
    lines, fallback = fit_lines(ms, y, keys, min_rows=50)
    assert set(lines) == {"LIRF"}  # EDDF tem 2 linhas: usa a reta global
    a, b = lines["LIRF"]
    assert abs(a + 2818) < 1e-4 and abs(b - 1.062) < 1e-7
    out = apply_lines(np.array([3000.0, 1000.0, np.nan]), np.array(["LIRF", "EDDF", "LIRF"]),
                      lines, fallback)
    assert abs(out[0] - (-2818 + 1.062 * 3000)) < 1e-3
    assert out[1] == max(0.0, fallback[0] + fallback[1] * 1000.0)
    assert np.isnan(out[2])


def test_retas_nunca_preveem_negativo():
    lines = {"LIRF": (-2818.0, 1.062)}
    assert apply_lines(np.array([0.0]), np.array(["LIRF"]), lines, (0.0, 0.0)).tolist() == [0.0]


def test_reta_so_troca_voos_sem_nm_acima_do_limiar():
    nm = np.array([1, 1, 1, 0, 1], bool)
    ms = np.array([30_000.0, 3_000.0, np.nan, 30_000.0, 21_600.0])
    line = np.array([29_000.0, 2_000.0, np.nan, 29_000.0, 20_000.0])
    assert line_rows(nm, ms, line, 21_600).tolist() == [True, False, False, False, False]
    assert line_rows(nm, ms, line, 0).tolist() == [True, True, False, False, True]


def test_grupos_com_corte_de_2h():
    df = pd.DataFrame({
        "AIRPORT": pd.Categorical(["LIRF", "LIRF", "EDDF"]),
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [9000.0, 600.0, np.nan],
    })
    assert nm_groups(df, split_ms=True).tolist() == ["LIRF|>2h", "LIRF|<=2h", "EDDF|<=2h"]
    assert nm_groups(df, split_ms=False).tolist() == ["LIRF", "LIRF", "EDDF"]


class _ModeloDaSeed:
    """Modelo falso: prevê a própria seed em todas as linhas."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.best_iter = None

    def fit(self, train, cols, run=None, valid=None) -> "_ModeloDaSeed":
        self.treinou = (len(train), tuple(cols), valid)
        self.best_iter = 100 + int(self.cfg["seed"])
        return self

    def predict(self, df) -> np.ndarray:
        return np.full(len(df), float(self.cfg["seed"]))


class _Run:
    """Run falso: só guarda as mensagens de progresso."""

    def __init__(self) -> None:
        self.linhas = []

    def log(self, msg: str) -> None:
        self.linhas.append(msg)


@pytest.fixture
def modelo_da_seed(monkeypatch):
    monkeypatch.setitem(models.MODELS, "falso", _ModeloDaSeed)
    return "falso"


def test_media_de_seeds_usa_seeds_consecutivas_e_loga_o_progresso(modelo_da_seed):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    run = _Run()

    m = build_model({"model": modelo_da_seed, "seed": 3, "seeds": 4})
    m.fit(df, ["x"], run=run, valid=None)

    assert [c["seed"] for c in m.cfgs] == [3, 4, 5, 6]
    assert m.predict(df).tolist() == [4.5, 4.5, 4.5]  # média de 3, 4, 5 e 6
    assert m.best_iter == 103  # o da primeira cópia
    assert run.linhas == [f"seed {k}/4 (seed={k + 2})" for k in range(1, 5)]


def test_cada_copia_treina_com_seeds_1_para_nao_recursar(modelo_da_seed):
    m = build_model({"model": modelo_da_seed, "seed": 0, "seeds": 2})

    assert [c["seeds"] for c in m.cfgs] == [1, 1]


def test_todas_as_copias_veem_o_mesmo_treino_e_holdout(modelo_da_seed):
    df = pd.DataFrame({"x": [1.0, 2.0]})
    hold = pd.DataFrame({"x": [9.0]})

    m = build_model({"model": modelo_da_seed, "seed": 0, "seeds": 3}).fit(df, ["x"], valid=hold)

    assert [c.treinou for c in m.models] == [(2, ("x",), hold)] * 3


def test_sem_seeds_ou_com_uma_seed_o_modelo_e_o_de_hoje(modelo_da_seed):
    assert isinstance(build_model({"model": modelo_da_seed, "seed": 0}), _ModeloDaSeed)
    assert isinstance(build_model({"model": modelo_da_seed, "seed": 0, "seeds": 1}), _ModeloDaSeed)


def _voos(linhas: list[dict]) -> pd.DataFrame:
    """Voos com MVT, LOBT e SCHED em horas do dia (None = horário ausente)."""
    dia = pd.Timestamp("2025-07-01", tz="UTC")

    def t(h):
        return pd.NaT if h is None else dia + pd.Timedelta(hours=h)

    def coluna(chave):  # como no cache: datetime UTC mesmo quando tudo é nulo
        return pd.to_datetime([t(r.get(chave)) for r in linhas], utc=True)

    df = pd.DataFrame({
        "MVT_TIME_UTC_mvt": coluna("mvt"),
        "LOBT_flt": coluna("lobt"),
        "SCHED_TIME_UTC_mvt": coluna("sched"),
    })
    df[models.SCHED_GAP] = (df["MVT_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    df["x"] = 1.0
    return df


def test_janela_do_lobt_e_o_gap_mais_menos_3606_s():
    df = _voos([{"mvt": 10.0, "lobt": 9 + 40 / 60}, {"mvt": 10.0}])

    lo, hi = janela_lobt(df)

    assert JANELA_LOBT_S == 3606
    assert (lo[0], hi[0]) == (1200 - 3606, 1200 + 3606)
    assert np.isnan(lo[1]) and np.isnan(hi[1])


def test_limitar_janela_projeta_e_depois_aplica_o_piso_zero():
    df = _voos([
        {"mvt": 10.0, "lobt": 9 + 40 / 60},  # janela [−2406, 4806]
        {"mvt": 10.0, "lobt": 9 + 40 / 60},
        {"mvt": 10.0, "lobt": 9 + 40 / 60},
        {"mvt": 10.0},                       # sem LOBT: fica como está
        {"mvt": 10.0, "lobt": 4.0},          # janela [17_994, 25_206]: sobe até o piso dela
        {"mvt": 10.0, "lobt": 13.0},         # janela [−14_406, −7194]: a projeção é negativa
    ])
    pred = np.array([9000.0, -9000.0, 900.0, 40_000.0, 10.0, 0.0])

    # acima do teto vira teto, abaixo do piso vira piso, e o piso 0 vem depois da projeção
    assert limitar_janela(pred, df).tolist() == [4806.0, 0.0, 900.0, 40_000.0, 17_994.0, 0.0]


class _Fixo:
    """Estágio falso: devolve sempre o mesmo valor em todas as linhas."""

    def __init__(self, valor: float) -> None:
        self.valor = valor

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.valor)


def _dois_estagios(janela: bool) -> models.TwoStage:
    m = models.TwoStage({"janela_lobt": janela} if janela else {})
    m.cols, m.cls, m.reg = ["x"], _Fixo(1.0), _Fixo(900.0)
    return m


def test_dois_estagios_zera_p_com_sched_fora_da_janela():
    # MVT 12:00, LOBT 09:40 → janela [4794, 12_006]
    df = _voos([
        {"mvt": 12.0, "lobt": 9 + 40 / 60, "sched": 11 + 40 / 60},  # ms 1200: fora
        {"mvt": 12.0, "lobt": 9 + 40 / 60, "sched": 9.5},           # ms 9000: dentro
        {"mvt": 12.0, "sched": 11 + 40 / 60},                       # sem LOBT: p segue
    ])

    assert _dois_estagios(janela=True).predict(df).tolist() == [900.0, 9000.0, 1200.0]
    assert _dois_estagios(janela=False).predict(df).tolist() == [1200.0, 9000.0, 1200.0]


def test_build_model_limita_a_previsao_final_so_com_janela_ligada(modelo_da_seed):
    df = _voos([{"mvt": 10.0, "lobt": 10.0}])  # janela [−3606, 3606]
    cfg = {"model": modelo_da_seed, "seed": 5000}

    assert build_model(cfg).predict(df).tolist() == [5000.0]
    assert build_model({**cfg, "janela_lobt": True}).predict(df).tolist() == [3606.0]


def test_janela_limita_a_media_das_seeds_uma_vez(modelo_da_seed):
    df = _voos([{"mvt": 10.0, "lobt": 10.0}])  # janela [−3606, 3606]
    m = build_model({"model": modelo_da_seed, "seed": 5000, "seeds": 2, "janela_lobt": True})

    m.fit(df, ["x"])

    assert m.predict(df).tolist() == [3606.0]  # a média 5000,5 projetada
    assert [type(c) for c in m.models] == [_ModeloDaSeed] * 2  # cópias sem projeção própria
    assert m.best_iter == 5100  # o do modelo de dentro continua visível


def _frame_prepare() -> pd.DataFrame:
    """Voos sintéticos com as colunas que prepare() exige, mais duas colunas adsb_*."""
    import features as F
    from cache import TRUTH

    n = 12  # ≥ features.REF_MIN_FLIGHTS: a P10 do grupo só existe com voos suficientes
    t = pd.Timestamp("2025-03-05 08:00", tz="UTC")
    df = pd.DataFrame({
        F.ID: np.arange(n, dtype=float),
        "MVT_TIME_UTC_mvt": [t + pd.Timedelta(minutes=i) for i in range(n)],
        F.TARGET: np.full(n, 900.0),
        TRUTH: np.full(n, 900.0),
    })
    fixas = {F.AIRPORT: "LIRF", "RUNWAY_mvt": "25", "STAND_mvt": "A1"}
    for col in F.CATEGORICAL:
        df[col] = fixas.get(col, "X")
    df["hour"], df["dow"], df["nm_missing"] = 8.0, 1, 0
    df["adsb_lat0"], df["adsb_lon0"] = 41.8, 12.2
    return df


def test_sem_features_tira_so_as_colunas_pedidas():
    train, outro = _frame_prepare(), _frame_prepare()

    todas = models.prepare(train, [outro])
    menos = models.prepare(_frame_prepare(), [_frame_prepare()], sem=("adsb_lat0",))

    assert "adsb_lat0" in todas
    assert menos == [c for c in todas if c != "adsb_lat0"]


def test_sem_features_com_nome_inexistente_e_erro():
    with pytest.raises(ValueError, match="nao_existe"):
        models.prepare(_frame_prepare(), [_frame_prepare()], sem=("nao_existe",))


def _frame_base(mes: int, voo: str = "AZ100", atraso_h: float = 2.0) -> pd.DataFrame:
    """`_frame_prepare` com os relógios que `--base-ret` e `--base-ext` pedem."""
    import features as F

    df = _frame_prepare()
    t = pd.Timestamp(f"2025-{mes:02d}-05 08:00", tz="UTC")
    df["MVT_TIME_UTC_mvt"] = [t + pd.Timedelta(minutes=i) for i in range(len(df))]
    df["FLIGHT_mvt"] = voo
    df["SCHED_TIME_UTC_mvt"] = df["MVT_TIME_UTC_mvt"] - pd.Timedelta(hours=atraso_h)
    df["AOBT_3_flt"] = df["MVT_TIME_UTC_mvt"] - pd.Timedelta(minutes=15)
    df[F.AIRPORT] = "LIRF"
    return df


def _bruto_dep(mes: int, n: int, copias: int) -> pd.DataFrame:
    """DEP brutas de um mês: `copias` delas com BLOCK colado no SCHED (atraso > 1 h)."""
    import features as F

    t = pd.Timestamp(f"2025-{mes:02d}-05 08:00", tz="UTC")
    sched = [t - pd.Timedelta(hours=2) + pd.Timedelta(minutes=i) for i in range(n)]
    block = [s if i < copias else s + pd.Timedelta(hours=1, minutes=50)
             for i, s in enumerate(sched)]
    return pd.DataFrame({
        "PHASE_mvt": "DEP",
        F.AIRPORT: "LIRF",
        "FLIGHT_mvt": "AZ100",
        "FLIGHT_ID_mvt": 1.0,
        "MVT_TIME_UTC_mvt": [s + pd.Timedelta(hours=2) for s in sched],
        "SCHED_TIME_UTC_mvt": sched,
        "BLOCK_TIME_UTC_mvt": block,
    })


def test_base_ext_so_aprende_a_taxa_de_copia_nos_meses_de_treino(monkeypatch):
    """A taxa de um voo nunca vê o BLOCK do mês dele nem o do mês previsto."""
    from externos import CopiaCia

    copia = CopiaCia(k=0.0)  # sem suavização: a taxa é a do próprio grupo
    copia.fit(_bruto_dep(1, 10, copias=0))    # janeiro: nenhuma cópia
    copia.fit(_bruto_dep(2, 10, copias=10))   # fevereiro: tudo cópia
    copia.fit(_bruto_dep(7, 10, copias=10))   # julho: o mês previsto, tudo cópia
    monkeypatch.setattr(models, "colunas_ext", lambda df, c, meses: {
        nome: valores.to_numpy(float) for nome, valores in c.transform(df, meses).items()})

    train = pd.concat([_frame_base(1), _frame_base(2)], ignore_index=True)
    hold = _frame_base(7)
    cols = models.prepare(train, [hold], ext=True, copia=copia)

    assert "ext_taxa_cia" in cols and "ext_taxa_cia_ms" in cols
    # julho não está no treino: a taxa dele vem de jan + fev (10 cópias em 20 voos)
    np.testing.assert_allclose(hold["ext_taxa_cia"].to_numpy(float), 0.5)
    # cada mês de treino só enxerga o outro: janeiro vê fevereiro (1,0) e vice-versa (0,0)
    mes = train["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
    np.testing.assert_allclose(train["ext_taxa_cia"].to_numpy(float)[mes == 1], 1.0)
    np.testing.assert_allclose(train["ext_taxa_cia"].to_numpy(float)[mes == 2], 0.0)


def test_base_ext_sem_tabela_pronta_usa_a_do_modulo(monkeypatch):
    from externos import CopiaCia

    copia = CopiaCia(k=0.0).fit(_bruto_dep(1, 10, copias=10)).fit(_bruto_dep(2, 10, copias=10))
    chamadas = []
    monkeypatch.setattr(models, "copia_cia_2025", lambda: chamadas.append(1) or copia)
    monkeypatch.setattr(models, "colunas_ext", lambda df, c, meses: {
        nome: valores.to_numpy(float) for nome, valores in c.transform(df, meses).items()})

    train = pd.concat([_frame_base(1), _frame_base(2)], ignore_index=True)
    models.prepare(train, [_frame_base(7)], ext=True)

    assert chamadas == [1]


def test_base_ret_grava_as_colunas_de_retencao_em_todos_os_frames(tmp_path, monkeypatch):
    import features as F
    import pista

    t = pd.Timestamp("2025-03-05 08:00", tz="UTC")
    bruto = pd.DataFrame([{
        "PHASE_mvt": "DEP", "ADEP_mvt": "LIRF", "RUNWAY_mvt": "25",
        "MVT_TIME_UTC_mvt": t + pd.Timedelta(minutes=i),
        "AOBT_3_flt": t + pd.Timedelta(minutes=i - 15),
        "EOBT_1_flt": t + pd.Timedelta(minutes=i - 20),
        "SCHED_TIME_UTC_mvt": t + pd.Timedelta(minutes=i - 10),
    } for i in range(12)])
    bruto.to_parquet(tmp_path / "training_2025-03-01_2025-04-01.parquet", index=False)
    monkeypatch.setattr(models, "colunas_retencao", lambda df: pista.colunas_retencao(df, tmp_path))

    train, outro = _frame_base(3), _frame_base(3)
    cols = models.prepare(train, [outro], ret=True)

    assert [c for c in pista.COLS_RET if c not in cols] == []
    assert train["ret_ativos_rwy"].notna().all() and outro["ret_ativos_rwy"].notna().all()
    assert F.TARGET not in cols  # o alvo nunca entra como feature


def _treino_regressor() -> pd.DataFrame:
    """Voos sintéticos: uma cópia do SCHED, cauda em LIRF sem NM e voos normais."""
    import features as F

    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    s = pd.Timedelta(seconds=1)
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "SCHED_TIME_UTC_mvt": [t, t, t, t, t],
        "BLOCK_TIME_UTC_mvt": [t, t + 900 * s, t + 900 * s, t + 900 * s, t + 900 * s],
        F.AIRPORT: pd.Categorical(["LIRF", "LIRF", "LIRF", "EDDF", "EDDF"]),
        "nm_missing": [1, 1, 0, 1, 0],
        F.TARGET: [500.0, 80_000.0, 9000.0, 30_000.0, 600.0],
    })


def _treina_capturando(cfg: dict, df: pd.DataFrame, monkeypatch) -> list:
    """Roda TwoStage.fit com lgb.train falso; devolve [(dados, alvo)] por estágio."""
    vistos = []

    def falso(params, dataset, rounds, callbacks=None):
        vistos.append((dataset.data, np.asarray(dataset.label, float)))
        return SimpleNamespace(free_dataset=lambda: None)

    monkeypatch.setattr(models.lgb, "train", falso)
    models.TwoStage(cfg).fit(df, ["x"])
    return vistos


def test_reg_corte_limita_so_o_alvo_do_regressor(monkeypatch):
    df = _treino_regressor()

    (cls_x, cls_y), (reg_x, reg_y) = _treina_capturando({"reg_corte": 7200}, df, monkeypatch)

    assert cls_x["x"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # classificador vê tudo
    assert cls_y.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert reg_x["x"].tolist() == [2.0, 3.0, 4.0, 5.0]  # sem a cópia, como hoje
    assert reg_y.tolist() == [7200.0, 7200.0, 7200.0, 600.0]


def test_sem_reg_corte_o_alvo_do_regressor_e_o_bruto(monkeypatch):
    df = _treino_regressor()

    (_, _), (reg_x, reg_y) = _treina_capturando({}, df, monkeypatch)

    assert reg_x["x"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert reg_y.tolist() == [80_000.0, 9000.0, 30_000.0, 600.0]


def test_reg_sem_lirf_nm_tira_so_as_linhas_de_roma_sem_nm(monkeypatch):
    df = _treino_regressor()

    (cls_x, _), (reg_x, reg_y) = _treina_capturando({"reg_sem_lirf_nm": True}, df, monkeypatch)

    assert cls_x["x"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # classificador não muda
    # sai a LIRF sem NM (x=2); ficam LIRF com NM (3) e EDDF sem NM (4) e com NM (5)
    assert reg_x["x"].tolist() == [3.0, 4.0, 5.0]
    assert reg_y.tolist() == [9000.0, 30_000.0, 600.0]


class _Const:
    """Regressor falso que devolve sempre o mesmo valor, para checar o roteamento."""

    def __init__(self, valor: float) -> None:
        self.valor = valor

    def predict(self, x) -> np.ndarray:
        return np.full(len(x), self.valor)


def _dois_estagios_por_apt(regs_apt: dict) -> models.TwoStage:
    m = models.TwoStage({"base_por_apt": True})
    m.cols = ["x"]
    m.cls = _Const(0.0)  # nenhuma cópia: a previsão é só a do regressor
    m.reg = _Const(1000.0)
    m.regs_apt = regs_apt
    return m


def _frame_apt() -> pd.DataFrame:
    import features as F

    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        models.SCHED_GAP: [np.nan, np.nan, np.nan],  # sem SCHED: a mistura é só o regressor
        F.AIRPORT: pd.Categorical(["LTFM", "EDDF", "LTFM"]),
    })


def test_por_apt_mistura_o_global_com_o_do_aeroporto_so_onde_ele_existe():
    m = _dois_estagios_por_apt({"LTFM": _Const(2000.0)})

    pred = m.predict(_frame_apt())

    assert pred.tolist() == [1500.0, 1000.0, 1500.0]  # EDDF sem modelo próprio fica no global


def test_sem_regressores_por_aeroporto_a_previsao_e_a_do_global():
    m = _dois_estagios_por_apt({})

    assert m.predict(_frame_apt()).tolist() == [1000.0, 1000.0, 1000.0]


def test_por_apt_so_treina_aeroporto_com_linhas_bastantes(monkeypatch):
    import features as F

    n = models.MIN_LINHAS_APT
    t = pd.Timestamp("2025-07-01 10:00", tz="UTC")
    df = pd.DataFrame({
        "x": np.arange(n + 5, dtype=float),
        "SCHED_TIME_UTC_mvt": [t] * (n + 5),
        "BLOCK_TIME_UTC_mvt": [t + pd.Timedelta(seconds=900)] * (n + 5),
        F.AIRPORT: pd.Categorical(["LTFM"] * n + ["EDDF"] * 5),
        "nm_missing": [0] * (n + 5),
        F.TARGET: np.full(n + 5, 600.0),
    })
    vistos = []

    def falso(params, dataset, rounds, callbacks=None):
        vistos.append(len(dataset.data))
        return SimpleNamespace(free_dataset=lambda: None)

    monkeypatch.setattr(models.lgb, "train", falso)
    modelo = models.TwoStage({"base_por_apt": True}).fit(df, ["x"])

    assert list(modelo.regs_apt) == ["LTFM"]  # EDDF tem 5 linhas, fica só com o global
    assert vistos == [n + 5, n + 5, n]  # classificador, regressor global, regressor do LTFM
