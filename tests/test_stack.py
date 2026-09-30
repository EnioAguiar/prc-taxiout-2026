import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import features as F
import stack
from adsb_events import FEATURES as ADSB
from cache import TRUTH
from stack import (
    apply_corrector,
    base_config,
    config_da_base,
    corrector_frame,
    previsao_corrigida,
)


class _Corretor:
    """Corretor falso: devolve sempre a mesma correção."""

    def __init__(self, correcao: float) -> None:
        self.correcao = correcao

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.correcao)


def _voos() -> pd.DataFrame:
    """Linhas com as colunas que o corretor usa, mais colunas que ele não pode ver.

    Janela do LOBT (MVT − LOBT ± 3606 s): voo 1 em [−2406, 4806], voo 2 sem LOBT (sem
    janela), voo 3 em [3594, 10806].
    """
    t = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    df = pd.DataFrame({
        F.ID: [1.0, 2.0, 3.0],
        F.AIRPORT: ["LIRF", "EDDF", "LIRF"],
        "nm_missing": [0, 1, 0],
        "hour": [8.0, 9.0, 10.0],
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [900.0, 1200.0, np.nan],
        "to_takeoff_from_LOBT_flt": [880.0, np.nan, 1000.0],
        "ref_p10": [600.0, 700.0, 800.0],
        "MVT_TIME_UTC_mvt": [t, t, t],
        "LOBT_flt": [t - pd.Timedelta(seconds=1200), pd.NaT, t - pd.Timedelta(seconds=7200)],
        F.TARGET: [1000.0, 1100.0, 1200.0],
        TRUTH: [1000.0, 1100.0, 1200.0],
    })
    for i, col in enumerate(ADSB):
        df[col] = [10.0 * i + 1, np.nan, 10.0 * i + 3]
    df["adsb_taxi_move"] = [700.0, np.nan, 950.0]
    return df


def test_correcao_negativa_grande_vira_zero_e_correcao_nula_devolve_a_base():
    base = np.array([100.0, 500.0])
    X = pd.DataFrame({"pred": base})

    assert apply_corrector(_Corretor(-1e6), X, base).tolist() == [0.0, 0.0]
    np.testing.assert_allclose(apply_corrector(_Corretor(0.0), X, base), base)
    np.testing.assert_allclose(apply_corrector(_Corretor(60.0), X, base), base + 60)


def test_sem_adsb_o_corretor_nao_ve_nenhuma_coluna_adsb():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])

    X = corrector_frame(df, pred, adsb=False)

    assert [c for c in X.columns if c.startswith("adsb")] == []
    np.testing.assert_allclose(X["pred"], pred)
    assert len(X) == len(df)


def test_com_adsb_o_corretor_ve_a_diferenca_entre_o_adsb_e_a_previsao():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])

    X = corrector_frame(df, pred)

    assert set(ADSB) <= set(X.columns)
    np.testing.assert_allclose(
        X["adsb_menos_pred"].to_numpy(float), df["adsb_taxi_move"].to_numpy(float) - pred
    )


def test_o_corretor_nunca_recebe_o_alvo():
    X = corrector_frame(_voos(), np.array([800.0, 900.0, 1000.0]))

    assert F.TARGET not in X.columns and TRUTH not in X.columns


def test_com_janela_o_corretor_ve_a_distancia_ate_os_limites_do_lobt():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])

    X = corrector_frame(df, pred, janela=True)

    np.testing.assert_allclose(X["dist_lo"].to_numpy(float), [800 + 2406, np.nan, 1000 - 3594])
    np.testing.assert_allclose(X["dist_hi"].to_numpy(float), [4806 - 800, np.nan, 10806 - 1000])


def test_sem_janela_o_corretor_nao_ve_os_limites_do_lobt():
    X = corrector_frame(_voos(), np.array([800.0, 900.0, 1000.0]), janela=False)

    assert "dist_lo" not in X.columns and "dist_hi" not in X.columns


def test_com_janela_a_correcao_enorme_para_nos_limites_do_lobt():
    df = _voos()
    base = np.array([800.0, 900.0, 1000.0])

    alta = previsao_corrigida(_Corretor(1e6), df, base, adsb=False, janela=True)
    baixa = previsao_corrigida(_Corretor(-1e6), df, base, adsb=False, janela=True)

    np.testing.assert_allclose(alta, [4806.0, 900.0 + 1e6, 10806.0])  # voo 2 não tem janela
    np.testing.assert_allclose(baixa, [0.0, 0.0, 3594.0])  # piso 0 depois da projeção


def test_sem_janela_a_previsao_corrigida_e_a_de_hoje():
    df = _voos()
    base = np.array([800.0, 900.0, 1000.0])

    pred = previsao_corrigida(_Corretor(120.0), df, base, adsb=False, janela=False)

    np.testing.assert_allclose(pred, [920.0, 1020.0, 1120.0])  # voo 3 ficaria em 3594 com janela



def test_base_config_pega_a_config_da_ultima_corrida_com_aquele_id(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text("\n".join(json.dumps(r) for r in [
        {"id": "20260101-a", "config": {"model": "single", "rounds": 100}},
        {"id": "20260102-b", "config": {"model": "two_stage_nm"}},
        {"id": "20260101-a", "config": {"model": "single", "rounds": 400}},
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    assert base_config("20260101-a") == {"model": "single", "rounds": 400}


def test_base_config_falha_quando_a_corrida_nao_existe(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps({"id": "outra", "config": {}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    with pytest.raises(SystemExit, match="20260101-a"):
        base_config("20260101-a")


def test_config_da_base_soma_a_media_de_seeds_aos_blocos(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    assert config_da_base("20260101-a", 1) == {"model": "two_stage_nm", "seed": 0}
    assert config_da_base("20260101-a", 5) == {"model": "two_stage_nm", "seed": 0, "seeds": 5}


def _entradas(n: int = 60) -> pd.DataFrame:
    """Entradas sintéticas do corretor: dois aeroportos, colunas numéricas simples."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        F.AIRPORT: pd.Series(["LIRF", "EDDF"] * (n // 2), dtype="category"),
        "nm_missing": rng.integers(0, 2, n).astype(float),
        "hour": rng.uniform(0, 24, n),
        "pred": rng.uniform(500, 1500, n),
    })


def test_conjunto_preve_a_media_dos_tres_submodelos():
    X = _entradas(6)
    conj = stack.Conjunto(
        _Corretor(10.0),
        {"LIRF": _Corretor(20.0), "EDDF": _Corretor(20.0)},
        _Corretor(60.0),
    )

    np.testing.assert_allclose(conj.predict(X), np.full(len(X), 30.0))


def test_aeroporto_visto_so_na_previsao_usa_o_modelo_global():
    X = _entradas(4)
    X[F.AIRPORT] = pd.Series(["LIRF", "LFPG", "LIRF", "LFPG"], dtype="category")
    conj = stack.Conjunto(_Corretor(10.0), {"LIRF": _Corretor(70.0)}, _Corretor(10.0))

    # LIRF: (10 + 70 + 10)/3 = 30; LFPG sem modelo próprio cai no global: 10
    np.testing.assert_allclose(conj.predict(X), [30.0, 10.0, 30.0, 10.0])


def test_sem_conjunto_o_corretor_e_o_booster_de_hoje(monkeypatch):
    X = _entradas()
    y, base = X["pred"].to_numpy() + 100, X["pred"].to_numpy()
    monkeypatch.setattr(stack, "ROUNDS", 5)

    assert isinstance(stack.fit_corrector(X, y, base), lgb.Booster)


def test_com_conjunto_o_corretor_e_a_media_dos_tres(monkeypatch):
    X = _entradas()
    y = X["pred"].to_numpy() + 100 + np.linspace(-5, 5, len(X))
    base = X["pred"].to_numpy()
    monkeypatch.setattr(stack, "ROUNDS", 5)
    monkeypatch.setattr(stack, "CATBOOST", {**stack.CATBOOST, "iterations": 5})

    model = stack.fit_corrector(X, y, base, conjunto=True)

    assert isinstance(model, stack.Conjunto)
    assert set(model.aeroportos) == {"EDDF", "LIRF"}
    # o conjunto corrige de verdade: a correção aprendida é ~100 (y − base)
    np.testing.assert_allclose(model.predict(X), np.full(len(X), 100.0), atol=40)


def test_a_config_do_crossfit_so_tem_corretor_com_a_flag(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(
        stack.parser().parse_args(["v12", "--crossfit", "--conjunto"]), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v12", "--crossfit"]), "20260101-a")

    assert com["corretor"] == "conjunto"
    assert "corretor" not in sem


def test_com_externos_o_corretor_ganha_so_as_colunas_ext():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])
    hoje = corrector_frame(df, pred)

    X = corrector_frame(df, pred, externos={
        "ext_taxa_cia": [0.1, 0.2, 0.3], "ext_solo_s": [60.0, np.nan, 90.0],
    })

    assert [c for c in hoje.columns if c.startswith("ext_")] == []
    assert list(X.columns) == [*hoje.columns, "ext_taxa_cia", "ext_solo_s"]
    pd.testing.assert_frame_equal(X[hoje.columns], hoje)
    np.testing.assert_allclose(X["ext_solo_s"].to_numpy(float), [60.0, np.nan, 90.0])


def _plano13() -> pd.DataFrame:
    """Quadro do plano 13 de três voos, com a companhia categórica de vocabulário fixo."""
    return pd.DataFrame({
        "met_temp": [1.0, np.nan, 12.0],
        "rot_idade": [3600.0, np.nan, 7200.0],
        "cia": pd.Categorical(["AZA", "outro", "DLH"], categories=["AZA", "DLH", "outro"]),
    }, index=[7, 8, 9])  # índice diferente do quadro: as colunas entram por posição


def test_com_plano13_o_corretor_ganha_as_colunas_novas_e_a_cia_categorica():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])
    hoje = corrector_frame(df, pred)

    X = corrector_frame(df, pred, plano13=_plano13())

    assert list(X.columns) == [*hoje.columns, "met_temp", "rot_idade", "cia"]
    pd.testing.assert_frame_equal(X[hoje.columns], hoje)
    np.testing.assert_allclose(X["rot_idade"].to_numpy(float), [3600.0, np.nan, 7200.0])
    assert list(X["cia"].cat.categories) == ["AZA", "DLH", "outro"]


def test_o_catboost_recebe_todas_as_categoricas_em_texto():
    X = corrector_frame(_voos(), np.array([800.0, 900.0, 1000.0]), plano13=_plano13())

    Xc = stack.catboost_frame(X)

    assert stack.colunas_cat(X) == [F.AIRPORT, "cia"]
    assert Xc[F.AIRPORT].tolist() == ["LIRF", "EDDF", "LIRF"]
    assert Xc["cia"].tolist() == ["AZA", "outro", "DLH"]
    assert not stack.colunas_cat(Xc)


def test_a_config_do_crossfit_so_tem_plano13_com_a_flag(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(
        stack.parser().parse_args(["v18", "--crossfit", "--externos", "--plano13"]), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v18", "--crossfit", "--externos"]), "20260101-a")

    assert com["plano13"] is True and com["externos"] is True
    assert "plano13" not in sem


def test_a_config_do_crossfit_so_tem_externos_com_a_flag(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(
        stack.parser().parse_args(["v16", "--crossfit", "--externos"]), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v16", "--crossfit"]), "20260101-a")

    assert com["externos"] is True
    assert "externos" not in sem


def test_sem_feature_tira_as_colunas_do_corretor_e_mantem_as_outras():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])
    todas = corrector_frame(df, pred)

    X = corrector_frame(df, pred, sem=("adsb_lat0", "adsb_lon0"))

    assert "adsb_lat0" not in X.columns and "adsb_lon0" not in X.columns
    assert list(X.columns) == [c for c in todas.columns if c not in ("adsb_lat0", "adsb_lon0")]
    assert "adsb_menos_pred" in X.columns
    assert [c for c in X.columns if c.startswith("adsb_")]


def test_sem_a_lista_o_corretor_ve_as_colunas_de_hoje():
    df = _voos()
    pred = np.array([800.0, 900.0, 1000.0])

    assert list(corrector_frame(df, pred, sem=()).columns) == list(corrector_frame(df, pred).columns)


def test_a_config_do_crossfit_grava_sem_features_no_topo_e_na_base(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(stack.parser().parse_args(
        ["v14", "--crossfit", "--sem-feature", "adsb_lat0", "--sem-feature", "adsb_lon0"],
    ), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v14", "--crossfit"]), "20260101-a")

    assert com["sem_features"] == ["adsb_lat0", "adsb_lon0"]
    assert com["base_config"]["sem_features"] == ["adsb_lat0", "adsb_lon0"]
    assert "sem_features" not in sem and "sem_features" not in sem["base_config"]


def test_sem_feature_sem_crossfit_e_recusado(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stack.py", "v14", "--sem-feature", "adsb_lat0"])

    with pytest.raises(SystemExit):
        stack.main()

    assert "--sem-feature só vale com --crossfit" in capsys.readouterr().err


def _registro_v20(tmp_path, monkeypatch, cfg_base: dict, oof: str | None = "runs/v20_oof.parquet"):
    """Registro com uma corrida `stack_cf` que gravou o oof daquela base."""
    registro = tmp_path / "experiments.jsonl"
    rec = {"id": "20260101-v20", "config": {"model": "stack_cf", "base": "20260101-a",
                                            "base_config": cfg_base}}
    if oof:
        rec["oof"] = oof
        caminho = tmp_path / oof
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_bytes(b"")
    registro.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)
    monkeypatch.setattr(stack, "ROOT", tmp_path)
    return registro


def test_reusar_oof_devolve_o_caminho_quando_a_base_e_a_mesma(tmp_path, monkeypatch):
    cfg_base = {"model": "two_stage_nm", "seed": 0, "janela_lobt": True}
    _registro_v20(tmp_path, monkeypatch, cfg_base)

    assert stack.oof_reusado("20260101-v20", "20260101-a", dict(cfg_base)) == \
        tmp_path / "runs/v20_oof.parquet"


def test_reusar_oof_recusa_base_config_diferente(tmp_path, monkeypatch):
    _registro_v20(tmp_path, monkeypatch, {"model": "two_stage_nm", "seed": 0, "rounds": 400})

    with pytest.raises(SystemExit, match="não é a desta"):
        stack.oof_reusado("20260101-v20", "20260101-a",
                          {"model": "two_stage_nm", "seed": 0, "rounds": 800})


def test_reusar_oof_recusa_outra_corrida_base(tmp_path, monkeypatch):
    cfg_base = {"model": "two_stage_nm", "seed": 0}
    _registro_v20(tmp_path, monkeypatch, cfg_base)

    with pytest.raises(SystemExit, match="não é a desta"):
        stack.oof_reusado("20260101-v20", "20260101-outra", dict(cfg_base))


def test_reusar_oof_recusa_corrida_sem_oof_gravado(tmp_path, monkeypatch):
    cfg_base = {"model": "two_stage_nm", "seed": 0}
    _registro_v20(tmp_path, monkeypatch, cfg_base, oof=None)

    with pytest.raises(SystemExit, match="não gravou `oof`"):
        stack.oof_reusado("20260101-v20", "20260101-a", dict(cfg_base))


def test_a_config_do_crossfit_so_tem_reusar_oof_com_a_flag(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(stack.parser().parse_args(
        ["v21", "--crossfit", "--reusar-oof", "20260101-v20"]), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v21", "--crossfit"]), "20260101-a")

    assert com["reusar_oof"] == "20260101-v20"
    assert "reusar_oof" not in sem


def test_reusar_oof_sem_crossfit_e_recusado(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stack.py", "v21", "--reusar-oof", "20260101-v20"])

    with pytest.raises(SystemExit):
        stack.main()

    assert "--reusar-oof só vale com --crossfit" in capsys.readouterr().err


def test_a_config_do_crossfit_grava_corretor_params(tmp_path, monkeypatch):
    registro = tmp_path / "experiments.jsonl"
    registro.write_text(json.dumps(
        {"id": "20260101-a", "config": {"model": "two_stage_nm", "seed": 0}}
    ) + "\n", encoding="utf-8")
    monkeypatch.setattr(stack, "REGISTRY", registro)

    com = stack.config_da_corrida(stack.parser().parse_args(
        ["v31", "--crossfit", "--corretor-params", '{"num_leaves": 127}']), "20260101-a")
    sem = stack.config_da_corrida(
        stack.parser().parse_args(["v31", "--crossfit"]), "20260101-a")

    assert com["corretor_params"] == {"num_leaves": 127}
    assert "corretor_params" not in sem


def test_corretor_params_sem_crossfit_e_recusado(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stack.py", "v31", "--corretor-params", '{"num_leaves": 7}'])

    with pytest.raises(SystemExit):
        stack.main()

    assert "--corretor-params só vale com --crossfit" in capsys.readouterr().err
