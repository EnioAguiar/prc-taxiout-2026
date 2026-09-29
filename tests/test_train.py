import json

import numpy as np
import pandas as pd
import pytest

import features as F
import runlog
import train
from cache import TRUTH
from train import build_submission, corrigir_ranking, final_config


class _Corretor:
    """Corretor falso: devolve sempre a mesma correção."""

    def __init__(self, correcao: float) -> None:
        self.correcao = correcao

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.correcao)


def _ranking() -> pd.DataFrame:
    """Dois voos do ranking: o primeiro com LOBT 2 h antes do MVT, o segundo sem LOBT."""
    t = pd.Timestamp("2026-01-15 10:00", tz="UTC")
    return pd.DataFrame({
        F.ID: [1.0, 2.0],
        F.AIRPORT: pd.Series(["LIRF", "EDDF"], dtype=object),
        "nm_missing": [0, 1],
        "hour": [10.0, 10.0],
        "to_takeoff_from_LOBT_flt": [7200.0, np.nan],
        "MVT_TIME_UTC_mvt": [t, t],
        "LOBT_flt": [t - pd.Timedelta(seconds=7200), pd.NaT],
    })


def test_submissao_segue_a_ordem_do_template_sem_nulos():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    out, preenchidas = build_submission(
        template, np.array([3.0, 1.0]), np.array([30.0, 10.0]), max_fill_frac=0.5
    )
    assert out[F.ID].tolist() == [1.0, 2.0, 3.0]
    assert out[F.TARGET].tolist() == [10.0, 20.0, 30.0]  # ID 2 sem previsão: mediana
    assert preenchidas == 1


def test_submissao_recusa_preencher_demais():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    with pytest.raises(SystemExit, match="sem previsão"):
        build_submission(template, np.array([1.0]), np.array([10.0]))


def test_rodadas_finais_escalam_o_melhor_ponto():
    single = final_config({"config": {"model": "single", "rounds": 400}, "best_iter": 325})
    assert single["rounds"] == 390
    two = final_config({"config": {"model": "two_stage", "cls_rounds": 400, "reg_rounds": 500},
                        "best_iter": None})
    assert (two["cls_rounds"], two["reg_rounds"]) == (480, 600)


def test_rodadas_finais_da_stack_escalam_so_a_base():
    champ = {"config": {"model": "stack_cf", "rounds": 300, "adsb": True,
                        "base_config": {"model": "two_stage_nm", "cls_rounds": 400,
                                        "reg_rounds": 400}},
             "best_iter": 325}
    cfg = final_config(champ)
    assert cfg["base_config"]["cls_rounds"] == 480
    assert cfg["base_config"]["reg_rounds"] == 480
    assert cfg["rounds"] == 300  # corretor intocado, best_iter ignorado
    # a campeã original continua com as rodadas dos blocos (sem escala)
    assert champ["config"]["base_config"]["cls_rounds"] == 400
    assert champ["config"]["rounds"] == 300


def test_envio_com_janela_na_base_para_nos_limites_do_lobt():
    rk = _ranking()
    base = np.array([1000.0, 1000.0])
    cfg_bloco = {"model": "two_stage_nm", "janela_lobt": True}

    alta = corrigir_ranking(_Corretor(1e6), cfg_bloco, False, rk, base)
    baixa = corrigir_ranking(_Corretor(-1e6), cfg_bloco, False, rk, base)

    # voo 1: janela [3594, 10806]; voo 2 sem LOBT segue como hoje
    np.testing.assert_allclose(alta, [10806.0, 1000.0 + 1e6])
    np.testing.assert_allclose(baixa, [3594.0, 0.0])


def test_envio_sem_janela_na_base_e_o_de_hoje():
    rk = _ranking()
    base = np.array([1000.0, 1000.0])

    pred = corrigir_ranking(_Corretor(120.0), {"model": "two_stage_nm"}, False, rk, base)

    np.testing.assert_allclose(pred, [1120.0, 1120.0])  # voo 1 ficaria em 3594 com janela


class _Run:
    def log(self, *a, **k) -> None:
        pass


def _cegas() -> pd.DataFrame:
    df = _ranking()
    df[TRUTH] = [900.0, 1100.0]
    return df


def _corretor_final_com(config: dict, monkeypatch, tmp_path) -> bool:
    """Roda `corretor_final` com dados falsos e devolve o `conjunto` que chegou ao fit."""
    cegas = _cegas()
    visto = {}
    monkeypatch.setattr(train, "load_split", lambda nome: cegas)
    monkeypatch.setattr(train, "oof_base", lambda *a, **k: pd.DataFrame(
        {F.ID: cegas[F.ID], TRUTH: cegas[TRUTH], "pred": [800.0, 1000.0]}))
    monkeypatch.setattr(train, "fit_corrector",
                        lambda X, y, base, conjunto=False, xgb=False: visto.setdefault("conjunto", conjunto))
    train.corretor_final(config.get("base_config", {}), False, cegas, cegas,
                         tmp_path / "oof.parquet", _Run(),
                         config.get("corretor") == "conjunto")
    return visto["conjunto"]


def test_sem_corretor_na_config_o_envio_segue_o_caminho_antigo(monkeypatch, tmp_path):
    assert _corretor_final_com({"base_config": {}}, monkeypatch, tmp_path) is False


def test_com_corretor_conjunto_o_envio_usa_o_conjunto(monkeypatch, tmp_path):
    assert _corretor_final_com(
        {"base_config": {}, "corretor": "conjunto"}, monkeypatch, tmp_path) is True


def _registro(tmp_path, monkeypatch, *ids) -> None:
    """experiments.jsonl sintético; `champion.json` aponta para um arquivo que não existe."""
    linhas = [json.dumps({"id": i, "config": {"model": "single", "rounds": 400},
                          "src_hash": runlog.src_hash(), "best_iter": 300}) for i in ids]
    registro = tmp_path / "experiments.jsonl"
    registro.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    monkeypatch.setattr(train, "REGISTRY", registro)
    monkeypatch.setattr(train, "CHAMPION", tmp_path / "nao_existe.json")
    monkeypatch.setenv("TEAM_NAME", "equipe")


def test_submit_com_corrida_usa_a_linha_do_registro_e_nao_o_campeao(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, "20260101-a", "20260101-b")
    visto = {}

    def parar(champ):
        visto["id"] = champ["id"]
        raise SystemExit("parou depois de escolher a corrida")

    monkeypatch.setattr(train, "final_config", parar)
    with pytest.raises(SystemExit, match="parou"):
        train.submit(14, corrida="20260101-b")

    assert visto["id"] == "20260101-b"


def test_submit_com_corrida_inexistente_cita_o_id(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, "20260101-a")

    with pytest.raises(SystemExit, match="20260101-z"):
        train.submit(14, corrida="20260101-z")
