import json

import numpy as np
import pandas as pd
import pytest

import campeao
import features as F
import runlog
import train
from cache import TRUTH
from train import build_submission, corrigir_ranking


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


def test_escala_base_multiplica_as_rodadas_sem_mutar():
    cfg = {"model": "two_stage_nm", "cls_rounds": 400, "reg_rounds": 400}
    assert train.escala_base(cfg)["cls_rounds"] == 480
    assert cfg["cls_rounds"] == 400


def test_envio_e_a_media_dos_membros_com_pos_regras(monkeypatch, tmp_path):
    rk = _ranking()
    rk["FLIGHT_ID_mvt"] = [1.0, 2.0]
    rk["to_takeoff_from_SCHED_TIME_UTC_mvt"] = [100.0, 100.0]
    membros = [{"config": {"base_config": {"model": "two_stage_nm"}, "adsb": False}},
               {"config": {"base_config": {"model": "two_stage_nm"}, "adsb": False}}]
    corretores = iter([_Corretor(100.0), _Corretor(300.0)])
    monkeypatch.setattr(train, "prever_membros",
                        lambda ms, full, rk_, run: [(m["config"], next(corretores), {}) for m in ms])
    pred = train.media_membros(membros, rk, np.array([1000.0, 1000.0]), None, None, ["roma"])
    np.testing.assert_allclose(pred, [1200.0, 1200.0])


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
                        lambda X, y, base, conjunto=False, xgb=False, rounds=0, params=None:
                        visto.setdefault("conjunto", conjunto))
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
    """experiments.jsonl sintético no lugar do real; a campeã não pode ser lida."""
    linhas = [json.dumps({"id": i, "config": {"base_config": {"model": "single"}, "adsb": False},
                          "src_hash": runlog.src_hash()}) for i in ids]
    registro = tmp_path / "experiments.jsonl"
    registro.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    monkeypatch.setattr(campeao, "REGISTRY", registro)
    monkeypatch.setattr(campeao, "CAMPEA", tmp_path / "nao_existe.json")
    monkeypatch.setenv("TEAM_NAME", "equipe")


def test_submit_com_corrida_usa_so_essa_corrida_e_nao_a_campea(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, "20260101-a", "20260101-b")
    visto = {}

    def parar(membros, full, rk, run):
        visto["ids"] = [m["id"] for m in membros]
        raise SystemExit("parou depois de escolher os membros")

    monkeypatch.setattr(train, "prever_membros", parar)
    monkeypatch.setattr(train, "load_split", lambda nome: _ranking())
    monkeypatch.setattr(train.pd, "read_parquet", lambda *a, **k: _ranking())
    with pytest.raises(SystemExit, match="parou"):
        train.submit(14, corrida="20260101-b")

    assert visto["ids"] == ["20260101-b"]


def test_submit_com_corrida_inexistente_cita_o_id(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, "20260101-a")

    with pytest.raises(SystemExit, match="20260101-z"):
        train.submit(14, corrida="20260101-z")
