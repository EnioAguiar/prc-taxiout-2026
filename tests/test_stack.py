import json

import numpy as np
import pandas as pd
import pytest

import features as F
import stack
from adsb_events import FEATURES as ADSB
from cache import TRUTH
from stack import apply_corrector, base_config, corrector_frame


class _Corretor:
    """Corretor falso: devolve sempre a mesma correção."""

    def __init__(self, correcao: float) -> None:
        self.correcao = correcao

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.correcao)


def _voos() -> pd.DataFrame:
    """Linhas com as colunas que o corretor usa, mais colunas que ele não pode ver."""
    df = pd.DataFrame({
        F.ID: [1.0, 2.0, 3.0],
        F.AIRPORT: ["LIRF", "EDDF", "LIRF"],
        "nm_missing": [0, 1, 0],
        "hour": [8.0, 9.0, 10.0],
        "to_takeoff_from_SCHED_TIME_UTC_mvt": [900.0, 1200.0, np.nan],
        "to_takeoff_from_LOBT_flt": [880.0, np.nan, 1000.0],
        "ref_p10": [600.0, 700.0, 800.0],
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
