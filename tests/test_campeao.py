import json

import numpy as np
import pandas as pd
import pytest

import campeao
from cache import TRUTH
from features import ID


def _corrida(tmp_path, rid, base, preds):
    path = tmp_path / f"{rid}.parquet"
    pd.DataFrame({ID: [1.0, 2.0], "dia": ["2025-01-01", "2025-01-02"], TRUTH: [100.0, 200.0],
                  "pred": preds}).to_parquet(path)
    return {"id": rid, "config": {"model": "stack_cf", "base": base, "reusar_oof": "OOF"},
            "previsoes": str(path), "metricas": {"completo": 1.0}}


def _registro(tmp_path, monkeypatch, *recs):
    reg = tmp_path / "experiments.jsonl"
    reg.write_text("".join(json.dumps(r) + "\n" for r in recs))
    monkeypatch.setattr(campeao, "REGISTRY", reg)
    monkeypatch.setattr(campeao, "ROOT", tmp_path)


def test_previsao_e_a_media_simples_dos_membros(tmp_path, monkeypatch):
    a = _corrida(tmp_path, "a", "B", [110.0, 190.0])
    b = _corrida(tmp_path, "b", "B", [130.0, 230.0])
    _registro(tmp_path, monkeypatch, a, b)
    p = campeao.previsao(["a", "b"])
    assert p["pred"].tolist() == [120.0, 210.0]
    assert p[TRUTH].tolist() == [100.0, 200.0]


def test_nova_exige_a_mesma_base(tmp_path, monkeypatch):
    a, b = _corrida(tmp_path, "a", "B1", [1.0, 2.0]), _corrida(tmp_path, "b", "B2", [1.0, 2.0])
    with pytest.raises(ValueError, match="mesma base"):
        campeao.nova([a, b], medir=False)


def test_salvar_e_carregar_ida_e_volta(tmp_path, monkeypatch):
    a = _corrida(tmp_path, "a", "B", [110.0, 190.0])
    _registro(tmp_path, monkeypatch, a)
    c = campeao.nova([a], enviada={"versao": 7, "sem_loteria": 1.0}, medir=False)
    campeao.salvar(c, tmp_path / "c.json")
    assert campeao.carregar(tmp_path / "c.json") == c
    assert c["base"] == "B" and c["oof"] == "OOF" and c["pos_regras"] == ["roma"]
