import json

import numpy as np
import pandas as pd

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


def test_previsao_alinha_por_voo_membros_de_bases_diferentes(tmp_path, monkeypatch):
    """A base nova grava o holdout na ordem dela: a média é por voo, não por posição."""
    a = _corrida(tmp_path, "a", "B1", [110.0, 190.0])
    b = _corrida(tmp_path, "b", "B2", [230.0, 130.0])  # voos 2 e 1, nessa ordem
    pd.DataFrame({ID: [2.0, 1.0], "dia": ["2025-01-02", "2025-01-01"], TRUTH: [200.0, 100.0],
                  "pred": [230.0, 130.0]}).to_parquet(tmp_path / "b.parquet")
    _registro(tmp_path, monkeypatch, a, b)
    p = campeao.previsao(["a", "b"])
    assert p[ID].tolist() == [1.0, 2.0]
    assert p["pred"].tolist() == [120.0, 210.0]


def test_bases_sao_derivadas_dos_membros_na_ordem_em_que_aparecem(tmp_path, monkeypatch):
    a = _corrida(tmp_path, "a", "B1", [1.0, 2.0])
    b = _corrida(tmp_path, "b", "B2", [1.0, 2.0])
    c = _corrida(tmp_path, "c", "B1", [1.0, 2.0])
    _registro(tmp_path, monkeypatch, a, b, c)
    champ = campeao.nova([a, b, c], medir=False)
    assert campeao.bases(champ) == [{"base": "B1", "oof": "OOF"}, {"base": "B2", "oof": "OOF"}]
    assert campeao.principal(champ) == {"base": "B1", "oof": "OOF"}


def test_base_sem_reusar_oof_aponta_para_a_propria_corrida(tmp_path, monkeypatch):
    a = _corrida(tmp_path, "a", "B", [1.0, 2.0])
    a["config"].pop("reusar_oof")
    _registro(tmp_path, monkeypatch, a)
    assert campeao.bases(campeao.nova([a], medir=False)) == [{"base": "B", "oof": "a"}]


def test_salvar_e_carregar_ida_e_volta(tmp_path, monkeypatch):
    a = _corrida(tmp_path, "a", "B", [110.0, 190.0])
    _registro(tmp_path, monkeypatch, a)
    c = campeao.nova([a], enviada={"versao": 7, "sem_loteria": 1.0}, medir=False)
    campeao.salvar(c, tmp_path / "c.json")
    assert campeao.carregar(tmp_path / "c.json") == c
    assert c["pos_regras"] == ["roma"] and c["membros"][0]["id"] == "a"
