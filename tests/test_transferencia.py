import numpy as np
import pandas as pd
import pytest

import transferencia as T
from compare import paired_bootstrap


def _cenario(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    dias = np.array([f"2025-01-{d:02d}" for d in range(1, 21)])[rng.integers(0, 20, n)]
    y = rng.gamma(2.0, 400.0, n)
    base = y + rng.normal(0, 200, n)
    return y, base, dias


def test_medir_sem_mascara_e_sem_peso_e_o_bootstrap_pareado_de_sempre():
    y, base, dias = _cenario()
    novo = base + 0.3 * (y - base)
    a = T.medir(y, base, novo, dias)
    b = paired_bootstrap(y, base, novo, dias)
    for k in ("ganho", "ic_baixo", "ic_alto"):
        assert a[k] == pytest.approx(b[k], rel=1e-9)


def test_corpo_mais_cauda_somam_o_ganho_completo():
    """A decomposição é exata, não aproximada: é o que autoriza ler cada parcela sozinha."""
    y, base, dias = _cenario()
    novo = base + 0.3 * (y - base)
    corpo = y <= T.TAIL_S
    total = T.medir(y, base, novo, dias)["ganho"]
    soma = (T.medir(y, base, novo, dias, mascara=corpo)["ganho"]
            + T.medir(y, base, novo, dias, mascara=~corpo)["ganho"])
    assert soma == pytest.approx(total, rel=1e-9)


def test_ganho_concentrado_em_poucos_voos_aparece_na_parte_do_topo():
    y, base, dias = _cenario()
    novo = base.copy()
    novo[:20] = y[:20]  # 20 voos corrigidos na mão, o resto intocado
    assert T.parte_do_topo(y, base, novo, k=20)["parte"] == pytest.approx(1.0)
    espalhado = base + 0.3 * (y - base)
    assert T.parte_do_topo(y, base, espalhado, k=20)["parte"] < 0.2


def test_parte_do_topo_passa_de_100_por_cento_quando_o_resto_piora():
    """Caso v37: 20 voos deram 129 % do ganho porque os demais perderam."""
    y, base, dias = _cenario()
    novo = base + 2.0 * (y - base)  # piora em todo mundo...
    novo[:20] = y[:20]              # ...menos nos 20 que o topo pega
    assert T.parte_do_topo(y, base, novo, k=20)["parte"] > 1.0


def test_peso_adversarial_vira_ganho_negativo_quando_o_ganho_mora_so_em_2025():
    """O candidato só melhora nos voos que o adversário diz que são raros em 2026."""
    y, base, dias = _cenario()
    so_2025 = np.arange(len(y)) % 2 == 0
    novo = np.where(so_2025, base + 0.5 * (y - base), base + 0.1 * (base - y))
    w = np.where(so_2025, 0.02, 1.98)  # média 1, quase todo peso fora da metade que ganhou
    assert T.medir(y, base, novo, dias)["ganho"] > 0
    assert T.medir(y, base, novo, dias, w=w)["ganho"] < 0


def _rel(corpo, corpo_ic, corpo_p, topo):
    return {"corpo": {"ganho": corpo, "ic_baixo": corpo_ic, "ic_alto": 9.0},
            "corpo_p": {"ganho": corpo_p, "ic_baixo": -1.0, "ic_alto": 9.0},
            "topo_corpo": {"k": 20, "parte": topo, "ganho_sem_topo": 0.0}}


def test_veredito_aprova_ganho_de_corpo_firme_e_espalhado():
    assert T.veredito(_rel(0.5, 0.2, 0.4, 0.3))["passa"]


@pytest.mark.parametrize("rel,pedaco", [
    (_rel(-0.1, -0.5, 0.4, 0.3), "corpo > 0"),
    (_rel(0.5, -0.01, 0.4, 0.3), "IC do corpo"),
    (_rel(0.5, 0.2, 0.4, 2.5), "top-20"),
    (_rel(0.5, 0.2, -0.2, 0.3), "peso 2026"),
])
def test_veredito_reprova_e_diz_qual_teste_falhou(rel, pedaco):
    v = T.veredito(rel)
    assert not v["passa"] and pedaco in v["motivo"]


def _previsoes_falsas(monkeypatch, quadros):
    import campeao

    def previsao(ids):
        partes = [quadros[i] for i in ids]
        saida = partes[0][[T.ID, "dia", "y_true"]].copy()
        preds = [saida[[T.ID]].merge(p[[T.ID, "pred"]], on=T.ID)["pred"].to_numpy(float)
                 for p in partes]
        saida["pred"] = np.mean(preds, axis=0)
        return saida

    monkeypatch.setattr(campeao, "previsao", previsao)


def test_relatorio_decompoe_o_ganho_e_traz_o_veredito(monkeypatch):
    y, base, dias = _cenario()
    ids = np.arange(len(y), dtype=float)
    novo = base + 0.3 * (y - base)
    quadros = {
        "campea": pd.DataFrame({T.ID: ids, "dia": dias, "y_true": y, "pred": base}),
        "cand": pd.DataFrame({T.ID: ids, "dia": dias, "y_true": y, "pred": novo}),
    }
    _previsoes_falsas(monkeypatch, quadros)
    w = pd.DataFrame({T.ID: ids, "w": np.ones(len(ids))})
    rel = T.relatorio(["cand"], ["campea"], w_tabela=w)
    assert rel["corpo"]["ganho"] + rel["cauda"]["ganho"] == pytest.approx(
        rel["completo"]["ganho"], rel=1e-9)
    assert rel["completo_p"]["ganho"] == pytest.approx(rel["completo"]["ganho"], rel=1e-9)
    assert rel["veredito"]["passa"]
    assert "top-20" in T.texto(rel)


def test_relatorio_reprova_candidato_que_so_ganha_na_cauda(monkeypatch):
    """O `completo` melhora, mas tudo vem de y > 1 h: é o perfil que a v37 tinha."""
    y, base, dias = _cenario()
    y = y.copy()
    y[:300] += 9000.0  # 300 voos de cauda, como os de loteria do holdout
    base = base + np.where(np.arange(len(y)) < 300, 9000.0 + 3000.0, 0.0)
    ids = np.arange(len(y), dtype=float)
    cauda = y > T.TAIL_S
    novo = np.where(cauda, y, base + 0.002 * (base - y))
    quadros = {
        "campea": pd.DataFrame({T.ID: ids, "dia": dias, "y_true": y, "pred": base}),
        "cand": pd.DataFrame({T.ID: ids, "dia": dias, "y_true": y, "pred": novo}),
    }
    _previsoes_falsas(monkeypatch, quadros)
    w = pd.DataFrame({T.ID: ids, "w": np.ones(len(ids))})
    rel = T.relatorio(["cand"], ["campea"], w_tabela=w)
    assert rel["completo"]["ganho"] > 0 and rel["corpo"]["ganho"] < 0
    assert not rel["veredito"]["passa"]
