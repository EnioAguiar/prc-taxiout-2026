import numpy as np

import regua


def _dias(n=40):
    return np.array([f"2025-01-{d:02d}" for d in range(1, 21)] * (n // 20))


def test_metades_alternam_dias_ordenados_e_sao_disjuntas():
    d = np.array(["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-01"])
    m = regua.metades(d)
    assert m.tolist() == ["A", "A", "B", "A"]  # 01→A, 02→B, 03→A


def test_propostas_troca_soma_e_sozinho():
    p = regua.propostas(["m1", "m2"], "n", mesma_base=True)
    assert p == {"troca:m1": ["n", "m2"], "troca:m2": ["m1", "n"], "soma": ["m1", "m2", "n"],
                 "sozinho": ["n"]}
    assert regua.propostas(["m1"], "n", mesma_base=False) == {"sozinho": ["n"]}


def _cenario(ganho_a, ganho_b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    dias = np.repeat([f"2025-01-{d:02d}" for d in range(1, 21)], n // 20)
    y = rng.normal(1000, 200, n)
    base = y + rng.normal(0, 200, n)
    lado = regua.metades(dias)
    melhora = np.where(lado == "A", ganho_a, ganho_b)
    novo = y + (base - y) * (1 - melhora)
    return y, base, novo, dias


def test_aprova_quem_melhora_nas_duas_metades():
    y, base, novo, dias = _cenario(0.05, 0.05)
    r = regua.decidir(y, base, novo, dias, np.ones(len(y), bool))
    assert r["aprovado"], r["motivo"]


def test_reprova_quem_so_melhora_na_metade_de_selecao():
    y, base, novo, dias = _cenario(0.05, -0.05)
    r = regua.decidir(y, base, novo, dias, np.ones(len(y), bool))
    assert not r["aprovado"] and "B" in r["motivo"]


def test_reprova_quem_nao_melhora_em_nenhuma_metade():
    y, base, novo, dias = _cenario(0.0, 0.0)
    r = regua.decidir(y, base, novo, dias, np.ones(len(y), bool))
    assert not r["aprovado"] and "A" in r["motivo"] and r["b"] is None


def test_a_escolhe_entre_as_que_passam_e_nao_a_de_maior_ganho(monkeypatch):
    import pandas as pd

    import campeao
    from cache import TRUTH

    rng = np.random.default_rng(0)
    dias = np.repeat([f"2025-01-{d:02d}" for d in range(1, 21)], 100)
    n = len(dias)
    y = rng.normal(1000, 200, n)
    base = y + rng.normal(0, 150, n)
    tres = np.isin(dias, dias[[0, 100, 200]])
    # A1 (sozinho/troca): ganho alto vindo de três dias, IC atravessando o zero.
    a1 = np.where(tres, y, base + (base - y) * 0.02)
    a2 = y + (base - y) * 0.99  # A2 (soma): ganho menor, consistente em todo dia

    preds = {("m1",): base, ("n",): a1, ("m1", "n"): a2}

    def previsao(ids):
        return pd.DataFrame({TRUTH: y, "dia": dias, "pred": preds[tuple(ids)]})

    monkeypatch.setattr(campeao, "previsao", previsao)
    monkeypatch.setattr(campeao, "registro", lambda i: {"config": {"base": "B"}})
    monkeypatch.setattr(regua, "slice_masks", lambda ref: {"sem loteria": np.ones(n, bool)})

    r = regua.avaliar(["m1"], "n")
    assert r["proposta"] == "soma" and r["membros"] == ["m1", "n"]
    assert r["a"]["ganho"] >= regua.GANHO_A and r["a"]["ic_baixo"] > 0
