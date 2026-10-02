import numpy as np

import regua


def _dias(n=40):
    return np.array([f"2025-01-{d:02d}" for d in range(1, 21)] * (n // 20))


def test_metades_alternam_dias_ordenados_e_sao_disjuntas():
    d = np.array(["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-01"])
    m = regua.metades(d)
    assert m.tolist() == ["A", "A", "B", "A"]  # 01→A, 02→B, 03→A


def test_propostas_troca_soma_e_sozinho():
    p = regua.propostas(["m1", "m2"], "n")
    assert p == {"troca:m1": ["n", "m2"], "troca:m2": ["m1", "n"], "soma": ["m1", "m2", "n"],
                 "sozinho": ["n"]}


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
    from features import ID

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
        return pd.DataFrame({ID: np.arange(n, dtype=float), TRUTH: y, "dia": dias,
                             "pred": preds[tuple(ids)]})

    monkeypatch.setattr(campeao, "previsao", previsao)
    monkeypatch.setattr(regua, "slice_masks", lambda ref: {"sem loteria": np.ones(n, bool)})

    r = regua.avaliar(["m1"], "n")
    assert r["proposta"] == "soma" and r["membros"] == ["m1", "n"]
    assert r["a"]["ganho"] >= regua.GANHO_A and r["a"]["ic_baixo"] > 0


def _quadros_de_duas_bases(ruido_novo=150.0, seed=0):
    """Campeã e base nova com erros independentes e o mesmo holdout em ordens diferentes."""
    import pandas as pd

    from cache import TRUTH
    from features import ID

    rng = np.random.default_rng(seed)
    dias = np.repeat([f"2025-01-{d:02d}" for d in range(1, 21)], 100)
    n = len(dias)
    y = rng.normal(1000, 200, n)
    voos = np.arange(n, dtype=float)
    campea, nova = y + rng.normal(0, 150, n), y + rng.normal(0, ruido_novo, n)
    ordem = rng.permutation(n)  # a base nova grava o holdout na ordem dela
    return n, {
        "m": pd.DataFrame({ID: voos, TRUTH: y, "dia": dias, "pred": campea}),
        "n": pd.DataFrame({ID: voos[ordem], TRUTH: y[ordem], "dia": dias[ordem],
                           "pred": nova[ordem]}),
    }


def _previsao_falsa(quadros):
    """Como `campeao.previsao`: média alinhada pelo voo, na ordem do primeiro membro."""
    import numpy as np

    from features import ID

    def previsao(ids):
        partes = [quadros[i[0]] for i in ids]
        ref = partes[0].drop(columns="pred").copy()
        ref["pred"] = np.mean([ref[[ID]].merge(p[[ID, "pred"]], on=ID,
                                               validate="one_to_one")["pred"].to_numpy(float)
                               for p in partes], axis=0)
        return ref

    return previsao


def test_avaliar_conjunto_prefere_somar_as_duas_bases_a_trocar_por_uma(monkeypatch):
    import campeao

    n, quadros = _quadros_de_duas_bases()
    monkeypatch.setattr(campeao, "previsao", _previsao_falsa(quadros))
    monkeypatch.setattr(regua, "slice_masks", lambda ref: {"sem loteria": np.ones(n, bool)})

    r = regua.avaliar_conjunto(["m1", "m2"], ["n1", "n2"])

    assert r["avaliadas"] == 2 and r["aprovado"], r["motivo"]
    assert r["proposta"] == "soma" and r["membros"] == ["m1", "m2", "n1", "n2"]
    assert r["a"]["ganho"] > 20  # erros independentes: a média corta ~30% do RMSE


def test_avaliar_conjunto_alinha_a_base_nova_pelo_voo_antes_de_comparar(monkeypatch):
    """Base nova bem melhor sozinha: só é escolhida se o holdout dela for alinhado por voo
    (sem alinhar, a previsão vira ruído e a proposta despenca)."""
    import campeao

    n, quadros = _quadros_de_duas_bases(ruido_novo=15.0)
    monkeypatch.setattr(campeao, "previsao", _previsao_falsa(quadros))
    monkeypatch.setattr(regua, "slice_masks", lambda ref: {"sem loteria": np.ones(n, bool)})

    r = regua.avaliar_conjunto(["m1"], ["n1"])

    assert r["proposta"] == "base_nova" and r["membros"] == ["n1"] and r["aprovado"], r["motivo"]


def test_confirmacao_em_b_so_exige_o_sinal_quando_os_dias_todos_confirmam():
    # B com ganho pequeno e IC largo (como a base com plano 13, 30/09) passa se os dias todos
    # sem loteria têm IC baixo > 0
    y, base, novo, dias = _cenario(0.05, 0.004)
    r = regua.decidir(y, base, novo, dias, np.ones(len(y), bool))
    assert r["b"]["ganho"] > 0 and r["aprovado"], r["motivo"]


def test_reprova_quando_b_piora_mesmo_com_a_forte():
    y, base, novo, dias = _cenario(0.05, -0.01)
    r = regua.decidir(y, base, novo, dias, np.ones(len(y), bool))
    assert not r["aprovado"] and r["motivo"].startswith("B")
