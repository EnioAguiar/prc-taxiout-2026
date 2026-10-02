import numpy as np
import pandas as pd

from compare import by_month, gain_without_top, is_better, may_promote, paired_bootstrap, verdict


def _data(seed=1):
    rng = np.random.default_rng(seed)
    y = rng.normal(900, 400, 20_000)
    days = np.repeat(np.arange(62), 20_000 // 62 + 1)[:20_000]
    return rng, y, days


def test_mesmo_modelo_nao_ganha():
    rng, y, days = _data()
    p = y + rng.normal(0, 300, y.size)
    r = paired_bootstrap(y, p, p, days)
    assert r["ganho"] == 0
    assert r["ic_baixo"] <= 0 <= r["ic_alto"]
    assert not is_better(r)


def test_novo_claramente_melhor_vence_com_ganho_positivo():
    rng, y, days = _data()
    base = y + rng.normal(0, 300, y.size)
    new = y + rng.normal(0, 200, y.size)
    r = paired_bootstrap(y, base, new, days)
    assert r["ganho"] > 50 and r["ic_baixo"] > 0
    assert is_better(r)
    assert not is_better(paired_bootstrap(y, new, base, days))


def test_so_promove_comparando_com_a_campea():
    champ = {"membros": [{"id": "A", "config": {}}]}
    assert may_promote(None, champ)        # sem <id_base>: comparou com a média da campeã
    assert may_promote("A", champ)         # base = o único membro
    assert not may_promote("B", champ)
    assert may_promote("B", None)


def _jan_jul(n=20_000, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-01-01", periods=31).append(pd.date_range("2025-07-01", periods=31))
    dia = np.array(days.strftime("%Y-%m-%d"))[rng.integers(0, 62, n)]
    y = rng.normal(900, 400, n)
    return rng, y, dia


def test_ganho_espalhado_e_melhor():
    rng, y, dia = _jan_jul()
    base = y + rng.normal(0, 300, y.size)
    new = y + rng.normal(0, 200, y.size)
    res = paired_bootstrap(y, base, new, dia)
    assert verdict(res, gain_without_top(y, base, new), by_month(y, base, new, dia)) == "MELHOR"


def test_ganho_de_poucos_voos_e_fragil():
    rng, y, dia = _jan_jul()
    base = y + rng.normal(0, 300, y.size)
    new = base + rng.normal(0, 30, y.size)  # um pouco pior em quase tudo
    idx = rng.choice(y.size, 10, replace=False)
    base[idx] = y[idx] + 40_000  # 10 voos (= TOP_K) em que a base erra horas
    new[idx] = y[idx]
    res = paired_bootstrap(y, base, new, dia)
    sem_top = gain_without_top(y, base, new)
    assert res["ganho"] >= 10 and sem_top <= 0
    assert verdict(res, sem_top, by_month(y, base, new, dia)) == "FRÁGIL"


def test_nao_promove_corrida_destilada_do_proprio_holdout(monkeypatch):
    """`--pseudo campea` marca `pseudo_vazado`: este holdout não julga a corrida."""
    import campeao
    import compare
    import pytest

    salvos = []
    monkeypatch.setattr(campeao, "salvar", lambda c: salvos.append(c))

    with pytest.raises(SystemExit, match="pseudo_vazado"):
        compare.promote({"id": "x", "config": {"pseudo": "campea", "pseudo_vazado": True}})

    assert not salvos
