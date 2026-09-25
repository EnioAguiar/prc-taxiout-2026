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


def test_so_promove_contra_o_campeao():
    champ = {"id": "A", "config": {"model": "single", "rounds": 400}}
    assert may_promote({"id": "A", "config": champ["config"]}, champ)
    assert may_promote({"id": "A2", "config": champ["config"]}, champ)  # repetição do campeão
    assert not may_promote({"id": "B", "config": {"model": "single", "rounds": 200}}, champ)
    assert may_promote({"id": "B", "config": {"model": "single", "rounds": 200}}, None)


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
