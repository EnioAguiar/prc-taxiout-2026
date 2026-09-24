import numpy as np

from compare import is_better, may_promote, paired_bootstrap


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
