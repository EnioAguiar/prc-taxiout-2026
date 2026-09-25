import numpy as np

from teto import ceiling_gain


def test_teto_so_ganha_se_o_novo_chega_mais_perto_do_oraculo():
    oracle = np.array([10_000.0, 20_000.0])
    base = np.array([5_000.0, 5_000.0])
    assert ceiling_gain(base, oracle, oracle, 1_000, 300.0) > 0
    assert ceiling_gain(base, base, oracle, 1_000, 300.0) == 0
    assert ceiling_gain(oracle, base, oracle, 1_000, 300.0) < 0
    # conta exata: ΔSSE = −(5000² + 15000²) em 1000 voos, a partir de 300 s
    expected = 300 - np.sqrt(max(300**2 - (5000**2 + 15000**2) / 1000, 0))
    assert abs(ceiling_gain(base, oracle, oracle, 1_000, 300.0) - expected) < 1e-9
