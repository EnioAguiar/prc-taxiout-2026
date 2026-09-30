import numpy as np
import pandas as pd
import pytest

import features as F
import pos_regras

MS = "to_takeoff_from_SCHED_TIME_UTC_mvt"


def _voos():
    return pd.DataFrame({
        F.AIRPORT: ["LIRF", "LIRF", "LIRF", "EDDF", "LIRF"],
        "FLIGHT_ID_mvt": [np.nan, np.nan, 123.0, np.nan, np.nan],
        MS: [16 * 3600.0, 14 * 3600.0, 16 * 3600.0, 16 * 3600.0, 30 * 3600.0],
    })


def test_roma_troca_so_lirf_sem_nm_com_atraso_entre_15_e_30_h():
    pred = np.full(5, 1000.0)
    out = pos_regras.roma(_voos(), pred)
    esperado = 54310.76 + 0.38 * 16 * 3600
    np.testing.assert_allclose(out, [esperado, 1000.0, 1000.0, 1000.0, 54310.76 + 0.38 * 30 * 3600])
    assert pred.tolist() == [1000.0] * 5  # não muta a entrada


def test_aplicar_recusa_regra_desconhecida():
    with pytest.raises(ValueError, match="regra"):
        pos_regras.aplicar(_voos(), np.zeros(5), ["nao_existe"])
