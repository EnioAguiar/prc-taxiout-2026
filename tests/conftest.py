import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features as F  # noqa: E402


@pytest.fixture
def raw_movements() -> pd.DataFrame:
    """3 decolagens e 1 pouso no mesmo aeroporto, no formato de features.load()."""
    t = pd.Timestamp("2025-01-15 10:00", tz="UTC")
    m = pd.Timedelta(minutes=1)
    df = pd.DataFrame(
        {
            F.ID: [1.0, 2.0, 3.0, 4.0],
            "FLIGHT_ID_mvt": [10.0, np.nan, 30.0, 40.0],
            "PHASE_mvt": ["DEP", "DEP", "DEP", "ARR"],
            "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EDDF"],
            "ADES_mvt": ["EDDF", "EDDF", "EGLL", "LIRF"],
            "RUNWAY_mvt": ["25", "25", "16L", "16R"],
            "STAND_mvt": ["A1", "A2", "B1", "C1"],
            "MVT_TIME_UTC_mvt": [t, t + 5 * m, t + 9 * m, t + 3 * m],
            "BLOCK_TIME_UTC_mvt": [t - 15 * m, t - 200 * m, t - 12 * m, t + 8 * m],
            "SCHED_TIME_UTC_mvt": [t - 30 * m, t - 200 * m, t - 20 * m, t],
            "LOBT_flt": [t - 25 * m, pd.NaT, t - 20 * m, t - 90 * m],
            "IOBT_flt": [t - 25 * m, pd.NaT, t - 20 * m, t - 90 * m],
            "EOBT_1_flt": [t - 25 * m, pd.NaT, t - 20 * m, t - 90 * m],
            "AOBT_3_flt": [t - 15 * m, pd.NaT, t - 12 * m, t - 80 * m],
            F.TARGET: [900.0, 12300.0, 1260.0, 300.0],
        }
    )
    df[F.AIRPORT] = np.where(df["PHASE_mvt"] == "DEP", df["ADEP_mvt"], df["ADES_mvt"])
    return df
