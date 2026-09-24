import features as F
from cache import TRUTH, build_blind


def test_holdout_montado_sem_ver_o_alvo(raw_movements):
    out = build_blind(raw_movements)
    assert out["PHASE_mvt"].eq("DEP").all()
    assert out[F.TARGET].isna().all()
    assert out["BLOCK_TIME_UTC_mvt"].isna().all()
    truth = raw_movements[raw_movements["PHASE_mvt"] == "DEP"].set_index(F.ID)[F.TARGET]
    assert out.set_index(F.ID)[TRUTH].to_dict() == truth.to_dict()
    assert raw_movements[F.TARGET].notna().all()  # não altera a entrada
