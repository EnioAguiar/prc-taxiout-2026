import json

import pytest

from runlog import Run


def test_registra_mesmo_quando_o_experimento_falha(tmp_path):
    reg = tmp_path / "experiments.jsonl"
    with pytest.raises(ValueError):
        with Run("t", {"model": "x"}, registry=reg, logs=tmp_path / "logs", sample_s=3600) as run:
            with run.phase("dados", 0.5):
                pass
            run.metric(completo=451.2)
            raise ValueError("boom")
    rec = json.loads(reg.read_text().splitlines()[-1])
    assert rec["ok"] is False
    assert "boom" in rec["erro"]
    assert list(rec["fases"]) == ["dados"]
    assert rec["metricas"] == {"completo": 451.2}
    assert (tmp_path / "logs" / f"{rec['id']}.log").read_text().count("dados") >= 2
