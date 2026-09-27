import cache
import contexto
import features as F
from cache import BLOCK, TRUTH, build_blind


def test_holdout_montado_sem_ver_o_alvo(raw_movements):
    out = build_blind(raw_movements)
    assert out["PHASE_mvt"].eq("DEP").all()
    assert out[F.TARGET].isna().all()
    assert out["BLOCK_TIME_UTC_mvt"].isna().all()
    truth = raw_movements[raw_movements["PHASE_mvt"] == "DEP"].set_index(F.ID)[F.TARGET]
    assert out.set_index(F.ID)[TRUTH].to_dict() == truth.to_dict()
    assert raw_movements[F.TARGET].notna().all()  # não altera a entrada


def _cria_meses(tmp_path) -> list[str]:
    nomes = []
    for m in range(1, 13):
        fim = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
        nome = f"training_2025-{m:02d}-01_{fim}.parquet"
        (tmp_path / nome).write_bytes(b"")
        nomes.append(nome)
    return nomes


def test_blind2025_pega_os_doze_meses(tmp_path, monkeypatch):
    nomes = _cria_meses(tmp_path)
    monkeypatch.setattr(cache, "DATA", tmp_path)
    assert [p.name for p in cache.split_paths("blind2025")] == sorted(nomes)
    assert "blind2025" in cache.SPLITS


def test_blind2025_montado_como_o_ranking(tmp_path, monkeypatch, raw_movements):
    _cria_meses(tmp_path)
    monkeypatch.setattr(cache, "DATA", tmp_path)
    monkeypatch.setattr(cache, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cache, "add_features", lambda df: df)
    monkeypatch.setattr(F, "load", lambda paths: raw_movements.copy())
    visto = {}

    def build(df):
        visto["block_dep_nulo"] = df.loc[df["PHASE_mvt"] == "DEP", BLOCK].isna().all()
        visto["block_arr"] = df.loc[df["PHASE_mvt"] == "ARR", BLOCK].notna().all()
        return df[df["PHASE_mvt"] == "DEP"].copy()

    monkeypatch.setattr(F, "build", build)

    out = cache.load_split("blind2025")

    assert visto == {"block_dep_nulo": True, "block_arr": True}
    assert out[F.TARGET].isna().all()
    dep = raw_movements[raw_movements["PHASE_mvt"] == "DEP"]
    assert out.set_index(F.ID)[TRUTH].to_dict() == dep.set_index(F.ID)[F.TARGET].to_dict()


def test_chave_do_cache_muda_com_o_arquivo_de_contexto(tmp_path, monkeypatch):
    paths = [tmp_path / "training_2025-01-01_2025-02-01.parquet"]
    paths[0].write_bytes(b"")
    fonte = tmp_path / "contexto.py"
    fonte.write_text("A = 1")
    monkeypatch.setattr(contexto, "__file__", str(fonte))
    antes = cache.cache_key(paths)
    fonte.write_text("A = 2")
    assert cache.cache_key(paths) != antes
