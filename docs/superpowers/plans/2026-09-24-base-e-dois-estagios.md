# Base de experimentos + modelo em dois estágios — plano de implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline que roda experimentos em ~2–3 min com metade do PC, mostra progresso (% por fase, ETA, RAM/CPU/GPU), registra tudo e decide por comparação pareada; e o primeiro ganho de modelo (dois estágios, −70 s medidos na simulação).

**Architecture:** Módulos pequenos em `src/`: `runlog` (progresso e registro), `cache` (features prontas por split), `models` (interface única), `experiment` (simulação calibrada), `compare` (bootstrap pareado e campeão), `train` (versão final, sem enviar). `bin/run` aplica o orçamento de hardware em todo comando pesado.

**Tech Stack:** Python 3.14, pandas 3, LightGBM 4.7 (CPU, 12 threads), psutil, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`

**Escopo deste plano:** parte 1 inteira + itens 0–2 da parte 2. Os itens 3–9 ganham planos próprios depois, porque dependem dos resultados medidos aqui.

## Global Constraints

- Todo processo pesado roda por `bin/run` (`taskset -c 0-5,12-17 nice -n 5`); LightGBM com `num_threads=12`; RAM somada ≤ 7 GB.
- Processos longos (> 1 min) rodam supervisionados (hub `start`), nunca num shell com limite de 5 min.
- Nada é enviado ao placar neste plano sem ok explícito do usuário. `s3.py submit` só depois do ok.
- L2 no alvo bruto; sem log, Huber, corte de outliers ou limite superior de previsão.
- `experiments.jsonl` e `champion.json` ficam na raiz e são versionados; `runs/`, `logs/`, `data/` não.
- Textos de saída e documentação em português.
- Commit + push ao fim de cada tarefa.

## Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `bin/run` | criar | Orçamento de hardware |
| `requirements.txt` | alterar | + psutil, pytest |
| `tests/conftest.py` | criar | `src/` no path; movimentos sintéticos |
| `src/runlog.py` | criar | Progresso, recursos, registro |
| `src/cache.py` | criar | Splits com features em cache |
| `src/models.py` | criar | `PARAMS`, `prepare`, `SingleLGBM`, `TwoStage`, `combine` |
| `src/experiment.py` | criar | Experimento na simulação calibrada |
| `src/compare.py` | criar | Bootstrap pareado, veredito, campeão |
| `src/train.py` | reescrever | Só `submit N` a partir do campeão |
| `src/features.py` | alterar | + `nm_missing` (tarefa 7) |
| `sim_ranking.py` | apagar | Substituído por `experiment.py` |
| `README.md` | alterar | Uso, roadmap, correção da hipótese do AOBT_3 |
| `tests/test_*.py` | criar | Contratos que um bug plausível quebraria |

---

### Task 1: Orçamento de hardware e dependências

**Files:**
- Create: `bin/run`
- Create: `tests/conftest.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `bin/run <script> [args]` (roda com o `.venv` do projeto, 6 núcleos físicos, nice 5); fixture pytest `raw_movements`.

- [ ] **Step 1: Criar `bin/run`**

```sh
#!/usr/bin/env sh
# Roda com metade do PC: 6 núcleos físicos (0-5 e os irmãos de hyperthreading 12-17)
# e prioridade baixa, para o PC continuar usável. Ex.: bin/run src/experiment.py base
cd "$(dirname "$0")/.." || exit 1
exec taskset -c 0-5,12-17 nice -n 5 .venv/bin/python "$@"
```

Run: `chmod +x bin/run`

- [ ] **Step 2: Dependências**

`requirements.txt` final:

```
pandas>=2.2
pyarrow>=17
numpy>=2.0
lightgbm>=4.5
minio>=7.2
python-dotenv>=1.0
psutil>=6.0
pytest>=8.0
```

Run: `.venv/bin/pip install -q -r requirements.txt`

- [ ] **Step 3: `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Verificar o orçamento**

Run: `bin/run -c "import os; print(sorted(os.sched_getaffinity(0)), os.nice(0))"`
Expected: `[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17] 5`

Run: `.venv/bin/python -m pytest -q`
Expected: `no tests ran` (sem erro de import).

- [ ] **Step 5: Commit**

```bash
git add bin/run requirements.txt tests/conftest.py
git commit -m "Orçamento de hardware (bin/run) e base de testes"
git push
```

---

### Task 2: `runlog.py` — progresso, recursos e registro

**Files:**
- Create: `src/runlog.py`
- Test: `tests/test_runlog.py`

**Interfaces:**
- Produces: `ROOT: Path`, `REGISTRY: Path` (`experiments.jsonl`), `LOGS: Path`; `hms(s) -> str`; `class Run(name: str, config: dict, registry=REGISTRY, logs=LOGS, sample_s=30.0)` com `.id: str`, `.log(msg)`, `.phase(label, weight)` (context manager), `.progress(frac, detail)`, `.lgb_callback(rounds, label="rodada", every=50, start=0.0, span=1.0)`, `.metric(**kw)`, `.set(**kw)`.

- [ ] **Step 1: Teste que falha**

`tests/test_runlog.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_runlog.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'runlog'`).

- [ ] **Step 3: Implementar `src/runlog.py`**

```python
"""Progresso, uso de recursos e registro de experimentos.

    with Run("dois_estagios", config) as run:
        with run.phase("dados", 0.15):
            ...
        with run.phase("treino", 0.75):
            lgb.train(..., callbacks=[run.lgb_callback(400)])
        run.metric(completo=391.1)

Cada linha mostra fase, % da fase, % total, ETA e RAM/CPU/GPU. Tudo vai para o
terminal e para logs/<id>.log. Ao sair (também com erro, ok=false) uma linha
JSON é acrescentada a experiments.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "experiments.jsonl"
LOGS = ROOT / "logs"


def src_hash() -> str:
    """Identidade do código que gerou o resultado (pega mudança não commitada)."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def git_commit() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return r.stdout.strip() or "?"


def gpu_usage() -> tuple[int, int] | None:
    """(% de uso, MB de VRAM) da GPU 0, ou None sem nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        util, mem = out.splitlines()[0].split(", ")
        return int(util), int(mem)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


class Run:
    def __init__(
        self, name: str, config: dict, registry: Path = REGISTRY, logs: Path = LOGS,
        sample_s: float = 30.0,
    ) -> None:
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
        self.rec: dict = {
            "id": self.id, "nome": name, "data": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(), "src_hash": src_hash(), "config": config,
            "fases": {}, "metricas": {},
        }
        self._registry = registry
        logs.mkdir(parents=True, exist_ok=True)
        self._log = (logs / f"{self.id}.log").open("a")
        self._proc = psutil.Process()
        self._proc.cpu_percent()
        self._t0 = time.perf_counter()
        self._sample_s = sample_s
        self._stop = threading.Event()
        self._done = 0.0  # soma dos pesos das fases concluídas
        self._phase = "início"
        self._weight = 0.0
        self._frac = 0.0  # fração concluída da fase atual
        self._rss_peak = 0.0
        self._gpu: tuple[int, int] | None = None

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self._log.write(msg + "\n")
        self._log.flush()

    def _resources(self) -> str:
        rss = self._proc.memory_info().rss / 2**30
        self._rss_peak = max(self._rss_peak, rss)
        text = f"RAM {rss:.1f} GB · CPU {self._proc.cpu_percent():.0f}%"
        if self._gpu:
            text += f" · GPU {self._gpu[0]}% {self._gpu[1]} MB"
        return text

    def _status(self) -> str:
        total = 100 * (self._done + self._weight * self._frac)
        return f"[{self._phase} {100 * self._frac:3.0f}% | total {total:3.0f}%]"

    def _watch(self) -> None:
        while not self._stop.wait(self._sample_s):
            self._gpu = gpu_usage()
            elapsed = hms(time.perf_counter() - self._t0)
            self.log(f"{self._status()} {self._resources()} · decorrido {elapsed}")

    def __enter__(self) -> "Run":
        threading.Thread(target=self._watch, daemon=True).start()
        self.log(f"== {self.id} (código {self.rec['src_hash']}, commit {self.rec['git_commit']})")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._resources()
        self.rec["total_s"] = round(time.perf_counter() - self._t0, 1)
        self.rec["rss_pico_gb"] = round(self._rss_peak, 2)
        self.rec["ok"] = exc_type is None
        if exc is not None:
            self.rec["erro"] = repr(exc)
        with self._registry.open("a") as f:
            f.write(json.dumps(self.rec, ensure_ascii=False) + "\n")
        self.log(
            f"== fim {self.id}: {hms(self.rec['total_s'])} · pico RAM "
            f"{self.rec['rss_pico_gb']} GB · ok={self.rec['ok']}"
        )
        self._log.close()

    @contextmanager
    def phase(self, label: str, weight: float):
        """Fase com peso no % total (a soma dos pesos de um Run deve ser 1)."""
        self._phase, self._weight, self._frac = label, weight, 0.0
        t = time.perf_counter()
        self.log(f"{self._status()} {label}...")
        yield
        self._frac = 1.0
        dt = time.perf_counter() - t
        self.rec["fases"][label] = round(dt, 1)
        self.log(f"{self._status()} {label} concluída em {hms(dt)} · {self._resources()}")
        self._done += weight
        self._weight, self._frac = 0.0, 0.0

    def progress(self, frac: float, detail: str) -> None:
        self._frac = min(max(frac, 0.0), 1.0)
        self.log(f"{self._status()} {detail} · {self._resources()}")

    def lgb_callback(
        self, rounds: int, label: str = "rodada", every: int = 50,
        start: float = 0.0, span: float = 1.0,
    ):
        """Callback do LightGBM: % da fase, ms/rodada, ETA e métrica de validação.

        start/span: fatia da fase ocupada por este treino (ex.: 2º de 2 modelos: 0.5/0.5).
        """
        t0 = time.perf_counter()

        def cb(env) -> None:
            i = env.iteration + 1
            if i % every and i != rounds:
                return
            el = time.perf_counter() - t0
            ev = " ".join(
                f"{name}={val:.1f}" for _, name, val, *_ in env.evaluation_result_list or []
            )
            detail = (
                f"{label} {i}/{rounds} · {1000 * el / i:.0f} ms/r · "
                f"ETA {hms(el * (rounds - i) / i)} {ev}"
            ).rstrip()
            self.progress(start + span * i / rounds, detail)

        cb.order = 30
        return cb

    def metric(self, **values) -> None:
        self.rec["metricas"].update(values)

    def set(self, **values) -> None:
        self.rec.update(values)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest tests/test_runlog.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/runlog.py tests/test_runlog.py
git commit -m "runlog: progresso por fase com ETA, RAM/CPU/GPU e registro em experiments.jsonl"
git push
```

---

### Task 3: `cache.py` — splits com features prontas

**Files:**
- Create: `src/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `features.load`, `features.build`, `features.TARGET`, `features.ID`; `runlog.Run`, `runlog.LOGS`.
- Produces: `DATA: Path`, `TRUTH = "y_true"`, `SPLITS`, `split_paths(name) -> list[Path]`, `build_blind(raw) -> DataFrame`, `load_split(name) -> DataFrame` (índice 0..n-1; `train2025`/`full2025` só com alvo não nulo; `holdout2025` com `TARGET` e BLOCK nulos e verdade em `y_true`).

- [ ] **Step 1: Teste que falha**

`tests/test_cache.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: FAIL (`No module named 'cache'`).

- [ ] **Step 3: Implementar `src/cache.py`**

```python
"""Features prontas em data/cache/, refeitas só quando features.py ou os dados mudam.

Splits:
    train2025    10 meses de 2025 (sem jan e jul), com alvo
    holdout2025  jan + jul/2025 montados como o ranking (BLOCK e alvo apagados nas
                 DEP antes das features); a verdade fica na coluna y_true
    full2025     12 meses de 2025, com alvo (treino da versão final)
    ranking2026  ranking.parquet

    bin/run src/cache.py    # monta os 4 em sequência (pico ~5 GB; rodar sozinho)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import features as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
HOLDOUT_MONTHS = {1, 7}
TRUTH = "y_true"
BLOCK = "BLOCK_TIME_UTC_mvt"
SPLITS = ("train2025", "holdout2025", "full2025", "ranking2026")


def _month(p: Path) -> int:
    """training_2025-01-01_2025-02-01.parquet -> 1"""
    return int(p.name.split("_")[1][5:7])


def split_paths(name: str) -> list[Path]:
    training = sorted(DATA.glob("training_2025-*.parquet"))
    match name:
        case "train2025":
            return [p for p in training if _month(p) not in HOLDOUT_MONTHS]
        case "holdout2025":
            return [p for p in training if _month(p) in HOLDOUT_MONTHS]
        case "full2025":
            return training
        case "ranking2026":
            return [DATA / "ranking.parquet"]
    raise ValueError(f"split desconhecido: {name}")


def cache_key(paths: list[Path]) -> str:
    h = hashlib.sha256(Path(F.__file__).read_bytes())
    for p in paths:
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]


def build_blind(raw: pd.DataFrame) -> pd.DataFrame:
    """Monta as DEP como no ranking: BLOCK e alvo apagados antes das features."""
    dep = raw["PHASE_mvt"] == "DEP"
    truth = pd.Series(raw.loc[dep, F.TARGET].to_numpy(), index=raw.loc[dep, F.ID].to_numpy())
    blind = raw.copy()
    blind.loc[dep, BLOCK] = pd.NaT
    blind.loc[dep, F.TARGET] = np.nan
    out = F.build(blind)
    out[TRUTH] = out[F.ID].map(truth)
    return out


def load_split(name: str) -> pd.DataFrame:
    paths = split_paths(name)
    if not paths or any(not p.exists() for p in paths):
        raise SystemExit(f"Faltam dados de {name}. Rode: .venv/bin/python src/s3.py download")
    target = CACHE / f"{name}-{cache_key(paths)}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    raw = F.load(paths)
    df = build_blind(raw) if name == "holdout2025" else F.build(raw)
    del raw
    if name in ("train2025", "full2025"):
        df = df[df[F.TARGET].notna()]
    df = df.reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for old in CACHE.glob(f"{name}-*.parquet"):
        old.unlink()
    df.to_parquet(target, compression="zstd", index=False)
    return df


if __name__ == "__main__":
    from runlog import LOGS, Run

    with Run("cache", {"splits": list(SPLITS)}, registry=LOGS / "infra.jsonl") as run:
        for split in SPLITS:
            with run.phase(split, 1 / len(SPLITS)):
                run.log(f"{split}: {len(load_split(split)):,} linhas")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest tests/test_cache.py -q`
Expected: `1 passed`

- [ ] **Step 5: Montar o cache real (sozinho, supervisionado)**

Run (hub start, nome `prc-cache`): `bin/run src/cache.py`
Expected: 4 fases concluídas; `train2025` ~1,74 M linhas, `holdout2025` ~344 k, `full2025` ~2,08 M, `ranking2026` 344.841; `data/cache/` com 4 arquivos (~90 MB no total por split grande); pico de RAM ≤ 5,5 GB.

Run de novo: `bin/run src/cache.py` → Expected: termina em < 15 s (leitura do cache).

- [ ] **Step 6: Commit**

```bash
git add src/cache.py tests/test_cache.py
git commit -m "cache: splits com features prontas e holdout montado como o ranking"
git push
```

---

### Task 4: `models.py` + `experiment.py` + linha de base

**Files:**
- Create: `src/models.py`
- Create: `src/experiment.py`
- Delete: `sim_ranking.py`

**Interfaces:**
- Consumes: `cache.load_split`, `cache.TRUTH`; `runlog.Run`, `runlog.ROOT`; `features.*`.
- Produces: `models.PARAMS: dict`; `models.prepare(train, others) -> list[str]`; `models.SingleLGBM(cfg)` com `.fit(train, cols, run=None, valid=None) -> self`, `.predict(df) -> np.ndarray`, `.best_iter: int | None`; `models.MODELS: dict[str, type]`; `experiment.rmse(y, p) -> float`; `experiment.metrics(df, pred) -> dict`; linha no registro com `metricas.completo`, `best_iter`, `previsoes` (`runs/<id>.parquet` com colunas `MVT_ID_mvt, dia, y_true, pred`).

- [ ] **Step 1: `src/models.py`**

```python
"""Modelos com interface única.

    cols = prepare(train, [outros...])
    modelo = MODELS[nome](cfg).fit(train, cols, run=run, valid=holdout)
    pred = modelo.predict(df)
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

import features as F
from cache import TRUTH

PARAMS = dict(
    objective="regression",  # L2 no alvo bruto, alinhado ao RMSE
    metric="rmse",
    learning_rate=0.05,
    num_leaves=255,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    cat_smooth=20,
    max_cat_to_onehot=8,
    num_threads=12,  # 6 núcleos físicos × 2; 24 threads é 2–4× mais lento (benchmark 24/09)
    verbose=-1,
)


def prepare(train: pd.DataFrame, others: list[pd.DataFrame]) -> list[str]:
    """Referência P10 (só do treino) e o mesmo vocabulário de categorias em todos."""
    ref = F.fit_reference(train)
    for df in (train, *others):
        df["ref_p10"] = F.apply_reference(df, ref)
    F.as_categories([train, *others])
    return F.feature_columns(train)


class SingleLGBM:
    """Um LightGBM L2 no alvo bruto (modelo da v2)."""

    def __init__(self, cfg: dict) -> None:
        self.rounds = int(cfg.get("rounds", 400))
        self.best_iter: int | None = None

    def fit(self, train, cols, run=None, valid=None) -> "SingleLGBM":
        self.cols = cols
        data = lgb.Dataset(train[cols], train[F.TARGET])
        callbacks = [run.lgb_callback(self.rounds)] if run else []
        valid_sets, curve = None, {}
        if valid is not None:
            valid_sets = [lgb.Dataset(valid[cols], valid[TRUTH], reference=data)]
            callbacks.append(lgb.record_evaluation(curve))
        self.model = lgb.train(
            PARAMS, data, self.rounds, valid_sets=valid_sets,
            valid_names=["holdout"] if valid_sets else None,
            callbacks=callbacks,
        )
        if curve:
            self.best_iter = int(np.argmin(curve["holdout"]["rmse"])) + 1
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.cols])


MODELS = {"single": SingleLGBM}
```

- [ ] **Step 2: `src/experiment.py`**

```python
"""Experimento na simulação calibrada do ranking.

    bin/run src/experiment.py <nome> --model single [--rounds 400] [--nota "texto"]
    bin/run src/experiment.py <nome> --model two_stage [--cls-rounds 400] [--reg-rounds 400]

Treina em train2025 (10 meses), prevê holdout2025 (jan+jul/2025 montados como o
ranking) e registra RMSE completo, sem outliers, por grupo e por aeroporto.
Previsões em runs/<id>.parquet; resumo em experiments.jsonl.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import features as F
from cache import TRUTH, load_split
from models import MODELS, prepare
from runlog import ROOT, Run

RUNS = ROOT / "runs"
MAX_NORMAL_S = 3 * 3600


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    y = df[TRUTH].to_numpy(float)
    pred = np.asarray(pred, float)
    nm = df["FLIGHT_ID_mvt"].isna().to_numpy()
    normal = (y > 0) & (y < MAX_NORMAL_S)
    tail = y > 3600
    m = {
        "completo": rmse(y, pred),
        "sem_outliers": rmse(y[normal], pred[normal]),
        "nm_presente": rmse(y[~nm], pred[~nm]),
        "nm_ausente": rmse(y[nm], pred[nm]),
        "y_gt_1h": rmse(y[tail], pred[tail]),
    }
    m = {k: round(v, 2) for k, v in m.items()}
    groups = df.groupby(F.AIRPORT, observed=True).indices
    m["por_aeroporto"] = {str(a): round(rmse(y[i], pred[i]), 1) for a, i in groups.items()}
    return m


def config(a: argparse.Namespace) -> dict:
    if a.model == "single":
        return {"model": a.model, "rounds": a.rounds}
    return {"model": a.model, "cls_rounds": a.cls_rounds, "reg_rounds": a.reg_rounds}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--model", choices=sorted(MODELS), default="single")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--cls-rounds", type=int, default=400)
    ap.add_argument("--reg-rounds", type=int, default=400)
    ap.add_argument("--nota", default="")
    a = ap.parse_args()
    cfg = config(a)

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        with run.phase("dados", 0.15):
            train, hold = load_split("train2025"), load_split("holdout2025")
            cols = prepare(train, [hold])
            run.log(f"treino {len(train):,} · holdout {len(hold):,} · {len(cols)} features")
        with run.phase("treino", 0.75):
            model = MODELS[a.model](cfg).fit(train, cols, run=run, valid=hold)
        with run.phase("métricas", 0.10):
            pred = model.predict(hold)
            m = metrics(hold, pred)
            RUNS.mkdir(exist_ok=True)
            path = RUNS / f"{run.id}.parquet"
            pd.DataFrame({
                F.ID: hold[F.ID].to_numpy(),
                "dia": hold["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy(),
                TRUTH: hold[TRUTH].to_numpy(),
                "pred": pred,
            }).to_parquet(path, index=False)
        run.metric(**m)
        run.set(best_iter=model.best_iter, previsoes=str(path.relative_to(ROOT)))
        run.log(" · ".join(f"{k} {v}" for k, v in m.items() if k != "por_aeroporto"))
        worst = sorted(m["por_aeroporto"].items(), key=lambda kv: -kv[1])
        run.log("por aeroporto: " + ", ".join(f"{k} {v}" for k, v in worst))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Apagar o script antigo**

Run: `git rm sim_ranking.py`

- [ ] **Step 4: Linha de base (supervisionado, nome `prc-exp`)**

Run: `bin/run src/experiment.py base_v2 --model single --rounds 400 --nota "modelo da v2 na base nova"`
Expected:
- progresso a cada 50 rodadas no formato `[treino  45% | total  49%] rodada 150/400 · 230 ms/r · ETA 58s holdout-rmse=... · RAM 3.1 GB · CPU 950%`;
- `completo` entre 450 e 465 (v2 na simulação: 460,4; a v2 tinha 1500 rodadas, aqui 400);
- tempo total ≤ 3 min; pico de RAM ≤ 4 GB;
- `best_iter` perto de 325; linha nova em `experiments.jsonl`; `runs/<id>.parquet` com 344 k linhas.

Se `completo` sair de 450–465: parar e investigar (cache/holdout diferente do `sim_ranking.py`) antes de seguir.

- [ ] **Step 5: Commit**

```bash
git add src/models.py src/experiment.py experiments.jsonl
git commit -m "experiment: simulação calibrada com métricas por grupo; linha de base registrada"
git push
```

---

### Task 5: `compare.py` — comparação pareada e campeão

**Files:**
- Create: `src/compare.py`
- Test: `tests/test_compare.py`
- Create (gerado): `champion.json`

**Interfaces:**
- Consumes: `runlog.REGISTRY`, `runlog.ROOT`, `cache.TRUTH`, `features.ID`.
- Produces: `CHAMPION: Path`; `MIN_GAIN_S = 10.0`; `paired_bootstrap(y, base, new, groups, n=1000, seed=0) -> {"ganho", "ic_baixo", "ic_alto"}` (ganho = RMSE(base) − RMSE(novo)); `is_better(res, min_gain=MIN_GAIN_S) -> bool`; `registry_entry(run_id) -> dict`; `promote(rec)`; `champion.json` = `{"id", "config", "best_iter", "rmse_simulacao"}`.

- [ ] **Step 1: Testes que falham**

`tests/test_compare.py`:

```python
import numpy as np

from compare import is_better, paired_bootstrap


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
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: FAIL (`No module named 'compare'`).

- [ ] **Step 3: Implementar `src/compare.py`**

```python
"""Comparação pareada entre dois experimentos (bootstrap por dia).

    bin/run src/compare.py <id_novo> [<id_base>] [--promover]

Sem <id_base>, compara com o campeão (champion.json). Veredito "MELHOR" exige
ganho ≥ 10 s na simulação e IC 95% acima de zero. --promover grava o novo
campeão quando o veredito é "MELHOR" ou quando ainda não há campeão.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from cache import TRUTH
from features import ID
from runlog import REGISTRY, ROOT

CHAMPION = ROOT / "champion.json"
MIN_GAIN_S = 10.0


def paired_bootstrap(y, base, new, groups, n: int = 1000, seed: int = 0) -> dict:
    """Ganho = RMSE(base) − RMSE(novo); reamostra dias inteiros com reposição."""
    y, base, new = (np.asarray(v, float) for v in (y, base, new))
    codes, _ = pd.factorize(np.asarray(groups))
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    sse_b = np.bincount(codes, (y - base) ** 2, minlength=k)
    sse_n = np.bincount(codes, (y - new) ** 2, minlength=k)
    gain = np.sqrt(sse_b.sum() / cnt.sum()) - np.sqrt(sse_n.sum() / cnt.sum())
    draws = np.random.default_rng(seed).integers(0, k, size=(n, k))
    c = cnt[draws].sum(1)
    boot = np.sqrt(sse_b[draws].sum(1) / c) - np.sqrt(sse_n[draws].sum(1) / c)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ganho": float(gain), "ic_baixo": float(lo), "ic_alto": float(hi)}


def is_better(res: dict, min_gain: float = MIN_GAIN_S) -> bool:
    return res["ganho"] >= min_gain and res["ic_baixo"] > 0


def registry_entry(run_id: str) -> dict:
    for line in REGISTRY.read_text().splitlines():
        rec = json.loads(line)
        if rec["id"] == run_id:
            return rec
    raise SystemExit(f"{run_id} não está em {REGISTRY.name}")


def promote(rec: dict) -> None:
    champ = {
        "id": rec["id"], "config": rec["config"], "best_iter": rec.get("best_iter"),
        "rmse_simulacao": rec["metricas"]["completo"],
    }
    CHAMPION.write_text(json.dumps(champ, indent=2, ensure_ascii=False) + "\n")
    print(f"campeão agora: {rec['id']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("novo")
    ap.add_argument("base", nargs="?")
    ap.add_argument("--promover", action="store_true")
    a = ap.parse_args()

    new = registry_entry(a.novo)
    if a.base:
        base = registry_entry(a.base)
    elif CHAMPION.exists():
        base = registry_entry(json.loads(CHAMPION.read_text())["id"])
    else:
        print("ainda não há campeão")
        if a.promover:
            promote(new)
        return

    pb = pd.read_parquet(ROOT / base["previsoes"])
    pn = pd.read_parquet(ROOT / new["previsoes"])
    m = pb.merge(pn[[ID, "pred"]], on=ID, suffixes=("_base", "_novo"))
    if not len(m) == len(pb) == len(pn):
        raise SystemExit("as previsões não cobrem o mesmo holdout; refaça o experimento base")
    res = paired_bootstrap(m[TRUTH], m["pred_base"], m["pred_novo"], m["dia"])
    better = is_better(res)
    print(f"{base['id']} → {new['id']}")
    print(f"  RMSE simulação {base['metricas']['completo']} → {new['metricas']['completo']}")
    print(f"  ganho {res['ganho']:.1f} s (IC 95% {res['ic_baixo']:.1f} a {res['ic_alto']:.1f})")
    print(f"  veredito: {'MELHOR' if better else 'não comprovado'} (exige ≥ {MIN_GAIN_S:.0f} s e IC > 0)")
    if a.promover and better:
        promote(new)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: `2 passed`

- [ ] **Step 5: Promover a linha de base e conferir**

Run: `bin/run src/compare.py <id_base_v2> --promover` (id da Task 4, em `experiments.jsonl`)
Expected: `ainda não há campeão` e `campeão agora: <id>`; `champion.json` criado.

Run: `bin/run src/compare.py <id_base_v2>`
Expected: ganho 0,0 s, IC 0,0 a 0,0, `não comprovado`.

- [ ] **Step 6: Commit**

```bash
git add src/compare.py tests/test_compare.py champion.json
git commit -m "compare: bootstrap pareado por dia, veredito e campeão"
git push
```

---

### Task 6: `train.py submit` a partir do campeão

**Files:**
- Modify (reescrever): `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `compare.CHAMPION`, `models.MODELS`, `models.prepare`, `cache.load_split`, `cache.DATA`, `runlog.Run`, `runlog.ROOT`.
- Produces: `ROUNDS_SCALE = 1.2`; `final_config(champ) -> dict`; `build_submission(template, ids, pred) -> DataFrame`; `leaky_columns(train, ranking, cols) -> list[str]`; CLI `bin/run src/train.py submit N` → `submissions/<TEAM>_vN.parquet` (não envia).

- [ ] **Step 1: Testes que falham**

`tests/test_train.py`:

```python
import numpy as np
import pandas as pd

import features as F
from train import build_submission, final_config


def test_submissao_segue_a_ordem_do_template_sem_nulos():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    out = build_submission(template, np.array([3.0, 1.0]), np.array([30.0, 10.0]))
    assert out[F.ID].tolist() == [1.0, 2.0, 3.0]
    assert out[F.TARGET].tolist() == [10.0, 20.0, 30.0]  # ID 2 sem previsão: mediana


def test_rodadas_finais_escalam_o_melhor_ponto():
    single = final_config({"config": {"model": "single", "rounds": 400}, "best_iter": 325})
    assert single["rounds"] == 390
    two = final_config({"config": {"model": "two_stage", "cls_rounds": 400, "reg_rounds": 500},
                        "best_iter": None})
    assert (two["cls_rounds"], two["reg_rounds"]) == (480, 600)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: FAIL (`cannot import name 'build_submission'`).

- [ ] **Step 3: Reescrever `src/train.py`**

```python
"""Versão final para envio, a partir do campeão (champion.json).

    bin/run src/train.py submit N    # gera submissions/<TEAM>_vN.parquet (NÃO envia)

Envio separado, só depois de aprovado: .venv/bin/python src/s3.py submit <arquivo>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import features as F
from cache import DATA, load_split
from compare import CHAMPION
from models import MODELS, prepare
from runlog import ROOT, Run

OUT = ROOT / "submissions"
ROUNDS_SCALE = 1.2  # full2025 tem 2,085 M linhas contra 1,741 M do train2025


def leaky_columns(train: pd.DataFrame, ranking: pd.DataFrame, cols: list[str]) -> list[str]:
    """Colunas preenchidas no treino mas apagadas no ranking: o modelo não pode usá-las."""
    return [
        c for c in cols
        if c in train and c in ranking
        and train[c].isna().mean() < 0.5 and ranking[c].isna().mean() > 0.95
    ]


def final_config(champ: dict) -> dict:
    cfg = dict(champ["config"])
    if champ.get("best_iter"):
        cfg["rounds"] = champ["best_iter"]
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in cfg:
            cfg[key] = round(cfg[key] * ROUNDS_SCALE)
    return cfg


def build_submission(template: pd.DataFrame, ids, pred) -> pd.DataFrame:
    by_id = pd.Series(np.asarray(pred, float), index=np.asarray(ids))
    out = template[[F.ID]].copy()
    out[F.TARGET] = out[F.ID].map(by_id)
    missing = out[F.TARGET].isna()
    if missing.any():
        # Decolagem sem MVT_TIME no ranking não gera features: usa a mediana das previsões.
        out.loc[missing, F.TARGET] = float(np.median(pred))
    return out


def submit(version: int) -> None:
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME") or sys.exit("Falta TEAM_NAME no .env")
    champ = json.loads(CHAMPION.read_text())
    cfg = final_config(champ)

    with Run(f"submit_v{version}", {**cfg, "campeao": champ["id"]}) as run:
        with run.phase("dados", 0.15):
            full, rk = load_split("full2025"), load_split("ranking2026")
            template = pd.read_parquet(DATA / "submitting.parquet")
            cols = prepare(full, [rk])
            drop = leaky_columns(full, rk, cols)
            cols = [c for c in cols if c not in drop]
            run.log(f"treino {len(full):,} · ranking {len(rk):,} · ignoradas: {drop or 'nenhuma'}")
        with run.phase("treino", 0.75):
            model = MODELS[cfg["model"]](cfg).fit(full, cols, run=run)
        with run.phase("arquivo", 0.10):
            out = build_submission(template, rk[F.ID], model.predict(rk))
            OUT.mkdir(exist_ok=True)
            path = OUT / f"{team}_v{version}.parquet"
            out.to_parquet(path, index=False)
            run.set(arquivo=str(path.relative_to(ROOT)))
            run.log(
                f"gerado {path.name}: {len(out):,} linhas, mediana {out[F.TARGET].median():.0f} s. "
                "NÃO enviado."
            )


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["submit", n] if n.isdigit():
            submit(int(n))
        case _:
            sys.exit(__doc__)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q`
Expected: todos passam (6).

- [ ] **Step 5: Ensaio sem envio (supervisionado)**

Run: `bin/run src/train.py submit 0`
Expected: `submissions/outgoing-boat_v0.parquet` com 344.841 linhas, sem nulos, mediana ~950 s; total ≤ 4 min.
Depois: `rm submissions/outgoing-boat_v0.parquet` (ensaio; nunca enviar).

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_train.py experiments.jsonl
git commit -m "train: versão final a partir do campeão, rodadas escaladas, sem envio"
git push
```

---

### Task 7: Dois estágios + `nm_missing` (itens 1–2 da parte 2)

**Files:**
- Modify: `src/features.py` (`build`, `feature_columns`)
- Modify: `src/models.py` (+ `copied_from_sched`, `combine`, `TwoStage`, registro em `MODELS`)
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `models.PARAMS`, `run.lgb_callback(rounds, label, start=, span=)`.
- Produces: feature `nm_missing` (int8); `models.SCHED_GAP = "to_takeoff_from_SCHED_TIME_UTC_mvt"`; `models.combine(p, ms, reg) -> np.ndarray`; `models.TwoStage(cfg)` (`cls_rounds`, `reg_rounds`; `best_iter = None`); `MODELS["two_stage"]`.

- [ ] **Step 1: Teste que falha**

`tests/test_models.py`:

```python
import numpy as np

from models import combine


def test_combina_pela_esperanca_da_mistura():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    ms = np.array([5000.0, 5000.0, 5000.0, np.nan])  # sem SCHED: só o regressor
    reg = np.array([900.0, 900.0, 900.0, 900.0])
    assert combine(p, ms, reg).tolist() == [900.0, 5000.0, 2950.0, 900.0]


def test_previsao_nunca_negativa():
    assert combine(np.array([0.0]), np.array([0.0]), np.array([-50.0])).tolist() == [0.0]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `.venv/bin/python -m pytest tests/test_models.py -q`
Expected: FAIL (`cannot import name 'combine'`).

- [ ] **Step 3: `nm_missing` em `src/features.py`**

Em `build`, logo depois de `dep["dow"] = t.dt.dayofweek`:

```python
    # Sem registro no Network Manager: 1% das DEP e 42% do erro quadrático (forense 24/09).
    dep["nm_missing"] = dep["FLIGHT_ID_mvt"].isna().astype("int8")
```

Em `feature_columns`, trocar o `return` por:

```python
    return [*CATEGORICAL, "hour", "dow", "nm_missing", "ref_p10", *derived]
```

- [ ] **Step 4: `TwoStage` em `src/models.py`**

Acrescentar antes de `MODELS` e trocar o `MODELS`:

```python
SCHED_GAP = "to_takeoff_from_SCHED_TIME_UTC_mvt"  # MVT − SCHED, em segundos
COPY_TOL_S = 60


def copied_from_sched(df: pd.DataFrame) -> pd.Series:
    """Rótulo do estágio 1: BLOCK oficial a ≤ 60 s do horário programado."""
    gap = (df["BLOCK_TIME_UTC_mvt"] - df["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return gap.abs() <= COPY_TOL_S


def combine(p, ms, reg) -> np.ndarray:
    """Esperança da mistura p·(MVT−SCHED) + (1−p)·regressor, com piso 0.

    Sem SCHED (ms nulo) usa só o regressor. Nunca argmax: errar a classe custa horas².
    """
    p, ms, reg = (np.asarray(v, float) for v in (p, ms, reg))
    mix = np.where(np.isnan(ms), reg, p * ms + (1 - p) * reg)
    return np.clip(mix, 0, None)


class TwoStage:
    """Classificador 'BLOCK copiado do SCHED' + regressor L2 nos voos normais."""

    def __init__(self, cfg: dict) -> None:
        self.cls_rounds = int(cfg.get("cls_rounds", 400))
        self.reg_rounds = int(cfg.get("reg_rounds", 400))
        self.best_iter: int | None = None

    def fit(self, train, cols, run=None, valid=None) -> "TwoStage":
        self.cols = cols
        copied = copied_from_sched(train)
        cls_params = {**PARAMS, "objective": "binary", "metric": "binary_logloss"}

        def cb(rounds: int, label: str, start: float) -> list:
            return [run.lgb_callback(rounds, label, start=start, span=0.5)] if run else []

        self.cls = lgb.train(
            cls_params, lgb.Dataset(train[cols], copied.astype("int8")), self.cls_rounds,
            callbacks=cb(self.cls_rounds, "classificador", 0.0),
        )
        normal = train[~copied]
        self.reg = lgb.train(
            PARAMS, lgb.Dataset(normal[cols], normal[F.TARGET]), self.reg_rounds,
            callbacks=cb(self.reg_rounds, "regressor", 0.5),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols]
        return combine(self.cls.predict(x), df[SCHED_GAP], self.reg.predict(x))


MODELS = {"single": SingleLGBM, "two_stage": TwoStage}
```

- [ ] **Step 5: Rodar e ver passar**

Run: `.venv/bin/python -m pytest -q`
Expected: todos passam (8).

- [ ] **Step 6: Refazer o cache e a base com `nm_missing`**

`features.py` mudou → o cache é refeito sozinho. Rodar sozinho, supervisionado:

Run: `bin/run src/cache.py`
Run: `bin/run src/experiment.py base_nm --model single --nota "v2 + nm_missing"`
Run: `bin/run src/compare.py <id_base_nm>`
Expected: registrado; veredito informativo (nm_missing sozinho no modelo único pode não passar de 10 s).

- [ ] **Step 7: Experimento dois estágios**

Run: `bin/run src/experiment.py dois_estagios --model two_stage --cls-rounds 400 --reg-rounds 400 --nota "item 1-2: P(BLOCK≈SCHED) + regressor"`
Expected: `completo` perto de 391 (forense: 391,1; tolerância ±15); `nm_presente` < 260; progresso mostra `classificador` 0–50% e `regressor` 50–100% da fase de treino.

Run: `bin/run src/compare.py <id_dois_estagios> --promover`
Expected: ganho ≈ 60–70 s, IC acima de 0, `MELHOR`, campeão atualizado.

Se não for `MELHOR`: não promover; registrar no roadmap com os números e parar para decidir com o usuário.

- [ ] **Step 8: Gerar a v3 e pedir aprovação (sem enviar)**

Run: `bin/run src/train.py submit 3`
Expected: `submissions/outgoing-boat_v3.parquet`, 344.841 linhas, sem nulos.

Mostrar ao usuário: simulação (base → dois estágios), ganho ± IC, projeção oficial = simulação × 0,836, e perguntar se envia. **Só com ok:**

```bash
.venv/bin/python src/s3.py submit submissions/outgoing-boat_v3.parquet
```

Depois ler `outgoing-boat_v3.parquet_result.json` no bucket e registrar a nota oficial e a nova relação oficial/simulação.

- [ ] **Step 9: Commit**

```bash
git add src/features.py src/models.py tests/test_models.py experiments.jsonl champion.json
git commit -m "Dois estágios (P(BLOCK≈SCHED) + regressor) e nm_missing"
git push
```

---

### Task 8: Documentação

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md` (só se algo medido contradisser o spec)

- [ ] **Step 1: README**

Atualizar:
- **Dados → "Vazamento"**: trocar a frase "O off-block oficial costuma ser *cópia* de um desses horários" por: "O BLOCK oficial tem resolução de segundo e AOBT_3/EOBT/SCHED de minuto: a proximidade de 38% com AOBT_3 não é cópia. Cópia existe só na cauda, e do SCHED: 84,7% dos y > 3 h têm |BLOCK − SCHED| ≤ 60 s (docs/research/2026-09-24-forense-dados.md)."
- **Modelo atual**: dois estágios (se promovido), `nm_missing`, `num_threads=12`, rodadas pelo `best_iter` × 1,2.
- **Uso**: substituir o bloco por:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                              # chaves e TEAM_NAME
.venv/bin/python src/s3.py download               # dados em data/
bin/run src/cache.py                              # features em cache (uma vez, sozinho)
bin/run src/experiment.py <nome> --model two_stage
bin/run src/compare.py <id> --promover            # decide contra o campeão
bin/run src/train.py submit N                     # gera a vN (não envia)
.venv/bin/python src/s3.py submit submissions/<TEAM>_vN.parquet   # só após aprovação
.venv/bin/python -m pytest -q
```

- **Estrutura**: listar `bin/run`, `src/runlog.py`, `src/cache.py`, `src/models.py`, `src/experiment.py`, `src/compare.py`, `tests/`, `experiments.jsonl`, `champion.json`, `docs/`; remover `sim_ranking.py`.
- **Roadmap**: marcar itens 0–2 da parte 2 com o resultado medido; próximo plano = itens 3–4.
- **Submissões**: linha da v3 (se enviada) com simulação e nota oficial.

- [ ] **Step 2: Commit**

```bash
git add README.md docs/
git commit -m "README: uso da base nova, resultados e roadmap"
git push
```

---

## Depois deste plano

Plano seguinte (escrito com os números medidos aqui): itens 3–4 da parte 2
(modelo linear por aeroporto para `nm_missing`; célula LIRF ∧ nm_missing ∧ ms > 2 h),
depois 5–9. Cada um entra como experimento → `compare.py` → aprovação de envio.
