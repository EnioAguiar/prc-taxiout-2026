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
import resource
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

from dispositivo import DEVICE

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
            "dispositivo": DEVICE,
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
        self.log(f"== {self.id} (código {self.rec['src_hash']}, commit {self.rec['git_commit']}, "
                 f"LightGBM na {DEVICE.upper()})")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._resources()
        self.rec["total_s"] = round(time.perf_counter() - self._t0, 1)
        kernel_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # KiB → GB
        self.rec["rss_pico_gb"] = round(max(self._rss_peak, kernel_peak), 2)
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
