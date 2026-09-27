# Plano 6 — média de seeds na base

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Medir se a média de N seeds do LightGBM na base melhora a simulação e, se sim, montar a v8 = v7 (`stack_cf`) + seeds.

**Architecture:** Chave nova `seeds` na config da base (padrão 1 = comportamento atual). Uma função `build_model(cfg)` em `models.py` devolve o modelo de hoje com `seeds == 1` ou um `SeedAvg` que treina N cópias com seeds `seed, seed+1, …, seed+N−1` e devolve a média das previsões. Todo lugar que hoje faz `MODELS[...](cfg)` passa a chamar `build_model(cfg)`; `crossfit`, `stack --crossfit` e `train.py` herdam as seeds sem mudança própria.

**Tech Stack:** Python 3.14, pandas 3, numpy, LightGBM 4.7, pytest (`.venv/bin/python -m pytest -q`).

**Spec:** `docs/superpowers/specs/2026-09-24-pipeline-melhoria-continua-design.md`, parte 2 item 7 ("3–5 seeds LightGBM") e `saltos.json` (`seeds`: PRC 2024 1º lugar −3,0 % com 10 seeds, `docs/research/2026-09-26-praticas-ml.md`).

## Global Constraints

- `seeds` ausente ou 1: previsões **idênticas** às de hoje (mesma seed, mesmo caminho de código).
- `SeedAvg`: média aritmética das previsões finais de cada cópia (cada cópia é o modelo completo, inclusive retas NM e combinação por esperança); piso 0 já vem de cada cópia.
- `best_iter` do `SeedAvg`: o da primeira cópia (as rodadas são iguais em todas).
- Progresso: cada cópia loga `seed k/N`; sem exigência de % correto por cópia.
- Config registrada inclui `seeds` só quando > 1 (as configs antigas continuam iguais, `compare.may_promote` segue funcionando).
- Processo pesado só por `bin/run`, um por vez; subagentes só rodam pytest sintético.
- Português; commit + push em `main` por tarefa. Nada é enviado ao placar sem ok do usuário.

---

### Task 1: `build_model` e `SeedAvg`

**Files:**
- Modify: `src/models.py`, `src/experiment.py` (flag `--seeds N`, `config`), `src/crossfit.py:48`, `src/train.py:93` (e qualquer outro `MODELS[...](cfg)`), `src/stack.py` (flag `--seeds N` no `--crossfit`: grava `base_config` com `seeds`)
- Test: `tests/test_models.py`, `tests/test_experiment.py`

**Interfaces:**
- Produces: `models.build_model(cfg: dict)` → objeto com `.fit(train, cols, run=None, valid=None) -> self`, `.predict(df) -> ndarray`, `.best_iter`.

```python
class SeedAvg:
    """Média de N cópias do mesmo modelo com seeds consecutivas."""

    def __init__(self, cfg: dict) -> None:
        n, s0 = int(cfg["seeds"]), int(cfg.get("seed", 0))
        self.cfgs = [{**cfg, "seed": s0 + k, "seeds": 1} for k in range(n)]
        self.best_iter: int | None = None

    def fit(self, train, cols, run=None, valid=None) -> "SeedAvg":
        self.models = []
        for k, c in enumerate(self.cfgs):
            if run:
                run.log(f"seed {k + 1}/{len(self.cfgs)} (seed={c['seed']})")
            self.models.append(MODELS[c["model"]](c).fit(train, cols, run=run, valid=valid))
        self.best_iter = self.models[0].best_iter
        return self

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.models], axis=0)


def build_model(cfg: dict):
    return SeedAvg(cfg) if int(cfg.get("seeds", 1)) > 1 else MODELS[cfg["model"]](cfg)
```

- [x] Teste: com um modelo falso registrado em `MODELS` (monkeypatch) que prevê a própria seed, `build_model({"model": falso, "seed": 3, "seeds": 4})` treina seeds 3,4,5,6 e prevê 4,5 em todas as linhas.
- [x] Teste: `build_model` com `seeds` ausente e com `seeds: 1` devolve uma instância da classe de `MODELS` (sem `SeedAvg`).
- [x] Teste: `experiment.config` com `--seeds 1` não inclui a chave `seeds`; com `--seeds 5` inclui `seeds: 5`.
- [x] Trocar todo `MODELS[...](cfg)` fora de `models.py` por `build_model(cfg)`; `pytest -q` verde.
- [x] Commit `models: média de seeds (SeedAvg, --seeds)`; push.

**Acceptance:** testes passam; `grep -n "MODELS\[" src/` só acha `models.py`.

### Task 2 (controlador): medir e decidir

- [x] Base: `bin/run src/experiment.py seeds5 --model two_stage_nm --nm-min-ms 21600 --seed 0 --seeds 5` (~25 min) → `compare.py <id>` contra a v6 (323,50). Portão barato: seguir só se `normais_nm` melhorar > 1 s (ruído entre seeds ~0,5 s) ou o completo > 2 s.
- [x] Se passar: `bin/run src/stack.py v8_cf --crossfit --base <id seeds5> --seeds 5` (~1 h 20) → `compare.py` contra a v6 e contra `20260927-112207-v7_cf`.
- [x] **Parar e mostrar ao usuário** números, IC e custo do envio (6 blocos × 5 seeds + base final ≈ 2–3 h).
- [x] Docs: README (Modelo atual, Uso, Roadmap), CONTEXTO "Retomar", `saltos.json` (`seeds` com ganho medido), caixas deste plano; commit + push.

## Ordem

1 → 2. Prazo 11/10/2026 23:59:59 CET.
