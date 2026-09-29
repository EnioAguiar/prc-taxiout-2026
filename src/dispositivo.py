"""Onde o LightGBM treina: CPU ou GPU (OpenCL do driver NVIDIA, sem instalar nada).

    PRC_DEVICE=gpu bin/run src/stack.py ...   # GPU
    PRC_DEVICE=cpu bin/run src/stack.py ...   # CPU

O wheel do LightGBM no PyPI já traz a versão OpenCL (`device_type="gpu"`); o runtime
OpenCL vem no driver da placa (`/etc/OpenCL/vendors/nvidia.icd`). Pesquisa e medição:
`docs/pesquisa/2026-09-28-gpu-em-ml.md` (raiz do repositório).

Na GPU, `deterministic` e `force_row_wise` saem: os dois só valem na CPU, e o resultado
de duas rodadas iguais pode variar um pouco (trate como ruído). O CatBoost do corretor
decide sozinho (GPU quando há placa, `stack.fit_catboost`).
"""
from __future__ import annotations

import os

DEVICE = os.environ.get("PRC_DEVICE", "cpu").strip().lower()
if DEVICE not in {"cpu", "gpu"}:
    raise SystemExit(f"PRC_DEVICE deve ser cpu ou gpu (recebido: {DEVICE!r})")

SO_CPU = ("deterministic", "force_row_wise")


def lgb_params(params: dict) -> dict:
    """Parâmetros do LightGBM para o dispositivo escolhido; na CPU, os mesmos de sempre."""
    if DEVICE == "cpu":
        return params
    return {**{k: v for k, v in params.items() if k not in SO_CPU}, "device_type": "gpu"}
