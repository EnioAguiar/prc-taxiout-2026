"""Devolver ao sistema a RAM dos quadros grandes já soltos.

`del` só tira a referência: o pedaço volta para o heap do glibc, que o guarda para o
próximo pedido. Como aqui os quadros têm 1–2 GB e o próximo pico vem depois (treinar com
a cópia filtrada, montar a matriz do LightGBM), esse resto fica contado no RSS e no pico
que o systemd-oomd enxerga. `soltar()` coleta os ciclos e pede ao glibc que devolva o
topo livre do heap.

Nada aqui muda cálculo nenhum: só é chamada depois de `del`, onde não há mais leitor.
"""

from __future__ import annotations

import ctypes
import gc

try:  # fora do glibc (musl, macOS) não existe malloc_trim; aí só o gc trabalha
    _trim = ctypes.CDLL("libc.so.6").malloc_trim
    _trim.argtypes = [ctypes.c_size_t]
except (OSError, AttributeError):  # pragma: no cover - depende da libc da máquina
    _trim = None


def soltar() -> None:
    """Coleta o lixo e devolve ao sistema o heap livre (nenhum efeito no resultado)."""
    gc.collect()
    if _trim is not None:
        _trim(0)
