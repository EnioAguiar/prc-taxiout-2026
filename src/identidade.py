"""Identidade dos artefatos derivados de uma base: config, código e dados de entrada.

Dois artefatos caem no mesmo arquivo de cache só quando saem exatamente das mesmas
entradas: a config da base, o código que a produz (`crossfit.py` e o que ele importa de
`src/`) e os parquets de dados. Serve à previsão fora do bloco de 2025
(`train.chave_oof`) e à previsão de 2026 do pseudo-rótulo (`pseudo.chave`).

Este módulo fica de fora do grafo de imports da base de propósito: mexer nele não
invalida os caches que ele nomeia.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from adsb_events import RAIZ as ADSB_RAIZ
from cache import DATA, ROOT


def modulos_da_base(inicio: str = "crossfit") -> list[Path]:
    """Arquivos de `src/` que a previsão da base usa: `inicio` e seus imports locais."""
    src = ROOT / "src"
    vistos: set[str] = set()
    fila = [inicio]
    while fila:
        nome = fila.pop()
        path = src / f"{nome}.py"
        if nome in vistos or not path.exists():
            continue
        vistos.add(nome)
        for no in ast.walk(ast.parse(path.read_text())):
            if isinstance(no, ast.Import):
                fila += [a.name.split(".")[0] for a in no.names]
            elif isinstance(no, ast.ImportFrom) and no.module and not no.level:
                fila.append(no.module.split(".")[0])
    return [src / f"{m}.py" for m in sorted(vistos)]


def arquivos_de_dados() -> list[Path]:
    """Parquets de entrada de qualquer previsão da base (dados do desafio e eventos ADS-B)."""
    return [*sorted(DATA.glob("*.parquet")), *sorted((DATA / "mapa").glob("*.parquet")),
            ADSB_RAIZ / "events.parquet"]


def chave_da_base(cfg_base: dict, marca: str = "") -> str:
    """Chave de cache de um artefato da base; `marca` separa artefatos da mesma base.

    Sem `marca` é a chave histórica da previsão fora do bloco, byte a byte.
    """
    h = hashlib.sha256(json.dumps(cfg_base, sort_keys=True).encode())
    if marca:
        h.update(marca.encode())
    for p in modulos_da_base():
        h.update(p.name.encode() + p.read_bytes())
    for p in arquivos_de_dados():  # caminho resolvido: data/adsb como link para outro disco
        p = p.resolve()           # dá a mesma chave
        st = p.stat() if p.exists() else None
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode() if st else f"{p}:-".encode())
    return h.hexdigest()[:16]
