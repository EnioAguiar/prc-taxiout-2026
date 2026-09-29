"""Mapa do aeroporto (`--mapa`): distância de táxi do stand até a cabeceira pelo grafo de taxiways.

Fonte: `apt.dat` do X-Plane Scenery Gateway (GPL v2+; https://gateway.x-plane.com), a cena
recomendada de cada aeroporto, baixada pela API pública para `data/mapa/<ICAO>.dat`:

    bin/run src/mapa.py            # baixa o que falta e grava data/mapa/distancias.parquet

Do `apt.dat` usamos três tipos de linha: `1300` (posição e nome de cada stand), `1201`/`1202`
(nós e arestas da rede de táxi) e `100` (as duas cabeceiras de cada pista). O caminho mais curto
(Dijkstra, arestas com o comprimento em metros) vai do nó mais perto do stand até o nó mais perto
da cabeceira, só na maior componente conectada da rede.

O nome do stand do dado (`STAND_mvt`) casa com o do mapa sem espaços e símbolos, sem o sufixo
"-(C)" do LTFM e, se não houver igual, pelo sufixo numérico ("Ramp 175" ↔ "175"). Conferido
contra a mediana do primeiro ponto ADS-B parado de cada stand (29/09): erro mediano 14–68 m em
EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LIRF e LSZH; LFPG e LTFM quase sem ADS-B no pátio para conferir.

Colunas (NaN quando o stand não está no mapa):
- `map_dist`: metros do stand à cabeceira da pista do voo (`RUNWAY_mvt`);
- `map_dmin`: metros do stand à cabeceira mais próxima (independe da pista);
- `map_extra`: `map_dist − map_dmin`, o quanto a pista escolhida alonga o táxi.
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import features as F
from cache import DATA

DIR = DATA / "mapa"
TABELA = DIR / "distancias.parquet"
API = "https://gateway.x-plane.com/apiv1"
COLS = ["map_dist", "map_dmin", "map_extra"]


def norm(s) -> str:
    return re.sub(r"[^A-Z0-9]", "", re.sub(r"-\([A-Z]\)$", "", str(s).upper().strip()))


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "prc-taxiout-2026"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def baixar(apt: str) -> Path:
    """`apt.dat` da cena recomendada do aeroporto (uma vez; depois fica em `data/mapa/`)."""
    f = DIR / f"{apt}.dat"
    if not f.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        sid = _get(f"{API}/airport/{apt}")["airport"]["recommendedSceneryId"]
        blob = _get(f"{API}/scenery/{sid}")["scenery"]["masterZipBlob"]
        zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(blob)))
        f.write_bytes(zf.read(next(n for n in zf.namelist() if n.endswith(".dat"))))
    return f


def _xy(lat, lon, lat0: float) -> np.ndarray:
    """Metros numa projeção local (equirretangular em volta de `lat0`)."""
    lat, lon = np.atleast_1d(lat).astype(float), np.atleast_1d(lon).astype(float)
    return np.c_[np.radians(lon) * 6371e3 * np.cos(np.radians(lat0)), np.radians(lat) * 6371e3]


def distancias_aeroporto(apt: str) -> pd.DataFrame:
    """Uma linha por (stand do mapa, cabeceira): `stand`, `pista`, `dist` em metros."""
    nos, arestas, stands, cab = {}, [], {}, {}
    for linha in baixar(apt).read_text(errors="ignore").splitlines():
        p = linha.split()
        if not p:
            continue
        if p[0] == "1201":
            nos[int(p[4])] = (float(p[1]), float(p[2]))
        elif p[0] == "1202":
            arestas.append((int(p[1]), int(p[2])))
        elif p[0] == "1300" and len(p) > 6:
            stands[norm(" ".join(p[6:]))] = (float(p[1]), float(p[2]))
        elif p[0] == "100":
            cab[norm(p[8])] = (float(p[9]), float(p[10]))
            cab[norm(p[17])] = (float(p[18]), float(p[19]))
    ids = list(nos)
    pos = {k: i for i, k in enumerate(ids)}
    ll = np.array([nos[k] for k in ids])
    lat0 = float(ll[:, 0].mean())
    P = _xy(ll[:, 0], ll[:, 1], lat0)
    i, j = np.array([(pos[a], pos[b]) for a, b in arestas if a in pos and b in pos]).T
    w = np.linalg.norm(P[i] - P[j], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(len(ids),) * 2).tocsr()
    _, comp = connected_components(G, directed=False)
    fora = comp != np.bincount(comp).argmax()

    def mais_perto(la: float, lo: float) -> int:
        d = np.linalg.norm(P - _xy(la, lo, lat0), axis=1)
        d[fora] = np.inf
        return int(np.argmin(d))

    pistas = list(cab)
    D = dijkstra(G, indices=[mais_perto(*cab[r]) for r in pistas])
    linhas = [(s, r, float(D[k, mais_perto(*c)])) for s, c in stands.items() for k, r in enumerate(pistas)]
    return pd.DataFrame(linhas, columns=["stand", "pista", "dist"]).assign(apt=apt)


def gerar(aeroportos) -> pd.DataFrame:
    t = pd.concat([distancias_aeroporto(a) for a in sorted(aeroportos)], ignore_index=True)
    t.to_parquet(TABELA, index=False)
    return t


@lru_cache(maxsize=1)
def _tabela() -> tuple[dict, dict, dict]:
    """(dist por (apt, stand, pista), dmin por (apt, stand), nome do dado → stand do mapa)."""
    t = pd.read_parquet(TABELA)
    dist = {(a, s, r): d for a, s, r, d in zip(t["apt"], t["stand"], t["pista"], t["dist"])}
    dmin = t.groupby(["apt", "stand"])["dist"].min().to_dict()
    idx: dict = {}
    for a, s in dmin:
        idx[(a, s)] = s
    for a, s in dmin:  # sufixo numérico só onde não há nome igual
        for rx in (r"([A-Z]?\d+[A-Z]?)$", r"(\d+[A-Z]?)$"):
            m = re.search(rx, s)
            if m:
                idx.setdefault((a, m.group(1)), s)
    return dist, dmin, idx


def colunas(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """`map_*` das linhas de `df`, na ordem delas."""
    dist, dmin, idx = _tabela()
    apt = df[F.AIRPORT].astype(str).to_numpy()
    st = [idx.get((a, norm(s))) for a, s in zip(apt, df["STAND_mvt"].to_numpy())]
    rw = [norm(r) for r in df["RUNWAY_mvt"].to_numpy()]
    d = np.array([dist.get((a, s, r), np.nan) if s else np.nan for a, s, r in zip(apt, st, rw)], float)
    m = np.array([dmin.get((a, s), np.nan) if s else np.nan for a, s in zip(apt, st)], float)
    return {"map_dist": d, "map_dmin": m, "map_extra": d - m}


if __name__ == "__main__":
    rk = pd.read_parquet(DATA / "ranking.parquet", columns=["ADEP_mvt", "PHASE_mvt"])
    aeroportos = rk.loc[rk["PHASE_mvt"] == "DEP", "ADEP_mvt"].unique()
    t = gerar(aeroportos)
