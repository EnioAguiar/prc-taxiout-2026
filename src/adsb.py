"""Recorte do adsb.lol (globe_history, ODbL) perto dos 10 aeroportos.

Por dia: baixa o tar do GitHub, lê os rastros em fluxo, guarda só os pontos
dentro da caixa de algum aeroporto e abaixo de ALT_MAX_FT (ou no chão) em
`<raiz>/cut/AAAA-MM-DD.parquet` (zstd) e apaga o tar. Retomável: dia com
parquet pronto é pulado.

Uso:
    bin/run src/adsb.py baixar [--raiz DIR] [--dias 2025-01,2025-07,2026-01,2026-07] [--procs 5]
    bin/run src/adsb.py baixar --dia 2025-01-15

Fonte: https://github.com/adsblol/globe_history_2025 (e _2026). Licença ODbL 1.0.
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Onde ficam os recortes e o events.parquet; em outra máquina, defina PRC_ADSB_RAIZ.
RAIZ = Path(os.environ.get("PRC_ADSB_RAIZ", "/mnt/c0399cd8-7cca-4664-884d-e89d4a1e81a2/prc-adsb"))
MESES = ["2025-01", "2025-07", "2026-01", "2026-07"]
ALT_MAX_FT = 3000
BOX_DEG = 0.10  # meia largura em latitude (~11 km); longitude corrigida por cos(lat). Cobre a Polderbaan (EHAM).

AEROPORTOS = {  # ARP (graus)
    "EDDF": (50.0333, 8.5706),
    "EDDM": (48.3538, 11.7861),
    "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639),
    "LEBL": (41.2971, 2.0785),
    "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479),
    "LIRF": (41.8003, 12.2389),
    "LTFM": (41.2753, 28.7519),
    "LSZH": (47.4647, 8.5492),
}
# Cada repositório lista, por dia, a réplica preferida (partes .tar.aa/.ab… ou um .tar único).
PREFERRED = "https://raw.githubusercontent.com/adsblol/globe_history_{ano}/main/PREFERRED_RELEASES.txt"


def caixas() -> np.ndarray:
    """[lat_min, lat_max, lon_min, lon_max] por aeroporto, na ordem de AEROPORTOS."""
    out = []
    for lat, lon in AEROPORTOS.values():
        dlon = BOX_DEG / np.cos(np.radians(lat))
        out.append([lat - BOX_DEG, lat + BOX_DEG, lon - dlon, lon + dlon])
    return np.array(out)


def aeroporto_do_ponto(lat: float, lon: float, cx: np.ndarray) -> int:
    """Índice do aeroporto cuja caixa contém o ponto, ou −1."""
    hit = (cx[:, 0] <= lat) & (lat <= cx[:, 1]) & (cx[:, 2] <= lon) & (lon <= cx[:, 3])
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else -1


COLS = ["icao", "reg", "tipo", "t", "apt", "lat", "lon", "chao", "alt", "gs", "track", "vrate", "fonte", "callsign"]


def _campo(p: list, i: int):
    return p[i] if len(p) > i else None


def recortar_rastro(d: dict, cx: np.ndarray) -> list[tuple]:
    """Pontos do rastro dentro de uma caixa e baixos, na ordem de COLS.

    Ponto do readsb: [dt, lat, lon, alt|"ground", gs, track, flags, vrate, aircraft, fonte, ...].
    """
    t0 = d["timestamp"]
    icao, reg, tipo = d["icao"], d.get("r"), d.get("t")
    cs = None
    rows = []
    for p in d.get("trace", []):
        if isinstance(_campo(p, 8), dict) and p[8].get("flight"):
            cs = p[8]["flight"].strip()
        alt = p[3]
        chao = alt == "ground"
        if not chao and (alt is None or alt > ALT_MAX_FT):
            continue
        a = aeroporto_do_ponto(p[1], p[2], cx)
        if a < 0:
            continue
        rows.append((icao, reg, tipo, t0 + p[0], a, p[1], p[2], chao, None if chao else alt,
                     p[4], _campo(p, 5), _campo(p, 7), _campo(p, 9), cs))
    return rows


def _url_partes(dia: str) -> list[str]:
    """URLs das partes do dia segundo PREFERRED_RELEASES.txt (sem API: evita o limite de 60/h).

    Olha também a lista do ano seguinte: 2025-12-31 está publicado em globe_history_2026.
    """
    tag = f"/v{dia.replace('-', '.')}-planes-readsb-"
    for ano in (int(dia[:4]), int(dia[:4]) + 1):
        try:
            with urllib.request.urlopen(PREFERRED.format(ano=ano), timeout=60) as r:
                linhas = r.read().decode().split()
        except urllib.error.HTTPError:
            continue  # repositório do ano seguinte ainda não existe
        for linha in linhas:
            if tag in linha:
                return sorted(linha.split(","))
    raise RuntimeError(f"{dia}: fora do PREFERRED_RELEASES.txt")


def processar_dia(dia: str, raiz: Path) -> str:
    out = raiz / "cut" / f"{dia}.parquet"
    if out.exists():
        return f"{dia}: já existe"
    tmp = raiz / "tmp" / dia
    tmp.mkdir(parents=True, exist_ok=True)
    t = time.time()
    partes = []
    for i, url in enumerate(_url_partes(dia)):
        dest = tmp / f"p{i:02d}"
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(dest), url], check=True)
        partes.append(str(dest))
    t_dl = time.time() - t
    cx = caixas()
    rows, ruins = [], 0
    cat = subprocess.Popen(["cat", *partes], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and "trace_full_" in m.name):
                    continue
                raw = tf.extractfile(m).read()
                try:
                    d = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                except (OSError, ValueError, EOFError, zlib.error):
                    ruins += 1  # rastro corrompido: pula só ele (o resto do dia vale)
                    continue
                rows.extend(recortar_rastro(d, cx))
    finally:
        cat.stdout.close()
        cat.wait()
    df = pd.DataFrame(rows, columns=COLS)
    df["apt"] = pd.Categorical.from_codes(df["apt"], list(AEROPORTOS))
    for c in ["alt", "gs", "track", "vrate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ["reg", "tipo", "fonte", "callsign"]:
        df[c] = df[c].astype("category")
    part = out.with_suffix(".part")
    df.to_parquet(part, compression="zstd", index=False)
    part.rename(out)
    for p in partes:
        Path(p).unlink()
    tmp.rmdir()
    mb = out.stat().st_size / 1e6
    return (f"{dia}: {len(df):,} pontos, {mb:.1f} MB, {ruins} rastros corrompidos pulados"
            f" · download {t_dl:.0f}s · total {time.time() - t:.0f}s")


def dias_dos_meses(meses: list[str]) -> list[str]:
    out = []
    for m in meses:
        d = date.fromisoformat(m + "-01")
        while d.strftime("%Y-%m") == m:
            out.append(d.isoformat())
            d += timedelta(days=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("acao", choices=["baixar"])
    ap.add_argument("--raiz", type=Path, default=RAIZ)
    ap.add_argument("--dias", default=",".join(MESES), help="meses AAAA-MM separados por vírgula")
    ap.add_argument("--dia", help="um dia só (AAAA-MM-DD)")
    ap.add_argument("--procs", type=int, default=5)
    a = ap.parse_args()
    (a.raiz / "cut").mkdir(parents=True, exist_ok=True)
    dias = [a.dia] if a.dia else dias_dos_meses(a.dias.split(","))
    pend = [d for d in dias if not (a.raiz / "cut" / f"{d}.parquet").exists()]
    print(f"{len(dias)} dias, {len(pend)} pendentes → {a.raiz}", flush=True)
    feitos, erros, t0 = 0, 0, time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        futs = {ex.submit(processar_dia, d, a.raiz): d for d in pend}
        for f in futs:
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001 — um dia ruim não derruba o lote; rodar de novo retoma
                msg = f"{futs[f]}: ERRO {e}"
                erros += 1
            feitos += 1
            eta = (time.time() - t0) / feitos * (len(pend) - feitos)
            print(f"[{feitos}/{len(pend)} · ETA {eta / 3600:.1f} h] {msg}", flush=True)
    # código ≠ 0 faz o serviço (Restart=on-failure) tentar de novo os dias que falharam
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
