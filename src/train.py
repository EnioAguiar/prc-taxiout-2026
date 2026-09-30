"""Versão final para envio, a partir do campeão (champion.json).

    bin/run src/train.py submit N [--corrida <id>]   # gera submissions/<TEAM>_vN.parquet (NÃO envia)

Sem `--corrida`, a receita é a da campeã (`champion.json` v2: média dos membros e as
`pos_regras`); com `--corrida <id>`, a da última linha dessa corrida no `experiments.jsonl`,
sozinha e com a pós-regra de Roma (o `champion.json` não é lido nem mexido).

Grava o `src_hash` atual no registro do envio.
Envio separado, só depois de aprovado: .venv/bin/python src/s3.py submit <arquivo>

Campeã `stack_cf`: o corretor treina nas cegas com a previsão da base fora do bloco
e corrige a base final do ranking. A previsão fora do bloco (~50 min) fica guardada em
`data/cache/oof_base/<chave>.parquet`; a chave junta a config da base, o código de que
ela depende (`crossfit.py` e o que ele importa de `src/`) e os arquivos de dados. Envio
que só muda o corretor reaproveita o arquivo: a base é determinística (mesma chave,
mesma previsão, conferido v29 × v30).

Com `externos: true` na config, as cegas e o ranking ganham as colunas `ext_*` de
`src/externos.py`: a taxa de cópia do SCHED por companhia vem dos 12 meses de 2025,
sempre sem o mês da própria linha.

Com `plano13: true`, as cegas e o ranking ganham também as colunas de `src/plano13.py`
(METAR, rotação no stand, consistência NM e a companhia como categoria); o vocabulário
de companhias sai do `full2025` e é o mesmo nos dois quadros.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import campeao
import features as F
import pos_regras
import refcel
import runlog
from adsb_events import RAIZ as ADSB_RAIZ
from cache import CACHE, DATA, TRUTH, load_split
from crossfit import oof_base
from externos import MESES_2025, CopiaCia, colunas_ext, copia_cia_2025
from models import build_model, leaky_columns, prepare
from plano13 import colunas_p13, vocabulario
from runlog import ROOT, Run
from stack import (ROUNDS as ROUNDS_CORRETOR, corrector_frame, fit_corrector,
                   previsao_corrigida, tabelas_celula)

OUT = ROOT / "submissions"
OOF_CACHE = CACHE / "oof_base"
ROUNDS_SCALE = 1.2  # full2025 tem 2,085 M linhas contra 1,741 M do train2025


def modulos_da_base(inicio: str = "crossfit") -> list[Path]:
    """Arquivos de `src/` que a previsão fora do bloco usa: `crossfit.py` e seus imports locais."""
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


def chave_oof(cfg_base: dict) -> str:
    """Identidade da previsão fora do bloco: config da base, código dela e dados de entrada."""
    h = hashlib.sha256(json.dumps(cfg_base, sort_keys=True).encode())
    for p in modulos_da_base():
        h.update(p.name.encode() + p.read_bytes())
    dados = [*sorted(DATA.glob("*.parquet")), *sorted((DATA / "mapa").glob("*.parquet")),
             ADSB_RAIZ / "events.parquet"]
    for p in dados:
        st = p.stat() if p.exists() else None
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode() if st else f"{p}:-".encode())
    return h.hexdigest()[:16]


def escalar_rodadas(cfg: dict) -> dict:
    """Mais linhas no treino final pedem proporcionalmente mais rodadas."""
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in cfg:
            cfg[key] = round(cfg[key] * ROUNDS_SCALE)
    return cfg


def escala_base(base_config: dict) -> dict:
    """Só a base final vê o full2025: rodadas × 1,2; corretor e blocos ficam como medidos."""
    return escalar_rodadas(dict(base_config))


def build_submission(
    template: pd.DataFrame, ids, pred, max_fill_frac: float = 0.001
) -> tuple[pd.DataFrame, int]:
    """Previsões na ordem do template; poucos IDs sem previsão viram a mediana."""
    by_id = pd.Series(np.asarray(pred, float), index=np.asarray(ids))
    out = template[[F.ID]].copy()
    out[F.TARGET] = out[F.ID].map(by_id)
    missing = out[F.TARGET].isna()
    n = int(missing.sum())
    if n > max_fill_frac * len(out):
        raise SystemExit(
            f"{n:,} de {len(out):,} IDs do template ficaram sem previsão "
            f"(limite {max_fill_frac:.1%}): ranking e template não batem, confira os IDs"
        )
    if n:
        # Decolagem sem MVT_TIME no ranking não gera features: usa a mediana das previsões.
        out.loc[missing, F.TARGET] = float(np.median(pred))
    return out, n


def base_final(cfg: dict, full: pd.DataFrame, rk: pd.DataFrame, run: Run) -> np.ndarray:
    """Base treinada no ano inteiro (muta `full` e `rk`), prevendo o ranking."""
    cols = prepare(full, [rk], cfg.get("sem_features", ()),
                   cfg.get("base_ctx", False), cfg.get("base_p13", False),
                   int(cfg.get("cat_max", 0)), cfg.get("base_mapa", False))
    drop = leaky_columns(full, rk, cols)
    cols = [c for c in cols if c not in drop]
    run.log(f"treino {len(full):,} · ranking {len(rk):,} · ignoradas: {drop or 'nenhuma'}")
    return build_model(cfg).fit(full, cols, run=run).predict(rk)


def corretor_final(cfg_bloco: dict, adsb: bool, full: pd.DataFrame, rk: pd.DataFrame,
                   caminho_oof, run: Run, conjunto: bool = False, sem: Iterable[str] = (),
                   copia: CopiaCia | None = None, cias: list[str] | None = None,
                   fila: bool | str = False, dist_plano: bool = False, sem_ctx: bool = False,
                   xgb: bool = False, superficie: bool = False, rounds: int = ROUNDS_CORRETOR,
                   mapa: bool = False, cel: tuple | None = None):
    """Corretor treinado nas cegas com a previsão de uma base que não viu o mês delas.

    A previsão fora do bloco sai de `caminho_oof` quando ele já existe (ver `chave_oof`);
    senão é calculada e gravada lá.

    Com `copia` (config `externos`), as cegas ganham as colunas `ext_*`: a taxa de cópia
    vem dos 12 meses de 2025, sempre sem o mês da própria linha. Com `cias` (config
    `plano13`), ganham também as colunas do plano 13. Com `cel` (config `ref_cel`), as
    estatísticas de célula do bloco de meses de cada linha (`(tabelas, mês → bloco)`).
    """
    blind = load_split("blind2025")
    run.log(f"cegas {len(blind):,}")
    if caminho_oof.exists():
        oof = pd.read_parquet(caminho_oof)
        run.log(f"fora do bloco: {len(oof):,} previsões reaproveitadas de {caminho_oof.name}")
    else:
        oof = oof_base(cfg_bloco, full, blind, rk, run)
        caminho_oof.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(caminho_oof, index=False)
        run.log(f"fora do bloco: {len(oof):,} previsões em {caminho_oof.name}")
    pred_oof = oof["pred"].to_numpy(float)
    cegas = blind.set_index(F.ID).loc[oof[F.ID]].reset_index()  # mesma ordem do oof
    del blind
    ext = colunas_ext(cegas, copia, MESES_2025) if copia else None
    p13 = colunas_p13(cegas, cias, com_fila=fila) if cias is not None else None
    cel_cegas = None
    if cel is not None:
        tabs_bloco, bloco_do_mes = cel
        bloco = pd.Series(oof["mes"].to_numpy()).map(bloco_do_mes).to_numpy()
        cel_cegas = refcel.aplicar_por_bloco(cegas, bloco, tabs_bloco)
    X = corrector_frame(cegas, pred_oof, adsb, bool(cfg_bloco.get("janela_lobt")), sem, ext, p13,
                        dist_plano, sem_ctx, superficie, mapa, cel_cegas)
    del cegas
    if adsb:
        run.log(f"adsb no treino do corretor: {X['adsb_taxi'].notna().mean():.1%}")
    return fit_corrector(X, oof[TRUTH].to_numpy(float), pred_oof, conjunto, xgb, rounds)


def corrigir_ranking(corretor, cfg_bloco: dict, adsb: bool, rk: pd.DataFrame,
                     pred: np.ndarray, sem: Iterable[str] = (),
                     copia: CopiaCia | None = None,
                     cias: list[str] | None = None, fila: bool | str = False,
                     dist_plano: bool = False, sem_ctx: bool = False,
                     superficie: bool = False, mapa: bool = False,
                     tabs_cel: list | None = None) -> np.ndarray:
    """Previsão final do ranking: na janela do LOBT quando os blocos da base usam."""
    ext = colunas_ext(rk, copia, MESES_2025) if copia else None
    p13 = colunas_p13(rk, cias, com_fila=fila) if cias is not None else None
    cel = refcel.aplicar(rk, tabs_cel) if tabs_cel is not None else None
    return previsao_corrigida(corretor, rk, pred, adsb, bool(cfg_bloco.get("janela_lobt")),
                              sem, ext, p13, dist_plano, sem_ctx, superficie, mapa, cel)


def prever_membros(membros: list[dict], full, rk, run) -> list[tuple[dict, object, dict]]:
    """Um corretor por membro, treinado nas cegas com o oof do cache da base comum."""
    saida = []
    for m in membros:
        c = m["config"]
        copia = copia_cia_2025(run) if c.get("externos") else None
        cias = vocabulario(full) if c.get("plano13") else None
        if cias is not None:
            run.log(f"plano 13: {len(cias)} companhias no vocabulário do full2025")
        cel = tabs_cel = None
        if c.get("ref_cel"):
            tabs_bloco, tabs_cel, bloco_do_mes = tabelas_celula(full, run)
            cel = (tabs_bloco, bloco_do_mes)
        corretor = corretor_final(
            c["base_config"], c["adsb"], full, rk,
            OOF_CACHE / f"{chave_oof(c['base_config'])}.parquet",
            run, c.get("corretor") == "conjunto", c.get("sem_features", ()), copia, cias,
            c.get("fila", False), bool(c.get("dist_plano")), bool(c.get("corretor_sem_ctx")),
            bool(c.get("corretor_xgb")), bool(c.get("superficie")), c.get("rounds", ROUNDS_CORRETOR),
            bool(c.get("mapa")), cel,
        )
        saida.append((c, corretor, {"copia": copia, "cias": cias, "tabs_cel": tabs_cel}))
    return saida


def media_membros(membros, rk, pred_base, full, run, regras, prontos=None) -> np.ndarray:
    """Média simples das previsões corrigidas de cada membro, depois as pós-regras.

    `prontos` = saída de `prever_membros` já calculada antes da `base_final` (que muta `full`
    e `rk`); sem ela, calcula aqui (usado nos testes)."""
    preds = []
    for c, corretor, ex in prontos or prever_membros(membros, full, rk, run):
        preds.append(corrigir_ranking(
            corretor, c["base_config"], c["adsb"], rk, pred_base, c.get("sem_features", ()),
            ex.get("copia"), ex.get("cias"), c.get("fila", False), bool(c.get("dist_plano")),
            bool(c.get("corretor_sem_ctx")), bool(c.get("superficie")), bool(c.get("mapa")),
            ex.get("tabs_cel")))
    return pos_regras.aplicar(rk, np.mean(preds, axis=0), regras)


def submit(version: int, corrida: str | None = None) -> None:
    """Gera a versão N da campeã (champion.json v2) ou de uma corrida sozinha (`--corrida`)."""
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME") or sys.exit("Falta TEAM_NAME no .env")
    if corrida:
        membros, regras = [campeao.registro(corrida)], ["roma"]
    else:
        champ = campeao.carregar()
        membros, regras = [campeao.registro(m["id"]) for m in champ["membros"]], champ["pos_regras"]
    base_cfg = membros[0]["config"]["base_config"]
    OUT.mkdir(exist_ok=True)
    with Run(f"submit_v{version}", {"membros": [m["id"] for m in membros], "pos_regras": regras,
                                    "src_hash": runlog.src_hash()}) as run:
        with run.phase("dados", 0.05):
            full, rk = load_split("full2025"), load_split("ranking2026")
            template = pd.read_parquet(DATA / "submitting.parquet")
        with run.phase("corretor", 0.55):
            prontos = prever_membros(membros, full, rk, run)
        with run.phase("base final", 0.30):
            pred_base = base_final(escala_base(base_cfg), full, rk, run)
            pred = media_membros(membros, rk, pred_base, full, run, regras, prontos)
        with run.phase("arquivo", 0.10):
            out, preenchidas = build_submission(template, rk[F.ID], pred)
            path = OUT / f"{team}_v{version}.parquet"
            out.to_parquet(path, index=False)
            run.set(arquivo=str(path.relative_to(ROOT)), preenchidas=preenchidas)
            run.log(f"gerado {path.name}: {len(out):,} linhas, {len(membros)} membro(s), "
                    f"pós-regras {regras}. NÃO enviado.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("submit", help="gera submissions/<TEAM>_vN.parquet (não envia)")
    sp.add_argument("version", type=int)
    sp.add_argument("--corrida", help="id no experiments.jsonl (padrão: champion.json)")
    a = ap.parse_args()
    submit(a.version, corrida=a.corrida)
