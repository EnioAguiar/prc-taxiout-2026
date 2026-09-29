"""Versão final para envio, a partir do campeão (champion.json).

    bin/run src/train.py submit N [--forcar] [--corrida <id>]   # gera submissions/<TEAM>_vN.parquet (NÃO envia)

Sem `--corrida`, a receita é a do campeão; com `--corrida <id>`, a da última linha dessa
corrida no `experiments.jsonl` (o `champion.json` não é lido nem mexido).

Aborta se o código mudou desde a corrida escolhida (src_hash); --forcar ignora.
Envio separado, só depois de aprovado: .venv/bin/python src/s3.py submit <arquivo>

Campeã `stack_cf`: o corretor treina nas cegas com a previsão da base fora do bloco
(guardada em submissions/<TEAM>_vN_oof.parquet) e corrige a base final do ranking.

Com `externos: true` na config, as cegas e o ranking ganham as colunas `ext_*` de
`src/externos.py`: a taxa de cópia do SCHED por companhia vem dos 12 meses de 2025,
sempre sem o mês da própria linha.

Com `plano13: true`, as cegas e o ranking ganham também as colunas de `src/plano13.py`
(METAR, rotação no stand, consistência NM e a companhia como categoria); o vocabulário
de companhias sai do `full2025` e é o mesmo nos dois quadros.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import features as F
import runlog
from cache import DATA, TRUTH, load_split
from compare import CHAMPION
from crossfit import oof_base
from externos import MESES_2025, CopiaCia, colunas_ext, copia_cia_2025
from models import build_model, leaky_columns, prepare
from plano13 import colunas_p13, vocabulario
from runlog import REGISTRY, ROOT, Run
from stack import ROUNDS as ROUNDS_CORRETOR, corrector_frame, fit_corrector, previsao_corrigida

OUT = ROOT / "submissions"
ROUNDS_SCALE = 1.2  # full2025 tem 2,085 M linhas contra 1,741 M do train2025


def check_code(champ: dict, forcar: bool) -> str:
    """O campeão vale para o código que o mediu; mudou, refaça o experimento."""
    atual = runlog.src_hash()
    esperado = champ.get("src_hash")
    if esperado and esperado != atual and not forcar:
        raise SystemExit(
            f"o código mudou desde o campeão {champ['id']} (src_hash {esperado} → {atual}): "
            "refaça o experimento dele e promova de novo, ou repita com --forcar"
        )
    return atual


def escalar_rodadas(cfg: dict) -> dict:
    """Mais linhas no treino final pedem proporcionalmente mais rodadas."""
    for key in ("rounds", "cls_rounds", "reg_rounds"):
        if key in cfg:
            cfg[key] = round(cfg[key] * ROUNDS_SCALE)
    return cfg


def final_config(champ: dict) -> dict:
    cfg = dict(champ["config"])
    if cfg.get("model") == "stack_cf":
        # Só a base final vê o full2025; o corretor e os blocos ficam como foram medidos
        # (as rodadas sem escala seguem em champ["config"]["base_config"]).
        cfg["base_config"] = escalar_rodadas(dict(cfg["base_config"]))
        return cfg
    if champ.get("best_iter"):
        cfg["rounds"] = champ["best_iter"]
    return escalar_rodadas(cfg)


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
                   mapa: bool = False):
    """Corretor treinado nas cegas com a previsão de uma base que não viu o mês delas.

    Com `copia` (config `externos`), as cegas ganham as colunas `ext_*`: a taxa de cópia
    vem dos 12 meses de 2025, sempre sem o mês da própria linha. Com `cias` (config
    `plano13`), ganham também as colunas do plano 13.
    """
    blind = load_split("blind2025")
    run.log(f"cegas {len(blind):,}")
    oof = oof_base(cfg_bloco, full, blind, rk, run)
    oof.to_parquet(caminho_oof, index=False)
    run.log(f"fora do bloco: {len(oof):,} previsões em {caminho_oof.name}")
    pred_oof = oof["pred"].to_numpy(float)
    cegas = blind.set_index(F.ID).loc[oof[F.ID]].reset_index()  # mesma ordem do oof
    del blind
    ext = colunas_ext(cegas, copia, MESES_2025) if copia else None
    p13 = colunas_p13(cegas, cias, com_fila=fila) if cias is not None else None
    X = corrector_frame(cegas, pred_oof, adsb, bool(cfg_bloco.get("janela_lobt")), sem, ext, p13,
                        dist_plano, sem_ctx, superficie, mapa)
    del cegas
    if adsb:
        run.log(f"adsb no treino do corretor: {X['adsb_taxi'].notna().mean():.1%}")
    return fit_corrector(X, oof[TRUTH].to_numpy(float), pred_oof, conjunto, xgb, rounds)


def corrigir_ranking(corretor, cfg_bloco: dict, adsb: bool, rk: pd.DataFrame,
                     pred: np.ndarray, sem: Iterable[str] = (),
                     copia: CopiaCia | None = None,
                     cias: list[str] | None = None, fila: bool | str = False,
                     dist_plano: bool = False, sem_ctx: bool = False,
                     superficie: bool = False, mapa: bool = False) -> np.ndarray:
    """Previsão final do ranking: na janela do LOBT quando os blocos da base usam."""
    ext = colunas_ext(rk, copia, MESES_2025) if copia else None
    p13 = colunas_p13(rk, cias, com_fila=fila) if cias is not None else None
    return previsao_corrigida(corretor, rk, pred, adsb, bool(cfg_bloco.get("janela_lobt")),
                              sem, ext, p13, dist_plano, sem_ctx, superficie, mapa)


def corrida_registrada(corrida_id: str) -> dict:
    """A última linha de `experiments.jsonl` com esse id (como `stack.base_config`)."""
    for line in reversed(REGISTRY.read_text().splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("id") == corrida_id:
            return rec
    raise SystemExit(f"corrida {corrida_id} não está em {REGISTRY.name}")


def submit(version: int, forcar: bool = False, corrida: str | None = None) -> None:
    """Gera a versão N do campeão, ou da corrida `corrida` do registro."""
    load_dotenv(ROOT / ".env")
    team = os.environ.get("TEAM_NAME") or sys.exit("Falta TEAM_NAME no .env")
    champ = corrida_registrada(corrida) if corrida else json.loads(CHAMPION.read_text())
    check_code(champ, forcar)
    cfg = final_config(champ)
    empilhado = cfg["model"] == "stack_cf"
    OUT.mkdir(exist_ok=True)

    with Run(f"submit_v{version}", {**cfg, "campeao": champ["id"], "forcar": forcar}) as run:
        with run.phase("dados", 0.05):
            full, rk = load_split("full2025"), load_split("ranking2026")
            template = pd.read_parquet(DATA / "submitting.parquet")
        if empilhado:
            with run.phase("corretor", 0.55):
                copia = copia_cia_2025(run) if champ["config"].get("externos") else None
                cias = vocabulario(full) if champ["config"].get("plano13") else None
                if cias is not None:
                    run.log(f"plano 13: {len(cias)} companhias no vocabulário do full2025")
                corretor = corretor_final(
                    champ["config"]["base_config"], champ["config"]["adsb"], full, rk,
                    OUT / f"{team}_v{version}_oof.parquet", run,
                    champ["config"].get("corretor") == "conjunto",
                    champ["config"].get("sem_features", ()), copia, cias,
                    champ["config"].get("fila", False),
                    bool(champ["config"].get("dist_plano")),
                    bool(champ["config"].get("corretor_sem_ctx")),
                    bool(champ["config"].get("corretor_xgb")),
                    bool(champ["config"].get("superficie")),
                    champ["config"].get("rounds", ROUNDS_CORRETOR),
                    bool(champ["config"].get("mapa")),
                )
            with run.phase("base final", 0.30):
                pred = base_final(cfg["base_config"], full, rk, run)
                pred = corrigir_ranking(
                    corretor, champ["config"]["base_config"], champ["config"]["adsb"], rk, pred,
                    champ["config"].get("sem_features", ()), copia, cias,
                    champ["config"].get("fila", False),
                    bool(champ["config"].get("dist_plano")),
                    bool(champ["config"].get("corretor_sem_ctx")),
                    bool(champ["config"].get("superficie")),
                    bool(champ["config"].get("mapa")),
                )
        else:
            with run.phase("treino", 0.85):
                pred = base_final(cfg, full, rk, run)
        with run.phase("arquivo", 0.10):
            out, preenchidas = build_submission(template, rk[F.ID], pred)
            path = OUT / f"{team}_v{version}.parquet"
            out.to_parquet(path, index=False)
            run.set(arquivo=str(path.relative_to(ROOT)), preenchidas=preenchidas)
            run.log(
                f"gerado {path.name}: {len(out):,} linhas, mediana {out[F.TARGET].median():.0f} s, "
                f"preenchidas com a mediana: {preenchidas}. NÃO enviado."
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("submit", help="gera submissions/<TEAM>_vN.parquet (não envia)")
    sp.add_argument("version", type=int)
    sp.add_argument("--forcar", action="store_true", help="ignora a mudança de src_hash")
    sp.add_argument("--corrida", help="id no experiments.jsonl (padrão: champion.json)")
    a = ap.parse_args()
    submit(a.version, forcar=a.forcar, corrida=a.corrida)
