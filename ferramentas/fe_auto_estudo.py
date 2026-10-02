"""Estudo do item 3 (geração automática de features no resíduo da campeã).

Três etapas, cada uma num processo (o holdout e o ranking não cabem juntos na RAM):

    bin/run ferramentas/fe_auto_estudo.py peneira   # estágio 1 do OpenFE (FeatureBoost)
    bin/run ferramentas/fe_auto_estudo.py deriva    # AUC adversarial 2025 × 2026
    bin/run ferramentas/fe_auto_estudo.py bloco     # ganho do bloco final, fora do mês

Resultados em docs/research/2026-10-03-fe-auto/ (JSON), lidos pelo relatório.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import campeao  # noqa: E402
import fe_auto  # noqa: E402
import features as F  # noqa: E402
import stack  # noqa: E402
from cache import TRUTH, load_split  # noqa: E402

SAIDA = ROOT / "docs" / "research" / "2026-10-03-fe-auto"
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              verbose=-1, num_threads=4, seed=0, deterministic=True, force_row_wise=True)
PARAMS_1 = {**PARAMS, "num_leaves": 15, "min_data_in_leaf": 500}
RODADAS, RODADAS_1 = 300, 60
FOLDS = 5
CORPO = 3600.0
N_PARES, N_GRUPO = 14, 12  # colunas do pool de pares e de agregação, por importância


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e))))


def dados() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Holdout alinhado à previsão da campeã, o quadro do corretor e o resíduo."""
    champ = campeao.carregar()
    prev = campeao.previsao([m["id"] for m in champ["membros"]])
    df = stack.na_ordem(load_split("holdout2025"), prev[F.ID].to_numpy())
    pred = prev["pred"].to_numpy(float)
    X = stack.corrector_frame(df, pred, adsb=True, janela=True, dist_plano=True)
    resid = prev[TRUTH].to_numpy(float) - pred
    log(f"holdout {len(df):,} · rmse da campeã {rmse(resid):.2f} · {X.shape[1]} colunas base")
    return df, X, resid


def mes(df: pd.DataFrame) -> np.ndarray:
    return df["MVT_TIME_UTC_mvt"].dt.month.to_numpy()


def dias(df: pd.DataFrame) -> np.ndarray:
    return df["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d").to_numpy()


def fora_do_mes(X: pd.DataFrame, resid: np.ndarray, dia: np.ndarray,
                treino: np.ndarray, aval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(oof dentro do mês de treino, previsão média no mês de avaliação).

    O oof é o init_score do estágio 1: a candidata é avaliada sobre a previsão que já
    existe, nunca sobre uma previsão que viu a própria linha.
    """
    Xt, rt = X[treino], resid[treino]
    folds = stack.day_folds(dia[treino], FOLDS)
    oof = np.empty(len(rt))
    pred_aval = np.zeros(int(aval.sum()))
    Xa = X[aval]
    for k in range(FOLDS):
        tr = folds != k
        ds = lgb.Dataset(Xt[tr], label=rt[tr], free_raw_data=True)
        m = lgb.train(PARAMS, ds, RODADAS)
        oof[~tr] = m.predict(Xt[~tr])
        pred_aval += m.predict(Xa) / FOLDS
        del m, ds
    return oof, pred_aval


def pool(X: pd.DataFrame, df: pd.DataFrame, resid: np.ndarray) -> list[str]:
    """Colunas-fonte das candidatas, por importância num modelo do resíduo.

    O modelo da seleção vê as entradas do corretor **e** as numéricas do quadro bruto que
    ele não usa (as contagens `apt_*`/`rwy_*` e os `gap_*`, que estão na base e não no
    corretor): a fonte de uma candidata boa pode ser uma coluna que o corretor não tem.
    Só entram colunas do quadro bruto — a previsão da base e o que deriva dela ficam fora,
    senão a candidata não é calculável no ranking de 2026 e não dá para medir a deriva.
    Identificadores (`*_ID_mvt`) também ficam de fora: são numéricos e crescem com a data,
    então o modelo os usa como relógio e qualquer feature feita deles é deriva pura.
    """
    proibidas = {TRUTH, F.TARGET, F.ID, "FLIGHT_ID_mvt"}
    numerica = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and c not in proibidas and not c.endswith("_ID_mvt")]
    extras = [c for c in numerica if c not in X.columns]
    Xp = pd.concat([X, df[extras].astype("float32")], axis=1)
    m = lgb.train(PARAMS, lgb.Dataset(Xp, label=resid), RODADAS)
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name())
    nums = [c for c in imp.sort_values(ascending=False).index if c in numerica]
    log(f"pool ({len(extras)} colunas brutas somadas): {nums[:N_PARES]}")
    return nums


def bases_do_mes(X: pd.DataFrame, resid: np.ndarray, m: np.ndarray,
                 dia: np.ndarray) -> dict[int, dict]:
    """Modelo do resíduo de um mês aplicado no outro, nos dois sentidos."""
    base: dict[int, dict] = {}
    for treino_m, aval_m in ((1, 7), (7, 1)):
        tr, av = m == treino_m, m == aval_m
        oof, pa = fora_do_mes(X, resid, dia, tr, av)
        base[aval_m] = {"tr": tr, "av": av, "oof": oof, "pred": pa,
                        "rmse": rmse(resid[av] - pa)}
        log(f"base {treino_m}→{aval_m}: rmse {base[aval_m]['rmse']:.3f}")
    return base


def avaliar(v: np.ndarray, base: dict[int, dict], resid: np.ndarray,
            corpo: np.ndarray) -> dict:
    """FeatureBoost: a candidata sozinha sobre a previsão que já existe, fora do mês."""
    reg: dict = {}
    for aval_m, b in base.items():
        tr, av = b["tr"], b["av"]
        ds = lgb.Dataset(v[tr].reshape(-1, 1), label=resid[tr], init_score=b["oof"])
        mdl = lgb.train(PARAMS_1, ds, RODADAS_1)
        novo = b["pred"] + mdl.predict(v[av].reshape(-1, 1))
        e0, e1 = resid[av] - b["pred"], resid[av] - novo
        c = corpo[av]
        reg[f"ganho_{aval_m}"] = round(b["rmse"] - rmse(e1), 4)
        reg[f"corpo_{aval_m}"] = round(rmse(e0[c]) - rmse(e1[c]), 4)
        del mdl, ds
    reg["min_ganho"] = round(min(reg["ganho_1"], reg["ganho_7"]), 4)
    reg["min_corpo"] = round(min(reg["corpo_1"], reg["corpo_7"]), 4)
    return reg


def peneira() -> None:
    df, X, resid = dados()
    m, dia = mes(df), dias(df)
    ordem = pool(X, df, resid)
    specs = fe_auto.candidatas(ordem[:N_GRUPO], fe_auto.CHAVES, ordem[:N_PARES])
    log(f"{len(specs)} candidatas")
    base = bases_do_mes(X, resid, m, dia)
    corpo = (resid + X["pred"].to_numpy(float)) <= CORPO  # o que não é loteria
    cache: dict[str, np.ndarray] = {}
    linhas = []
    for i, spec in enumerate(specs):
        v = fe_auto.valores(df, spec, cache)
        if not np.isfinite(v).any():
            continue
        linhas.append({"spec": list(spec), "nome": fe_auto.nome(spec),
                       **avaliar(v, base, resid, corpo)})
        if (i + 1) % 50 == 0:
            log(f"{i + 1}/{len(specs)}")
    linhas.sort(key=lambda r: -r["min_ganho"])
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "peneira.json").write_text(json.dumps(
        {"base": {k: round(v["rmse"], 4) for k, v in base.items()},
         "pool_pares": ordem[:N_PARES], "pool_grupo": ordem[:N_GRUPO],
         "candidatas": linhas}, indent=1))
    bons = [r for r in linhas if r["min_ganho"] > 0 and r["min_corpo"] > 0]
    log(f"{len(bons)} candidatas ganham nos dois meses e no corpo; top 10:")
    for r in bons[:10]:
        log(f"  {r['nome']}: jan {r['ganho_1']:+.3f} jul {r['ganho_7']:+.3f} "
            f"corpo {r['min_corpo']:+.3f}")


def placebo(n: int = 40) -> None:
    """Banda de ruído: as mesmas contas com a candidata embaralhada por linha.

    Sem isto não dá para ler a peneira — uma feature que "ganha 0,05 s fora do mês" só
    quer dizer alguma coisa se o embaralhamento dela ganhar menos que isso.
    """
    reg = json.loads((SAIDA / "peneira.json").read_text())
    specs = [tuple(r["spec"]) for r in reg["candidatas"]]
    df, X, resid = dados()
    m, dia = mes(df), dias(df)
    base = bases_do_mes(X, resid, m, dia)
    corpo = (resid + X["pred"].to_numpy(float)) <= CORPO
    rng = np.random.default_rng(7)
    cache: dict[str, np.ndarray] = {}
    linhas = []
    for i in rng.choice(len(specs), size=n, replace=False):
        v = fe_auto.valores(df, specs[int(i)], cache)
        linhas.append({"nome": fe_auto.nome(specs[int(i)]),
                       **avaliar(rng.permutation(v), base, resid, corpo)})
    g = np.array([r["min_ganho"] for r in linhas])
    c = np.array([r["min_corpo"] for r in linhas])
    resumo = {"n": n, "min_ganho": {"p50": float(np.median(g)), "p95": float(np.percentile(g, 95)),
                                    "max": float(g.max())},
              "min_corpo": {"p50": float(np.median(c)), "p95": float(np.percentile(c, 95)),
                            "max": float(c.max())},
              "positivos_nos_dois": int(((g > 0)).sum()),
              "corpo_positivo_nos_dois": int((c > 0).sum()), "corridas": linhas}
    log(f"placebo: min_ganho p95 {resumo['min_ganho']['p95']:+.3f} max "
        f"{resumo['min_ganho']['max']:+.3f} · min_corpo p95 {resumo['min_corpo']['p95']:+.3f} "
        f"max {resumo['min_corpo']['max']:+.3f} · {resumo['corpo_positivo_nos_dois']}/{n} "
        "embaralhadas ganham no corpo nos dois meses")
    (SAIDA / "placebo.json").write_text(json.dumps(resumo, indent=1))


def _amostra(n: int, k: int, semente: int) -> np.ndarray:
    rng = np.random.default_rng(semente)
    return rng.choice(n, size=min(k, n), replace=False)


def _auc(rotulo: np.ndarray, score: np.ndarray) -> float:
    """AUC pela estatística de Mann-Whitney (sem sklearn no ambiente)."""
    postos = pd.Series(score).rank().to_numpy()
    pos = rotulo == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((postos[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sobreviventes(k: int = 40, com_ganho: bool = True) -> list[dict]:
    """As candidatas que passam do ruído: ganho **positivo** no corpo nos dois meses (e o
    zero já está acima do p95 do placebo, que é negativo) e ganho completo acima do p95 do
    placebo — ou seja, a cauda pode tirar o ganho, mas não mais do que tira de uma coluna
    embaralhada.

    Ordenadas pelo pior dos dois meses no corpo — o critério conservador."""
    reg = json.loads((SAIDA / "peneira.json").read_text())
    pl = json.loads((SAIDA / "placebo.json").read_text())
    corte_corpo = max(0.0, pl["min_corpo"]["p95"])
    corte_ganho = pl["min_ganho"]["p95"]
    bons = [r for r in reg["candidatas"] if r["min_corpo"] > corte_corpo
            and (r["min_ganho"] > corte_ganho or not com_ganho)]
    bons.sort(key=lambda r: -r["min_corpo"])
    log(f"corte: corpo > {corte_corpo:+.3f} · completo > {corte_ganho:+.3f} (p95 do placebo)")
    return bons[:k]


def deriva(k: int = 40, n: int = 120_000) -> None:
    """AUC adversarial 2025 (holdout) × 2026 (ranking), candidata a candidata.

    De cada agregação de nível vai também a **variante sem deriva** (`fe_auto.SEM_DERIVA`,
    média do grupo centrada na média do quadro e desvio em desvios do quadro): dentro de um
    quadro é a mesma coluna a menos de deslocamento/escala, então o ganho da peneira vale
    igual, e é só a AUC que muda. Junto vão as colunas-fonte cruas como **referência**:
    0,92 numa feature gerada só quer dizer alguma coisa comparado com a AUC da coluna de
    onde ela saiu.
    """
    bons = sobreviventes(k, com_ganho=False)
    for r in list(bons):
        op = r["spec"][0]
        if op in fe_auto.SEM_DERIVA:
            variante = [fe_auto.SEM_DERIVA[op], *r["spec"][1:]]
            bons.append({**r, "spec": variante, "nome": fe_auto.nome(tuple(variante)),
                         "variante_de": r["nome"]})
    specs = [tuple(r["spec"]) for r in bons]
    reg = json.loads((SAIDA / "peneira.json").read_text())
    fontes = sorted({p for s in specs for p in s[1:] if p not in fe_auto.CHAVES}
                    | set(reg["pool_pares"][:6]))
    log(f"{len(specs)} candidatas + {len(fontes)} colunas cruas de referência")
    colunas = {}
    for nome_split in ("holdout2025", "ranking2026"):
        d = load_split(nome_split)
        cache: dict[str, np.ndarray] = {}
        idx = _amostra(len(d), n, 0)
        colunas[nome_split] = np.column_stack(
            [fe_auto.valores(d, s, cache)[idx] for s in specs]
            + [np.asarray(d[c], float)[idx] for c in fontes]).astype("float32")
        log(f"{nome_split}: {colunas[nome_split].shape}")
        del d, cache
    A, B = colunas["holdout2025"], colunas["ranking2026"]
    rotulo = np.r_[np.zeros(len(A)), np.ones(len(B))]
    saida, referencia = [], []
    p = {**PARAMS, "objective": "binary", "metric": "auc", "num_leaves": 15,
         "min_data_in_leaf": 500}
    linhas = [{**r} for r in bons] + [{"nome": c, "crua": True} for c in fontes]
    for j, r in enumerate(linhas):
        x = np.r_[A[:, j], B[:, j]].reshape(-1, 1)
        corte = _amostra(len(x), len(x) // 2, 1)
        teste = np.setdiff1d(np.arange(len(x)), corte)
        m = lgb.train(p, lgb.Dataset(x[corte], label=rotulo[corte]), 60)
        auc = _auc(rotulo[teste], m.predict(x[teste]))
        faltam = (float(np.isnan(A[:, j]).mean()), float(np.isnan(B[:, j]).mean()))
        linha = {**r, "auc": round(auc, 4), "nan_2025": round(faltam[0], 4),
                 "nan_2026": round(faltam[1], 4)}
        (referencia if r.get("crua") else saida).append(linha)
        log(f"  {'[crua] ' if r.get('crua') else ''}{r['nome']}: auc {auc:.3f} · "
            f"nan {faltam[0]:.2%}/{faltam[1]:.2%}")
        del m
    (SAIDA / "deriva.json").write_text(json.dumps(
        {"candidatas": saida, "referencia": referencia}, indent=1))


def bloco(limite: int = 10, auc_max: float = 0.6) -> None:
    """Ganho fora do mês do bloco final, com o recorte da cauda e dos 20 piores voos.

    Entram as candidatas que passam nos três portões: ganho no corpo nos dois meses,
    ganho completo acima do placebo e AUC adversarial 2025×2026 ≤ `auc_max`."""
    reg = json.loads((SAIDA / "deriva.json").read_text())["candidatas"]
    corte_ganho = json.loads((SAIDA / "placebo.json").read_text())["min_ganho"]["p95"]
    escolhidas = [r for r in reg
                  if r["auc"] <= auc_max and r["min_ganho"] > corte_ganho][:limite]
    specs = [tuple(r["spec"]) for r in escolhidas]
    log(f"bloco com {len(specs)}: {[r['nome'] for r in escolhidas]}")
    df, X, resid = dados()
    m, dia = mes(df), dias(df)
    cache: dict[str, np.ndarray] = {}
    Xf = X.copy()
    for s in specs:
        Xf[fe_auto.nome(s)] = fe_auto.valores(df, s, cache)
    y_true = resid + X["pred"].to_numpy(float)
    saida = {"features": [r["nome"] for r in escolhidas], "meses": {}}
    for treino_m, aval_m in ((1, 7), (7, 1)):
        tr, av = m == treino_m, m == aval_m
        _, p0 = fora_do_mes(X, resid, dia, tr, av)
        _, p1 = fora_do_mes(Xf, resid, dia, tr, av)
        e0, e1 = resid[av] - p0, resid[av] - p1
        yv = y_true[av]
        corpo = yv <= CORPO
        top = np.argsort(-np.square(e0))[:20]
        e_sem = e1.copy()
        e_sem[top] = e0[top]  # bloco aplicado em todo mundo menos os 20 piores voos
        saida["meses"][str(aval_m)] = {
            "rmse_base": round(rmse(e0), 4), "rmse_bloco": round(rmse(e1), 4),
            "ganho": round(rmse(e0) - rmse(e1), 4),
            "ganho_corpo": round(rmse(e0[corpo]) - rmse(e1[corpo]), 4),
            "ganho_cauda": round(rmse(e0[~corpo]) - rmse(e1[~corpo]), 4),
            "n_corpo": int(corpo.sum()), "n_cauda": int((~corpo).sum()),
            "ganho_sem_top20": round(rmse(e0) - rmse(e_sem), 4),
            "parte_top20": round(1 - (rmse(e0) - rmse(e_sem)) / (rmse(e0) - rmse(e1)), 4)
            if rmse(e0) != rmse(e1) else None,
        }
        log(f"{treino_m}→{aval_m}: {saida['meses'][str(aval_m)]}")
    (SAIDA / "bloco.json").write_text(json.dumps(saida, indent=1))


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "peneira"
    {"peneira": peneira, "placebo": placebo, "deriva": deriva, "bloco": bloco}[etapa]()
