"""Empilhamento sobre uma corrida base: um LightGBM aprende a correção `y − pred_base`.

Duas simulações do mesmo corretor:

- padrão (plano 4): 5 folds por dia no holdout jan+jul/2025 — cada voo é corrigido por um
  modelo que não viu o dia dele;
- `--crossfit` (plano 5): o corretor treina no ano inteiro, sobre as previsões da base
  fora do bloco (`crossfit.oof_base` nas linhas cegas de `blind2025`), e é medido no
  holdout com as previsões da corrida base.

Entradas do corretor: `pred`, aeroporto, `nm_missing`, hora, `to_takeoff_from_*` e, sem
`--sem-adsb`, as colunas `adsb_*` de `events.parquet` (NaN sem cobertura); a saída é
`pred + correção` com piso 0. A corrida é gravada em `runs/` e no `experiments.jsonl`,
então passa pelo `compare.py` como qualquer outra.

Quando a corrida base usa `janela_lobt`, o corretor de `--crossfit` ganha `dist_lo` e
`dist_hi` (folga até as bordas da janela do LOBT, NaN sem LOBT) e sua saída é projetada
na janela antes do piso 0; sem `janela_lobt` na base, tudo segue como antes.

Uso:
    bin/run src/stack.py <nome> [--base <id>] [--crossfit [--seeds N] [--conjunto]] [--sem-adsb]
                                [--sem-feature COLUNA] [--corretor-sem-feature COLUNA]
                                [--externos] [--plano13] [--reusar-oof <id>]

`--sem-feature COLUNA` (pode repetir, só com `--crossfit`) tira a coluna da base e do
corretor e grava `sem_features` na config da corrida e na `base_config`; sem a flag, nada
muda. Nos folds a base vem pronta e ninguém confere o nome, então a flag é recusada.

`--corretor-sem-feature COLUNA` (pode repetir, só com `--crossfit`) tira a coluna só das
entradas do corretor e grava `corretor_sem_features` na config da corrida — a
`base_config` fica intacta, então a previsão fora do bloco pode vir de `--reusar-oof`.
É a flag das ablações de grupo do corretor (`ctx_*`, `met_*`, `pista_*`…), que custam
~10 min em vez dos ~75 min de uma base nova.

`--conjunto` (só com `--crossfit`) troca o corretor único pela média de três treinados
nas mesmas entradas: LightGBM global, um LightGBM por aeroporto (aeroporto sem modelo
próprio usa o global) e CatBoost; grava `corretor: "conjunto"` na config, que o
`train.py` lê no envio.

Com `--crossfit`, `--seeds N` (N > 1) manda a base de cada bloco ser a média de N seeds:
sobrescreve `seeds` na config da corrida base. A base do holdout vem pronta de `--base`,
que já deve ser a corrida de N seeds.

`--externos` (só com `--crossfit`) soma ao corretor as colunas `ext_*` de
`src/externos.py` (taxa de cópia do SCHED por companhia, séries diárias da EUROCONTROL e
tempo em solo do OPDI) e grava `externos: true` na config. A taxa de cópia é aprendida só
nos meses do oof (os 10 do treino) e nunca no mês da própria linha.

`--plano13` (só com `--crossfit`) soma ao corretor os quatro sinais de `src/plano13.py`
(METAR do aeroporto, rotação no stand, consistência do plano NM e a companhia como
categoria) e grava `plano13: true` na config. O vocabulário de companhias é fixado nas
linhas de treino e vale igual para as cegas, o holdout e o ranking.

`--dist-plano` e `--corretor-sem-ctx` (só com `--crossfit`, laço de 28/09): o corretor
ganha a distância da previsão da base a cada horário planejado, e/ou perde as colunas
`ctx_*` quando a base já as usa (`--base-ctx`). Gravam `dist_plano` / `corretor_sem_ctx`.

`--fe-auto` (só com `--crossfit`) soma ao corretor as colunas `fe_*` de `src/fe_auto.py`:
as features geradas automaticamente (razão/diferença/produto de duas numéricas e
agregações por grupo) que passaram no estudo de 03/10 — ganho fora do mês em jan **e** em
jul, ganho também no corpo (`y ≤ 3600 s`) e AUC adversarial 2025×2026 abaixo de 0,6.
Grava `fe_auto: true` na config. Nada ali lê o alvo nem a previsão da base, então o valor
de um voo é o mesmo nas cegas, no holdout e no ranking.

`--reusar-oof <id>` (só com `--crossfit`) pula o recálculo da base fora do bloco e lê o
`oof` gravado por aquela corrida, recusando a troca se o `base` ou a `base_config` dela
não forem idênticos aos desta. A previsão fora do bloco depende só da base, então trocar
de corretor não exige refazê-la: a corrida cai de ~25 min para ~7 min. Grava
`reusar_oof: <id>` na config; o `train.py` ignora a chave e refaz o oof do ano inteiro no
envio, como sempre.

`--pseudo propria|campea` (só com `--crossfit`) põe as partidas de 2026 do
`ranking.parquet` no **treino** do corretor, com alvo previsto (nunca verdade de 2026) e
peso `--pseudo-peso` (0,3). Só o corpo entra: alvo até `--pseudo-corte` (3.600 s) e voo
com plano NM. `propria` tira o alvo do próprio corretor desta corrida, que só viu os 10
meses de treino — o holdout continua limpo; `campea` tira do último arquivo de
`submissions/`, que treinou no `full2025` **com** jan e jul e por isso marca
`pseudo_vazado: true` na config: o holdout daquela corrida não decide nada
(`docs/research/2026-10-03-pseudo-rotulo.md`).
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import campeao
import contexto
import features as F
from adsb_events import FEATURES as ADSB
from cache import TRUTH, load_split
from crossfit import month_blocks, oof_base
from externos import colunas_ext, copia_cia_2025
from experiment import RUNS, metrics, rmse
from models import janela_lobt, limitar_janela, linhas_de_regra as models_linhas_de_regra
from dispositivo import lgb_params
from pista import colunas as colunas_pista
from pista import colunas_retencao
from plano13 import colunas_p13, vocabulario
from runlog import REGISTRY, ROOT, Run
from superficie import contagens as contagens_superficie
import fe_auto as fe_auto_mod
import mapa as mapa_aeroporto
import memoria
import pseudo as pseudo_mod
import refcel
import roma as roma_tdg_mod

FOLDS = 5
ROUNDS = 300
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              verbose=-1, num_threads=12, seed=0, deterministic=True, force_row_wise=True)
PARAMS_AEROPORTO = {**PARAMS, "min_data_in_leaf": 100}  # menos dados por modelo
CATBOOST = dict(iterations=800, depth=8, learning_rate=0.08, loss_function="RMSE", random_seed=0)


def registro_da_corrida(run_id: str) -> dict:
    """Última linha de `run_id` no registro (a medição que vale)."""
    for line in reversed(REGISTRY.read_text().splitlines()):
        rec = json.loads(line) if line.strip() else {}
        if rec.get("id") == run_id:
            return rec
    raise SystemExit(f"corrida {run_id} não está em {REGISTRY.name}")


def base_config(base_id: str) -> dict:
    """Config com que a corrida base foi medida (a última linha dela no registro)."""
    return registro_da_corrida(base_id)["config"]


def oof_reusado(run_id: str, base_id: str, cfg_base: dict) -> Path:
    """Caminho do oof de `run_id`, só se ele saiu desta mesma base.

    A previsão fora do bloco depende apenas de `base` e `base_config`: com as duas iguais,
    recalcular dá o mesmo quadro. Qualquer diferença é recusada — corretor diferente pode
    reaproveitar, base diferente não.
    """
    rec = registro_da_corrida(run_id)
    cfg = rec.get("config", {})
    if cfg.get("base") != base_id or cfg.get("base_config") != cfg_base:
        raise SystemExit(
            f"--reusar-oof {run_id}: a base daquela corrida não é a desta.\n"
            f"  lá: base {cfg.get('base')} · {json.dumps(cfg.get('base_config'), sort_keys=True, ensure_ascii=False)}\n"
            f"  aqui: base {base_id} · {json.dumps(cfg_base, sort_keys=True, ensure_ascii=False)}"
        )
    if not rec.get("oof"):
        raise SystemExit(f"--reusar-oof {run_id}: aquela corrida não gravou `oof` no registro")
    caminho = ROOT / rec["oof"]
    if not caminho.exists():
        raise SystemExit(f"--reusar-oof {run_id}: falta o arquivo {caminho}")
    return caminho


def config_da_base(base_id: str, seeds: int) -> dict:
    """Config dos blocos: a da corrida base, com média de N seeds quando N > 1."""
    cfg = base_config(base_id)
    return {**cfg, "seeds": seeds} if seeds > 1 else cfg


def corrector_frame(df: pd.DataFrame, pred: np.ndarray, adsb: bool = True,
                    janela: bool = False, sem: Iterable[str] = (),
                    externos: dict | None = None,
                    plano13: pd.DataFrame | None = None, dist_plano: bool = False,
                    sem_ctx: bool = False, superficie: bool = False,
                    mapa: bool = False, ref_cel: pd.DataFrame | None = None,
                    pista: bool = False, retencao: bool = False,
                    roma: pd.DataFrame | None = None, fe_auto: bool = False,
                    exigir: Iterable[str] = ()) -> pd.DataFrame:
    """Entradas do corretor: a previsão da base, o contexto do voo e o rastro ADS-B.

    `sem` tira as colunas com esses nomes (nome que não está no quadro é ignorado: a
    lista é a mesma da base, que tem colunas que o corretor não usa). `exigir` são os
    nomes que *precisam* existir — a ablação de `--corretor-sem-feature` erra alto em vez
    de medir uma receita idêntica à campeã por causa de um nome escrito errado.

    `externos` (`src/externos.py`, `--externos`) são colunas `ext_*` prontas, uma por
    linha e na mesma ordem de `df`; sem elas o quadro é o de sempre.

    `plano13` (`src/plano13.py`, `--plano13`) é o quadro com as colunas `met_*`, `rot_*`,
    `nm_*` e `cia`, na mesma ordem de `df`; `cia` entra categórica.

    `dist_plano` (`--dist-plano`) soma a distância da previsão da base a cada horário
    planejado (`dist_*`, `dist_min`, `n_planos`, `n_planos_longos`): o sinal do hedge entre
    "taxi normal" e "cópia do planejado" (laço de 28/09). `sem_ctx` (`--corretor-sem-ctx`)
    tira as colunas `ctx_*` do corretor, para bases que já as usam (`base_ctx`).

    `superficie` (`src/superficie.py`, `--superficie`) soma as colunas `sup_*`: quantos
    aviões estão no solo no push estimado (`MVT − pred`) e quantos decolam ou pousam
    durante o táxi estimado, sem nunca ler o off-block de outra decolagem.

    `mapa` (`src/mapa.py`, `--mapa`) soma `map_*`: distância de táxi do stand à cabeceira
    pelo grafo de taxiways do `apt.dat` do X-Plane.

    `ref_cel` (`src/refcel.py`, `--corretor-ref`) são as colunas `cel_*` prontas: mediana,
    P90, desvio e tamanho da célula (aeroporto, stand, pista), ajustadas fora do bloco de
    meses. Entram com a distância da previsão à mediana da célula.

    `pista` (`src/pista.py`, `--pista`) soma as colunas `pista_*`: configuração de pista do
    aeroporto na hora do voo, fila com peso de esteira na mesma pista entre o `AOBT_3` e a
    decolagem e a cadência das últimas decolagens. Só colunas que o ranking também tem.

    `retencao` (`src/pista.py`, `--retencao`) soma as colunas `ret_*`: no instante do push
    real (`AOBT_3`), quantas partidas já passaram do `EOBT_1` sem empurrar, quantas estão
    taxiando na mesma pista, o atraso de push recente do aeroporto e o EWMA de decolagens.

    `roma` (`src/roma.py`, `--roma-tdg`) é o quadro com `roma_g_hat`/`roma_t_hat` pronto, na
    mesma ordem de `df`: o atraso de portão previsto no LIRF e o táxi que a identidade
    `T = D − G` reconstrói com ele. Entra com a distância da previsão da base a `roma_t_hat`;
    fora do LIRF as três colunas são nulas.

    `fe_auto` (`src/fe_auto.py`, `--fe-auto`) soma as colunas `fe_*`: as features geradas
    automaticamente (razão/diferença/produto de duas numéricas e agregações por grupo) que
    sobreviveram ao estudo de 03/10 — ganho fora do mês nos dois meses do holdout, ganho
    também no corpo (`y ≤ 3600 s`) e AUC adversarial 2025×2026 abaixo de 0,6.
    """
    cols = [F.AIRPORT, "nm_missing", "hour", *[c for c in df if c.startswith("to_takeoff_from_")],
            *([] if sem_ctx else [c for c in contexto.COLS if c in df])]
    X = df[cols].copy()  # cópia real: quem chama solta `df` logo depois
    X[F.AIRPORT] = X[F.AIRPORT].astype("category")
    X["pred"] = np.asarray(pred, float)
    if janela:
        lo, hi = janela_lobt(df)
        X["dist_lo"] = X["pred"].to_numpy(float) - lo  # folga até o fundo da janela do LOBT
        X["dist_hi"] = hi - X["pred"].to_numpy(float)
    if adsb:
        X = X.join(df[ADSB])
        X["adsb_menos_pred"] = X["adsb_taxi_move"] - X["pred"]
    for nome, valores in (externos or {}).items():
        X[nome] = np.asarray(valores, float)
    if plano13 is not None:
        for nome in plano13.columns:
            X[nome] = plano13[nome].array  # posicional; `cia` continua categórica
    if dist_plano:
        for nome, valores in distancias_plano(df, X["pred"].to_numpy(float)).items():
            X[nome] = valores
    if superficie:
        for nome, valores in contagens_superficie(df, X["pred"].to_numpy(float)).items():
            X[nome] = np.asarray(valores, float)
    if mapa:
        for nome, valores in mapa_aeroporto.colunas(df).items():
            X[nome] = valores
    if ref_cel is not None:
        for nome in refcel.COLS:
            X[nome] = np.asarray(ref_cel[nome], float)
        X["cel_pred_menos_p50"] = X["pred"].to_numpy(float) - X["cel_p50"].to_numpy(float)
    if pista:
        quadro = colunas_pista(df)
        for nome in quadro.columns:
            X[nome] = quadro[nome].array  # posicional; `pista_cfg` continua categórica
    if retencao:
        for nome, valores in colunas_retencao(df).items():
            X[nome] = np.asarray(valores, float)
    if fe_auto:
        for nome, valores in fe_auto_mod.colunas(df).items():
            X[nome] = np.asarray(valores, float)
    if roma is not None:
        for nome in roma_tdg_mod.COLS:
            X[nome] = np.asarray(roma[nome], float)
        X["roma_t_menos_pred"] = X["roma_t_hat"].to_numpy(float) - X["pred"].to_numpy(float)
    faltando = [c for c in exigir if c not in X.columns]
    if faltando:
        raise SystemExit(f"colunas que o corretor não tem: {faltando}")
    return X.drop(columns=[c for c in sem if c in X.columns])


def distancias_plano(df: pd.DataFrame, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Distância da previsão a cada `to_takeoff_from_*` planejado, a menor delas e quantos
    planos existem (e quantos passam de 1 h)."""
    gaps = df.reindex(columns=[f"to_takeoff_from_{c}" for c in F.PLAN_REFS]).to_numpy(float)
    d = gaps - np.asarray(pred, float)[:, None]
    out = {f"dist_{c.split('_')[0].lower()}": d[:, i] for i, c in enumerate(F.PLAN_REFS)}
    absd = np.where(np.isnan(d), np.inf, np.abs(d)).min(axis=1)
    out["dist_min"] = np.where(np.isfinite(absd), absd, np.nan)
    out["n_planos"] = np.isfinite(gaps).sum(1).astype(float)
    out["n_planos_longos"] = (gaps > 3600).sum(1).astype(float)
    return out


class Conjunto:
    """Média simples de três (ou quatro) corretores sobre as mesmas entradas.

    (1) LightGBM global; (2) um LightGBM por aeroporto (aeroporto sem modelo próprio cai
    no global); (3) CatBoost com as categóricas em texto; (4) com `--corretor-xgb`,
    XGBoost na GPU.
    """

    def __init__(self, global_, aeroportos: dict, catboost, xgb=None) -> None:
        self.global_, self.aeroportos, self.catboost = global_, aeroportos, catboost
        self.xgb = xgb

    def _por_aeroporto(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.global_.predict(X), float)  # aeroporto novo usa o global
        aero = X[F.AIRPORT].astype(str).to_numpy()
        for nome, modelo in self.aeroportos.items():
            sel = aero == nome
            if sel.any():
                out[sel] = modelo.predict(X[sel])
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        glob = np.asarray(self.global_.predict(X), float)
        cat = np.asarray(self.catboost.predict(catboost_frame(X)), float)
        partes = [glob, self._por_aeroporto(X), cat]
        if self.xgb is not None:
            partes.append(np.asarray(self.xgb.inplace_predict(X), float))
        return np.mean(partes, axis=0)


def colunas_cat(X: pd.DataFrame) -> list[str]:
    """Colunas categóricas do quadro: o aeroporto e, com `--plano13`, a companhia."""
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Entradas do CatBoost: as mesmas do LightGBM, com as categóricas em texto.

    A cópia é rasa: só as duas ou três colunas categóricas são reescritas, e o
    copy-on-write garante que `X` não muda. A cópia profunda de antes duplicava o quadro
    inteiro do corretor (~1,4 GB nas cegas) para trocar duas colunas.
    """
    Xc = X.copy(deep=False)
    for col in colunas_cat(X):  # no pandas 3 o astype(str) mantém o NaN: vira o texto "nan", como antes
        Xc[col] = Xc[col].astype(str).fillna("nan")
    return Xc


def fit_catboost(X: pd.DataFrame, alvo: np.ndarray, peso: np.ndarray | None = None):
    """CatBoost determinístico na correção; GPU quando há placa, senão 12 threads."""
    from catboost import CatBoostRegressor, utils

    params = dict(CATBOOST, verbose=0)
    if utils.get_gpu_device_count() > 0:
        params["task_type"] = "GPU"
    else:
        params["thread_count"] = 12
    modelo = CatBoostRegressor(**params)
    modelo.fit(catboost_frame(X), alvo, cat_features=colunas_cat(X), sample_weight=peso)
    return modelo


XGB_CORRETOR = dict(tree_method="hist", device="cuda", grow_policy="lossguide", max_leaves=63,
                    max_depth=0, learning_rate=0.05, min_child_weight=200, max_bin=256,
                    max_cat_to_onehot=8, objective="reg:squarederror", seed=0)


def fit_xgb(X: pd.DataFrame, alvo: np.ndarray, peso: np.ndarray | None = None):
    """XGBoost na GPU com os mesmos parâmetros do LightGBM global (folhas, taxa, rodadas)."""
    import xgboost as xgb

    d = xgb.QuantileDMatrix(X, alvo, weight=peso, enable_categorical=True,
                            max_bin=XGB_CORRETOR["max_bin"])
    return xgb.train(XGB_CORRETOR, d, ROUNDS)


def fit_corrector(X: pd.DataFrame, y: np.ndarray, base: np.ndarray,
                  conjunto: bool = False, xgb: bool = False,
                  rounds: int = ROUNDS, params: dict | None = None,
                  peso: np.ndarray | None = None) -> lgb.Booster | Conjunto:
    """Corretor L2 na diferença entre o alvo e a previsão da base.

    Com `conjunto`, devolve a média de três corretores sobre as mesmas entradas; com `xgb`
    também, soma um quarto (XGBoost na GPU). `rounds` vale só para os LightGBM, e `params`
    sobrescreve os parâmetros deles (o CatBoost fica intocado).

    `peso` (uma linha a uma linha de `X`, `--pseudo`) entra em todos os motores: é como as
    linhas de 2026 pesam menos que as de 2025.
    """
    alvo = np.asarray(y, float) - np.asarray(base, float)
    peso = None if peso is None else np.asarray(peso, float)
    pg = lgb_params({**PARAMS, **(params or {})})
    pa = lgb_params({**PARAMS_AEROPORTO, **(params or {})})
    global_ = lgb.train(pg, lgb.Dataset(X, alvo, weight=peso), rounds)
    global_.free_dataset()  # histograma binado: não serve para prever
    memoria.soltar()
    if not conjunto:
        return global_
    aeroportos = {}
    aero = X[F.AIRPORT].astype(str).to_numpy()
    for nome in np.unique(aero):
        sel = aero == nome
        dados = lgb.Dataset(X[sel], alvo[sel], weight=None if peso is None else peso[sel])
        aeroportos[nome] = lgb.train(pa, dados, rounds)
        aeroportos[nome].free_dataset()
    memoria.soltar()  # o CatBoost monta a matriz dele do zero: entra com o heap limpo
    return Conjunto(global_, aeroportos, fit_catboost(X, alvo, peso),
                    fit_xgb(X, alvo, peso) if xgb else None)


def apply_corrector(model: lgb.Booster | Conjunto, X: pd.DataFrame, base: np.ndarray,
                    df: pd.DataFrame | None = None) -> np.ndarray:
    """Base mais a correção, com piso 0; com `df`, projetada antes na janela do LOBT."""
    pred = np.asarray(base, float) + model.predict(X)
    return limitar_janela(pred, df) if df is not None else np.clip(pred, 0, None)


def linhas_de_regra(df: pd.DataFrame, cfg_base: dict) -> np.ndarray:
    """As linhas de `df` que a base entrega a uma regra fixa, pelos limiares da própria base.

    Hoje o corretor trata essas linhas como qualquer outra: ele aprende nelas e corrige a
    previsão da reta, e só depois — no envio — a regra de Roma reescreve as dela
    (`train.media_membros` → `pos_regras.aplicar`). Com `--corretor-sem-regra` elas saem do
    treino e ficam com a previsão da base intacta.
    """
    return models_linhas_de_regra(df, float(cfg_base.get("nm_min_ms", 0) or 0))


def previsao_corrigida(model: lgb.Booster | Conjunto, df: pd.DataFrame, base: np.ndarray,
                       adsb: bool, janela: bool, sem: Iterable[str] = (),
                       externos: dict | None = None,
                       plano13: pd.DataFrame | None = None, dist_plano: bool = False,
                       sem_ctx: bool = False, superficie: bool = False,
                       mapa: bool = False, ref_cel: pd.DataFrame | None = None,
                       pista: bool = False, retencao: bool = False,
                       roma: pd.DataFrame | None = None,
                       regra: np.ndarray | None = None, fe_auto: bool = False) -> np.ndarray:
    """Previsão dos voos de `df` corrigida: com `janela`, dentro da janela do LOBT.

    `regra` (de `linhas_de_regra`, `--corretor-sem-regra`) marca as linhas em que a
    previsão da base é mantida como está, sem correção.
    """
    X = corrector_frame(df, base, adsb, janela, sem, externos, plano13, dist_plano, sem_ctx,
                        superficie, mapa, ref_cel, pista, retencao, roma, fe_auto)
    pred = apply_corrector(model, X, base, df if janela else None)
    if regra is not None:
        pred[regra] = np.asarray(base, float)[regra]
    return pred


def day_folds(days: np.ndarray, k: int = FOLDS) -> np.ndarray:
    """Fold de cada linha: dias ordenados distribuídos em rodízio (um dia inteiro por fold)."""
    uniq = np.array(sorted(set(days)))
    return pd.Series(np.arange(uniq.size) % k, index=uniq)[days].to_numpy()


def oof_correction(X: pd.DataFrame, y: np.ndarray, base: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(y))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        model = fit_corrector(X[tr], y[tr], base[tr])
        out[te] = apply_corrector(model, X[te], base[te])
    return out


def na_ordem(df: pd.DataFrame, ids) -> pd.DataFrame:
    """As linhas de `df` na ordem de `ids`, esvaziando `df` coluna a coluna.

    Mesmo resultado de `df.set_index(ID).loc[ids].reset_index()` — que mantinha o quadro
    original e a reordenação vivos ao mesmo tempo (2 × 1,7 GB nas cegas). Aqui só uma
    coluna existe em duplicata por vez; `df` fica vazio no fim.
    """
    chaves = pd.Index(df[F.ID])
    if chaves.has_duplicates:
        raise ValueError(f"{F.ID} repetido no quadro a reordenar")
    onde = chaves.get_indexer(pd.Index(np.asarray(ids)))
    if (onde < 0).any():
        raise KeyError(f"{int((onde < 0).sum())} ids fora do quadro a reordenar")
    ordem = [F.ID, *[c for c in df.columns if c != F.ID]]  # como o set_index/reset_index
    return pd.DataFrame({c: df.pop(c).array.take(onde) for c in ordem}, copy=False)


def holdout_da_base(base: pd.DataFrame) -> pd.DataFrame:
    """Holdout na mesma ordem das previsões da corrida base."""
    return na_ordem(load_split("holdout2025"), base[F.ID].to_numpy())


def simulacao_folds(
    run: Run, base_id: str, adsb: bool, sem: Iterable[str] = ()
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor em 5 folds por dia dentro do holdout; devolve (holdout, base, previsão)."""
    with run.phase("dados", 0.3):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        X = corrector_frame(hold, base["pred"].to_numpy(float), adsb, sem=sem)
        if adsb and "adsb_taxi" in X:  # a ablação --corretor-sem-feature adsb_taxi tira a coluna
            run.log(f"adsb: {X['adsb_taxi'].notna().mean():.1%} dos voos com evento")
    with run.phase("treino", 0.6):
        pred = oof_correction(X, hold[TRUTH].to_numpy(float), base["pred"].to_numpy(float),
                              day_folds(base["dia"].to_numpy()))
    return hold, base, pred


def tabelas_celula(train: pd.DataFrame, run: Run | None = None):
    """Tabelas de célula do `refcel`: por bloco de meses (fora dele) e do treino inteiro.

    O corretor aprende nas cegas com a previsão de uma base que não viu o mês delas; as
    estatísticas de célula seguem a mesma regra, com os mesmos blocos de dois meses do
    `crossfit.oof_base`. Holdout e ranking usam o treino inteiro, que não os contém.
    """
    mes = train["MVT_TIME_UTC_mvt"].dt.month
    bloco_do_mes = month_blocks(mes)
    meses_por_bloco: dict[int, list[int]] = {}
    for m, k in bloco_do_mes.items():
        meses_por_bloco.setdefault(k, []).append(m)
    # `refcel.ajustar` só lê as três colunas da chave: recortar antes evita copiar o treino
    # inteiro (10/12 das linhas, ~1,2 GB) uma vez por bloco.
    chaves = train[list(dict.fromkeys(c for chave in refcel.CHAVES for c in chave))]
    tabs_bloco = {}
    for k, meses in sorted(meses_por_bloco.items()):
        fora = ~mes.isin(meses)
        tabs_bloco[k] = refcel.ajustar(chaves[fora], train.loc[fora, F.TARGET])
    if run:
        run.log(f"células: {len(tabs_bloco)} blocos · "
                f"{len(tabs_bloco[0][0]):,} (aeroporto, stand, pista) no bloco 1")
    return tabs_bloco, refcel.ajustar(chaves, train[F.TARGET]), bloco_do_mes


def modelos_roma(train: pd.DataFrame, run: Run | None = None):
    """Modelos de `G` do `roma`: por bloco de meses (fora dele) e do treino inteiro.

    Mesma regra do `tabelas_celula`: as cegas usam o modelo que não viu o mês delas, e
    holdout e ranking usam o treino inteiro, que não os contém.
    """
    mes = train["MVT_TIME_UTC_mvt"].dt.month
    bloco_do_mes = month_blocks(mes)
    meses_por_bloco: dict[int, list[int]] = {}
    for m, k in bloco_do_mes.items():
        meses_por_bloco.setdefault(k, []).append(m)
    # Só as linhas do LIRF entram no modelo: recortar antes evita copiar o treino inteiro.
    lirf = roma_tdg_mod.linhas(train)
    cols = [F.AIRPORT, roma_tdg_mod.MS, *roma_tdg_mod.CAT,
            *roma_tdg_mod.colunas_numericas(train)]
    sub = train.loc[lirf, [c for c in dict.fromkeys(cols) if c in train.columns]]
    alvo = train.loc[lirf, F.TARGET]
    mes_sub = mes[lirf]
    mods = {}
    for k, meses in sorted(meses_por_bloco.items()):
        fora = ~mes_sub.isin(meses)
        mods[k] = roma_tdg_mod.ajustar(sub[fora], alvo[fora])
    if run:
        run.log(f"roma: {int(lirf.sum()):,} partidas do LIRF no treino · {len(mods)} blocos")
    return mods, roma_tdg_mod.ajustar(sub, alvo), bloco_do_mes


def pseudo_2026(model, rk: pd.DataFrame, pred_rk: np.ndarray, X_rk: pd.DataFrame, fonte: str,
                corte: float, regra_rk: np.ndarray | None, janela: bool,
                run: Run | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Linhas de 2026 que entram no treino do corretor: `(entradas, alvo, previsão da base)`.

    `fonte` escolhe o professor: `propria` é o corretor `model` desta corrida (treinado só
    nos meses de treino, sem jan/jul — o holdout segue limpo); `campea` é o último arquivo
    de `submissions/`, que treinou no `full2025` e **viu o holdout**. Só o corpo entra
    (`pseudo.corpo`); ver `docs/research/2026-10-03-pseudo-rotulo.md`.
    """
    if fonte == "propria":
        alvo = apply_corrector(model, X_rk, pred_rk, rk if janela else None)
        if regra_rk is not None:
            alvo[regra_rk] = np.asarray(pred_rk, float)[regra_rk]
    else:
        alvo = pseudo_mod.alvo_da_submissao(rk[F.ID].to_numpy())
    pred_rk = np.asarray(pred_rk, float)
    corpo = pseudo_mod.corpo(alvo, rk["nm_missing"], regra_rk, corte, pred_rk)
    if run:
        medio = float(np.mean(alvo[corpo])) if corpo.any() else float("nan")
        run.log(f"pseudo ({fonte}): {int(corpo.sum()):,} de {len(rk):,} partidas de 2026 no "
                f"corpo (alvo ≤ {corte:.0f} s, com plano NM e com base) · alvo médio "
                f"{medio:.1f} s")
    return X_rk[corpo], alvo[corpo], pred_rk[corpo]


def simulacao_crossfit(
    run: Run, base_id: str, cfg_base: dict, adsb: bool, conjunto: bool = False,
    sem: Iterable[str] = (), externos: bool = False, plano13: bool = False,
    fila: bool | str = False, dist_plano: bool = False, sem_ctx: bool = False,
    reusar_oof: str | None = None, corretor_xgb: bool = False, superficie: bool = False,
    rounds: int = ROUNDS, mapa: bool = False, ref_cel: bool = False, params: dict | None = None,
    pista: bool = False, retencao: bool = False, sem_regra: bool = False,
    roma_tdg: bool = False, sem_cor: Iterable[str] = (), fe_auto: bool = False,
    pseudo: str | None = None, pseudo_peso: float = pseudo_mod.PESO,
    pseudo_corte: float = pseudo_mod.CORTE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Corretor treinado no ano fora do bloco; devolve (holdout, base, previsão).

    Com `reusar_oof`, a previsão fora do bloco vem pronta daquela corrida (mesma base e
    mesma `base_config`), e só o corretor é refeito. `sem_cor` (`--corretor-sem-feature`)
    sai só das entradas do corretor, por isso não mexe na `base_config` nem no `reusar_oof`.

    Com `pseudo`, o corretor é ajustado duas vezes: a primeira só nas cegas (é ela que gera
    o alvo das linhas de 2026 quando a fonte é `propria`) e a segunda nas cegas mais as
    partidas de 2026, com peso `pseudo_peso`. O holdout não entra no treino em momento
    algum.
    """
    sem = (*sem, *sem_cor)
    janela = bool(cfg_base.get("janela_lobt"))
    caminho_pronto = oof_reusado(reusar_oof, base_id, cfg_base) if reusar_oof else None
    with run.phase("dados", 0.1):
        base = pd.read_parquet(RUNS / f"{base_id}.parquet")
        hold = holdout_da_base(base)
        train, blind = load_split("train2025"), load_split("blind2025")
        rk = None if caminho_pronto and not pseudo else load_split("ranking2026")
        cias = vocabulario(train) if plano13 else None  # vocabulário fixo do treino
        tabs_bloco, tabs_todos, bloco_do_mes = tabelas_celula(train, run) if ref_cel else (None, None, None)
        roma_bloco, roma_todos, roma_bloco_do_mes = modelos_roma(train, run) if roma_tdg else (None, None, None)
        run.log(f"treino {len(train):,} · cegas {len(blind):,} · holdout {len(hold):,}")
    with run.phase("base fora do bloco", 0.7):
        if caminho_pronto:
            oof = pd.read_parquet(caminho_pronto)
            caminho_oof = caminho_pronto
            run.log(f"oof reaproveitado de {reusar_oof}: {caminho_pronto.name}")
        else:
            oof = oof_base(cfg_base, train, blind, rk, run)
            caminho_oof = RUNS / f"{run.id}_oof.parquet"
            oof.to_parquet(caminho_oof, index=False)
        pred_rk = None
        if pseudo:
            # a base de 2026 sai de um ajuste no train2025 (10 meses): nenhum rótulo de
            # jan/jul entra no pseudo-rótulo (docs/research/2026-10-03-pseudo-rotulo.md)
            pred_rk = pseudo_mod.base_2026(cfg_base, rk, run, train)
        else:
            rk = None
        del train
        memoria.soltar()  # ~1,4 GB do treino: sai do heap antes do pico do corretor
        run.set(oof=str(caminho_oof.relative_to(ROOT)))
        run.log(f"fora do bloco: {len(oof):,} previsões · rmse {rmse(oof[TRUTH], oof['pred']):.2f}")
    with run.phase("corretor", 0.1):
        pred_oof = oof["pred"].to_numpy(float)
        cegas = na_ordem(blind, oof[F.ID].to_numpy())  # mesma ordem do oof; esvazia `blind`
        del blind
        memoria.soltar()
        ext_cegas = ext_hold = None
        if externos:
            # meses do oof: os 10 do treino, nunca jan/jul — nem as cegas nem o holdout
            # veem a taxa de cópia do próprio mês.
            meses = sorted({int(m) for m in oof["mes"]})
            copia = copia_cia_2025(run)
            ext_cegas, ext_hold = (colunas_ext(d, copia, meses) for d in (cegas, hold))
            run.log(f"externos: meses de treino da taxa de cópia {meses}")
        p13_cegas = p13_hold = None
        if plano13:
            p13_cegas, p13_hold = (colunas_p13(d, cias, com_fila=fila) for d in (cegas, hold))
            run.log(f"plano 13: {len(cias)} companhias no vocabulário · rotação em "
                    f"{p13_cegas['rot_idade'].notna().mean():.1%} das cegas")
        cel_cegas = cel_hold = None
        if ref_cel:
            bloco = pd.Series(oof["mes"].to_numpy()).map(bloco_do_mes).to_numpy()
            cel_cegas = refcel.aplicar_por_bloco(cegas, bloco, tabs_bloco)
            cel_hold = refcel.aplicar(hold, tabs_todos)
            run.log(f"células: {cel_cegas['cel_p50'].notna().mean():.1%} das cegas · "
                    f"nível 0 em {(cel_hold['cel_nivel'] == 0).mean():.1%} do holdout")
        roma_cegas = roma_hold = None
        if roma_tdg:
            bloco = pd.Series(oof["mes"].to_numpy()).map(roma_bloco_do_mes).to_numpy()
            roma_cegas = roma_tdg_mod.aplicar_por_bloco(cegas, bloco, roma_bloco)
            roma_hold = roma_tdg_mod.aplicar(hold, roma_todos)
            run.log(f"roma: {roma_cegas['roma_t_hat'].notna().mean():.2%} das cegas · "
                    f"{roma_hold['roma_t_hat'].notna().mean():.2%} do holdout")
        regra_cegas = linhas_de_regra(cegas, cfg_base) if sem_regra else None
        X_oof = corrector_frame(cegas, pred_oof, adsb, janela, sem, ext_cegas, p13_cegas,
                                dist_plano, sem_ctx, superficie, mapa, cel_cegas, pista, retencao,
                                roma_cegas, fe_auto, exigir=sem_cor)
        del cegas
        memoria.soltar()
        if adsb and "adsb_taxi" in X_oof:  # idem
            run.log(f"adsb no treino do corretor: {X_oof['adsb_taxi'].notna().mean():.1%}")
        alvo_oof, treino_oof = oof[TRUTH].to_numpy(float), pred_oof
        if regra_cegas is not None:
            run.log(f"corretor sem regra: {int(regra_cegas.sum()):,} de {len(X_oof):,} cegas fora")
            X_oof, alvo_oof, treino_oof = (X_oof[~regra_cegas], alvo_oof[~regra_cegas],
                                           pred_oof[~regra_cegas])
        model = fit_corrector(X_oof, alvo_oof, treino_oof, conjunto, corretor_xgb,
                              rounds, params)
        if pseudo:
            ext_rk = colunas_ext(rk, copia, meses) if externos else None
            p13_rk = colunas_p13(rk, cias, com_fila=fila) if plano13 else None
            cel_rk = refcel.aplicar(rk, tabs_todos) if ref_cel else None
            roma_rk = roma_tdg_mod.aplicar(rk, roma_todos) if roma_tdg else None
            regra_rk = linhas_de_regra(rk, cfg_base) if sem_regra else None
            X_rk = corrector_frame(rk, pred_rk, adsb, janela, sem, ext_rk, p13_rk, dist_plano,
                                   sem_ctx, superficie, mapa, cel_rk, pista, retencao, roma_rk,
                                   fe_auto)
            X_ps, alvo_ps, base_ps = pseudo_2026(model, rk, pred_rk, X_rk, pseudo, pseudo_corte,
                                                 regra_rk, janela, run)
            del X_rk, rk, model, ext_rk, p13_rk, cel_rk, roma_rk
            memoria.soltar()
            X_oof, alvo_oof, treino_oof, pesos = pseudo_mod.juntar(
                X_oof, alvo_oof, treino_oof, X_ps, alvo_ps, base_ps, pseudo_peso)
            del X_ps
            memoria.soltar()
            run.log(f"pseudo: treino do corretor com {len(X_oof):,} linhas · peso de 2026 "
                    f"{pesos[pesos < 1].sum() / pesos.sum():.1%} do total")
            model = fit_corrector(X_oof, alvo_oof, treino_oof, conjunto, corretor_xgb,
                                  rounds, params, pesos)
        del X_oof
        memoria.soltar()
        pred_base = base["pred"].to_numpy(float)
        regra_hold = linhas_de_regra(hold, cfg_base) if sem_regra else None
        pred = previsao_corrigida(model, hold, pred_base, adsb, janela, sem, ext_hold, p13_hold,
                                  dist_plano, sem_ctx, superficie, mapa, cel_hold, pista, retencao,
                                  roma_hold, regra_hold, fe_auto)
    return hold, base, pred


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nome")
    ap.add_argument("--base", help="id da corrida base (padrão: campeã)")
    ap.add_argument("--crossfit", action="store_true",
                    help="corretor treinado no ano, nas previsões da base fora do bloco")
    ap.add_argument("--sem-adsb", action="store_true", help="controle: mesmo empilhamento sem adsb_*")
    ap.add_argument("--seeds", type=int, default=1,
                    help="--crossfit: base de cada bloco é a média de N seeds")
    ap.add_argument("--conjunto", action="store_true",
                    help="--crossfit: corretor = média de global, por aeroporto e CatBoost")
    ap.add_argument("--sem-feature", action="append", default=[], metavar="COLUNA",
                    help="tira a coluna da base e do corretor (pode repetir)")
    ap.add_argument("--corretor-sem-feature", action="append", default=[], metavar="COLUNA",
                    help="--crossfit: tira a coluna só do corretor (pode repetir); a base "
                         "fica igual, então `--reusar-oof` continua valendo")
    ap.add_argument("--externos", action="store_true",
                    help="--crossfit: soma as colunas ext_* (companhia, séries diárias, OPDI)")
    ap.add_argument("--plano13", action="store_true",
                    help="--crossfit: soma METAR, rotação no stand, consistência NM e a companhia")
    ap.add_argument("--fila", action="store_true",
                    help="--plano13: soma contagens de fila, pista e pátio do stand (diagnóstico de Roma)")
    ap.add_argument("--stand-prefixo", action="store_true",
                    help="--plano13: soma só a coluna stand_p do bloco --fila (laço fiel 2)")
    ap.add_argument("--corretor-rounds", type=int, default=ROUNDS, metavar="N",
                    help=f"rodadas dos LightGBM do corretor (padrão {ROUNDS}; CatBoost igual)")
    ap.add_argument("--dist-plano", action="store_true",
                    help="--crossfit: soma a distância da previsão a cada horário planejado")
    ap.add_argument("--mapa", action="store_true",
                    help="soma ao corretor a distância de táxi stand → cabeceira (src/mapa.py)")
    ap.add_argument("--superficie", action="store_true",
                    help="--crossfit: soma as colunas sup_* (aviões no solo no push estimado)")
    ap.add_argument("--corretor-sem-regra", action="store_true",
                    help="--crossfit: o corretor não treina nas linhas que a base entrega a uma "
                         "regra fixa (reta dos sem NM e regra de Roma) e deixa a previsão da base "
                         "intacta nelas")
    ap.add_argument("--pista", action="store_true",
                    help="--crossfit: soma as colunas pista_* (configuração de pista, fila com "
                         "peso de esteira e cadência das decolagens, src/pista.py)")
    ap.add_argument("--retencao", action="store_true",
                    help="--crossfit: soma as colunas ret_* (fila de portão no AOBT_3: partidas "
                         "vencidas sem push, ativas na pista, atraso recente e EWMA, src/pista.py)")
    ap.add_argument("--roma-tdg", action="store_true",
                    help="--crossfit: soma as colunas roma_* (atraso de portão previsto no LIRF e "
                         "o táxi que T = D − G reconstrói com ele, src/roma.py)")
    ap.add_argument("--fe-auto", action="store_true",
                    help="--crossfit: soma as colunas fe_* (features geradas automaticamente "
                         "e aprovadas fora do mês, src/fe_auto.py)")
    ap.add_argument("--corretor-sem-ctx", action="store_true",
                    help="--crossfit: tira ctx_* do corretor (a base já usa, --base-ctx)")
    ap.add_argument("--reusar-oof", metavar="ID",
                    help="--crossfit: usa o oof já gravado por essa corrida (mesma base)")
    ap.add_argument("--corretor-xgb", action="store_true",
                    help="--conjunto: soma um 4º corretor, XGBoost na GPU")
    ap.add_argument("--corretor-ref", action="store_true",
                    help="--crossfit: soma as colunas cel_* (mediana, P90, desvio e tamanho da "
                         "célula aeroporto × stand × pista, src/refcel.py)")
    ap.add_argument("--corretor-params", type=json.loads, metavar="JSON",
                    help="--crossfit: sobrescreve parâmetros dos LightGBM do corretor")
    ap.add_argument("--pseudo", choices=pseudo_mod.FONTES,
                    help="--crossfit: põe as partidas de 2026 no treino do corretor, com alvo "
                         "previsto por `propria` (corretor desta corrida, sem jan/jul) ou "
                         "`campea` (último envio, que viu o holdout: marca pseudo_vazado)")
    ap.add_argument("--pseudo-peso", type=float, default=pseudo_mod.PESO, metavar="W",
                    help=f"--pseudo: peso das linhas de 2026 (padrão {pseudo_mod.PESO})")
    ap.add_argument("--pseudo-corte", type=float, default=pseudo_mod.CORTE, metavar="S",
                    help=f"--pseudo: alvo máximo do corpo em segundos (padrão {pseudo_mod.CORTE:.0f})")
    ap.add_argument("--nota", default="")
    return ap


def config_da_corrida(a: argparse.Namespace, base_id: str) -> dict:
    """Config gravada no registro: com `--conjunto`, a chave `corretor`.

    Com `--sem-feature`, `sem_features` no topo (corretor) e na `base_config` (blocos);
    com `--externos`, `externos: true`; com `--plano13`, `plano13: true`.
    """
    adsb = not a.sem_adsb
    sem = list(a.sem_feature)
    if not a.crossfit:
        cfg = {"model": "stack", "base": base_id, "adsb": adsb, "folds": FOLDS,
               "rounds": ROUNDS, "seed": PARAMS["seed"]}
        if sem:
            cfg["sem_features"] = sem
        return cfg
    cfg_base = config_da_base(base_id, a.seeds)
    if sem:
        cfg_base = {**cfg_base, "sem_features": sem}
    cfg = {"model": "stack_cf", "base": base_id,
           "base_config": cfg_base,
           "adsb": adsb, "rounds": a.corretor_rounds, "seed": PARAMS["seed"]}
    if a.conjunto:
        cfg["corretor"] = "conjunto"
    if a.externos:
        cfg["externos"] = True
    if a.plano13:
        cfg["plano13"] = True
    if a.fila:
        cfg["fila"] = True
    elif a.stand_prefixo:
        cfg["fila"] = "stand"
    if a.dist_plano:
        cfg["dist_plano"] = True
    if a.superficie:
        cfg["superficie"] = True
    if a.corretor_sem_regra:
        cfg["corretor_sem_regra"] = True
    if a.pista:
        cfg["pista"] = True
    if a.retencao:
        cfg["retencao"] = True
    if a.roma_tdg:
        cfg["roma_tdg"] = True
    if a.fe_auto:
        cfg["fe_auto"] = True
    if a.mapa:
        cfg["mapa"] = True
    if a.corretor_sem_ctx:
        cfg["corretor_sem_ctx"] = True
    if a.corretor_xgb:
        cfg["corretor_xgb"] = True
    if a.corretor_ref:
        cfg["ref_cel"] = True
    if a.corretor_params:
        cfg["corretor_params"] = a.corretor_params
    if a.reusar_oof:
        cfg["reusar_oof"] = a.reusar_oof
    if a.pseudo:
        cfg["pseudo"] = a.pseudo
        cfg["pseudo_peso"] = a.pseudo_peso
        cfg["pseudo_corte"] = a.pseudo_corte
        if pseudo_mod.vazado(a.pseudo):
            cfg["pseudo_vazado"] = True
    if sem:
        cfg["sem_features"] = sem
    if a.corretor_sem_feature:
        cfg["corretor_sem_features"] = list(a.corretor_sem_feature)
    return cfg


def main() -> None:
    ap = parser()
    a = ap.parse_args()
    if a.seeds > 1 and not a.crossfit:
        ap.error("--seeds só vale com --crossfit (a base do holdout vem pronta em --base)")
    if a.conjunto and not a.crossfit:
        ap.error("--conjunto só vale com --crossfit")
    if a.sem_feature and not a.crossfit:
        ap.error("--sem-feature só vale com --crossfit (nos folds nada confere o nome)")
    if a.corretor_sem_feature and not a.crossfit:
        ap.error("--corretor-sem-feature só vale com --crossfit (nos folds nada confere o nome)")
    if a.externos and not a.crossfit:
        ap.error("--externos só vale com --crossfit (os folds não têm meses de treino separados)")
    if a.corretor_xgb and not a.conjunto:
        ap.error("--corretor-xgb só vale com --conjunto")
    if (a.fila or a.stand_prefixo) and not a.plano13:
        ap.error("--fila e --stand-prefixo só valem com --plano13")
    if a.fila and a.stand_prefixo:
        ap.error("--stand-prefixo já é parte de --fila")
    if a.corretor_rounds != ROUNDS and not a.crossfit:
        ap.error("--corretor-rounds só vale com --crossfit")
    if (a.dist_plano or a.corretor_sem_ctx) and not a.crossfit:
        ap.error("--dist-plano e --corretor-sem-ctx só valem com --crossfit")
    if a.mapa and not a.crossfit:
        ap.error("--mapa só vale com --crossfit")
    if a.pista and not a.crossfit:
        ap.error("--pista só vale com --crossfit (os folds não têm as cegas do ano)")
    if a.superficie and not a.crossfit:
        ap.error("--superficie só vale com --crossfit (os folds não têm as cegas do ano)")
    if a.corretor_sem_regra and not a.crossfit:
        ap.error("--corretor-sem-regra só vale com --crossfit (os folds não têm as cegas do ano)")
    if a.retencao and not a.crossfit:
        ap.error("--retencao só vale com --crossfit (os folds não têm as cegas do ano)")
    if a.roma_tdg and not a.crossfit:
        ap.error("--roma-tdg só vale com --crossfit (os folds não têm meses de treino separados)")
    if a.fe_auto and not a.crossfit:
        ap.error("--fe-auto só vale com --crossfit (os folds não têm as cegas do ano)")
    if a.plano13 and not a.crossfit:
        ap.error("--plano13 só vale com --crossfit (os folds não têm vocabulário de treino)")
    if a.reusar_oof and not a.crossfit:
        ap.error("--reusar-oof só vale com --crossfit (só ele calcula a base fora do bloco)")
    if a.corretor_ref and not a.crossfit:
        ap.error("--corretor-ref só vale com --crossfit (os folds não têm meses de treino separados)")
    if a.corretor_params and not a.crossfit:
        ap.error("--corretor-params só vale com --crossfit")
    if a.pseudo and not a.crossfit:
        ap.error("--pseudo só vale com --crossfit (os folds não têm base de 2026)")
    if (a.pseudo_peso, a.pseudo_corte) != (pseudo_mod.PESO, pseudo_mod.CORTE) and not a.pseudo:
        ap.error("--pseudo-peso e --pseudo-corte só valem com --pseudo")
    base_id = a.base or campeao.principal(campeao.carregar())["base"]
    adsb = not a.sem_adsb
    cfg = config_da_corrida(a, base_id)

    with Run(a.nome, cfg) as run:
        run.set(nota=a.nota)
        sem = cfg.get("sem_features", ())
        if a.crossfit:
            hold, base, pred = simulacao_crossfit(run, base_id, cfg["base_config"], adsb,
                                                  a.conjunto, sem, a.externos, a.plano13,
                                                  cfg.get("fila", False), a.dist_plano, a.corretor_sem_ctx,
                                                  a.reusar_oof, a.corretor_xgb, a.superficie,
                                                  a.corretor_rounds, a.mapa, a.corretor_ref,
                                                  a.corretor_params, a.pista, a.retencao,
                                                  a.corretor_sem_regra, a.roma_tdg,
                                                  cfg.get("corretor_sem_features", ()), a.fe_auto,
                                                  a.pseudo, a.pseudo_peso, a.pseudo_corte)
        else:
            hold, base, pred = simulacao_folds(run, base_id, adsb, sem)
        with run.phase("métricas", 0.1):
            m = metrics(hold, pred)
            path = RUNS / f"{run.id}.parquet"
            pd.DataFrame({
                F.ID: base[F.ID].to_numpy(),
                "dia": base["dia"].to_numpy(),
                TRUTH: hold[TRUTH].to_numpy(float),
                "pred": pred,
            }).to_parquet(path, index=False)
        run.metric(**m)
        run.set(previsoes=str(path.relative_to(ROOT)))
        run.log(" · ".join(f"{k} {v}" for k, v in m.items() if not isinstance(v, dict)))


if __name__ == "__main__":
    main()
