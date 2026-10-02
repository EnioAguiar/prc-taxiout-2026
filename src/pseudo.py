"""Pseudo-rotulagem transdutiva: as partidas de 2026 entram no treino do corretor.

O corretor aprende `y − pred_base` nas cegas de 2025. Com `--pseudo` ele ganha também as
partidas do `ranking.parquet` (jan e jul de **2026**, 344.841 linhas), com alvo vindo de um
modelo — nunca de verdade de 2026, que não existe aqui — e peso 0,3. O que se compra é
cobertura: o corretor passa a ver a distribuição de covariáveis de 2026 (cobertura ADS-B
diferente, dias de ruptura de jan/26), que é o problema declarado da deriva.

Duas fontes de alvo, e a diferença entre elas decide o que a medição no holdout vale:

- `propria`: o alvo é a previsão do **próprio corretor desta corrida** (treinado nas cegas
  dos 10 meses de treino) sobre uma base treinada no `train2025`. Nenhum dos dois viu
  rótulo de jan/jul de 2025 — o holdout continua limpo e o número medido é honesto.
- `campea`: o alvo é a previsão da campeã no último arquivo de `submissions/`. Aquele
  modelo treinou no `full2025`, **inclusive jan e jul de 2025, que são o holdout**, e o
  ranking de 2026 é de jan e jul. Medir no holdout uma corrida dessa fonte é medir um
  destilado do próprio holdout: a config sai marcada com `pseudo_vazado: true` e o ganho
  no holdout não decide nada (`docs/research/2026-10-03-pseudo-rotulo.md`).

Só o **corpo** entra: alvo pseudo ≤ 3.600 s e voo com plano NM (`nm_missing == 0`); com
`--corretor-sem-regra`, ficam de fora também as linhas que a base entrega à regra fixa. A
cauda é onde o pseudo-rótulo é pior (ali a previsão é hedge entre táxi e cópia do
planejado) e onde o erro² mora — pseudo-rotulá-la só reforçaria o próprio hedge.

A previsão da base para 2026 (um ajuste no `train2025`, ~10 min) fica em
`data/cache/pseudo/base-<chave>.parquet`; a chave junta a config da base e o código que a
determina (`models.py`, `crossfit.py`, `features.py`, `cache.py`), no espírito do
`train.chave_oof`. O mesmo arquivo serve à simulação e ao envio, então o membro medido e o
membro enviado usam exatamente as mesmas linhas de 2026.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from cache import CACHE, HOLDOUT_MONTHS, load_split
from models import build_model, leaky_columns, prepare
from runlog import ROOT

PASTA = CACHE / "pseudo"
SUBMISSOES = ROOT / "submissions"
CORTE = 3600.0  # corpo: acima disso a previsão já é hedge de cópia
PESO = 0.3
FONTES = ("propria", "campea")
MODULOS = ("models", "crossfit", "features", "cache")  # o que determina a previsão da base
VERSAO = re.compile(r"_v(\d+)\.parquet$")


def chave(cfg_base: dict) -> str:
    """Identidade da previsão de 2026: a config da base e o código que a produz."""
    h = hashlib.sha256(json.dumps(cfg_base, sort_keys=True).encode())
    for nome in MODULOS:
        h.update(nome.encode() + (ROOT / "src" / f"{nome}.py").read_bytes())
    return h.hexdigest()[:16]


def caminho_base(cfg_base: dict, pasta: Path = PASTA) -> Path:
    return pasta / f"base-{chave(cfg_base)}.parquet"


def meses_do_mestre(fonte: str) -> tuple[int, ...]:
    """Meses de 2025 cujos rótulos o modelo que gera o pseudo-rótulo viu."""
    todos = tuple(range(1, 13))
    return todos if fonte == "campea" else tuple(m for m in todos if m not in HOLDOUT_MONTHS)


def vazado(fonte: str) -> bool:
    """A fonte do alvo viu o holdout? Então o holdout não mede mais essa corrida."""
    return bool(set(meses_do_mestre(fonte)) & HOLDOUT_MONTHS)


def base_2026(cfg_base: dict, rk: pd.DataFrame, run=None, train: pd.DataFrame | None = None,
              pasta: Path = PASTA) -> np.ndarray:
    """Previsão da base para o ranking, de um modelo que só viu os 10 meses do `train2025`.

    Guardada por `chave(cfg_base)`: a segunda corrida com a mesma base não paga o ajuste.
    `train` é só para reaproveitar o quadro de quem já o carregou — é sempre o `train2025`,
    e tem que ser: um ajuste no `full2025` poria jan/jul de 2025 dentro do pseudo-rótulo e
    contaminaria o holdout de toda corrida que lesse este arquivo depois.
    Como `prepare` escreve colunas em `train` e `rk`, chame depois do oof fora do bloco.
    """
    destino = caminho_base(cfg_base, pasta)
    if destino.exists():
        pronto = pd.read_parquet(destino)
        if run:
            run.log(f"pseudo: base de 2026 reaproveitada de {destino.name}")
        return rk[F.ID].map(pd.Series(pronto["pred"].to_numpy(float),
                                      index=pronto[F.ID].to_numpy())).to_numpy(float)
    train = load_split("train2025") if train is None else train
    cols = prepare(train, [rk], cfg_base.get("sem_features", ()),
                   cfg_base.get("base_ctx", False), cfg_base.get("base_p13", False),
                   int(cfg_base.get("cat_max", 0)), cfg_base.get("base_mapa", False),
                   cfg_base.get("base_ret", False), cfg_base.get("base_ext", False))
    cols = [c for c in cols if c not in leaky_columns(train, rk, cols)]
    pred = build_model(cfg_base).fit(train, cols, run=run).predict(rk)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp")
    pd.DataFrame({F.ID: rk[F.ID].to_numpy(), "pred": np.asarray(pred, float)}).to_parquet(
        tmp, index=False)
    tmp.replace(destino)  # arquivo pela metade nunca vira cache válido
    if run:
        run.log(f"pseudo: base de 2026 em {destino.name} ({len(rk):,} partidas)")
    return np.asarray(pred, float)


def ultima_submissao(pasta: Path = SUBMISSOES) -> Path:
    """O arquivo de envio de maior versão (`<time>_vN.parquet`), que é a campeã de hoje."""
    arquivos = [(int(m.group(1)), p) for p in pasta.glob("*_v*.parquet")
                if (m := VERSAO.search(p.name)) and not p.name.endswith("_oof.parquet")]
    if not arquivos:
        raise SystemExit(f"--pseudo campea: nenhum envio em {pasta}")
    return max(arquivos)[1]


def alvo_da_submissao(ids, caminho: Path | None = None) -> np.ndarray:
    """Previsão da campeã para cada id de 2026 (NaN para quem não está no arquivo)."""
    sub = pd.read_parquet(caminho or ultima_submissao())
    por_id = pd.Series(sub[F.TARGET].to_numpy(float), index=sub[F.ID].to_numpy())
    return pd.Series(np.asarray(ids)).map(por_id).to_numpy(float)


def corpo(alvo, nm_missing, regra=None, corte: float = CORTE, base=None) -> np.ndarray:
    """Linhas de 2026 que viram treino: alvo finito, até `corte` e voo com plano NM.

    `regra` (de `stack.linhas_de_regra`, só com `--corretor-sem-regra`) tira as linhas que
    a base entrega à reta dos sem-NM: lá o corretor nem treina nem corrige em 2025.

    `base` é a previsão da base nessas linhas: sem ela finita o corretor aprenderia
    `alvo − NaN` e o ajuste inteiro iria junto. Um voo do ranking sem previsão guardada
    simplesmente não entra.
    """
    alvo = np.asarray(alvo, float)
    ok = np.isfinite(alvo) & (alvo <= corte) & (np.asarray(nm_missing, float) == 0)
    if base is not None:
        ok &= np.isfinite(np.asarray(base, float))
    if regra is not None:
        ok &= ~np.asarray(regra, bool)
    return ok


def alinhar_categorias(X: pd.DataFrame, X_ps: pd.DataFrame) -> pd.DataFrame:
    """As categóricas de 2026 passam a ter as categorias de 2025, na mesma ordem.

    O LightGBM lê categórica pelo código: concatenar dois quadros com vocabulários
    diferentes trocaria os códigos das linhas de 2025 e, com eles, o que o corretor
    prevê no holdout. Categoria que só existe em 2026 vira ausente nas linhas de 2026 —
    o modelo não saberia o que fazer com ela em 2025 de qualquer jeito.
    """
    X_ps = X_ps.copy(deep=False)
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype) and col in X_ps:
            X_ps[col] = X_ps[col].astype("category").cat.set_categories(X[col].cat.categories)
    return X_ps


def juntar(X: pd.DataFrame, y, base, X_ps: pd.DataFrame, y_ps, base_ps,
           peso: float = PESO) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Treino do corretor com as linhas de 2026 no fim e os pesos (1 em 2025, `peso` em 2026).

    Devolve `(X, y, base, pesos)` pronto para o `stack.fit_corrector`, que aprende
    `y − base` com esses pesos.
    """
    X_ps = alinhar_categorias(X, X_ps)[X.columns]
    juntos = pd.concat([X, X_ps], ignore_index=True)
    y = np.concatenate([np.asarray(y, float), np.asarray(y_ps, float)])
    base = np.concatenate([np.asarray(base, float), np.asarray(base_ps, float)])
    pesos = np.concatenate([np.ones(len(X)), np.full(len(X_ps), float(peso))])
    return juntos, y, base, pesos
