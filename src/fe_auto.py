"""Features geradas automaticamente no resíduo da campeã (`--fe-auto`).

Item 3 de `docs/pesquisa/2026-10-02-tecnicas-descoberta.md`: em vez de inventar colunas à
mão, enumerar transformações mecânicas sobre as colunas que o quadro já tem e ficar só com
as que pagam **fora do mês** no resíduo da campeã. É o OpenFE (ICML 2023,
arXiv:2211.12507) refeito aqui: o estágio de avaliação do OpenFE (*FeatureBoost*) treina a
candidata **sobre a previsão que já existe**, que é exatamente o formato do nosso corretor.

Os operadores (`candidatas`):

- `raz`, `dif`, `pro`: razão, diferença e produto de duas numéricas;
- `gmed`, `gdes`: média e desvio de uma numérica dentro de um grupo (`GroupByThenMean`);
- `gdif`, `gz`: a numérica menos a média do grupo, e a mesma diferença em desvios;
- `gpos`: posto da numérica dentro do grupo, em `[0, 1]`;
- `gcont`: tamanho do grupo.

As chaves de grupo estão em `CHAVES`. Duas regras que não são enfeite:

1. **`gcont` só em chave com dia.** O corretor treina nas cegas (10 meses) e é aplicado no
   holdout (2 meses) e no ranking (2026): uma contagem por `aeroporto` conta dez meses num
   quadro e dois no outro — a mesma feature com escala diferente no treino e na aplicação.
   Contagem por `dia × aeroporto` é comparável nos três quadros. Média, desvio e posto não
   dependem do tamanho do recorte, então valem em qualquer chave.
2. **Nada aqui lê o alvo nem a previsão da base.** Só covariáveis. Primeiro porque média de
   alvo por grupo é target encoding, que já existe no `refcel`/`features` com cross-fitting
   (fazê-lo aqui, dentro do quadro, vazaria); segundo porque é o que torna a candidata
   calculável no `ranking.parquet` de 2026 **sem previsão nenhuma** — sem isso não dá para
   medir a AUC adversarial 2025 × 2026 de cada candidata, que é o portão do item 5.

`ESCOLHIDAS` são as que sobreviveram ao estudo de 03/10
(`docs/research/2026-10-03-fe-auto.md`): ganho fora do mês nos **dois** sentidos
(jan→jul e jul→jan), ganho também no corpo (`y ≤ 3600 s`) e AUC adversarial 2025×2026
abaixo de 0,6. `colunas(df)` devolve só essas, e é o que a flag `--fe-auto` soma ao
corretor.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

import features as F

DIA = "__dia"  # chave de dia derivada do MVT (não é coluna do quadro)

# Chaves de grupo. As que começam com `dia_` são as únicas que podem contar linhas.
CHAVES: dict[str, tuple[str, ...]] = {
    "apt_stand": (F.AIRPORT, "STAND_mvt"),
    "apt_hora": (F.AIRPORT, "hour"),
    "apt_rwy": (F.AIRPORT, "RUNWAY_mvt"),
    "cia_apt": ("AIRCRAFT_OPERATOR_flt", F.AIRPORT),
    "tipo_apt": ("AIRCRAFT_TYPE_mvt", F.AIRPORT),
    "dia_apt": (DIA, F.AIRPORT),
    "dia_apt_hora": (DIA, F.AIRPORT, "hour"),
    "dia_apt_rwy": (DIA, F.AIRPORT, "RUNWAY_mvt"),
}
PARES = ("raz", "dif", "pro")
GRUPO = ("gmed", "gdes", "gdif", "gz", "gpos")
# Variante sem deriva de cada agregação de nível: a média do grupo menos a média do quadro
# e o desvio do grupo em desvios do quadro. Dentro de um quadro são a mesma coluna a menos
# de deslocamento/escala — nenhuma árvore distingue as duas, e o ganho medido na peneira
# vale igual —, mas entre quadros diferentes (cegas de 10 meses × holdout × ranking 2026) a
# versão crua carrega o nível do próprio quadro, que é o que a AUC adversarial pega (0,92).
SEM_DERIVA = {"gmed": "gmedc", "gdes": "gdesr"}
EPS = 1e-6


def _dia(df: pd.DataFrame) -> np.ndarray:
    """Dia do movimento (inteiro): o mesmo recorte que a `dia` das corridas."""
    return df["MVT_TIME_UTC_mvt"].to_numpy("datetime64[D]").astype("int64")


def _coluna(df: pd.DataFrame, nome: str) -> pd.Series:
    return pd.Series(_dia(df), index=df.index) if nome == DIA else df[nome]


def codigos(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Código inteiro do grupo, no estilo de `features._key`: texto colado e fatorado."""
    chave = _coluna(df, cols[0]).astype("string").fillna("?")
    for col in cols[1:]:
        chave = chave + "|" + _coluna(df, col).astype("string").fillna("?")
    return pd.factorize(chave, sort=False)[0]


def _n_grupos(cod: np.ndarray) -> int:
    return int(cod.max()) + 1 if cod.size else 0


def _media_desvio(cod: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média, desvio e contagem do grupo, por linha, ignorando NaN."""
    ok = np.isfinite(x)
    peso = ok.astype(float)
    n = np.bincount(cod, weights=peso, minlength=_n_grupos(cod))
    s = np.bincount(cod, weights=np.where(ok, x, 0.0), minlength=n.size)
    s2 = np.bincount(cod, weights=np.where(ok, x * x, 0.0), minlength=n.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, s2 / np.maximum(n, 1) - media**2, np.nan)
    return media[cod], np.sqrt(np.maximum(var, 0))[cod], n[cod]


def valores(df: pd.DataFrame, spec: tuple, cache: dict | None = None) -> np.ndarray:
    """A candidata `spec` calculada em `df`; `cache` guarda os códigos de grupo."""
    op = spec[0]
    if op in PARES:
        a = np.asarray(df[spec[1]], float)
        b = np.asarray(df[spec[2]], float)
        if op == "dif":
            return a - b
        if op == "pro":
            return a * b
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(b) > EPS, a / np.where(np.abs(b) > EPS, b, 1.0), np.nan)
    chave = spec[-1]
    cod = (cache if cache is not None else {}).get(chave)
    if cod is None:
        cod = codigos(df, CHAVES[chave])
        if cache is not None:
            cache[chave] = cod
    if op == "gcont":
        return np.bincount(cod, minlength=_n_grupos(cod))[cod].astype(float)
    x = np.asarray(df[spec[1]], float)
    if op == "gpos":
        return pd.Series(x).groupby(cod).rank(pct=True).to_numpy(float)
    media, desvio, _ = _media_desvio(cod, x)
    if op == "gmed":
        return media
    if op == "gdes":
        return desvio
    if op == "gmedc":  # média do grupo menos a média do quadro: mesma árvore, sem a deriva
        return media - np.nanmean(x) if np.isfinite(x).any() else media
    if op == "gdesr":  # desvio do grupo em desvios do quadro
        s = float(np.nanstd(x)) if np.isfinite(x).any() else 0.0
        return desvio / s if s > EPS else np.full(len(desvio), np.nan)
    if op == "gdif":
        return x - media
    if op == "gz":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(desvio > EPS, (x - media) / np.where(desvio > EPS, desvio, 1.0), np.nan)
    raise ValueError(f"operador desconhecido: {op}")


def nome(spec: tuple) -> str:
    """Nome da coluna: `fe_<op>_<argumentos>`, estável e sem espaço."""
    return "fe_" + "_".join(str(p) for p in spec)


def candidatas(numericas: Sequence[str], chaves: Iterable[str] = (),
               pares: Sequence[str] = ()) -> list[tuple]:
    """Todas as candidatas: `pares` entre si (razão/diferença/produto) e `numericas`
    agregadas em cada chave de `chaves` (mais a contagem, só nas chaves com dia)."""
    saida: list[tuple] = []
    for i, a in enumerate(pares):
        for b in pares[i + 1:]:
            saida += [(op, a, b) for op in PARES]
    for chave in chaves:
        saida += [(op, c, chave) for c in numericas for op in GRUPO]
        if chave.startswith("dia_"):
            saida.append(("gcont", chave))
    return saida


# Sobreviventes do estudo de 03/10 (`docs/research/2026-10-03-fe-auto.md`): de 755
# candidatas, **uma** passa nos três portões (ganho no corpo em jan e em jul, ganho
# completo acima do placebo, AUC adversarial 2025×2026 ≤ 0,6). É o posto do tempo medido
# entre o push real e a decolagem (`MVT − AOBT_3`) dentro do dia × aeroporto × hora: se o
# voo está entre os rápidos ou os lentos da hora dele naquele aeroporto, sem depender do
# nível absoluto — que é justamente o que faz as médias por grupo reprovarem na deriva.
ESCOLHIDAS: tuple[tuple, ...] = (("gpos", "to_takeoff_from_AOBT_3_flt", "dia_apt_hora"),)


def colunas(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """As features escolhidas, uma por chave, na ordem de `ESCOLHIDAS`."""
    cache: dict[str, np.ndarray] = {}
    return {nome(s): valores(df, s, cache) for s in ESCOLHIDAS}
