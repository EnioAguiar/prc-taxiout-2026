"""Configuração de pista, fila com peso de esteira e cadência (`--pista`).

O que atrasa uma decolagem num aeroporto saturado não é só quanta gente está no solo
(isso é o `--superficie`): é *como o aeroporto está operando* — quais pistas estão em uso
para decolar e pousar, se a configuração acabou de mudar, quantos aviões (e de que porte)
estão na frente na mesma pista e com que cadência a pista está despachando.

Todo o bloco sai de colunas que o ranking de 2026 traz igual ao treino — `MVT_TIME_UTC_mvt`,
`RUNWAY_mvt`, `ADEP_mvt`/`ADES_mvt`, `PHASE_mvt`, `WK_TBL_CAT_flt` e `AOBT_3_flt`. O
`BLOCK_TIME_UTC_mvt` das decolagens (o alvo da competição, apagado no ranking; ver
`leaky_columns` em `src/models.py`) **nunca é lido aqui**, nem o da própria linha nem o dos
vizinhos. Os movimentos vêm dos parquets brutos que cobrem o período do quadro (os meses de
2025 e o `ranking.parquet` de 2026, como em `plano13._pousos`), então o valor de um voo é o
mesmo no treino, no holdout e no ranking — não depende do recorte.

As colunas:

- `pista_cfg` (categoria): aeroporto + conjunto de pistas com decolagem e conjunto de pistas
  com pouso na janela de 60 min em volta do `MVT` (`D:07C+07R A:07L`). É a configuração de
  pista do aeroporto naquela hora, que no mundo real decide o comprimento do táxi;
- `pista_cfg_mudou`: 1 quando essa configuração difere da de uma hora antes (troca de
  cabeceira: táxi mais longo e fila enquanto a mudança acontece), NaN sem hora anterior;
- `pista_cfg_dep` / `pista_cfg_arr`: quantas pistas estão ativas na janela em cada sentido;
- `pista_dominante`: 1 quando a pista do voo é a que mais decola na janela (as secundárias
  costumam ficar do lado errado do pátio);
- `pista_parte_rwy`: parte das decolagens da janela que saem pela pista do voo;
- `pista_fila_peso`: decolagens da **mesma pista** com `MVT` em `(AOBT_3, MVT]` do voo — os
  aviões que entraram na pista enquanto ele taxiava —, pesadas pela categoria de esteira
  (`J` 2,0; `H` 1,6; `M` 1,0; `L` 0,7; sem categoria 1,0): separação atrás de um pesado é
  maior, então a mesma fila custa mais tempo. O próprio voo é descontado;
- `pista_fila_pesadas`: quantos desses vizinhos são `H` ou `J`;
- `pista_span20`: segundos entre a 20ª decolagem anterior do aeroporto e o `MVT` do voo —
  a cadência medida com janela adaptativa (quanto menor, mais apertado o fluxo), NaN quando
  não há 20 decolagens antes;
- `pista_dep_h30`: decolagens por hora do aeroporto nos 30 min anteriores.

Repetições evitadas (28/09–01/10):

- contagens de decolagens/pousos em janelas fixas por aeroporto e por pista
  (`apt_dep_*`, `apt_arr_*`, `rwy_dep_*`, `fila_rwy_hora`) já estão no bloco `--fila`
  (`plano13.FILA_CONTAGENS`), que as pega prontas de `features.py`: nada disso é refeito.
  `pista_dep_h30` é o único vizinho próximo (vale `2 ×` `apt_dep_prev_30m`), e fica porque
  `--pista` precisa funcionar sem `--fila`;
- `apt_rwy` (aeroporto + pista do voo como categoria) é do `--fila`; `pista_cfg` é outra
  coisa: o conjunto de pistas do aeroporto, não a pista do voo;
- a contagem *sem peso* da fila na mesma pista não entra: o `--superficie` já tem
  `sup_dep_decolam_durante_rwy`, a mesma ideia com o push estimado (`MVT − pred`) no lugar
  do `AOBT_3`. Aqui só entram o peso de esteira e a contagem de pesados, que não existem
  em lugar nenhum.

A janela de 60 min é discretizada em blocos de 10 min: o voo cai no bloco `b` e a janela vai
do bloco `b−3` ao `b+2` (dos 30 min antes aos 30 min depois do começo do bloco dele). Cada
configuração vira uma soma acumulada por pista, e não uma varredura por voo; a "hora
anterior" é a mesma janela seis blocos atrás.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import features as F
from cache import DATA
from externos import TIME
from plano13 import _brutos

AOBT = "AOBT_3_flt"
WAKE = "WK_TBL_CAT_flt"
RWY = "RUNWAY_mvt"
MOV_COLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", RWY, TIME, WAKE]

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
MARGEM = pd.Timedelta("6h")       # movimento de fora do quadro ainda conta na janela
BLOCO = 600.0                     # bloco de 10 min
JANELA = 6                        # 6 blocos = 60 min (de b−3 a b+2)
HORA = 6                          # blocos que separam a janela da hora anterior
N_ULTIMAS = 20                    # decolagens do `pista_span20`
JANELA_H30 = 1800.0               # 30 min do `pista_dep_h30`
PESO = {"J": 2.0, "H": 1.6, "M": 1.0, "L": 0.7}
PESO_PADRAO = 1.0                 # sem registro NM não há categoria: trata como média
PESADAS = ("H", "J")

CAT = "pista_cfg"
COLS_NUM = ["pista_cfg_mudou", "pista_cfg_dep", "pista_cfg_arr", "pista_dominante",
            "pista_parte_rwy", "pista_fila_peso", "pista_fila_pesadas", "pista_span20",
            "pista_dep_h30"]
COLS = [CAT, *COLS_NUM]


def _segundos(s: pd.Series) -> np.ndarray:
    """Horário em segundos desde 1970 (NaT vira NaN)."""
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return (t - EPOCH).dt.total_seconds().to_numpy(float)


def _grupos(chave: np.ndarray) -> dict:
    """Posições de cada chave (aeroporto, ou pista)."""
    return pd.Series(np.arange(len(chave))).groupby(chave, sort=False).indices


def _peso(wk: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Peso de esteira de cada movimento e se ele é pesado (`H`/`J`)."""
    cat = wk.astype(str).str.strip().str.upper().str[:1]
    return (cat.map(PESO).fillna(PESO_PADRAO).to_numpy(float),
            cat.isin(PESADAS).to_numpy(float))


def _movimentos(apts: set[str], t0: float, t1: float, dados: Path) -> dict:
    """Movimentos (DEP e ARR) dos aeroportos pedidos, em arrays compactos e ordenados.

    Aeroporto e pista viram códigos inteiros: carregar 4 M de strings na hora do corretor
    custaria centenas de MB à toa. Só colunas que existem iguais no treino e no ranking são
    lidas — nunca o `BLOCK_TIME_UTC_mvt`.
    """
    vocab_apt = {a: i for i, a in enumerate(sorted(apts))}
    vocab_rwy: dict[str, int] = {}
    margem = MARGEM.total_seconds()
    lim0 = EPOCH + pd.to_timedelta(t0 - margem, unit="s")
    lim1 = EPOCH + pd.to_timedelta(t1 + margem, unit="s")
    partes = []
    for ini, fim, p in _brutos(dados):
        if fim <= lim0 or ini > lim1:
            continue
        raw = pd.read_parquet(p, columns=MOV_COLS)
        fase = raw["PHASE_mvt"].to_numpy()
        nome = np.where(fase == "DEP", raw["ADEP_mvt"].to_numpy(), raw["ADES_mvt"].to_numpy())
        apt = pd.Series(nome).map(vocab_apt).to_numpy(float)
        t = _segundos(raw[TIME])
        ok = (np.isfinite(apt) & np.isfinite(t)
              & (t >= t0 - margem) & (t <= t1 + margem))
        if ok.any():
            codigos, nomes = pd.factorize(raw[RWY][ok].astype(str).to_numpy())
            tabela = np.array([vocab_rwy.setdefault(n, len(vocab_rwy)) for n in nomes],
                              dtype=np.int32)
            peso, pesada = _peso(raw[WAKE][ok])
            partes.append({"apt": apt[ok].astype(np.int16), "rwy": tabela[codigos],
                           "t": t[ok], "peso": peso, "pesada": pesada,
                           "dep": (fase[ok] == "DEP")})
        del raw
    vazio = {"apt": np.int16, "rwy": np.int32, "t": float, "peso": float, "pesada": float,
             "dep": bool}
    junto = {c: (np.concatenate([p[c] for p in partes]) if partes else np.empty(0, dtype=tipo))
             for c, tipo in vazio.items()}
    ordem = np.argsort(junto["t"], kind="stable")  # tudo ordenado no tempo, de uma vez
    junto = {c: v[ordem] for c, v in junto.items()}
    junto["vocab_apt"] = vocab_apt
    junto["nomes_rwy"] = [n for n, _ in sorted(vocab_rwy.items(), key=lambda kv: kv[1])]
    return junto


def _contagem(bins: np.ndarray, pistas: np.ndarray, nb: int, r: int) -> np.ndarray:
    """Movimentos de cada pista em cada bloco de 10 min."""
    if not bins.size:
        return np.zeros((nb, r))
    return np.bincount(bins * r + pistas, minlength=nb * r).reshape(nb, r).astype(float)


def _janela(cnt: np.ndarray) -> np.ndarray:
    """Soma dos `JANELA` blocos em volta de cada bloco (b−3 … b+2), por pista."""
    nb, r = cnt.shape
    cs = np.vstack([np.zeros((1, r)), np.cumsum(cnt, axis=0)])
    i = np.arange(nb)
    return cs[np.clip(i + JANELA // 2, None, nb)] - cs[np.clip(i - JANELA // 2, 0, None)]


def _codigos(win_dep: np.ndarray, win_arr: np.ndarray,
             nomes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Texto da configuração de cada bloco e o índice dela (para comparar com a hora anterior).

    As pistas saem em ordem alfabética, então o mesmo conjunto dá sempre o mesmo texto.
    """
    r = len(nomes)
    ordem = sorted(range(r), key=lambda i: nomes[i])
    ativo = np.concatenate([win_dep > 0, win_arr > 0], axis=1)
    unicos, inverso = np.unique(ativo, axis=0, return_inverse=True)
    textos = []
    for linha in unicos:
        d = "+".join(nomes[i] for i in ordem if linha[i]) or "-"
        a = "+".join(nomes[i] for i in ordem if linha[r + i]) or "-"
        textos.append(f"D:{d} A:{a}")
    return np.array(textos, dtype=object), np.asarray(inverso).reshape(-1)


def _configuracao(num: dict, cod: np.ndarray, pos: np.ndarray, apt: str, t: np.ndarray,
                  rwy: np.ndarray, dep: tuple, arr: tuple, nomes_rwy: list[str]) -> None:
    """Pistas ativas na janela, troca de configuração e peso da pista do próprio voo."""
    dep_t, dep_r = dep
    arr_t, arr_r = arr
    locais, inverso = np.unique(np.concatenate([dep_r, arr_r, rwy]), return_inverse=True)
    nomes = [nomes_rwy[c] if c >= 0 else "?" for c in locais]
    r = locais.size
    d_idx, a_idx = inverso[:dep_r.size], inverso[dep_r.size:dep_r.size + arr_r.size]
    meu = inverso[dep_r.size + arr_r.size:]

    bins = np.floor(np.concatenate([dep_t, arr_t, t]) / BLOCO).astype(np.int64)
    b0, nb = bins.min(), int(bins.max() - bins.min() + 1)
    bins -= b0
    win_d = _janela(_contagem(bins[:dep_t.size], d_idx, nb, r))
    win_a = _janela(_contagem(bins[dep_t.size:dep_t.size + arr_t.size], a_idx, nb, r))
    bi = bins[dep_t.size + arr_t.size:]

    textos, inverso_cfg = _codigos(win_d, win_a, nomes)
    cod[pos] = [f"{apt}|{textos[k]}" for k in inverso_cfg[bi]]
    antes = bi - HORA
    num["pista_cfg_mudou"][pos] = np.where(
        antes >= 0, (inverso_cfg[bi] != inverso_cfg[np.clip(antes, 0, None)]).astype(float),
        np.nan)
    wd, wa = win_d[bi], win_a[bi]
    num["pista_cfg_dep"][pos] = (wd > 0).sum(axis=1)
    num["pista_cfg_arr"][pos] = (wa > 0).sum(axis=1)
    total, minha = wd.sum(axis=1), wd[np.arange(bi.size), meu]
    num["pista_dominante"][pos] = np.where(total > 0, (minha >= wd.max(axis=1)).astype(float),
                                           np.nan)
    num["pista_parte_rwy"][pos] = np.divide(minha, total, out=np.full(total.shape, np.nan),
                                            where=total > 0)


def _cadencia(num: dict, pos: np.ndarray, t: np.ndarray, dep_t: np.ndarray) -> None:
    """Quanto tempo o aeroporto levou nas últimas 20 decolagens e o ritmo dos 30 min."""
    j = np.searchsorted(dep_t, t, "left")  # decolagens estritamente antes do MVT do voo
    antes = j - N_ULTIMAS
    num["pista_span20"][pos] = np.where(antes >= 0, t - dep_t[np.clip(antes, 0, None)], np.nan)
    j30 = np.searchsorted(dep_t, t - JANELA_H30, "left")
    num["pista_dep_h30"][pos] = (j - j30) * (3600.0 / JANELA_H30)


def _fila(num: dict, pos: np.ndarray, t: np.ndarray, aobt: np.ndarray, rwy: np.ndarray,
          dep: tuple, peso_proprio: np.ndarray, pesada_propria: np.ndarray) -> None:
    """Fila da mesma pista entre o off-block do NM e a decolagem, com peso de esteira.

    O intervalo é `(AOBT_3, MVT]`: quem decolou no instante do off-block já tinha ido, e a
    própria linha (que está nos parquets brutos) é descontada do total.
    """
    dep_t, dep_r, dep_peso, dep_pesada = dep
    for cod_rwy, linhas in _grupos(rwy).items():
        sel = dep_r == cod_rwy
        tt = dep_t[sel]  # subconjunto de um vetor ordenado: continua ordenado
        acc_peso = np.concatenate([[0.0], np.cumsum(dep_peso[sel])])
        acc_pesada = np.concatenate([[0.0], np.cumsum(dep_pesada[sel])])
        a, m = aobt[linhas], t[linhas]
        hi = np.searchsorted(tt, m, "right")
        lo = np.searchsorted(tt, np.where(np.isfinite(a), a, m), "right")
        onde = pos[linhas]
        num["pista_fila_peso"][onde] = np.where(
            np.isfinite(a), np.maximum(acc_peso[hi] - acc_peso[lo] - peso_proprio[linhas], 0.0),
            np.nan)
        num["pista_fila_pesadas"][onde] = np.where(
            np.isfinite(a),
            np.maximum(acc_pesada[hi] - acc_pesada[lo] - pesada_propria[linhas], 0.0), np.nan)


def _quadro(cod: np.ndarray, num: dict, index: pd.Index) -> pd.DataFrame:
    """Quadro final: a configuração como categoria e o resto como float."""
    out = pd.DataFrame({CAT: pd.Categorical(cod)}, index=index)
    for c in COLS_NUM:
        out[c] = num[c]
    return out[COLS]


def colunas(df: pd.DataFrame, dados: Path = DATA) -> pd.DataFrame:
    """Colunas `pista_*` das linhas de `df`, na ordem delas (`corrector_frame`).

    `df` são decolagens (cegas, holdout ou ranking); os vizinhos vêm sempre dos parquets
    brutos, nunca do recorte, e só das colunas que o ranking também tem.
    """
    n = len(df)
    num = {c: np.full(n, np.nan) for c in COLS_NUM}
    cod = np.full(n, None, dtype=object)
    if n == 0:
        return _quadro(cod, num, df.index)

    t = _segundos(df[TIME])
    aobt = _segundos(df[AOBT]) if AOBT in df else np.full(n, np.nan)
    apt_nome = df[F.AIRPORT].astype(str).to_numpy()
    rwy_nome = df[RWY].astype(str).to_numpy()
    if WAKE in df:
        peso_proprio, pesada_propria = _peso(df[WAKE])
    else:
        peso_proprio, pesada_propria = np.full(n, PESO_PADRAO), np.zeros(n)
    conhecido = np.isfinite(t)
    if not conhecido.any():
        return _quadro(cod, num, df.index)

    mov = _movimentos(set(apt_nome[conhecido]), float(np.nanmin(t)), float(np.nanmax(t)), dados)
    indice = {nome: i for i, nome in enumerate(mov["nomes_rwy"])}
    rwy_cod = pd.Series(rwy_nome).map(indice).fillna(-1).to_numpy(np.int64)
    for nome, pos in _grupos(apt_nome).items():
        pos = pos[conhecido[pos]]
        codigo_apt = mov["vocab_apt"].get(nome)
        if not pos.size or codigo_apt is None:
            continue
        no_apt = mov["apt"] == codigo_apt
        d, a = no_apt & mov["dep"], no_apt & ~mov["dep"]
        _configuracao(num, cod, pos, nome, t[pos], rwy_cod[pos],
                      (mov["t"][d], mov["rwy"][d]), (mov["t"][a], mov["rwy"][a]),
                      mov["nomes_rwy"])
        _cadencia(num, pos, t[pos], mov["t"][d])
        _fila(num, pos, t[pos], aobt[pos], rwy_cod[pos],
              (mov["t"][d], mov["rwy"][d], mov["peso"][d], mov["pesada"][d]),
              peso_proprio[pos], pesada_propria[pos])
    return _quadro(cod, num, df.index)
