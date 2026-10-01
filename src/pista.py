"""Estado da pista (`--pista`) e retenção no portão (`--retencao`).

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

## Bloco `--retencao` (`colunas_retencao`, ideia de `docs/research/2026-10-01-repos-concorrentes.md`)

O `--pista` e o `--superficie` olham o taxiway; este olha o **portão**, e com o relógio real
do push (`t = AOBT_3` do voo, presente em 98,5 % das DEP do ranking) no lugar do push
estimado `MVT − pred`. Sem `AOBT_3` as cinco colunas ficam NaN.

- `ret_overdue_rwy` / `ret_overdue_apt`: no instante `t`, quantas partidas da mesma pista (e
  do mesmo aeroporto) já passaram do `EOBT_1` e **ainda não empurraram** —
  `|{EOBT ≤ t}| − |{AOBT ≤ t}|`, que telescopa para `{EOBT ≤ t < AOBT}`. Guardas do
  unique-umbrella (`model.py:437-481`): só partidas com `EOBT ≤ AOBT`, espera < 7200 s
  (acima disso é virada de data) e que chegaram a decolar (`MVT ≥ AOBT`, o que também
  garante "ainda não decolou", já que `MVT ≥ AOBT > t`). O próprio voo se cancela nas duas
  contagens. Placar deles: 307,01 → 304,95;
- `ret_ativos_rwy`: partidas da mesma pista com `AOBT_3 ≤ t < MVT` — quem está mesmo no solo
  rodando, pelo relógio real dos vizinhos e não pelo push estimado. O próprio voo é
  descontado;
- `ret_atraso_15m`: média de `AOBT_3 − SCHED` das partidas do aeroporto com `AOBT_3` em
  `[t − 15 min, t)` — em colapso de capacidade o avião empurra cedo e segura o off-block;
  15 min é o ótimo que os dois repositórios mediram. A própria linha fica fora;
- `ret_ewma_dep`: decolagens por hora do aeroporto com EWMA de meia-vida 10 min no instante
  `t`, só com `MVT < t` dos outros (o decaimento suave ganhou do boxcar na varredura deles).

Repetições evitadas no `--retencao`:

- `sup_dep_taxiando_push` (`--superficie`) é a mesma contagem de `ret_ativos_rwy` com o push
  *estimado* (`MVT − pred`) nas duas pontas e sem separar por pista; é justamente a troca que
  a pesquisa recomenda, então o `ret_ativos_rwy` fica por pista e com o relógio real e nenhuma
  variante por aeroporto é acrescentada (a parcimoniosa ganhou na medição deles);
- `ctx_viz_*` (`src/contexto.py`) é a média de `MVT − AOBT_3` dos vizinhos (proxy de táxi),
  não o atraso de portão `AOBT_3 − SCHED`: `ret_atraso_15m` não repete nenhuma delas;
- `apt_dep_prev_*m` (`--fila`) e `pista_dep_h30` contam decolagens em janela retangular em
  volta do `MVT`; `ret_ewma_dep` é no instante do push, com decaimento exponencial — fica,
  mas é a coluna deste bloco mais próxima de algo que já existe;
- a fila de portão por aeroporto **e** por pista está nas duas colunas `ret_overdue_*`; a
  terceira variante (as duas somadas) foi medida pelos outros e não ganhou nada.
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
EOBT = "EOBT_1_flt"
SCHED = "SCHED_TIME_UTC_mvt"
WAKE = "WK_TBL_CAT_flt"
RWY = "RUNWAY_mvt"
MOV_COLS = ["PHASE_mvt", "ADEP_mvt", "ADES_mvt", RWY, TIME, WAKE]
DEP_COLS = ["PHASE_mvt", "ADEP_mvt", RWY, TIME, AOBT, EOBT, SCHED]

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

MAX_RETIDO = 7200.0               # AOBT − EOBT maior que isso é artefato de virada de data
JANELA_ATRASO = 900.0             # 15 min do `ret_atraso_15m` (o ótimo medido pelos outros)
MEIA_VIDA = 600.0                 # meia-vida do EWMA de decolagens
CORTE_EWMA = 7200.0               # 12 meias-vidas: o que é mais velho pesa < 0,03 %
PEDACO_EWMA = 21600.0             # origem local a cada 6 h: expoentes pequenos, soma estável

CAT = "pista_cfg"
COLS_NUM = ["pista_cfg_mudou", "pista_cfg_dep", "pista_cfg_arr", "pista_dominante",
            "pista_parte_rwy", "pista_fila_peso", "pista_fila_pesadas", "pista_span20",
            "pista_dep_h30"]
COLS = [CAT, *COLS_NUM]
COLS_RET = ["ret_overdue_rwy", "ret_overdue_apt", "ret_ativos_rwy", "ret_atraso_15m",
            "ret_ewma_dep"]


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


# ------------------------------------------------------------------ retenção no portão


def _partidas(apts: set[str], t0: float, t1: float, dados: Path) -> dict:
    """Partidas (só `PHASE_mvt == "DEP"`) com os três relógios que o ranking também traz.

    `MVT_TIME_UTC_mvt`, `AOBT_3_flt`, `EOBT_1_flt` e `SCHED_TIME_UTC_mvt` — nunca o
    `BLOCK_TIME_UTC_mvt`, que é o alvo e está apagado nas DEP do ranking.
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
        raw = pd.read_parquet(p, columns=DEP_COLS)
        apt = raw["ADEP_mvt"].map(vocab_apt).to_numpy(float)
        mvt = _segundos(raw[TIME])
        ok = ((raw["PHASE_mvt"].to_numpy() == "DEP") & np.isfinite(apt) & np.isfinite(mvt)
              & (mvt >= t0 - margem) & (mvt <= t1 + margem))
        if ok.any():
            codigos, nomes = pd.factorize(raw[RWY][ok].astype(str).to_numpy())
            tabela = np.array([vocab_rwy.setdefault(n, len(vocab_rwy)) for n in nomes],
                              dtype=np.int32)
            partes.append({"apt": apt[ok].astype(np.int16), "rwy": tabela[codigos],
                           "mvt": mvt[ok], "aobt": _segundos(raw[AOBT])[ok],
                           "eobt": _segundos(raw[EOBT])[ok],
                           "sched": _segundos(raw[SCHED])[ok]})
        del raw
    tipos = {"apt": np.int16, "rwy": np.int32, "mvt": float, "aobt": float, "eobt": float,
             "sched": float}
    junto = {c: (np.concatenate([p[c] for p in partes]) if partes else np.empty(0, dtype=tipo))
             for c, tipo in tipos.items()}
    junto["vocab_apt"], junto["vocab_rwy"] = vocab_apt, vocab_rwy
    return junto


def _retidas(par: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EOBT e AOBT ordenados das partidas que servem para a fila de portão.

    Guardas do `overdue_runway_queue`: os dois relógios existem, `EOBT ≤ AOBT`, a espera é
    menor que `MAX_RETIDO` (acima disso é virada de data) e o voo chegou a decolar
    (`MVT ≥ AOBT`). Com elas a diferença das duas contagens acumuladas sobra exatamente em
    quem já passou do EOBT e ainda não empurrou — e `MVT ≥ AOBT > t` garante que ninguém
    contado já decolou.
    """
    espera = par["aobt"] - par["eobt"]
    bom = (sel & np.isfinite(espera) & (espera >= 0) & (espera < MAX_RETIDO)
           & (par["mvt"] >= par["aobt"]))
    return np.sort(par["eobt"][bom]), np.sort(par["aobt"][bom])


def _ativas(par: dict, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AOBT e MVT ordenados das partidas que empurraram e decolaram."""
    bom = sel & np.isfinite(par["aobt"]) & np.isfinite(par["mvt"]) & (par["aobt"] <= par["mvt"])
    return np.sort(par["aobt"][bom]), np.sort(par["mvt"][bom])


def _conta_ate(ordenado: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Quantos eventos acontecem até `t` (inclusive)."""
    return np.searchsorted(ordenado, t, "right").astype(float)


def _atraso_recente(par: dict, sel: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Média de `AOBT_3 − SCHED` das partidas do aeroporto nos 15 min antes de `t`.

    A janela é `[t − 15 min, t)`: a própria linha (cujo `AOBT_3` é o próprio `t`) fica
    fora, e com ela qualquer empate exato no instante.
    """
    bom = sel & np.isfinite(par["aobt"]) & np.isfinite(par["sched"])
    ordem = np.argsort(par["aobt"][bom], kind="stable")
    quando = par["aobt"][bom][ordem]
    cs = np.concatenate([[0.0], np.cumsum((par["aobt"][bom] - par["sched"][bom])[ordem])])
    hi = np.searchsorted(quando, t, "left")
    lo = np.searchsorted(quando, t - JANELA_ATRASO, "left")
    n = hi - lo
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _ewma(mvt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Decolagens por hora com EWMA de meia-vida `MEIA_VIDA`, só com `MVT < t`.

    `Σ 2^((MVT_j − t)/meia-vida)` vira, em cada pedaço de 6 h, uma soma acumulada com
    origem local: os expoentes ficam entre −12 e 36 e a conta não estoura. O corte em
    `CORTE_EWMA` (12 meias-vidas) joga fora o que pesaria menos de 0,03 %.
    """
    saida = np.zeros(t.size)
    if not mvt.size:
        return saida
    for pedaco, linhas in _grupos(np.floor(t / PEDACO_EWMA)).items():
        origem = pedaco * PEDACO_EWMA
        ini = np.searchsorted(mvt, origem - CORTE_EWMA, "left")
        fim = np.searchsorted(mvt, origem + PEDACO_EWMA, "left")
        ev = mvt[ini:fim]
        cs = np.concatenate([[0.0], np.cumsum(np.exp2((ev - origem) / MEIA_VIDA))])
        tq = t[linhas]
        hi = np.searchsorted(ev, tq, "left")   # só o passado estrito
        lo = np.searchsorted(ev, tq - CORTE_EWMA, "left")
        saida[linhas] = np.exp2((origem - tq) / MEIA_VIDA) * (cs[hi] - cs[lo])
    return saida * (3600.0 * np.log(2.0) / MEIA_VIDA)


def colunas_retencao(df: pd.DataFrame, dados: Path = DATA) -> pd.DataFrame:
    """Colunas `ret_*` das linhas de `df`, na ordem delas (`corrector_frame`).

    Tudo é medido no instante do push de verdade do voo, `t = AOBT_3` (presente em 98,5 %
    das DEP do ranking); sem `AOBT_3` o bloco inteiro fica NaN.
    """
    n = len(df)
    num = {c: np.full(n, np.nan) for c in COLS_RET}
    if n == 0:
        return pd.DataFrame(num, index=df.index)[COLS_RET]
    t = _segundos(df[AOBT]) if AOBT in df else np.full(n, np.nan)
    mvt = _segundos(df[TIME])
    apt_nome = df[F.AIRPORT].astype(str).to_numpy()
    rwy_nome = df[RWY].astype(str).to_numpy()
    conhecido = np.isfinite(t)
    if not conhecido.any():
        return pd.DataFrame(num, index=df.index)[COLS_RET]

    par = _partidas(set(apt_nome[conhecido]), float(np.nanmin(t)), float(np.nanmax(t)), dados)
    rwy_cod = pd.Series(rwy_nome).map(par["vocab_rwy"]).fillna(-1).to_numpy(np.int64)
    # a própria linha entra nas contagens de quem está taxiando: sai aqui
    propria = (np.isfinite(mvt) & (mvt > t)).astype(float)
    for nome, pos in _grupos(apt_nome).items():
        pos = pos[conhecido[pos]]
        codigo_apt = par["vocab_apt"].get(nome)
        if not pos.size or codigo_apt is None:
            continue
        no_apt = par["apt"] == codigo_apt
        tq = t[pos]
        eobt_s, aobt_s = _retidas(par, no_apt)
        num["ret_overdue_apt"][pos] = _conta_ate(eobt_s, tq) - _conta_ate(aobt_s, tq)
        num["ret_atraso_15m"][pos] = _atraso_recente(par, no_apt, tq)
        num["ret_ewma_dep"][pos] = _ewma(np.sort(par["mvt"][no_apt]), tq)
        for cod_rwy, linhas in _grupos(rwy_cod[pos]).items():
            na_pista = no_apt & (par["rwy"] == cod_rwy)
            onde, quando = pos[linhas], tq[linhas]
            eobt_r, aobt_r = _retidas(par, na_pista)
            num["ret_overdue_rwy"][onde] = _conta_ate(eobt_r, quando) - _conta_ate(aobt_r, quando)
            push, decola = _ativas(par, na_pista)
            num["ret_ativos_rwy"][onde] = np.maximum(
                _conta_ate(push, quando) - _conta_ate(decola, quando) - propria[onde], 0.0)
    return pd.DataFrame(num, index=df.index)[COLS_RET]
