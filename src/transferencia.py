"""Checagem de transferência 2025 → 2026 antes de propor um envio.

    bin/run src/transferencia.py <id_novo...> [--contra <id...>] [--json] [--pesos]

A régua (`src/regua.py`) responde "o candidato ganha no holdout de 2025?". Este módulo
responde as três perguntas que faltam, e que juntas teriam rejeitado a v37 (simulação
−0,77 s de `completo`, oficial −0,02 s):

1. **De que regime vem o ganho?** O ganho de RMSE se decompõe **exatamente** em soma de
   contribuições por voo: `RMSE(base) − RMSE(novo) = Σᵢ wᵢ·dᵢ / (Σw · (RMSE_b + RMSE_n))`
   com `dᵢ = (y−base)² − (y−novo)²`. Somando `d` só nos voos normais (`y ≤ 3600`) sai a
   parcela do ganho que vem do corpo; o resto vem da cauda, que é loteria.
2. **De quantos voos vem o ganho?** A mesma soma restrita aos `k` voos de maior `d` dá a
   fração do ganho que mora neles. Na v37 foram **20 voos = 129 % do ganho**.
3. **O ganho sobrevive à mudança de distribuição?** Um classificador "esta linha é 2025 ou
   2026?" (AUC 0,880, `docs/research/2026-10-02-descoberta-bruta.md`) dá
   `w(x) = p/(1−p)`, truncado no P95 — o peso do IWCV de Sugiyama (2007), o estimador
   quase não enviesado do risco sob mudança de covariáveis. Com ele o holdout de 2025 é
   relido como se fosse a população de 2026.

Os três números do relatório (`corpo`, `topo`, `corpo_p`) viram o portão `veredito()`,
usado por `esteira.talvez_enviar` antes de montar uma submissão. Calibração e backtest
contra os deltas oficiais das v33–v37: `docs/research/2026-10-03-transferencia.md`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import campeao
from adsb_events import add_features
from cache import CACHE, TRUTH, cache_key, split_paths
from experiment import TAIL_S
from features import ID

# Colunas que a base e o corretor compartilham (mesmo conjunto da varredura de 02/10).
PREFIXOS = ("ctx_", "adsb_", "to_takeoff_from_", "gap_", "apt_", "rwy_", "round_")
EXTRAS = ("hour", "dow", "nm_missing")
CATEGORICA = "AIRPORT"

RODADAS, FOLHAS, SEMENTE = 200, 63, 0
CORTE_PESO = 0.95  # truncagem do peso no P95 (variância explosiva sem truncar)
TOP_K = 20  # os 20 voos da v37 valeram 129 % do ganho

# Portão de envio, calibrado no backtest das v30→v37 (docs/research/2026-10-03-transferencia.md),
# onde acerta 6 de 6 e rejeita a v37. O `corpo` sozinho prevê o delta oficial com
# razão oficial/simulado 1,03 e erro absoluto médio de 0,15 s.
CORPO_MIN = 0.0  # a parcela do ganho vinda dos voos normais, positiva e com IC baixo > 0
RAZAO_OFICIAL = 1.03  # delta oficial ≈ 1,03 × `corpo` (EAM 0,15 s nas seis transições)
# 50 % seria bonito e reprovaria tudo: com ganhos de ~1 s em 344 mil voos, até a v33
# (+2,21 s oficiais) concentra 125 % do ganho do corpo em 20 voos. O limite que separa as
# três transições que valeram ≥ 0,65 s das três que valeram ≤ 0,16 s é 200 %.
TOPO_MAX = 2.0


# --------------------------------------------------------------------------- medição


def medir(y, base, novo, dias, w=None, mascara=None, n: int = 1000,
          semente: int = 0) -> dict:
    """Parcela do ganho de RMSE que vem de `mascara`, com IC por bootstrap de dias.

    Sem `mascara` e sem `w` é idêntico ao `compare.paired_bootstrap`: a decomposição é
    exata, não uma aproximação (`RMSE_b − RMSE_n = Σd / (Σw · (RMSE_b + RMSE_n))`).
    """
    y, base, novo = (np.asarray(v, float) for v in (y, base, novo))
    w = np.ones(y.size) if w is None else np.asarray(w, float)
    codes, _ = pd.factorize(np.asarray(dias).astype(str))
    k = int(codes.max()) + 1
    d = w * ((y - base) ** 2 - (y - novo) ** 2)
    if mascara is not None:
        d = np.where(np.asarray(mascara, bool), d, 0.0)
    soma = lambda v: np.bincount(codes, v, minlength=k)  # noqa: E731
    peso, sse_b, sse_n, dia_d = (soma(w), soma(w * (y - base) ** 2),
                                 soma(w * (y - novo) ** 2), soma(d))

    def ganho(pw, b, nv, dd):
        return dd / pw / (np.sqrt(b / pw) + np.sqrt(nv / pw))

    ponto = ganho(peso.sum(), sse_b.sum(), sse_n.sum(), dia_d.sum())
    sorteio = np.random.default_rng(semente).integers(0, k, size=(n, k))
    boot = ganho(peso[sorteio].sum(1), sse_b[sorteio].sum(1),
                 sse_n[sorteio].sum(1), dia_d[sorteio].sum(1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"ganho": float(ponto), "ic_baixo": float(lo), "ic_alto": float(hi)}


def parte_do_topo(y, base, novo, w=None, k: int = TOP_K) -> dict:
    """Fração do ganho que mora nos `k` voos que mais ganharam (129 % na v37)."""
    y, base, novo = (np.asarray(v, float) for v in (y, base, novo))
    w = np.ones(y.size) if w is None else np.asarray(w, float)
    d = w * ((y - base) ** 2 - (y - novo) ** 2)
    total = float(d.sum())
    topo = float(np.sort(d)[-k:].sum())
    return {"k": k, "parte": topo / total if total else float("nan"),
            "ganho_sem_topo": float(total - topo)}


# ------------------------------------------------------------------- peso adversarial


def _caminho_cache(nome: str) -> Path:
    caminho = CACHE / f"{nome}-{cache_key(split_paths(nome))}.parquet"
    if not caminho.exists():
        raise SystemExit(f"falta o cache de {nome}; rode: bin/run src/cache.py")
    return caminho


def _quadro(nome: str) -> pd.DataFrame:
    """Só as colunas do classificador adversário (o split inteiro não cabe na bancada)."""
    caminho = _caminho_cache(nome)
    nomes = pq.ParquetFile(caminho).schema_arrow.names
    cols = [c for c in nomes
            if c.startswith(PREFIXOS) or c in EXTRAS or c in (CATEGORICA, ID)]
    df = pq.read_table(caminho, columns=cols).to_pandas(split_blocks=True, self_destruct=True)
    return add_features(df)


def _chave() -> str:
    partes = [cache_key(split_paths("holdout2025")), cache_key(split_paths("ranking2026")),
              str(RODADAS), str(FOLHAS), str(SEMENTE), str(CORTE_PESO)]
    return hashlib.sha256("|".join(partes).encode()).hexdigest()[:12]


def calcular_pesos(semente: int = SEMENTE) -> pd.DataFrame:
    """Treina o adversário 2025 × 2026 e devolve `w = p/(1−p)` fora da amostra, por voo.

    Duas dobras: cada linha do holdout recebe `p` de um modelo que não a viu, senão o peso
    carrega o sobreajuste do próprio classificador.
    """
    import lightgbm as lgb

    h, r = _quadro("holdout2025"), _quadro("ranking2026")
    ids = h[ID].to_numpy()
    feats = sorted((set(h.columns) & set(r.columns)) - {ID})
    X = pd.concat([h[feats], r[feats]], ignore_index=True)
    del h, r
    for c in feats:
        if c != CATEGORICA:
            X[c] = X[c].astype("float32")
    X[CATEGORICA] = X[CATEGORICA].astype("category")
    alvo = np.r_[np.zeros(len(ids)), np.ones(len(X) - len(ids))]

    dobra = np.random.default_rng(semente).integers(0, 2, size=len(X))
    p = np.full(len(ids), np.nan)
    auc = []
    for f in (0, 1):
        tr, te = dobra != f, dobra == f
        modelo = lgb.train(
            {"objective": "binary", "learning_rate": 0.05, "num_leaves": FOLHAS,
             "max_bin": 63, "verbose": -1, "seed": semente, "num_threads": 4},
            lgb.Dataset(X[tr], alvo[tr]), RODADAS)
        saida = modelo.predict(X[te])
        auc.append(_auc(alvo[te], saida))
        te_h = te[:len(ids)]  # o holdout são as primeiras linhas; `saida` preserva a ordem
        p[te_h] = saida[:te_h.sum()]
        del modelo, saida
    del X

    razao = p / np.clip(1 - p, 1e-6, None)
    w = np.clip(razao, 0, float(np.quantile(razao, CORTE_PESO)))
    return pd.DataFrame({ID: ids, "p": p, "w": w / w.mean(),
                         "auc": float(np.mean(auc))})


def _auc(y, s) -> float:
    y, s = np.asarray(y, float), np.asarray(s, float)
    n1 = y.sum()
    n0 = len(y) - n1
    if not n0 or not n1:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def pesos(semente: int = SEMENTE, recalcular: bool = False) -> pd.DataFrame:
    """`w` por voo do holdout, com cache em `data/cache/` (o treino custa ~4 min)."""
    alvo = CACHE / f"transferencia-{_chave()}.parquet"
    if alvo.exists() and not recalcular:
        return pd.read_parquet(alvo)
    df = calcular_pesos(semente)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = alvo.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd", index=False)
    tmp.replace(alvo)
    for velho in CACHE.glob("transferencia-*.parquet"):
        if velho != alvo:
            velho.unlink()
    return df


# ----------------------------------------------------------------------- relatório


def _alinhado(ref: pd.DataFrame, ids: list[str]) -> np.ndarray:
    p = ref[[ID]].merge(campeao.previsao(ids)[[ID, "pred"]], on=ID, how="left",
                        validate="one_to_one")["pred"].to_numpy(float)
    if np.isnan(p).any():
        raise SystemExit(f"{'+'.join(ids)} não cobre o mesmo holdout da referência")
    return p


def relatorio(novos: list[str], contra: list[str] | None = None, semente: int = 0,
              k: int = TOP_K, w_tabela: pd.DataFrame | None = None) -> dict:
    """Decomposição do ganho de `novos` contra `contra` (padrão: a campeã)."""
    contra = contra or [m["id"] for m in campeao.carregar()["membros"]]
    ref = campeao.previsao(contra)
    y = ref[TRUTH].to_numpy(float)
    dias = ref["dia"].to_numpy()
    base = ref["pred"].to_numpy(float)
    novo = _alinhado(ref, novos)
    tab = pesos() if w_tabela is None else w_tabela
    w = ref[[ID]].merge(tab[[ID, "w"]], on=ID, how="left",
                        validate="one_to_one")["w"].to_numpy(float)
    if np.isnan(w).any():
        raise SystemExit("peso adversarial não cobre o holdout da referência; "
                         "rode bin/run src/transferencia.py --pesos")
    corpo = y <= TAIL_S
    m = lambda **kw: medir(y, base, novo, dias, semente=semente, **kw)  # noqa: E731
    saida = {
        "novos": novos, "contra": contra, "n": int(y.size),
        "completo": m(), "corpo": m(mascara=corpo), "cauda": m(mascara=~corpo),
        "completo_p": m(w=w), "corpo_p": m(w=w, mascara=corpo), "cauda_p": m(w=w, mascara=~corpo),
        "topo": parte_do_topo(y, base, novo, k=k),
        "topo_corpo": parte_do_topo(y[corpo], base[corpo], novo[corpo], k=k),
    }
    saida["previsao_oficial"] = RAZAO_OFICIAL * saida["corpo"]["ganho"]
    return saida | {"veredito": veredito(saida)}


def veredito(rel: dict) -> dict:
    """Portão de envio: ganho no corpo com IC > 0, não concentrado, e firme sob o peso.

    `topo_corpo` é a fração do ganho **do corpo** que mora nos 20 voos de maior ganho; é
    nela que a v37 estourou (249 %), não no ganho completo (129 %).
    """
    corpo, corpo_p, topo = rel["corpo"], rel["corpo_p"], rel["topo_corpo"]
    testes = {
        "corpo > 0": corpo["ganho"] > CORPO_MIN,
        "IC do corpo > 0": corpo["ic_baixo"] > 0,
        f"top-{topo['k']} do corpo < {TOPO_MAX:.0%}": topo["parte"] < TOPO_MAX,
        "corpo com peso 2026 > 0": corpo_p["ganho"] > CORPO_MIN,
    }
    faltou = [nome for nome, ok in testes.items() if not ok]
    return {"passa": not faltou, "testes": testes,
            "motivo": "transferência ok" if not faltou else "falha: " + "; ".join(faltou)}


def texto(rel: dict) -> str:
    f = lambda d: f"{d['ganho']:+.3f} (IC {d['ic_baixo']:+.2f} a {d['ic_alto']:+.2f})"  # noqa: E731
    linhas = [f"novos:  {' + '.join(rel['novos'])}",
              f"contra: {' + '.join(rel['contra'])}   ({rel['n']:,} voos)", "",
              "| parcela | sem peso | com peso 2026 |", "|---|---|---|",
              f"| completo | {f(rel['completo'])} | {f(rel['completo_p'])} |",
              f"| corpo (y ≤ {TAIL_S} s) | {f(rel['corpo'])} | {f(rel['corpo_p'])} |",
              f"| cauda (y > {TAIL_S} s) | {f(rel['cauda'])} | {f(rel['cauda_p'])} |", "",
              f"top-{rel['topo']['k']} voos: {rel['topo']['parte']:.1%} do ganho completo, "
              f"{rel['topo_corpo']['parte']:.1%} do ganho do corpo",
              f"delta oficial previsto: {rel['previsao_oficial']:+.2f} s "
              f"(±0,15 s de erro médio no backtest)", ""]
    for nome, ok in rel["veredito"]["testes"].items():
        linhas.append(f"  [{'x' if ok else ' '}] {nome}")
    return "\n".join(linhas + ["", rel["veredito"]["motivo"]])


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Checagem de transferência 2025 → 2026")
    p.add_argument("ids", nargs="*", help="ids das corridas do candidato (média)")
    p.add_argument("--contra", nargs="+", default=None, help="ids da referência (padrão: campeã)")
    p.add_argument("--semente", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--pesos", action="store_true", help="só (re)calcula o peso adversarial")
    a = p.parse_args(argv)
    if a.pesos:
        t = pesos(a.semente, recalcular=True)
        print(f"{len(t):,} voos · AUC {t['auc'].iloc[0]:.4f} · "
              f"w médio {t['w'].mean():.3f} · P99 {t['w'].quantile(0.99):.3f}")
        return
    if not a.ids:
        p.error("informe os ids do candidato (ou use --pesos)")
    rel = relatorio(a.ids, a.contra, semente=a.semente)
    print(json.dumps(rel, ensure_ascii=False, indent=2) if a.json else texto(rel))


if __name__ == "__main__":
    main()
