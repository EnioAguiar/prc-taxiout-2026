"""Projeção até o fim da competição: onde estamos, quanto os líderes andam, se alcançamos.

Entradas (versionadas):
  placar/AAAA-MM-DD.json  fotos diárias do placar público (baixadas daqui)
  submissions.jsonl       nossos envios com a nota oficial
  saltos.json             fila de saltos candidatos (ganho oficial estimado, chance, dias)

Saída: docs/projecao.md (reescrito a cada rodada).

Método: a nota de corte de cada posição (1º, 3º, 10º, 50º) no prazo é projetada pela
velocidade dos últimos dias (reta nas fotos do placar), em dois cenários: ritmo atual e
metade do ritmo (retornos decrescentes). A nossa nota final sai de um Monte Carlo sobre
a fila de saltos que cabe nos dias restantes (cada salto funciona com a sua chance).

Uso:
    .venv/bin/python ferramentas/projecao.py [--sem-baixar]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLACAR = ROOT / "placar"
URL = "https://prc-leaderboard.fly.dev/data.json"
TEAM = "outgoing-boat"
# 11/10/2026 23:59:59 CET; em outubro a Europa está em CEST (UTC+2) → prazo conservador em UTC
PRAZO = datetime(2026, 10, 11, 21, 59, 59, tzinfo=timezone.utc)
POSICOES = [1, 3, 10, 50]
JANELA_DIAS = 7
MEIA_VIDA_DIAS = 7  # cenário desacelerando
SIMULACOES = 20_000
USO_DOS_DIAS = 0.8  # fração dos dias restantes que vira trabalho útil


def baixar() -> Path:
    with urllib.request.urlopen(URL, timeout=60) as r:
        d = json.load(r)
    path = PLACAR / f"{d['updated'][:10]}.json"
    PLACAR.mkdir(exist_ok=True)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return path


def fotos() -> list[tuple[datetime, dict]]:
    out = []
    for p in sorted(PLACAR.glob("*.json")):
        d = json.loads(p.read_text())
        t = datetime.fromisoformat(d["updated"].replace("Z", "+00:00"))
        out.append((t, d))
    return out


def corte(d: dict, k: int) -> float | None:
    notas = sorted(x["best"] for x in d["teams"])
    return notas[k - 1] if len(notas) >= k else None


def velocidade(serie: list[tuple[datetime, float]], agora: datetime) -> float | None:
    """s/dia pela reta dos pontos dos últimos JANELA_DIAS (negativo = melhora)."""
    pts = [(t, v) for t, v in serie if (agora - t).days <= JANELA_DIAS]
    if len(pts) < 2:
        return None
    x = np.array([(t - pts[0][0]).total_seconds() / 86400 for t, _ in pts])
    y = np.array([v for _, v in pts])
    return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0.5 else None


def posicao(d: dict, nota: float) -> int:
    return 1 + sum(x["best"] < nota for x in d["teams"] if x["team"] != TEAM)


def monte_carlo(nota: float, saltos: list[dict], dias: float, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Notas finais simuladas com os saltos pendentes que cabem nos dias (melhor valor/dia primeiro)."""
    pend = [s for s in saltos if s["status"] == "pendente"]
    pend.sort(key=lambda s: -s["ganho_s"] * s["prob"] / max(s["dias"], 0.25))
    cabem, usado = [], 0.0
    for s in pend:
        if usado + s["dias"] <= dias:
            cabem.append(s)
            usado += s["dias"]
    finais = np.full(SIMULACOES, nota)
    for s in cabem:
        finais -= (rng.random(SIMULACOES) < s["prob"]) * s["ganho_s"]
    return finais, cabem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sem-baixar", action="store_true")
    a = ap.parse_args()
    if not a.sem_baixar:
        baixar()
    agora = datetime.now(timezone.utc)
    dias = max((PRAZO - agora).total_seconds() / 86400, 0)
    fs = fotos()
    ultima = next(d for _, d in reversed(fs) if not d.get("parcial"))

    envios = [json.loads(l) for l in (ROOT / "submissions.jsonl").read_text().splitlines() if l.strip()]
    nossa = min(e["oficial"] for e in envios)
    nossa_placar = next((x["best"] for x in ultima["teams"] if x["team"] == TEAM), None)
    serie_nossa, melhor = [], np.inf
    for e in envios:
        melhor = min(melhor, e["oficial"])
        serie_nossa.append((datetime.fromisoformat(e["enviado_utc"]).replace(tzinfo=timezone.utc), melhor))

    linhas = [f"# Projeção — gerada em {agora:%Y-%m-%d %H:%M} UTC", "",
              f"Dias até o prazo (11/10 23:59:59, horário da Europa): **{dias:.1f}**. "
              f"Placar público: {ultima['updated']}, {len(ultima['teams'])} equipes.", "",
              f"Nossa melhor nota: **{nossa:.2f}** → posição **{posicao(ultima, nossa)}**"
              + ("" if nossa_placar is None or abs(nossa_placar - nossa) < 0.01
                 else f" (o placar ainda mostra {nossa_placar:.2f})"), "",
              "## Corte por posição", "",
              "| Posição | Hoje | Velocidade (s/dia) | No prazo: parado / desacelerando / ritmo atual | Falta para nós hoje |",
              "|---|---|---|---|---|"]
    alvos = {}
    for k in POSICOES:
        serie = [(t, c) for t, d in fs if (c := corte(d, k)) is not None]
        hoje = corte(ultima, k)
        v = velocidade(serie, agora) or 0.0
        # desacelerando: o ritmo cai pela metade a cada MEIA_VIDA_DIAS (retornos decrescentes)
        desac = hoje + v * MEIA_VIDA_DIAS / np.log(2) * (1 - 2 ** (-dias / MEIA_VIDA_DIAS))
        alvos[k] = {"parado": hoje, "desacelerando": desac, "ritmo atual": hoje + v * dias}
        cen = " / ".join(f"{x:.1f}" for x in alvos[k].values())
        linhas.append(f"| {k}º | {hoje:.2f} | {'—' if v == 0 else f'{v:+.2f}'} | {cen} | {nossa - hoje:.1f} s |")

    rng = np.random.default_rng(0)
    saltos = json.loads((ROOT / "saltos.json").read_text())["saltos"]
    finais, cabem = monte_carlo(nossa, saltos, dias * USO_DOS_DIAS, rng)
    v_nossa = velocidade(serie_nossa, agora)
    linhas += ["", "## Nossa projeção (fila de saltos em `saltos.json`)", "",
               f"Saltos que cabem em {dias * USO_DOS_DIAS:.1f} dias úteis: "
               + ", ".join(f"`{s['id']}` ({s['ganho_s']} s × {s['prob']:.0%})" for s in cabem) + ".", "",
               f"- Esperado: **{finais.mean():.1f}**; faixa 10–90 %: {np.percentile(finais, 10):.1f} a {np.percentile(finais, 90):.1f}; "
               f"se tudo funcionar: {nossa - sum(s['ganho_s'] for s in cabem):.1f}.",
               f"- Nosso ritmo nos últimos {JANELA_DIAS} dias: "
               + ("—" if v_nossa is None else f"{v_nossa:+.1f} s/dia (não extrapolar: vem de saltos, não de tendência)") + ".",
               "", "| Meta | Chance: parado / desacelerando / ritmo atual |", "|---|---|"]
    for k in POSICOES:
        linhas.append(f"| top {k} | " + " / ".join(f"{np.mean(finais <= x):.0%}" for x in alvos[k].values()) + " |")
    alvo3 = alvos[3]["desacelerando"]
    precisa = nossa - alvo3
    linhas += ["", f"Para o top 3 (cenário desacelerando, {alvo3:.1f}) faltam **{precisa:.0f} s**: "
               f"{precisa / max(dias, 1):.1f} s por dia, ou {100 * (1 - (alvo3 / nossa) ** (1 / max(dias, 1))):.2f} % por dia composto.",
               "", "Leitura: a chance vem só da fila de saltos. Salto novo com evidência → entra em `saltos.json`; "
               "experimento feito → status e ganho real atualizados. Estimativas sem medida são inferência."]
    (ROOT / "docs" / "projecao.md").write_text("\n".join(linhas) + "\n")
    print("\n".join(linhas))


if __name__ == "__main__":
    main()
