"""Auditoria diária: acha o que ficou desatualizado entre código, dados, docs e placar.

Checa fatos verificáveis por máquina e grava docs/auditoria/AAAA-MM-DD.md com ✅/⚠️.
O que exige leitura (texto velho no meio de um .md) fica para a sessão: o relatório lista
o que conferir. Roda sozinho todo dia pelo timer `prc-auditoria` (systemd do usuário).

Uso:
    .venv/bin/python ferramentas/auditoria.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAI = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ferramentas"))

itens: list[tuple[bool, str]] = []


def ok(cond: bool, texto: str) -> None:
    itens.append((bool(cond), texto))


def sh(*cmd: str, cwd: Path = ROOT) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()


def envios() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "submissions.jsonl").read_text().splitlines() if l.strip()]


def checar_placar_e_envios() -> None:
    import projecao

    try:
        projecao.baixar()
        subprocess.run([sys.executable, str(ROOT / "ferramentas" / "projecao.py"), "--sem-baixar"],
                       cwd=ROOT, capture_output=True, check=True)
        ok(True, "placar do dia salvo em `placar/` e `docs/projecao.md` regenerado")
    except Exception as e:  # noqa: BLE001 — sem rede não derruba o resto
        ok(False, f"placar/projeção não atualizados: {e}")
    melhor = min(e["oficial"] for e in envios() if e["oficial"] is not None)
    ultima = json.loads(sorted((ROOT / "placar").glob("*.json"))[-1].read_text())
    no_placar = next((x["best"] for x in ultima["teams"] if x["team"] == projecao.TEAM), None)
    ok(no_placar is not None and abs(no_placar - melhor) < 0.01,
       f"placar público ({no_placar}) = melhor nota em submissions.jsonl ({melhor:.2f})")
    try:
        from s3 import client

        s, bucket = client(), f"prc-2026-{projecao.TEAM}"
        feitos = {int(m.group(1)) for o in s.list_objects(bucket)
                  if (m := re.search(r"_v(\d+)\.parquet_result\.json$", o.object_name))}
        faltam = sorted(feitos - {e["versao"] for e in envios()})
        ok(not faltam, "todo resultado do bucket está em submissions.jsonl"
           + (f" — faltam v{faltam}" if faltam else ""))
    except Exception as e:  # noqa: BLE001
        ok(False, f"não consegui ler o bucket: {e}")


def checar_campea() -> None:
    champ = json.loads((ROOT / "champion.json").read_text())
    reg = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "experiments.jsonl").read_text().splitlines() if l.strip()}
    ids = [m["id"] for m in champ["membros"]]
    ok(all(i in reg for i in ids), f"membros da campeã existem no experiments.jsonl ({', '.join(ids)})")
    ok(all(reg[i]["config"]["base"] == champ["base"] for i in ids if i in reg), "membros com a mesma base")
    readme = (ROOT / "README.md").read_text()
    ok(all(i in readme for i in ids), "README cita os membros da campeã")
    melhor = min(e["oficial"] for e in envios() if e["oficial"] is not None)
    txt = f"{melhor:.2f}".replace(".", ",")
    ok(txt in readme, f"README cita a melhor nota oficial ({txt})")
    ctx = (PAI / "CONTEXTO.md").read_text()
    ok(txt in ctx, f"CONTEXTO.md cita a melhor nota oficial ({txt})")


def checar_estrutura() -> None:
    readme = (ROOT / "README.md").read_text()
    faltam = [f"{d}/{p.name}" for d in ("src", "ferramentas") for p in sorted((ROOT / d).glob("*.py"))
              if f"{d}/{p.name}" not in readme]
    ok(not faltam, "todo `src/*.py` e `ferramentas/*.py` aparece no README" + (f" — faltam {faltam}" if faltam else ""))
    planos = sorted((ROOT / "docs/superpowers/plans").glob("*.md"))
    ativo = planos[-1]
    abertos = sum(1 for l in ativo.read_text().splitlines() if l.lstrip().startswith("- [ ]"))
    ok(True, f"plano mais recente `{ativo.name}`: {abertos} caixas abertas (conferir se ainda valem)")
    saltos = json.loads((ROOT / "saltos.json").read_text())
    idade = (datetime.now(timezone.utc).date() - datetime.fromisoformat(saltos["atualizado"]).date()).days
    ok(idade <= 2, f"saltos.json atualizado há {idade} dia(s)")


def checar_testes_e_git() -> None:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    ok(r.returncode == 0, f"pytest: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}")
    for repo in (ROOT, PAI):
        # o que a própria auditoria gera (placar, projeção, relatório) não conta como pendência
        gerados = ("placar/", "docs/projecao.md", "docs/auditoria/")
        sujo = "\n".join(l for l in sh("git", "status", "--porcelain", cwd=repo).splitlines()
                         if not any(g in l for g in gerados))
        sh("git", "fetch", "-q", cwd=repo)
        atras = sh("git", "rev-list", "--count", "@{u}..HEAD", cwd=repo)
        ok(not sujo, f"`{repo.name}`: sem mudanças fora do git" + (f" — {len(sujo.splitlines())} arquivo(s)" if sujo else ""))
        ok(atras in ("0", ""), f"`{repo.name}`: tudo enviado ao GitHub ({atras or '?'} commit(s) locais)")


def checar_dados() -> None:
    from adsb import RAIZ

    ev = RAIZ / "events.parquet"
    cortes = sorted((RAIZ / "cut").glob("*.parquet"))
    ok(ev.exists(), f"events.parquet existe ({len(cortes)} dias recortados)")
    if ev.exists() and cortes:
        ok(ev.stat().st_mtime >= max(p.stat().st_mtime for p in cortes),
           "events.parquet mais novo que o último recorte (senão: rodar src/adsb_events.py)")
    falhos = sh("systemctl", "--user", "--failed", "--no-legend", "--plain")
    ok("prc-" not in falhos, "nenhum serviço prc-* falhou" + (f": {falhos}" if "prc-" in falhos else ""))


def checar_esteira() -> None:
    import esteira

    if not esteira.DB.exists():
        ok(False, "esteira ainda não rodou (data/esteira.db ausente)")
        return
    f = esteira.Fila(esteira.DB)
    ult = f.ultimos(1000)
    dia = [c for c in ult if c["fim"] and time.time() - c["fim"] < 86400]
    falhas = sum(c["estado"] == "falhou" for c in dia)
    ok(len(dia) >= 24, f"esteira: {len(dia)} candidatos nas últimas 24 h (esperado ≥ 24)")
    ok(falhas <= max(2, len(dia) // 10), f"esteira: {falhas} falhas nas últimas 24 h")
    ok(int(f.meta("consultas_b") or 0) <= 50, f"esteira: {f.meta('consultas_b') or 0} consultas à metade B (≤ 50)")
    ok(not esteira.PAUSA.exists(), "esteira: não está pausada")


def main() -> int:
    for f in (checar_placar_e_envios, checar_campea, checar_estrutura, checar_testes_e_git, checar_dados,
              checar_esteira):
        try:
            f()
        except Exception as e:  # noqa: BLE001 — uma checagem quebrada vira item, não aborta
            ok(False, f"{f.__name__} quebrou: {e}")
    hoje = datetime.now(timezone.utc)
    ruins = sum(not c for c, _ in itens)
    linhas = [f"# Auditoria {hoje:%Y-%m-%d %H:%M} UTC — {ruins} pendência(s)", ""]
    linhas += [f"- {'✅' if c else '⚠️'} {t}" for c, t in itens]
    linhas += ["", "Conferir na sessão (leitura): seção 'Modelo atual' e 'Roadmap' do README, "
               "'Retomar' do CONTEXTO.md, caixas do plano ativo e `saltos.json` contra o que foi feito ontem."]
    out = ROOT / "docs" / "auditoria" / f"{hoje:%Y-%m-%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(linhas) + "\n")
    print("\n".join(linhas))
    return 1 if ruins else 0


if __name__ == "__main__":
    sys.exit(main())
