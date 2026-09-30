"""Esteira de experimentos: fila SQLite, trabalhador 24/7, gerador de vizinhos, relatório.

    bin/run src/esteira.py add --tipo corretor --receita '{"--fila": true}' [--prioridade N] [--origem X]
    bin/run src/esteira.py fila | status | pausar | retomar | gerar
    bin/run src/esteira.py trabalhar          # o serviço prc-esteira roda isto

Spec: docs/superpowers/specs/2026-09-30-esteira-design.md
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from runlog import ROOT

DB = ROOT / "data" / "esteira.db"
PAUSA = ROOT / "data" / "esteira.pausa"
RELATORIO = ROOT / "docs" / "esteira.md"

# config da corrida → flag do stack.py (booleanas)
FLAGS = {"externos": "--externos", "plano13": "--plano13", "dist_plano": "--dist-plano",
         "superficie": "--superficie", "mapa": "--mapa", "corretor_sem_ctx": "--corretor-sem-ctx",
         "ref_cel": "--corretor-ref"}


def hash_receita(tipo: str, receita: dict) -> str:
    return hashlib.sha256(json.dumps([tipo, receita], sort_keys=True).encode()).hexdigest()[:16]


def receita_de_config(cfg: dict) -> dict:
    r = {"--conjunto": cfg.get("corretor") == "conjunto"}
    r |= {flag: True for chave, flag in FLAGS.items() if cfg.get(chave)}
    if cfg.get("fila") is True:
        r["--fila"] = True
    elif cfg.get("fila") == "stand":
        r["--stand-prefixo"] = True
    if cfg.get("rounds"):
        r["--corretor-rounds"] = int(cfg["rounds"])
    if cfg.get("corretor_params"):
        r["--corretor-params"] = cfg["corretor_params"]
    return {k: v for k, v in r.items() if v}


def argv_corretor(receita: dict, base: str, oof: str | None) -> list[str]:
    """Argumentos do `stack.py`; `oof=None` (base nova) calcula o oof fora do bloco do zero."""
    argv = ["--base", base, "--crossfit", *(["--reusar-oof", oof] if oof else [])]
    for flag, valor in sorted(receita.items()):
        if valor is True:
            argv.append(flag)
        elif isinstance(valor, dict):
            argv += [flag, json.dumps(valor, sort_keys=True)]
        else:
            argv += [flag, str(valor)]
    return argv


class Fila:
    def __init__(self, caminho: Path = DB) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(caminho)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            create table if not exists candidatos (
              id integer primary key, criado real, origem text, tipo text, receita text,
              hash text, prioridade integer, estado text, campea text, run_id text,
              resultado text, motivo text, inicio real, fim real);
            create unique index if not exists dedup on candidatos(hash, campea);
            create table if not exists meta (chave text primary key, valor text);""")

    def add(self, tipo: str, receita: dict, origem: str, prioridade: int = 0,
            campea: str = "") -> int | None:
        try:
            cur = self.db.execute(
                "insert into candidatos (criado, origem, tipo, receita, hash, prioridade, estado, campea)"
                " values (?,?,?,?,?,?, 'fila', ?)",
                (time.time(), origem, tipo, json.dumps(receita, sort_keys=True),
                 hash_receita(tipo, receita), prioridade, campea))
        except sqlite3.IntegrityError:
            return None
        self.db.commit()
        return cur.lastrowid

    def proximo(self) -> dict | None:
        row = self.db.execute("select * from candidatos where estado='fila'"
                              " order by prioridade desc, id asc limit 1").fetchone()
        return dict(row) if row else None

    def marcar(self, cid: int, estado: str, **campos) -> None:
        sets = ", ".join(["estado=?", *(f"{k}=?" for k in campos)])
        vals = [estado, *(json.dumps(v) if isinstance(v, dict) else v for v in campos.values()), cid]
        self.db.execute(f"update candidatos set {sets} where id=?", vals)
        self.db.commit()

    def recuperar(self) -> int:
        n = self.db.execute("update candidatos set estado='fila' where estado='rodando'").rowcount
        self.db.commit()
        return n

    def ultimos(self, n: int = 30) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "select * from candidatos where estado in ('feito','falhou','pulado')"
            " order by fim desc limit ?", (n,))]

    def contar(self, estado: str) -> int:
        return self.db.execute("select count(*) from candidatos where estado=?", (estado,)).fetchone()[0]

    def meta(self, chave: str, valor: str | None = None) -> str | None:
        if valor is not None:
            self.db.execute("insert or replace into meta values (?,?)", (chave, valor))
            self.db.commit()
            return valor
        row = self.db.execute("select valor from meta where chave=?", (chave,)).fetchone()
        return row[0] if row else None


def relatorio(fila: Fila, champ: dict) -> str:
    agora = time.time()
    feitos24 = [c for c in fila.ultimos(1000) if c["fim"] and agora - c["fim"] < 86400]
    linhas = [
        "# Esteira de experimentos", "",
        f"Atualizado em {time.strftime('%Y-%m-%d %H:%M')}. Spec: "
        "`docs/superpowers/specs/2026-09-30-esteira-design.md`.", "",
        "## Campeã", "",
        f"- Membros: {', '.join(m['id'] for m in champ['membros'])}",
        f"- Simulação: completo {champ.get('rmse_simulacao')} · sem loteria {champ.get('sem_loteria')}",
        f"- Última enviada: {json.dumps(champ.get('enviada'))}",
        f"- Arquivo pronto esperando ok: {fila.meta('pronto') or 'nenhum'}", "",
        "## Vazão", "",
        f"- Fila: {fila.contar('fila')} · rodando: {fila.contar('rodando')} · "
        f"feitos nas últimas 24 h: {len(feitos24)} · consultas à metade B: {fila.meta('consultas_b') or 0}", "",
        "## Últimos candidatos", "",
        "| id | origem | receita | decisão | A sem lot. | B sem lot. | min |", "|---|---|---|---|---|---|---|",
    ]
    for c in fila.ultimos(30):
        r = json.loads(c["resultado"]) if c["resultado"] else {}
        a, b = (r.get("a") or {}), (r.get("b") or {})
        minutos = round((c["fim"] - c["inicio"]) / 60, 1) if c["fim"] and c["inicio"] else ""
        fmt = lambda d: f"{d['ganho']:+.2f}" if d else "—"  # noqa: E731
        linhas.append(f"| {c['id']} | {c['origem']} | `{c['receita']}` | {c['motivo'] or c['estado']} | "
                      f"{fmt(a)} | {fmt(b)} | {minutos} |")
    revisoes = fila.meta("revisoes") or ""
    return "\n".join(linhas + ["", "## Revisões", "", revisoes, ""])
