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
import subprocess
import time
from pathlib import Path

import campeao
import regua
from runlog import ROOT

DB = ROOT / "data" / "esteira.db"
PAUSA = ROOT / "data" / "esteira.pausa"
RELATORIO = ROOT / "docs" / "esteira.md"

MEMORIA_MIN_GB = 10.0
ROUNDS = (300, 500, 700)
PARAMS_GRADE = {"learning_rate": (0.03, 0.05), "num_leaves": (63, 127, 255),
                "lambda_l2": (0, 10, 50), "min_data_in_leaf": (100, 200)}
BLOCOS = ("--superficie", "--mapa", "--corretor-ref", "--dist-plano", "--corretor-sem-ctx")
GANHO_ENVIO_S, ENVIO_INTERVALO_S = 0.5, 6 * 3600

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
    erros = [f"- Último erro de {nome}: {fila.meta(f'erro_{nome}')}"
             for nome in ("envio", "commit") if fila.meta(f"erro_{nome}")]
    linhas = [
        "# Esteira de experimentos", "",
        f"Atualizado em {time.strftime('%Y-%m-%d %H:%M')}. Spec: "
        "`docs/superpowers/specs/2026-09-30-esteira-design.md`.", "",
        "## Campeã", "",
        f"- Membros: {', '.join(m['id'] for m in champ['membros'])}",
        f"- Simulação: completo {champ.get('rmse_simulacao')} · sem loteria {champ.get('sem_loteria')}",
        f"- Última enviada: {json.dumps(champ.get('enviada'))}",
        f"- Arquivo pronto esperando ok: {fila.meta('pronto') or 'nenhum'}",
        *erros,
        "",
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


def carregar_campea() -> dict:
    return campeao.carregar()


def salvar_campea(membros: list[str]) -> None:
    atual = campeao.carregar()
    nova = campeao.nova([campeao.registro(i) for i in membros], atual["pos_regras"], atual["enviada"])
    campeao.salvar(nova)


def id_campea(champ: dict) -> str:
    """Chave estável da campeã para deduplicar candidatos (muda quando os membros mudam)."""
    return "+".join(sorted(m["id"] for m in champ["membros"]))


def commitar_promocao(membros: list[str], fila: "Fila | None" = None,
                      rodar=subprocess.run) -> str | None:
    """Commit da campeã nova junto com o relatório já reescrito.

    Devolve o motivo quando o commit falha (e grava em `erro_commit` se a fila veio)."""
    alvos = ["champion.json", "docs/esteira.md"]
    try:
        rodar(["git", "add", *alvos], cwd=ROOT, check=True)
        rodar(["git", "commit", "-qm", f"esteira: promove {' + '.join(membros)}", *alvos],
              cwd=ROOT, check=True)
    except subprocess.CalledProcessError as e:
        erro = f"git falhou (código {e.returncode})"
        if fila is not None:
            fila.meta("erro_commit", erro)
        return erro
    return None


def memoria_livre_gb() -> float:
    for linha in Path("/proc/meminfo").read_text().splitlines():
        if linha.startswith("MemAvailable:"):
            return int(linha.split()[1]) / 1024 / 1024
    return 0.0


def vizinhos(champ: dict) -> list[dict]:
    """Receitas a um passo da campeã: liga/desliga um bloco, troca o bloco de fila,
    varia os rounds ou um parâmetro do corretor por vez."""
    saida = []
    for m in champ["membros"]:
        r = receita_de_config(m["config"])
        for b in BLOCOS:
            v = dict(r)
            if b in v:
                v.pop(b)
            else:
                v[b] = True
            saida.append(v)
        for alt in ("--stand-prefixo", "--fila", None):  # três estados do bloco fila
            v = {k: x for k, x in r.items() if k not in ("--stand-prefixo", "--fila")}
            if alt:
                v[alt] = True
            saida.append(v)
        for n in ROUNDS:
            saida.append(r | {"--corretor-rounds": n})
        atual = r.get("--corretor-params", {})
        for p, valores in PARAMS_GRADE.items():
            for x in valores:
                if atual.get(p) != x:
                    saida.append(r | {"--corretor-params": atual | {p: x}})
    atuais = {json.dumps(receita_de_config(m["config"]), sort_keys=True) for m in champ["membros"]}
    unicos = {json.dumps(v, sort_keys=True): v for v in saida}
    return [v for k, v in unicos.items() if k not in atuais]


def executar(c: dict, champ: dict, rodar=subprocess.run) -> str:
    nome = f"e{c['id']}"
    receita = json.loads(c["receita"])
    if c["tipo"] == "corretor":
        argv = argv_corretor(receita, champ["base"], champ["oof"])
        rodar(["bin/run", "src/stack.py", nome, *argv], cwd=ROOT, check=True)
    else:  # base: experiment.py com a receita da base, depois stack com a receita do 1º membro
        rodar(["bin/run", "src/experiment.py", f"{nome}_base", *receita["base"]], cwd=ROOT, check=True)
        base_id = campeao.ultimo_por_nome(f"{nome}_base")["id"]
        corr = receita_de_config(champ["membros"][0]["config"])
        argv = argv_corretor(corr, base_id, None)  # base nova: oof calculado do zero
        rodar(["bin/run", "src/stack.py", nome, *argv], cwd=ROOT, check=True)
    return campeao.ultimo_por_nome(nome)["id"]


def passo(fila: Fila, rodar=subprocess.run, avaliar=regua.avaliar) -> bool:
    """Um candidato do começo ao fim; False quando não há nada a fazer (ou pausa)."""
    if PAUSA.exists():
        return False
    champ = carregar_campea()
    c = fila.proximo()
    if c is None:
        for v in vizinhos(champ):
            fila.add("corretor", v, "gerador", 0, id_campea(champ))
        c = fila.proximo()
        if c is None:
            talvez_enviar(fila, champ, rodar)
            return False
    while memoria_livre_gb() < MEMORIA_MIN_GB:
        time.sleep(60)
    fila.marcar(c["id"], "rodando", inicio=time.time())
    try:
        run_id = executar(c, champ, rodar)
        membros = [m["id"] for m in champ["membros"]]
        r = avaliar(membros, run_id, semente=int(fila.meta("semente") or 0))
    except subprocess.CalledProcessError as e:
        fila.marcar(c["id"], "falhou", fim=time.time(), motivo=f"código {e.returncode}")
        talvez_enviar(fila, carregar_campea(), rodar)
        return True
    except (SystemExit, Exception) as e:  # noqa: B014 — SystemExit não é Exception
        fila.marcar(c["id"], "falhou", fim=time.time(),
                    motivo=f"{type(e).__name__}: {e}"[:200])
        talvez_enviar(fila, carregar_campea(), rodar)
        return True
    if r.get("b") is not None:
        fila.meta("consultas_b", str(int(fila.meta("consultas_b") or 0) + 1))
    fila.marcar(c["id"], "feito" if r["aprovado"] else "pulado", fim=time.time(), run_id=run_id,
                resultado={k: r[k] for k in ("a", "b", "completo", "proposta", "membros")},
                motivo=r["motivo"])
    if r["aprovado"]:
        salvar_campea(r["membros"])
    talvez_enviar(fila, carregar_campea(), rodar)
    RELATORIO.write_text(relatorio(fila, carregar_campea()))
    if r["aprovado"]:
        commitar_promocao(r["membros"], fila)
    return True


def talvez_enviar(fila: Fila, champ: dict, rodar=subprocess.run) -> str | None:
    """Arquiva uma submissão quando a campeã ganhou o bastante e faz tempo desde a última."""
    enviada = (champ.get("enviada") or {}).get("sem_loteria")
    ultimo = float(fila.meta("ultimo_arquivo_em") or 0)
    if (fila.meta("pronto") or enviada is None or champ["sem_loteria"] is None
            or enviada - champ["sem_loteria"] < GANHO_ENVIO_S
            or time.time() - ultimo < ENVIO_INTERVALO_S):
        return None
    versao = 1 + max(json.loads(l)["versao"]
                     for l in (ROOT / "submissions.jsonl").read_text().splitlines() if l.strip())
    while memoria_livre_gb() < MEMORIA_MIN_GB:
        time.sleep(60)
    fila.meta("ultimo_arquivo_em", str(time.time()))  # a trava de 6 h vale também para a falha
    try:
        rodar(["bin/run", "src/train.py", "submit", str(versao)], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as e:
        fila.meta("erro_envio", f"submit v{versao}: código {e.returncode}")
        return None
    except (SystemExit, Exception) as e:  # noqa: B014 — SystemExit não é Exception
        fila.meta("erro_envio", f"submit v{versao}: {type(e).__name__}: {e}"[:200])
        return None
    fila.meta("pronto_sem_loteria", str(champ["sem_loteria"]))
    return fila.meta("pronto", f"submissions/outgoing-boat_v{versao}.parquet "
                               f"(campeã {', '.join(m['id'] for m in champ['membros'])})")


def main(argv: list[str] | None = None) -> None:
    import argparse
    import signal

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # `... | head` não vira traceback

    p = argparse.ArgumentParser(description="Esteira de experimentos")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="enfileira um candidato")
    a.add_argument("--tipo", choices=("corretor", "base"), default="corretor")
    a.add_argument("--receita", required=True, help="JSON da receita")
    a.add_argument("--prioridade", type=int, default=0)
    a.add_argument("--origem", default="usuario")
    for nome in ("fila", "status", "pausar", "retomar", "gerar", "trabalhar"):
        sub.add_parser(nome)
    sub.add_parser("semente").add_argument("n", type=int)
    sub.add_parser("revisao").add_argument("texto")
    sub.add_parser("enviado").add_argument("n", type=int)
    args = p.parse_args(argv)

    fila = Fila()
    if args.cmd == "add":
        campea = id_campea(carregar_campea())
        cid = fila.add(args.tipo, json.loads(args.receita), args.origem, args.prioridade, campea)
        print(f"candidato {cid}" if cid else "já estava na fila para esta campeã")
    elif args.cmd == "fila":
        for c in fila.db.execute("select * from candidatos where estado='fila'"
                                 " order by prioridade desc, id asc"):
            print(f"{c['id']:>5} p{c['prioridade']} {c['origem']:<8} {c['tipo']:<8} {c['receita']}")
    elif args.cmd == "status":
        print(relatorio(fila, carregar_campea()))
    elif args.cmd == "pausar":
        PAUSA.parent.mkdir(parents=True, exist_ok=True)
        PAUSA.write_text(time.strftime("%Y-%m-%d %H:%M\n"))
        print(f"pausada ({PAUSA})")
    elif args.cmd == "retomar":
        PAUSA.unlink(missing_ok=True)
        print("retomada")
    elif args.cmd == "gerar":
        champ = carregar_campea()
        n = sum(fila.add("corretor", v, "gerador", 0, id_campea(champ)) is not None
                for v in vizinhos(champ))
        print(f"{n} vizinhos novos na fila")
    elif args.cmd == "semente":
        fila.meta("semente", str(args.n))
        fila.meta("consultas_b", "0")
        print(f"semente {args.n}; consultas à metade B zeradas")
    elif args.cmd == "revisao":
        anterior = fila.meta("revisoes") or ""
        fila.meta("revisoes", f"{anterior}{time.strftime('%Y-%m-%d')}: {args.texto}\n")
        print("revisão anotada")
    elif args.cmd == "enviado":
        fila.meta("pronto", "")
        champ = carregar_campea()
        gerado = fila.meta("pronto_sem_loteria")  # valor do momento em que o arquivo foi gerado
        sem_lot = float(gerado) if gerado else champ["sem_loteria"]
        champ["enviada"] = {"versao": args.n, "sem_loteria": sem_lot}
        campeao.salvar(champ)
        fila.meta("pronto_sem_loteria", "")
        print(f"campeã marcada como enviada na v{args.n} (sem loteria {sem_lot})")
    elif args.cmd == "trabalhar":
        recuperados = fila.recuperar()
        if recuperados:
            print(f"{recuperados} candidatos devolvidos à fila")
        while True:
            if not passo(fila):
                time.sleep(60)


if __name__ == "__main__":
    main()
