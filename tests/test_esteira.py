import esteira


def test_duplicado_contra_a_mesma_campea_nao_entra(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    assert f.add("corretor", {"--fila": True, "--corretor-rounds": 500}, "usuario", campea="C1")
    assert f.add("corretor", {"--corretor-rounds": 500, "--fila": True}, "gerador", campea="C1") is None
    assert f.add("corretor", {"--fila": True, "--corretor-rounds": 500}, "gerador", campea="C2")


def test_proximo_por_prioridade_depois_antiguidade(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    a = f.add("corretor", {"a": 1}, "usuario", 0, "C")
    b = f.add("corretor", {"b": 1}, "usuario", 5, "C")
    c = f.add("corretor", {"c": 1}, "usuario", 5, "C")
    assert [f.proximo()["id"] for _ in range(1)] == [b]
    f.marcar(b, "feito"); assert f.proximo()["id"] == c
    f.marcar(c, "feito"); assert f.proximo()["id"] == a


def test_recuperar_devolve_rodando_para_a_fila(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    i = f.add("corretor", {"a": 1}, "usuario", 0, "C")
    f.marcar(i, "rodando")
    assert f.recuperar() == 1 and f.proximo()["id"] == i


def test_receita_de_config_e_argv_ida_e_volta():
    cfg = {"corretor": "conjunto", "externos": True, "plano13": True, "fila": "stand",
           "dist_plano": True, "ref_cel": True, "mapa": True, "rounds": 500,
           "corretor_params": {"num_leaves": 127}}
    r = esteira.receita_de_config(cfg)
    argv = esteira.argv_corretor(r, "BASE", "OOF")
    for flag in ("--conjunto", "--externos", "--plano13", "--stand-prefixo", "--dist-plano",
                 "--corretor-ref", "--mapa", "--crossfit"):
        assert flag in argv
    assert argv[argv.index("--corretor-rounds") + 1] == "500"
    assert argv[argv.index("--reusar-oof") + 1] == "OOF"
    assert argv[argv.index("--base") + 1] == "BASE"


def test_vizinhos_mudam_uma_coisa_por_vez():
    champ = {"membros": [{"id": "m", "config": {"corretor": "conjunto", "fila": "stand", "rounds": 500,
                                               "mapa": True}}]}
    vs = esteira.vizinhos(champ)
    base = esteira.receita_de_config(champ["membros"][0]["config"])
    for v in vs:
        dif = {k for k in set(base) | set(v) if base.get(k) != v.get(k)}
        assert 1 <= len(dif) <= 2  # trocar stand-prefixo por fila mexe em duas chaves
    assert any(v.get("--corretor-rounds") == 700 for v in vs)
    assert any("--mapa" not in v for v in vs)


def test_passo_promove_e_registra(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = {"base": "B", "oof": "O", "membros": [{"id": "m", "config": {"rounds": 500}}],
             "pos_regras": ["roma"], "sem_loteria": 233.0, "enviada": {"versao": 32, "sem_loteria": 233.0}}
    salvo = {}
    monkeypatch.setattr(esteira, "carregar_campea", lambda: champ)
    monkeypatch.setattr(esteira, "salvar_campea", lambda membros: salvo.setdefault("m", membros))
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: 12.0)
    monkeypatch.setattr(esteira, "executar", lambda c, ch, rodar=None: "RUN1")
    ordem = []
    monkeypatch.setattr(esteira, "commitar_promocao",
                        lambda membros, fila=None: ordem.append((tmp_path / "esteira.md").exists()))
    monkeypatch.setattr(esteira, "RELATORIO", tmp_path / "esteira.md")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    f.add("corretor", {"--fila": True}, "usuario", 0, "m")
    ok = esteira.passo(f, avaliar=lambda membros, novo, semente=0: {
        "aprovado": True, "membros": ["m", "RUN1"], "proposta": "soma", "a": {"ganho": 0.9},
        "b": {"ganho": 0.5}, "completo": 0.3, "motivo": "aprovado", "avaliadas": 3})
    assert ok and salvo["m"] == ["m", "RUN1"]
    assert f.ultimos(1)[0]["estado"] == "feito"
    assert ordem == [True]  # o commit vê o relatório já reescrito


def test_passo_marca_falhou_quando_executar_ou_avaliar_estoura(tmp_path, monkeypatch):
    champ = {"base": "B", "oof": "O", "membros": [{"id": "m", "config": {"rounds": 500}}],
             "pos_regras": ["roma"], "sem_loteria": 233.0, "enviada": None}
    monkeypatch.setattr(esteira, "carregar_campea", lambda: champ)
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: 12.0)
    monkeypatch.setattr(esteira, "RELATORIO", tmp_path / "esteira.md")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")

    def estoura(c, ch, rodar=None):
        raise SystemExit("nenhuma corrida com nome e1")

    f = esteira.Fila(tmp_path / "a.db")
    f.add("corretor", {"--fila": True}, "usuario", 0, "m")
    monkeypatch.setattr(esteira, "executar", estoura)
    assert esteira.passo(f) is True
    c = f.ultimos(1)[0]
    assert c["estado"] == "falhou" and "SystemExit" in c["motivo"] and c["fim"]
    assert f.proximo() is None  # não volta ao topo da fila

    g = esteira.Fila(tmp_path / "b.db")
    g.add("corretor", {"--fila": True}, "usuario", 0, "m")
    monkeypatch.setattr(esteira, "executar", lambda c, ch, rodar=None: "RUN1")

    def avaliar_ruim(membros, novo, semente=0):
        raise ValueError("holdouts diferentes")

    assert esteira.passo(g, avaliar=avaliar_ruim) is True
    c = g.ultimos(1)[0]
    assert c["estado"] == "falhou" and c["motivo"].startswith("ValueError:")


def test_passo_respeita_pausa(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    (tmp_path / "pausa").write_text("")
    f.add("corretor", {"a": 1}, "usuario", 0, "m")
    assert esteira.passo(f) is False


def _champ(membros=("m",), sem_loteria=233.0, enviada=None):
    return {"base": "B", "oof": "O", "sem_loteria": sem_loteria, "pos_regras": ["roma"],
            "enviada": enviada, "membros": [{"id": i, "config": {"rounds": 500}} for i in membros]}


def test_commitar_promocao_versiona_o_relatorio_antes_nao_rastreado(tmp_path, monkeypatch):
    import subprocess

    raiz = tmp_path / "repo"
    (raiz / "docs").mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=raiz, check=True,  # noqa: E731
                                    capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (raiz / "champion.json").write_text("{}")
    git("add", "champion.json"); git("commit", "-qm", "inicial")
    (raiz / "champion.json").write_text('{"membros": []}')
    (raiz / "docs" / "esteira.md").write_text("# relatório\n")  # nunca rastreado

    monkeypatch.setattr(esteira, "ROOT", raiz)
    assert esteira.commitar_promocao(["m1", "m2"]) is None
    log = subprocess.run(["git", "show", "--stat", "--name-only", "--format=%s", "HEAD"],
                         cwd=raiz, capture_output=True, text=True).stdout
    assert "esteira: promove m1 + m2" in log
    assert "docs/esteira.md" in log and "champion.json" in log
    assert not subprocess.run(["git", "status", "--porcelain"], cwd=raiz,
                              capture_output=True, text=True).stdout.strip()


def test_commitar_promocao_anota_erro_e_segue(tmp_path):
    import subprocess

    f = esteira.Fila(tmp_path / "e.db")

    def rodar(argv, **kw):
        raise subprocess.CalledProcessError(128, argv)

    erro = esteira.commitar_promocao(["m"], f, rodar)
    assert erro and "128" in erro and "128" in f.meta("erro_commit")


def test_id_campea_soma_os_membros_e_solta_a_dedup(tmp_path):
    c1, c2 = _champ(["m"]), _champ(["m2", "m"])
    assert esteira.id_campea(c1) == "m" and esteira.id_campea(c2) == "m+m2"
    f = esteira.Fila(tmp_path / "e.db")
    receita = {"--fila": True}
    assert f.add("corretor", receita, "gerador", 0, esteira.id_campea(c1))
    assert f.add("corretor", receita, "gerador", 0, esteira.id_campea(c1)) is None
    assert f.add("corretor", receita, "gerador", 0, esteira.id_campea(c2))


def _pronta(tmp_path, monkeypatch, fila):
    """Campeã com 0,5 s de ganho sobre a enviada e a trava de 6 h vencida."""
    (tmp_path / "submissions.jsonl").write_text('{"versao": 32}\n')
    monkeypatch.setattr(esteira, "ROOT", tmp_path)
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: 12.0)
    fila.meta("ultimo_arquivo_em", "0")
    return _champ(sem_loteria=232.5, enviada={"versao": 32, "sem_loteria": 233.0})


def test_talvez_enviar_anota_erro_quando_o_submit_falha(tmp_path, monkeypatch):
    import subprocess

    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)

    def rodar(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)

    assert esteira.talvez_enviar(f, champ, rodar) is None
    assert "v33" in f.meta("erro_envio") and "código 1" in f.meta("erro_envio")
    assert not f.meta("pronto")


def test_talvez_enviar_espera_memoria(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    livres, dormiu = iter([1.0, 2.0, 12.0]), []
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: next(livres))
    monkeypatch.setattr(esteira.time, "sleep", lambda s: dormiu.append(s))
    cmds = []
    assert esteira.talvez_enviar(f, champ, lambda argv, **kw: cmds.append(argv))
    assert dormiu == [60, 60] and cmds[0][-3:] == ["src/train.py", "submit", "33"]


def test_passo_sem_aprovacao_ainda_gera_o_arquivo(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    monkeypatch.setattr(esteira, "carregar_campea", lambda: champ)
    monkeypatch.setattr(esteira, "executar", lambda c, ch, rodar=None: "RUN1")
    monkeypatch.setattr(esteira, "RELATORIO", tmp_path / "esteira.md")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    cmds = []
    f.add("corretor", {"--fila": True}, "usuario", 0, "m")
    assert esteira.passo(f, rodar=lambda argv, **kw: cmds.append(argv), avaliar=lambda *a, **k: {
        "aprovado": False, "membros": ["m"], "proposta": "soma", "a": {"ganho": 0.1}, "b": None,
        "completo": 0.0, "motivo": "A: não seleciona", "avaliadas": 3})
    assert [c for c in cmds if "submit" in c] and f.meta("pronto")


def test_enviado_grava_o_sem_loteria_do_momento_da_geracao(tmp_path, monkeypatch, capsys):
    import campeao

    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    assert esteira.talvez_enviar(f, champ, lambda argv, **kw: None)
    assert f.meta("pronto_sem_loteria") == "232.5" and f.meta("pronto_membros") == "m"

    outra = _champ(["m", "n"], sem_loteria=231.0, enviada=champ["enviada"])
    monkeypatch.setattr(esteira, "Fila", lambda *a, **k: f)
    monkeypatch.setattr(esteira, "carregar_campea", lambda: outra)
    salvo = {}
    monkeypatch.setattr(campeao, "salvar", lambda c: salvo.update(c))
    esteira.main(["enviado", "33"])
    assert salvo["enviada"] == {"versao": 33, "sem_loteria": 232.5}
    assert not f.meta("pronto") and not f.meta("pronto_sem_loteria")
