import json

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


def test_receita_com_lista_vira_flag_repetida():
    r = {"--conjunto": True, "--corretor-sem-feature": ["met_temp", "met_vis"]}

    argv = esteira.argv_corretor(r, "BASE", "OOF")

    assert argv.count("--corretor-sem-feature") == 2
    i = argv.index("--corretor-sem-feature")
    assert argv[i + 1 : i + 4] == ["met_temp", "--corretor-sem-feature", "met_vis"]


def test_vizinhos_mudam_uma_coisa_por_vez():
    champ = {"membros": [{"id": "m", "config": {"corretor": "conjunto", "fila": "stand", "rounds": 500,
                                               "mapa": True}}]}
    vs = [r for _, r in esteira.vizinhos(champ)]
    base = esteira.receita_de_config(champ["membros"][0]["config"])
    for v in vs:
        dif = {k for k in set(base) | set(v) if base.get(k) != v.get(k)}
        assert 1 <= len(dif) <= 2  # trocar stand-prefixo por fila mexe em duas chaves
    assert any(v.get("--corretor-params", {}).get("num_leaves") == 127 for v in vs)
    assert any("--mapa" not in v for v in vs)


def test_vizinhos_nao_variam_rodadas_nem_params_cortados():
    champ = {"membros": [{"id": "m", "config": {"rounds": 500, "corretor_params": {"num_leaves": 63}}}]}
    vs = [r for _, r in esteira.vizinhos(champ)]
    assert all(v.get("--corretor-rounds") == 500 for v in vs)
    params = [v.get("--corretor-params", {}) for v in vs]
    assert all("lambda_l2" not in p and "min_data_in_leaf" not in p for p in params)
    assert any(p.get("num_leaves") == 127 for p in params)


def test_executar_base_refaz_todos_os_membros_reusando_o_oof(monkeypatch):
    import campeao
    champ = {"membros": [
        {"id": "m1", "config": {"rounds": 500, "base": "B", "reusar_oof": "O"}},
        {"id": "m2", "config": {"mapa": True, "base": "B", "reusar_oof": "O"}}]}
    monkeypatch.setattr(campeao, "ultimo_por_nome", lambda n: {"id": f"ID:{n}"})
    cmds = []
    c = {"id": 7, "tipo": "base", "receita": json.dumps({"base": ["--superficie"]})}
    ids = esteira.executar(c, champ, rodar=lambda *a, **k: cmds.append(a[0]))
    assert ids == ["ID:e7_m0", "ID:e7_m1"]
    assert cmds[0][:4] == ["bin/run", "src/experiment.py", "e7_base", "--superficie"]
    assert len(cmds) == 3 and "--reusar-oof" not in cmds[1]
    assert cmds[1][cmds[1].index("--base") + 1] == "ID:e7_base"
    assert cmds[2][cmds[2].index("--reusar-oof") + 1] == "ID:e7_m0"


def test_executar_corretor_usa_a_base_do_membro_de_onde_a_receita_saiu(monkeypatch):
    import campeao
    champ = {"membros": [
        {"id": "m1", "config": {"corretor": "conjunto", "base": "B1", "reusar_oof": "O1"}},
        {"id": "m2", "config": {"corretor": "conjunto", "mapa": True, "base": "B2"}}]}
    monkeypatch.setattr(campeao, "ultimo_por_nome", lambda n: {"id": f"ID:{n}"})
    cmds = []
    rodar = lambda *a, **k: cmds.append(a[0])  # noqa: E731

    # vizinho de m2 (mapa + superfície): roda sobre B2, cujo oof é o próprio m2
    c = {"id": 9, "tipo": "corretor",
         "receita": json.dumps({"--conjunto": True, "--mapa": True, "--superficie": True})}
    assert esteira.executar(c, champ, rodar=rodar) == "ID:e9"
    assert cmds[0][cmds[0].index("--base") + 1] == "B2"
    assert cmds[0][cmds[0].index("--reusar-oof") + 1] == "m2"

    # vizinho de m1 (só liga a fila): roda sobre B1, com o oof que m1 reaproveita
    c = {"id": 10, "tipo": "corretor", "receita": json.dumps({"--conjunto": True, "--fila": True})}
    esteira.executar(c, champ, rodar=rodar)
    assert cmds[1][cmds[1].index("--base") + 1] == "B1"
    assert cmds[1][cmds[1].index("--reusar-oof") + 1] == "O1"


def test_passo_usa_avaliar_conjunto_quando_executar_devolve_lista(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = {"membros": [{"id": "m", "config": {"rounds": 500, "base": "B", "reusar_oof": "O"}}],
             "pos_regras": ["roma"], "sem_loteria": 233.0, "enviada": None}
    monkeypatch.setattr(esteira, "carregar_campea", lambda: champ)
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: 12.0)
    monkeypatch.setattr(esteira, "RELATORIO", tmp_path / "esteira.md")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    monkeypatch.setattr(esteira, "executar", lambda c, ch, rodar=None: ["R1", "R2"])
    vistos = {}

    def conjunto(membros, novos, semente=0):
        vistos["novos"] = novos
        return {"aprovado": False, "membros": novos, "proposta": "base_nova", "a": {"ganho": 0.1},
                "b": None, "completo": -1.0, "motivo": "A: não seleciona", "avaliadas": 1}

    f.add("base", {"base": ["--superficie"]}, "usuario", 0, "m")
    assert esteira.passo(f, avaliar=lambda *a, **k: 1 / 0, avaliar_conjunto=conjunto) is True
    c = f.ultimos(1)[0]
    assert vistos["novos"] == ["R1", "R2"] and c["run_id"] == "R1+R2" and c["estado"] == "pulado"


def test_passo_promove_e_registra(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = {"membros": [{"id": "m", "config": {"rounds": 500, "base": "B", "reusar_oof": "O"}}],
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
    champ = {"membros": [{"id": "m", "config": {"rounds": 500, "base": "B", "reusar_oof": "O"}}],
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
    return {"sem_loteria": sem_loteria, "pos_regras": ["roma"], "enviada": enviada,
            "membros": [{"id": i, "config": {"rounds": 500, "base": "B", "reusar_oof": "O"}}
                        for i in membros]}


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


def _pronta(tmp_path, monkeypatch, fila, transfere=True):
    """Campeã com 0,5 s de ganho sobre a enviada, trava de 6 h vencida e transferência ok."""
    (tmp_path / "submissions.jsonl").write_text('{"versao": 32}\n')
    monkeypatch.setattr(esteira, "ROOT", tmp_path)
    monkeypatch.setattr(esteira, "memoria_livre_gb", lambda: 12.0)
    monkeypatch.setattr(esteira, "transferencia_ok",
                        lambda champ: (transfere, "ok" if transfere else "falha: IC do corpo > 0"))
    fila.meta("ultimo_arquivo_em", "0")
    return _champ(["m", "n"], sem_loteria=232.5,
                  enviada={"versao": 32, "sem_loteria": 233.0, "membros": ["m"]})


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


def test_talvez_enviar_nao_reusa_a_versao_enviada_fora_do_registro(tmp_path, monkeypatch):
    """`esteira.py enviado 33` antes de o registro ganhar a linha da v33: o próximo é v34, não v33."""
    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    champ["enviada"] = {"versao": 33, "sem_loteria": 233.0}
    cmds = []
    assert esteira.talvez_enviar(f, champ, lambda argv, **kw: cmds.append(argv))
    assert cmds[0][-1] == "34" and "_v34.parquet" in f.meta("pronto")


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


def test_enviado_grava_o_sem_loteria_e_os_membros_do_momento_da_geracao(tmp_path, monkeypatch):
    import campeao

    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    assert esteira.talvez_enviar(f, champ, lambda argv, **kw: None)
    assert f.meta("pronto_sem_loteria") == "232.5"

    outra = _champ(["m", "n", "o"], sem_loteria=231.0, enviada=champ["enviada"])
    monkeypatch.setattr(esteira, "Fila", lambda *a, **k: f)
    monkeypatch.setattr(esteira, "carregar_campea", lambda: outra)
    salvo = {}
    monkeypatch.setattr(campeao, "salvar", lambda c: salvo.update(c))
    esteira.main(["enviado", "33"])
    # a campeã já tem 3 membros, mas o arquivo enviado foi o de 2: é esse o alvo da próxima
    # checagem de transferência
    assert salvo["enviada"] == {"versao": 33, "sem_loteria": 232.5, "membros": ["m", "n"]}
    assert not f.meta("pronto") and not f.meta("pronto_sem_loteria")


def test_talvez_enviar_segura_o_submit_quando_a_transferencia_reprova(tmp_path, monkeypatch):
    """O caso v37: a régua promoveu, mas o ganho não é de regime que transfere."""
    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f, transfere=False)
    cmds = []
    assert esteira.talvez_enviar(f, champ, lambda argv, **kw: cmds.append(argv)) is None
    assert not cmds and not f.meta("pronto")
    assert "transferência" in f.meta("erro_envio") and "IC do corpo" in f.meta("erro_envio")
    assert not f.meta("ultimo_arquivo_em") or f.meta("ultimo_arquivo_em") == "0"


def test_transferencia_reprovada_nao_e_remedida_para_a_mesma_campea(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f, transfere=False)
    medidas = []
    monkeypatch.setattr(esteira, "transferencia_ok",
                        lambda c: (medidas.append(c) or (False, "falha: corpo > 0")))
    for _ in range(3):
        assert esteira.talvez_enviar(f, champ, lambda argv, **kw: None) is None
    assert len(medidas) == 1


def test_transferencia_ok_compara_a_campea_com_os_membros_da_ultima_enviada(monkeypatch):
    champ = _champ(["m", "n"], enviada={"versao": 32, "sem_loteria": 233.0, "membros": ["m"]})
    vistos = {}

    def relatorio(novos, contra):
        vistos["novos"], vistos["contra"] = novos, contra
        return {"veredito": {"passa": True, "motivo": "transferência ok"}}

    monkeypatch.setattr(esteira.transferencia, "relatorio", relatorio)
    assert esteira.transferencia_ok(champ) == (True, "transferência ok")
    assert vistos == {"novos": ["m", "n"], "contra": ["m"]}


def test_transferencia_ok_reprova_campea_igual_a_enviada_e_enviada_sem_membros():
    igual = _champ(["m"], enviada={"versao": 32, "sem_loteria": 233.0, "membros": ["m"]})
    assert esteira.transferencia_ok(igual) == (False, "campeã igual à última enviada")
    velha = _champ(["m", "n"], enviada={"versao": 32, "sem_loteria": 233.0})
    passa, motivo = esteira.transferencia_ok(velha)
    assert not passa and "membros" in motivo


def test_falha_no_submit_respeita_a_trava_de_seis_horas(tmp_path, monkeypatch):
    import subprocess

    f = esteira.Fila(tmp_path / "e.db")
    champ = _pronta(tmp_path, monkeypatch, f)
    chamadas = []

    def rodar(argv, **kw):
        chamadas.append(argv)
        raise subprocess.CalledProcessError(1, argv)

    assert esteira.talvez_enviar(f, champ, rodar) is None
    assert esteira.talvez_enviar(f, champ, rodar) is None
    assert len(chamadas) == 1  # a segunda tentativa cai na trava, sem refazer o submit pesado


def test_relatorio_mostra_os_erros_de_envio_e_commit(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    champ = _champ()
    assert "erro de envio" not in esteira.relatorio(f, champ)
    f.meta("erro_envio", "submit v33: código 1")
    f.meta("erro_commit", "git falhou (código 128)")
    texto = esteira.relatorio(f, champ)
    assert "- Último erro de envio: submit v33: código 1" in texto
    assert "- Último erro de commit: git falhou (código 128)" in texto


def test_familia_rotula_a_unica_diferenca():
    base = {"--conjunto": True, "--fila": True, "--corretor-params": {"num_leaves": 63}}
    assert esteira.familia(base | {"--superficie": True}, base) == "bloco:--superficie"
    assert esteira.familia({k: v for k, v in base.items() if k != "--conjunto"},
                           base) == "bloco:--conjunto"
    assert esteira.familia(base | {"--corretor-rounds": 900}, base) == "rodadas"
    assert esteira.familia(base | {"--corretor-params": {"num_leaves": 127}}, base) == "param:num_leaves"
    assert esteira.familia(base | {"--corretor-params": {"num_leaves": 63, "learning_rate": 0.03}},
                           base) == "param:learning_rate"


def test_familia_dos_tres_estados_do_bloco_fila():
    base = {"--conjunto": True, "--fila": True}
    sem = {"--conjunto": True}
    assert esteira.familia(sem | {"--stand-prefixo": True}, base) == "fila:--stand-prefixo"
    assert esteira.familia(sem, base) == "fila:nenhum"
    assert esteira.familia(base, sem) == "fila:--fila"


def test_familia_outro_quando_muda_nada_ou_muita_coisa():
    base = {"--conjunto": True}
    assert esteira.familia(base, base) == "outro"
    assert esteira.familia(base | {"--mapa": True, "--superficie": True}, base) == "outro"
    assert esteira.familia({"--corretor-params": {"num_leaves": 127, "learning_rate": 0.03}},
                           {"--corretor-params": {"num_leaves": 63}}) == "outro"


_seq = iter(range(10_000))


def _termina(f, familia, ganho, estado="pulado"):
    i = f.add("corretor", {"x": next(_seq)}, "gerador", 0, "C", familia)
    f.marcar(i, estado, fim=1.0, resultado={"a": {"ganho": ganho}})
    return i


def test_notas_resume_cada_familia(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    _termina(f, "bloco:--mapa", -0.2)
    _termina(f, "bloco:--mapa", 0.4, "feito")
    _termina(f, "param:num_leaves", -0.1)
    f.add("corretor", {"z": 1}, "gerador", 0, "C", "bloco:--mapa")  # ainda na fila, não conta
    n = esteira.notas(f)
    assert n["bloco:--mapa"] == {"n": 2, "media": 0.1, "melhor": 0.4, "promovidos": 1}
    assert n["param:num_leaves"]["n"] == 1 and n["param:num_leaves"]["promovidos"] == 0


def test_repriorizar_explora_familia_nova_e_pontua_a_conhecida(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    for g in (0.2, 0.4, 0.3):
        _termina(f, "bloco:--mapa", g)
    nova = f.add("corretor", {"a": 1}, "gerador", 0, "C", "fila:--fila")
    conhecida = f.add("corretor", {"b": 1}, "gerador", 0, "C", "bloco:--mapa")
    esteira.repriorizar(f)
    pri = lambda i: f.db.execute("select prioridade from candidatos where id=?", (i,)).fetchone()[0]  # noqa: E731
    assert pri(nova) == 1 and pri(conhecida) == 3  # round(10 * 0,3)


def test_repriorizar_corta_familia_sem_rendimento(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    for _ in range(10):
        _termina(f, "bloco:--mapa", -0.3)
    alvo = f.add("corretor", {"b": 1}, "gerador", 0, "C", "bloco:--mapa")
    assert esteira.repriorizar(f) == {"bloco:--mapa"}
    c = dict(f.db.execute("select * from candidatos where id=?", (alvo,)).fetchone())
    assert c["estado"] == "pulado" and c["motivo"].startswith("família sem rendimento (n=10")


def test_repriorizar_nao_corta_familia_com_promocao(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    for _ in range(9):
        _termina(f, "bloco:--mapa", -0.4)
    _termina(f, "bloco:--mapa", 0.1, "feito")
    alvo = f.add("corretor", {"b": 1}, "gerador", 0, "C", "bloco:--mapa")
    assert esteira.repriorizar(f) == set()
    assert f.db.execute("select estado from candidatos where id=?", (alvo,)).fetchone()[0] == "fila"


def test_repriorizar_nao_toca_no_que_veio_da_mao(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    for _ in range(10):
        _termina(f, "bloco:--mapa", -0.3)
    mao = f.add("corretor", {"b": 1}, "agente", 9, "C", "bloco:--mapa")
    base = f.add("base", {"base": []}, "gerador", 7, "C", "base")
    esteira.repriorizar(f)
    for i, p in ((mao, 9), (base, 7)):
        c = dict(f.db.execute("select * from candidatos where id=?", (i,)).fetchone())
        assert c["estado"] == "fila" and c["prioridade"] == p


def test_vizinhos_devolve_familia_e_pula_as_cortadas():
    champ = {"membros": [{"id": "m", "config": {"corretor": "conjunto", "fila": "stand"}}]}
    vs = esteira.vizinhos(champ)
    assert all(isinstance(fam, str) and isinstance(r, dict) for fam, r in vs)
    assert {"bloco:--mapa", "fila:--fila", "param:num_leaves"} <= {fam for fam, _ in vs}
    cortadas = {fam for fam, _ in esteira.vizinhos(champ, {"bloco:--mapa"})}
    assert "bloco:--mapa" not in cortadas and "fila:--fila" in cortadas


def test_preencher_familias_usa_o_membro_mais_proximo(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    m1 = {"--conjunto": True, "--fila": True}
    m2 = {"--conjunto": True, "--mapa": True, "--superficie": True}
    a = f.add("corretor", m1 | {"--corretor-ref": True}, "gerador", 0, "C")
    b = f.add("corretor", {"--conjunto": True, "--mapa": True}, "agente", 0, "C")
    c = f.add("base", {"base": []}, "usuario", 0, "C")
    assert esteira.preencher_familias(f, [m1, m2]) == 3
    lido = lambda i: f.db.execute("select familia from candidatos where id=?", (i,)).fetchone()[0]  # noqa: E731
    assert lido(a) == "bloco:--corretor-ref" and lido(b) == "bloco:--superficie" and lido(c) == "base"


def test_coluna_familia_e_criada_num_banco_antigo(tmp_path):
    import sqlite3
    caminho = tmp_path / "velha.db"
    db = sqlite3.connect(caminho)
    db.executescript("""
        create table candidatos (
          id integer primary key, criado real, origem text, tipo text, receita text,
          hash text, prioridade integer, estado text, campea text, run_id text,
          resultado text, motivo text, inicio real, fim real);
        create table meta (chave text primary key, valor text);""")
    db.execute("insert into candidatos (id, origem, tipo, receita, hash, prioridade, estado, campea)"
               " values (1, 'gerador', 'corretor', '{}', 'h', 0, 'fila', 'C')")
    db.commit(); db.close()
    f = esteira.Fila(caminho)
    assert f.proximo()["familia"] is None
    assert f.add("corretor", {"--mapa": True}, "gerador", 0, "C", "bloco:--mapa")


def test_relatorio_traz_a_tabela_de_familias(tmp_path):
    f = esteira.Fila(tmp_path / "e.db")
    _termina(f, "bloco:--mapa", -0.2)
    texto = esteira.relatorio(f, _champ())
    assert "## Famílias" in texto and "| bloco:--mapa | 1 |" in texto and "explorando" in texto
