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
    monkeypatch.setattr(esteira, "RELATORIO", tmp_path / "esteira.md")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    f.add("corretor", {"--fila": True}, "usuario", 0, "m")
    ok = esteira.passo(f, avaliar=lambda membros, novo, semente=0: {
        "aprovado": True, "membros": ["m", "RUN1"], "proposta": "soma", "a": {"ganho": 0.9},
        "b": {"ganho": 0.5}, "completo": 0.3, "motivo": "aprovado", "avaliadas": 3})
    assert ok and salvo["m"] == ["m", "RUN1"]
    assert f.ultimos(1)[0]["estado"] == "feito"


def test_passo_respeita_pausa(tmp_path, monkeypatch):
    f = esteira.Fila(tmp_path / "e.db")
    monkeypatch.setattr(esteira, "PAUSA", tmp_path / "pausa")
    (tmp_path / "pausa").write_text("")
    f.add("corretor", {"a": 1}, "usuario", 0, "m")
    assert esteira.passo(f) is False
