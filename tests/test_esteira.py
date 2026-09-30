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
