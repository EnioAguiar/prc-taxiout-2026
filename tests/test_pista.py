import numpy as np
import pandas as pd
import pytest

import features as F
import pista as P

T = pd.Timestamp("2025-01-15 10:00", tz="UTC")  # 10:00 em ponto: começo de um bloco de 10 min
BLOCK = "BLOCK_TIME_UTC_mvt"


def _mov(fase, rwy, minuto, wk="M", apt="LIRF") -> dict:
    """Uma linha de parquet bruto: movimento do aeroporto `apt` em `minuto` depois de T."""
    return {"PHASE_mvt": fase,
            "ADEP_mvt": apt if fase == "DEP" else "EDDF",
            "ADES_mvt": apt if fase == "ARR" else "EDDF",
            "RUNWAY_mvt": rwy,
            P.TIME: T + pd.Timedelta(minutes=minuto),
            P.WAKE: wk,
            BLOCK: T - pd.Timedelta(hours=10)}  # off-block absurdo: nunca pode ser lido


def _dados(tmp_path, movimentos):
    pd.DataFrame(movimentos).to_parquet(
        tmp_path / "training_2025-01-01_2025-02-01.parquet", index=False)
    return tmp_path


def _dep(minutos, rwy="25", aobt=None, wk="M", apt="LIRF") -> pd.DataFrame:
    """Decolagens como o `corrector_frame` as entrega: hora, pista, AOBT_3 e esteira."""
    n = len(minutos)
    def lista(v):
        return list(v) if isinstance(v, (list, tuple)) else [v] * n
    return pd.DataFrame({
        F.AIRPORT: lista(apt),
        P.RWY: lista(rwy),
        P.TIME: [T + pd.Timedelta(minutes=m) for m in minutos],
        P.AOBT: [None if a is None else T + pd.Timedelta(minutes=a)
                 for a in (lista(aobt) if aobt is not None else [None] * n)],
        P.WAKE: lista(wk),
    })


# ---------------------------------------------------------------- configuração de pista


def test_a_configuracao_lista_as_pistas_de_decolagem_e_de_pouso(tmp_path):
    dados = _dados(tmp_path, [
        _mov("DEP", "25", 0), _mov("DEP", "07", 5),
        _mov("ARR", "16L", 10), _mov("ARR", "16L", 20),
    ])
    out = P.colunas(_dep([0]), dados)
    assert out[P.CAT].iloc[0] == "LIRF|D:07+25 A:16L"  # pistas em ordem alfabética
    assert out["pista_cfg_dep"].iloc[0] == 2
    assert out["pista_cfg_arr"].iloc[0] == 1


def test_a_janela_da_configuracao_e_de_60_min_em_volta_do_movimento(tmp_path):
    # voo às 10:00: a janela são os blocos de 09:30 (inclusive) a 10:30 (exclusive)
    dados = _dados(tmp_path, [
        _mov("DEP", "25", -30), _mov("DEP", "fora_antes", -31),
        _mov("ARR", "07", 29), _mov("ARR", "fora_depois", 30),
    ])
    out = P.colunas(_dep([0]), dados)
    assert out[P.CAT].iloc[0] == "LIRF|D:25 A:07"


def test_a_mudanca_de_configuracao_compara_com_a_hora_anterior(tmp_path):
    # pousos pela 07 até as 10:30 e pela 25 depois: quem decola às 11:00 vê a troca
    dados = _dados(tmp_path, [_mov("DEP", "25", m) for m in (-90, -30, 30, 90)]
                   + [_mov("ARR", "07", -90), _mov("ARR", "07", -15), _mov("ARR", "25", 60)])
    out = P.colunas(_dep([0, 60]), dados)
    assert out[P.CAT].tolist() == ["LIRF|D:25 A:07", "LIRF|D:25 A:25"]
    assert out["pista_cfg_mudou"].tolist() == [0.0, 1.0]


def test_sem_hora_anterior_a_mudanca_fica_sem_valor(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", 0)])
    out = P.colunas(_dep([0]), dados)
    assert np.isnan(out["pista_cfg_mudou"].iloc[0])


def test_a_pista_dominante_e_a_que_mais_decola_na_janela(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", 0), _mov("DEP", "25", 5),
                              _mov("DEP", "07", 10)])
    out = P.colunas(_dep([0, 0], rwy=["25", "07"]), dados)
    assert out["pista_dominante"].tolist() == [1.0, 0.0]
    np.testing.assert_allclose(out["pista_parte_rwy"], [2 / 3, 1 / 3])


def test_pista_sem_decolagem_na_janela_nao_e_dominante(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", 0), _mov("ARR", "07", 0)])
    out = P.colunas(_dep([0], rwy="07"), dados)
    assert out["pista_dominante"].iloc[0] == 0.0
    assert out["pista_parte_rwy"].iloc[0] == 0.0


def test_movimento_de_outro_aeroporto_nao_entra_na_configuracao(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", 0), _mov("DEP", "09", 0, apt="EDDF"),
                              _mov("ARR", "18", 0, apt="EDDF")])
    out = P.colunas(_dep([0]), dados)
    assert out[P.CAT].iloc[0] == "LIRF|D:25 A:-"
    assert out["pista_cfg_arr"].iloc[0] == 0


# ---------------------------------------------------------------- fila com peso de esteira


@pytest.fixture
def fila(tmp_path):
    """O voo decola às 11:00 com off-block (AOBT_3) às 10:40, pela pista 25."""
    return _dados(tmp_path, [
        _mov("DEP", "25", 60),           # o próprio voo, que o bruto também tem
        _mov("DEP", "25", 40),           # exatamente no AOBT: já tinha ido
        _mov("DEP", "25", 45, wk="H"),
        _mov("DEP", "25", 50),
        _mov("DEP", "25", 59, wk="J"),
        _mov("DEP", "16L", 50, wk="H"),  # outra pista
        _mov("DEP", "25", 61),           # depois da decolagem
    ])


def test_a_fila_pesa_so_a_mesma_pista_entre_o_aobt_e_a_decolagem(fila):
    out = P.colunas(_dep([60], aobt=[40]), fila)
    # H (1,6) + M (1,0) + J (2,0); o próprio voo (M, 1,0) entra na busca e é descontado
    np.testing.assert_allclose(out["pista_fila_peso"], [4.6])
    assert out["pista_fila_pesadas"].tolist() == [2.0]


def test_a_fila_do_proprio_voo_nao_se_conta(fila):
    # sem vizinhos no intervalo (AOBT às 10:59, decolagem às 11:00) sobra só ele mesmo
    out = P.colunas(_dep([60], aobt=[59.5]), fila)
    assert out["pista_fila_peso"].tolist() == [0.0]
    assert out["pista_fila_pesadas"].tolist() == [0.0]


def test_sem_aobt_a_fila_fica_sem_valor(fila):
    out = P.colunas(_dep([60]), fila)
    assert out["pista_fila_peso"].isna().all()
    assert out["pista_fila_pesadas"].isna().all()


def test_a_categoria_de_esteira_desconhecida_pesa_como_media(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", 60), _mov("DEP", "25", 50, wk=None)])
    out = P.colunas(_dep([60], aobt=[40]), dados)
    np.testing.assert_allclose(out["pista_fila_peso"], [P.PESO_PADRAO])
    assert out["pista_fila_pesadas"].tolist() == [0.0]


# ---------------------------------------------------------------- cadência


def test_o_span20_mede_as_20_decolagens_anteriores(tmp_path):
    # 25 decolagens de minuto em minuto antes do voo: a 20ª para trás saiu 20 min antes
    dados = _dados(tmp_path, [_mov("DEP", "25", -m) for m in range(1, 26)]
                   + [_mov("DEP", "25", 0)])
    out = P.colunas(_dep([0]), dados)
    assert out["pista_span20"].iloc[0] == 20 * 60.0


def test_sem_20_decolagens_antes_o_span_fica_sem_valor(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", -m) for m in range(1, 20)])
    out = P.colunas(_dep([0]), dados)
    assert np.isnan(out["pista_span20"].iloc[0])


def test_as_decolagens_por_hora_saem_dos_30_min_anteriores(tmp_path):
    dados = _dados(tmp_path, [_mov("DEP", "25", -1), _mov("DEP", "07", -29),
                              _mov("DEP", "25", -30),   # borda: entra
                              _mov("DEP", "25", -31),   # velha demais
                              _mov("DEP", "25", 0),     # o próprio voo não conta
                              _mov("DEP", "25", 5)])    # futuro não conta
    out = P.colunas(_dep([0]), dados)
    assert out["pista_dep_h30"].iloc[0] == 6.0  # 3 decolagens em 30 min = 6/h


# ---------------------------------------------------------------- nada de BLOCK nem recorte


def test_o_off_block_das_decolagens_nunca_muda_o_bloco(fila):
    df = _dep([60], aobt=[40])
    com = P.colunas(df.assign(**{BLOCK: T}), fila)
    sem = P.colunas(df.assign(**{BLOCK: pd.NaT}), fila)
    pd.testing.assert_frame_equal(com, sem)


def test_o_valor_de_um_voo_nao_depende_do_recorte(fila):
    df = _dep([60, 45, 50], aobt=[40, 30, 35])
    inteiro = P.colunas(df, fila)
    sozinho = P.colunas(df.iloc[[0]], fila)
    pd.testing.assert_frame_equal(sozinho.reset_index(drop=True),
                                  inteiro.iloc[[0]].reset_index(drop=True))


def test_o_quadro_sai_na_ordem_e_no_indice_das_linhas(fila):
    df = _dep([60, 45, 50], aobt=[40, 30, 35]).set_index(pd.Index([7, 8, 9]))
    out = P.colunas(df, fila)
    assert out.index.tolist() == [7, 8, 9]
    assert list(out.columns) == P.COLS
    assert out["pista_fila_peso"].iloc[0] == 4.6


def test_quadro_vazio_sai_com_as_colunas(tmp_path):
    out = P.colunas(_dep([]), _dados(tmp_path, [_mov("DEP", "25", 0)]))
    assert list(out.columns) == P.COLS
    assert out.empty


def test_decolagem_sem_hora_fica_sem_valor(fila):
    df = _dep([60, 60], aobt=[40, 40])
    df.loc[1, P.TIME] = pd.NaT
    out = P.colunas(df, fila)
    assert out.iloc[1].isna().all()
    assert out["pista_fila_peso"].iloc[0] == 4.6
