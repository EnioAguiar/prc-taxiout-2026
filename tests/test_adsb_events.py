import pandas as pd

from adsb_events import casar, decolagens


def _pts(icao, pts, cs="DLH1", apt="EDDM"):
    rows = [dict(icao=icao, t=p[0], chao=p[1], gs=p[2] if len(p) > 2 else 5.0, apt=apt, lat=48.0, lon=11.0,
                 fonte="adsb_icao", callsign=cs, reg="D-X", tipo="A20N") for p in pts]
    return pd.DataFrame(rows)


def test_decolagem_e_primeiro_ponto_no_chao():
    # parado no portão desde 1000, táxi, decola em 1900; lacuna de 700 s antes corta o segmento
    cut = _pts("a", [(100, True), (800, True), (1000, True), (1400, True), (1880, True), (1900, False)])
    ev = decolagens(cut)
    assert len(ev) == 1
    assert ev.loc[0, "takeoff"] == 1900 and ev.loc[0, "first_ground"] == 800
    assert ev.loc[0, "n_chao"] == 4


def test_primeiro_movimento_depois_de_parado_no_portao():
    cut = _pts("a", [(0, True, 0.0), (300, True, 0.0), (420, True, 3.0), (900, True, 15.0), (920, False)])
    ev = decolagens(cut)
    assert ev.loc[0, "first_ground"] == 0 and ev.loc[0, "first_move"] == 420 and ev.loc[0, "gs0"] == 0.0


def test_lacuna_grande_no_fim_nao_e_decolagem():
    cut = _pts("a", [(0, True), (60, True), (300, False)])  # Δt 240 s > 120 s
    assert decolagens(cut).empty


def test_casamento_prefere_callsign_e_nao_reusa_evento():
    ev = pd.concat([decolagens(_pts("a", [(0, True), (500, True), (520, False)], cs="DLH1")),
                    decolagens(_pts("b", [(0, True), (540, True), (560, False)], cs="AFR2"))])
    dep = pd.DataFrame({"MVT_ID_mvt": [1, 2, 3], "apt": "EDDM",
                        "mvt": [555.0, 530.0, 525.0], "callsign": ["DLH1", None, None]})
    m = casar(dep, ev).set_index("MVT_ID_mvt")
    assert m.loc[1, "icao"] == "a"  # callsign vence a proximidade (b está a 5 s)
    assert m.loc[2, "icao"] == "b"  # o mais próximo que sobrou
    assert 3 not in m.index  # sem evento livre
