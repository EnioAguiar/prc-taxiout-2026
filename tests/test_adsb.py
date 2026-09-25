from adsb import caixas, dias_dos_meses, recortar_rastro


def test_recorte_guarda_so_pontos_baixos_perto_de_aeroporto():
    d = {
        "icao": "3c6444",
        "timestamp": 1_736_900_000.0,
        "trace": [
            [0.0, 48.3538, 11.7861, "ground", 0.0, 0, 0, 0, {"flight": "DLH7LV  "}],  # EDDM, chão
            [60.0, 48.36, 11.79, 1500, 150.0, 0, 0, 0, None],  # EDDM, subindo
            [120.0, 48.36, 11.79, 8000, 250.0, 0, 0, 0, None],  # alto demais
            [180.0, 45.0, 10.0, "ground", 0.0, 0, 0, 0, None],  # longe de tudo
        ],
    }
    rows = recortar_rastro(d, caixas())
    assert [(r[1] - d["timestamp"], r[2], r[5], r[8]) for r in rows] == [
        (0.0, 1, True, "DLH7LV"),
        (60.0, 1, False, "DLH7LV"),
    ]


def test_dias_dos_meses():
    dias = dias_dos_meses(["2025-01", "2026-07"])
    assert len(dias) == 62 and dias[0] == "2025-01-01" and dias[-1] == "2026-07-31"
