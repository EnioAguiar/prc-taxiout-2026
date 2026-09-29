import importlib

import dispositivo


def _com(monkeypatch, valor):
    monkeypatch.setenv("PRC_DEVICE", valor)
    return importlib.reload(dispositivo)


def test_na_cpu_os_parametros_sao_os_de_sempre(monkeypatch):
    d = _com(monkeypatch, "cpu")
    p = {"num_leaves": 63, "deterministic": True, "force_row_wise": True}
    assert d.lgb_params(p) == p


def test_na_gpu_sai_o_que_so_vale_na_cpu_e_entra_o_device(monkeypatch):
    d = _com(monkeypatch, "gpu")
    p = {"num_leaves": 63, "deterministic": True, "force_row_wise": True}
    assert d.lgb_params(p) == {"num_leaves": 63, "device_type": "gpu"}
    assert p["deterministic"] is True  # não muta o dicionário de origem
    _com(monkeypatch, "cpu")
