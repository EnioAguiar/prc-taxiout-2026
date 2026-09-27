import numpy as np
import pandas as pd
import pytest

import features as F
from train import build_submission, final_config


def test_submissao_segue_a_ordem_do_template_sem_nulos():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    out, preenchidas = build_submission(
        template, np.array([3.0, 1.0]), np.array([30.0, 10.0]), max_fill_frac=0.5
    )
    assert out[F.ID].tolist() == [1.0, 2.0, 3.0]
    assert out[F.TARGET].tolist() == [10.0, 20.0, 30.0]  # ID 2 sem previsão: mediana
    assert preenchidas == 1


def test_submissao_recusa_preencher_demais():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    with pytest.raises(SystemExit, match="sem previsão"):
        build_submission(template, np.array([1.0]), np.array([10.0]))


def test_rodadas_finais_escalam_o_melhor_ponto():
    single = final_config({"config": {"model": "single", "rounds": 400}, "best_iter": 325})
    assert single["rounds"] == 390
    two = final_config({"config": {"model": "two_stage", "cls_rounds": 400, "reg_rounds": 500},
                        "best_iter": None})
    assert (two["cls_rounds"], two["reg_rounds"]) == (480, 600)


def test_rodadas_finais_da_stack_escalam_so_a_base():
    champ = {"config": {"model": "stack_cf", "rounds": 300, "adsb": True,
                        "base_config": {"model": "two_stage_nm", "cls_rounds": 400,
                                        "reg_rounds": 400}},
             "best_iter": 325}
    cfg = final_config(champ)
    assert cfg["base_config"]["cls_rounds"] == 480
    assert cfg["base_config"]["reg_rounds"] == 480
    assert cfg["rounds"] == 300  # corretor intocado, best_iter ignorado
    # a campeã original continua com as rodadas dos blocos (sem escala)
    assert champ["config"]["base_config"]["cls_rounds"] == 400
    assert champ["config"]["rounds"] == 300
