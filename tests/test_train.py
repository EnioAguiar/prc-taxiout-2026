import numpy as np
import pandas as pd

import features as F
from train import build_submission, final_config


def test_submissao_segue_a_ordem_do_template_sem_nulos():
    template = pd.DataFrame({F.ID: [1.0, 2.0, 3.0], F.TARGET: np.nan})
    out = build_submission(template, np.array([3.0, 1.0]), np.array([30.0, 10.0]))
    assert out[F.ID].tolist() == [1.0, 2.0, 3.0]
    assert out[F.TARGET].tolist() == [10.0, 20.0, 30.0]  # ID 2 sem previsão: mediana


def test_rodadas_finais_escalam_o_melhor_ponto():
    single = final_config({"config": {"model": "single", "rounds": 400}, "best_iter": 325})
    assert single["rounds"] == 390
    two = final_config({"config": {"model": "two_stage", "cls_rounds": 400, "reg_rounds": 500},
                        "best_iter": None})
    assert (two["cls_rounds"], two["reg_rounds"]) == (480, 600)
