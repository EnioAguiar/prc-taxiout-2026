# 6-busca-parametros-corretor

150 combinações do corretor barato (parâmetros do LightGBM). Referência (parâmetros atuais): sem loteria 248.63, normais 201.24. Tuning raramente transfere (Discord); só vale se ganhar ≥ 1 s em sem loteria **e** normais.

| sem loteria | normais | completo | rodadas | parâmetros |
|---|---|---|---|---|
| 246.84 | 200.73 | 309.21 | 200 | {'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 100, 'feature_fraction': 1.0, 'bagging_fraction': 0.7, 'lambda_l2': 50.0} |
| 247.18 | 200.49 | 308.93 | 300 | {'learning_rate': 0.03, 'num_leaves': 255, 'min_data_in_leaf': 100, 'feature_fraction': 1.0, 'bagging_fraction': 0.7, 'lambda_l2': 1.0} |
| 247.25 | 200.8 | 309.14 | 800 | {'learning_rate': 0.02, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 1.0, 'bagging_fraction': 0.7, 'lambda_l2': 50.0} |
| 247.41 | 200.54 | 309.13 | 300 | {'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 200, 'feature_fraction': 0.85, 'bagging_fraction': 1.0, 'lambda_l2': 0.0} |
| 247.45 | 200.87 | 309.57 | 800 | {'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'lambda_l2': 50.0} |
| 247.63 | 200.78 | 309.29 | 800 | {'learning_rate': 0.02, 'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.85, 'bagging_fraction': 0.7, 'lambda_l2': 50.0} |
| 247.67 | 200.23 | 308.91 | 500 | {'learning_rate': 0.03, 'num_leaves': 255, 'min_data_in_leaf': 100, 'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'lambda_l2': 50.0} |
| 247.77 | 200.81 | 309.96 | 500 | {'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 1.0, 'bagging_fraction': 0.85, 'lambda_l2': 10.0} |
| 247.77 | 201.29 | 309.6 | 500 | {'learning_rate': 0.05, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 1.0, 'bagging_fraction': 0.85, 'lambda_l2': 1.0} |
| 247.84 | 201.32 | 309.83 | 300 | {'learning_rate': 0.08, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'lambda_l2': 50.0} |
| 247.88 | 200.96 | 309.52 | 200 | {'learning_rate': 0.05, 'num_leaves': 255, 'min_data_in_leaf': 100, 'feature_fraction': 1.0, 'bagging_fraction': 0.7, 'lambda_l2': 1.0} |
| 247.92 | 201.28 | 310.06 | 300 | {'learning_rate': 0.02, 'num_leaves': 63, 'min_data_in_leaf': 200, 'feature_fraction': 0.7, 'bagging_fraction': 0.85, 'lambda_l2': 1.0} |
| 247.92 | 201.58 | 309.24 | 500 | {'learning_rate': 0.08, 'num_leaves': 31, 'min_data_in_leaf': 500, 'feature_fraction': 0.85, 'bagging_fraction': 1.0, 'lambda_l2': 10.0} |
| 247.94 | 200.94 | 310.32 | 200 | {'learning_rate': 0.02, 'num_leaves': 255, 'min_data_in_leaf': 200, 'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'lambda_l2': 50.0} |
| 247.94 | 200.42 | 309.25 | 500 | {'learning_rate': 0.02, 'num_leaves': 127, 'min_data_in_leaf': 100, 'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'lambda_l2': 10.0} |
