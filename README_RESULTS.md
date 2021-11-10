# Results

## Run Training
The performance of the models has been evaluated by running the following scripts:
| Model                 | Script                         | Comment
| -                     | -                              | -
| FC NN                 | run_fc_nn.py                   |
| Diff. DT – Routing    | run_differentiable_xgboost.py  | XGBoost = False
| Diff. DT – XGBoost    | run_differentiable_xgboost.py  | XGBoost = True
| XGBoost               | run_xgboost.py                 | 

## Results – Breast Cancer Wisconsin Dataset
| Model | Epochs/Iterations | Train Loss | Validation Loss | Hyperparameters
| - | - | - | - | -
| FC NN | 748   | 0.004422 | 0.023458 | {'learning_rate': 0.001, 'regularization': True, 'l2': 0.01, 'num_units': 256, 'optimizer': <class 'keras.optimizer_v2.adam.Adam'>, 'batch_norm': True, 'batch_size': 128, 'num_examples': None, 'hidden': True}
| Diff. DT – Routing (Ensemble) | 2311 | 0.006615 | 0.041459 | {'learning_rate': 0.001, 'optimizer': <class 'keras.optimizer_v2.adam.Adam'>, 'batch_norm': True, 'batch_norm_trainable': True, 'batch_size': 128, 'depth': 5, 'feature_selection_rate': 1.0, 'xgboost': False, 'num_models': 8, 'num_features': 30}
| Diff. DT – Routing | 2331 | 0.009385 | 0.054416 | {'learning_rate': 0.001, 'optimizer': <class 'keras.optimizer_v2.adam.Adam'>, 'batch_norm': True, 'batch_norm_trainable': True, 'batch_size': 128, 'depth': 5, 'used_features_rate': 1.0, 'xgboost': False}
| Diff. DT – XGBoost (Ensemble) | 1479 | 0.044350 | 0.067450 | {'learning_rate': 0.001, 'optimizer': <class 'keras.optimizer_v2.adam.Adam'>, 'batch_norm': True, 'batch_norm_trainable': True, 'batch_size': 128, 'depth': 4, 'feature_selection_rate': 0.95, 'xgboost': True, 'num_models': 6, 'num_features': 30}
| Diff DT – XGBoost | 1414 | 0.041864 | 0.076515 | 'learning_rate': 0.001, 'optimizer': <class 'keras.optimizer_v2.adam.Adam'>, 'batch_norm': True, 'batch_norm_trainable': True, 'batch_size': 128, 'depth': 4, 'feature_selection_rate': 1.0, 'xgboost': True, 'num_models': 1, 'num_features': 30}
| XGBoost | 72 | 0.018718 | 0.080153 | {'n_estimators': 2000, 'max_depth': 2, 'reg_lambda': 0.5, 'eval_metric': ['logloss'], 'objective': 'binary:logistic', 'subsample': 0.5}