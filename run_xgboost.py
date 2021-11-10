import time
from datasets.breast_cancer import Dataset
import xgboost as xgb
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

if __name__ == '__main__':
    kwargs = {
        'n_estimators': 2000,
        'max_depth': 2,
        # 'colsample_bytree': 0.4,
        # 'colsample_bynode': 1.0,
        # 'colsample_bylevel': 0.5,
        # 'missing': -1,
        # 'use_label_encoder': False,
        'reg_lambda': 0.5,
        # 'alpha': 0,
        # 'min_child_weight': 0.5,
        # 'base_score': 0.5,
        # 'learning_rate': 0.01,
        # 'tree_method': 'exact',
        # 'booster': 'gbtree',
        # 'nthread': 1,
        'eval_metric': ['logloss'],
        # 'early_stopping_rounds': 100,
        'objective': 'binary:logistic',
        'subsample': 0.5,
    }

    model = xgb.XGBClassifier(**kwargs)
    dataset = Dataset()
    inputs_train, labels_train, inputs_eval, labels_eval = dataset.load(
        'np', batch_size=None, num_examples=None)

    start_time = time.time()
    model.fit(inputs_train, labels_train,
              eval_set=[[inputs_train, labels_train],
                        [inputs_eval, labels_eval]],
              verbose=False,
              early_stopping_rounds=100
              )

    best_iteration = model.get_booster().best_iteration
    booster = model.get_booster()
    result = model.evals_result()

    i = 0
    for train_loss, eval_loss in zip(result['validation_0']['logloss'], result['validation_1']['logloss']):
        logger.info(f'[{i}] train_loss:{train_loss:.6f}\teval_loss:{eval_loss:.6f}')
        i += 1

    best_train_loss = result['validation_0']['logloss'][best_iteration]
    best_val_loss = result['validation_1']['logloss'][best_iteration]
    logger.info(
        f'Best iteration {best_iteration} – train_loss {best_train_loss} – eval_loss {best_val_loss}')
    logger.info(kwargs)