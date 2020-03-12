from sklearn.model_selection import ParameterGrid, train_test_split
import lightgbm as lgb

from src import config
from src.config import lgb_params_grid


def model_lgb_train(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=config.seed)

    # create dataset for lightgbm
    d_train = lgb.Dataset(X_train, y_train)
    d_eval = lgb.Dataset(X_val, y_val, reference=d_train)

    # fine the best parameters
    i = 0
    best_score = -1
    best_params = None
    best_model_lgb = None
    parameterGrid = ParameterGrid(lgb_params_grid)
    for params in parameterGrid:
        i = i + 1
        print('*************************************************')
        print(i, '/', len(parameterGrid), '|', "params=", params)
        model_lgb = lgb.train(params, d_train, valid_sets=d_eval, early_stopping_rounds=100, verbose_eval=False)
        score = model_lgb.best_score['valid_0']['multi_logloss']
        print("logistic loss:", score)
        if score < best_score or best_params is None:
            best_score = score
            best_params = params
            best_model_lgb = model_lgb

    print('*****best logistic loss for Validation set*****')
    print(best_score)
    print("params: ", best_params)

    return best_model_lgb, best_params
