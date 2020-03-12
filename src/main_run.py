
import sklearn
from src.feature_engineering import feature_engineering
from src.models import model_lgb_train
from src.preprocessing import load_data

if __name__ == '__main__':
    df_users, df_sessions = load_data()
    X_train, X_test, y_train, y_test = feature_engineering(df_users, df_sessions)
    model_lgb, params = model_lgb_train(X_train, y_train)

    y_pred = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
    model_lgb_logloss = round(sklearn.metrics.log_loss(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
                              3)
    print('\nlogistic loss/cross-entropy loss for Test set:', model_lgb_logloss)

