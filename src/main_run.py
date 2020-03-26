
import sklearn
from src.feature_engineering import feature_engineering
from src.model_evaluation import model_evaluate
from src.models import model_lgb_train
from src.preprocessing import load_data

if __name__ == '__main__':
    df_users, df_sessions = load_data()
    X_train, X_test, y_train, y_test = feature_engineering(df_users, df_sessions)
    model_lgb, params = model_lgb_train(X_train, y_train)
    model_lgb_logloss = model_evaluate(model_lgb, X_test, y_test)
    print('\nlogistic loss/cross-entropy loss for Test set:', model_lgb_logloss)

