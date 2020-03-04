"""
the main entrance of the project
"""
from src.feature_engineering import feature_engineering
from src.preprocessing import load_data

if __name__ == '__main__':
    df_users, df_sessions = load_data()
    print("CSV files loaded successfully!")
    X_train, X_test, y_train, y_test = feature_engineering(df_users, df_sessions)
    print("Feature engineering done!")
