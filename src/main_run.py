"""
the main entrance of the project
"""
from src.feature_engineering import feature_engineering
from src.preprocessing import load_data

if __name__ == '__main__':
    print("Loading CSV files...")
    df_users, df_sessions = load_data()
    print("CSV files loaded successfully!")
    print("Running Feature Engineering...")
    X_train, X_test, y_train, y_test = feature_engineering(df_users, df_sessions)
    print("Feature Engineering done!")
