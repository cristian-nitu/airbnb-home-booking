import pandas as pd

from src import config


def load_data():
    df_users = pd.read_csv(config.DATA_PATH + "\\train_users_2.csv")
    df_sessions = pd.read_csv(config.DATA_PATH + "\\sessions.csv")

    return df_users, df_sessions
