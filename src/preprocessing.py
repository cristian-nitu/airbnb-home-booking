import zipfile

import pandas as pd

from src import config


def load_data():
    print("Loading CSV files...")

    zf = zipfile.ZipFile(config.DATA_PATH + "\\new-user-bookings.zip")
    df_users = pd.read_csv(zf.open("train_users_2.csv"))
    df_sessions = pd.read_csv(zf.open("sessions.csv"))

    print("     CSV files loaded successfully!")
    return df_users, df_sessions
