import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder

from src import config


def feature_engineering(df_users, df_sessions):
    # drop instances with null values in 'date_first_booking' feature
    df_users = df_users.dropna(subset=['date_first_booking'])

    # replace the outliers by the maximum and minimum limit
    age_lower_limit = 18
    age_upper_limit = 99
    df_users['age'] = np.where(df_users['age'] > age_upper_limit, age_upper_limit,
                               np.where(df_users['age'] < age_lower_limit, age_lower_limit, df_users['age']))

    # fill missing values
    age_median = df_users['age'].median()
    df_users['age'].fillna(age_median, inplace=True)
    df_users['first_affiliate_tracked'].fillna('untracked', inplace=True)
    df_sessions['action_type'].fillna('-unknown-', inplace=True)
    df_sessions['action_detail'].fillna('-unknown-', inplace=True)
    df_sessions = df_sessions.dropna()

    # Categorical variable encoding
    users_one_hot = ['gender', 'signup_method', 'signup_app' 'language', 'affiliate_channel',
                     'affiliate_provider', 'first_affiliate_tracked', 'first_device_type', 'first_browser',
                     'country_destination']
    sessions_one_hot = ['action_type', 'device_type']
    sessions_label_encoding = ['action', 'action_detail']

    # TODO: remove original column
    for col in users_one_hot:
        tmp = pd.get_dummies(df_users[col], prefix=col)
        df_users = pd.concat([df_users, tmp], axis=1)

    for col in sessions_one_hot:
        tmp = pd.get_dummies(df_sessions[col], prefix=col)
        df_sessions = pd.concat([df_sessions, tmp], axis=1)

    label_encoders = {}
    for col in sessions_label_encoding:
        label_encoders[col] = LabelEncoder()
        df_sessions[col] = label_encoders[col].fit_transform(df_sessions[col])

    # extract features
    df_users['time_first_active'] = pd.to_datetime(df_users['timestamp_first_active'], format='%Y%m%d%H%M%S')
    df_users['year_first_active'] = df_users['time_first_active'].dt.year
    df_users['quarter_first_active'] = df_users['time_first_active'].dt.quarter
    df_users['month_first_active'] = df_users['time_first_active'].dt.month
    df_users['week_first_active'] = df_users['time_first_active'].dt.week
    df_users['dayofweek_first_active'] = df_users['time_first_active'].dt.dayofweek
    df_users['date_first_active'] = df_users['time_first_active'].dt.date

    # feature aggregation
    agg_dict = {
        'secs_elapsed': ['sum', 'mean'],
        'action': ['nunique', 'count'],
        'device_type': ['nunique', 'count']
    }

    df_agg = df_sessions.groupby('user_id').agg(agg_dict)
    df_agg.columns = [f'sessions_{col[0]}_{col[1]}' for col in df_agg.columns.tolist()]
    df = df_users.merge(df_agg, left_on='id', right_on='user_id', how='inner')

    y = df['country_destination']
    features = [f for f in df_users.columns if f not in ['id', 'user_id', 'country_destination']]
    X = df[features]

    # X_train, X_test, y_train, y_test
    return sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=config.seed)
