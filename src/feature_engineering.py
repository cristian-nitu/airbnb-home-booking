from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src import config


def feature_engineering(df_users, df_sessions):
    print("Running Feature Engineering...")
    # drop instances with null values in 'date_first_booking' feature
    df_users.dropna(subset=['date_first_booking'], inplace=True)
    print('     "date_first_booking" with NaN values, dropped!')

    # replace the outliers by the maximum and minimum limit
    age_lower_limit = 18
    age_upper_limit = 99
    df_users['age'] = np.where(df_users['age'] > age_upper_limit, age_upper_limit,
                               np.where(df_users['age'] < age_lower_limit, age_lower_limit, df_users['age']))
    print('     Outliers in "age", replaced!')

    # fill missing values
    age_median = df_users['age'].median()
    df_users['age'].fillna(age_median, inplace=True)
    df_users['first_affiliate_tracked'].fillna('untracked', inplace=True)
    df_sessions['action_type'].fillna('-unknown-', inplace=True)
    df_sessions['action_detail'].fillna('-unknown-', inplace=True)
    df_sessions.dropna(inplace=True)
    print('     Missing values, replaced!')

    # Categorical variable encoding
    users_one_hot = ['gender', 'signup_method', 'signup_app', 'language', 'affiliate_channel',
                     'affiliate_provider', 'first_affiliate_tracked', 'first_device_type', 'first_browser']
    users_label_encoding = ['country_destination']
    sessions_label_encoding = ['action', 'action_type', 'action_detail', 'device_type']

    for col in users_one_hot:
        df_users[col] = df_users[col].apply(lambda x: x.lower())
        tmp = pd.get_dummies(df_users[col], prefix=col)
        df_users = pd.concat([df_users, tmp], axis=1)
    df_users.drop(users_one_hot, axis=1, inplace=True)

    label_encoders = {}
    for col in users_label_encoding:
        df_users[col] = df_users[col].apply(lambda x: x.lower())
        label_encoders[col] = LabelEncoder()
        df_users[col] = label_encoders[col].fit_transform(df_users[col])

    for col in sessions_label_encoding:
        df_sessions[col] = df_sessions[col].apply(lambda x: x.lower())
        label_encoders[col] = LabelEncoder()
        df_sessions[col] = label_encoders[col].fit_transform(df_sessions[col])
    print('     Categorical variables, encoded!')

    # extract features
    df_users['time_first_active'] = pd.to_datetime(df_users['timestamp_first_active'], format='%Y%m%d%H%M%S')
    df_users['year_first_active'] = df_users['time_first_active'].dt.year
    df_users['quarter_first_active'] = df_users['time_first_active'].dt.quarter
    df_users['week_first_active'] = df_users['time_first_active'].dt.week
    df_users['dayofweek_first_active'] = df_users['time_first_active'].dt.dayofweek
    df_users.drop(['time_first_active', 'timestamp_first_active'], axis=1, inplace=True)

    df_users['date_account_created'] = pd.to_datetime(df_users['date_account_created'])
    df_users['year_account_created'] = df_users['date_account_created'].dt.year
    df_users['quarter_account_created'] = df_users['date_account_created'].dt.quarter
    df_users['week_account_created'] = df_users['date_account_created'].dt.week
    df_users['dayofweek_account_created'] = df_users['date_account_created'].dt.dayofweek
    df_users.drop(['date_account_created'], axis=1, inplace=True)

    df_users['date_first_booking'] = pd.to_datetime(df_users['date_first_booking'])
    df_users['year_first_booking'] = df_users['date_first_booking'].dt.year
    df_users['quarter_first_booking'] = df_users['date_first_booking'].dt.quarter
    df_users['week_first_booking'] = df_users['date_first_booking'].dt.week
    df_users['dayofweek_first_booking'] = df_users['date_first_booking'].dt.dayofweek
    df_users.drop(['date_first_booking'], axis=1, inplace=True)

    # feature aggregation
    agg_dict = {
        'secs_elapsed': ['sum', 'mean'],
        'action': ['nunique', 'count'],
        'device_type': ['nunique', 'count'],
        'action_type': ['nunique', 'count'],
        'action_detail': ['nunique', 'count']
    }

    df_agg = df_sessions.groupby('user_id').agg(agg_dict)
    df_agg.columns = [f'sessions_{col[0]}_{col[1]}' for col in df_agg.columns.tolist()]
    df = df_users.merge(df_agg, left_on='id', right_on='user_id', how='inner')
    print('     New features, created!')
    print("Feature Engineering done!")
    print('Two tables got merged!')

    y = df['country_destination']
    features = [f for f in df_users.columns if f not in ['id', 'user_id', 'country_destination']]
    X = df[features]

    print('Number of features:', len(df.columns))

    # X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=0.2, random_state=config.seed)