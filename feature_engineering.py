
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data
import datetime


df_train = pd.read_csv('train_users_2.csv')

#TODO sessions.csv - web sessions log for users
df_sessions = pd.read_csv('sessions.csv')

# TODO countries.csv - Summary statistics of destination countries
df_countries = pd.read_csv('countries.csv')

#TODO age_gender_bkts.csv - Summary statistics of users' age group, gender, country of destination
df_age = pd.read_csv('age_gender_bkts.csv')

# TODO - Counts

#print("In total we have", df_train.shape[0] , "users.")
#print("We have", df_sessions.shape[0], "Session Records for" , df_sessions.user_id.nunique() , "users." )
#print("We have", (df_train.shape[0] - df_sessions.user_id.nunique()) , "users with no session records." )
#print("We have", (df_countries.shape[0]) , "records in the countries dataset." )
#print("We have", (df_age.shape[0]) , "records in the age/gender dataset." )

#print(df_train.head())
#print(df_train.columns.values)

#print(df_sessions.head())

#print(df_countries)

#print(df_age.head())

#print(df_train['gender'])

# Identifying Outliers (Method 1): Using arbitary values for lower/upper limits.
# age

age_lower_limit = 18
age_upper_limit = 99

# Process outliers (Method 2): Capping Outliers
# age

# replace the outliers by the maximum and minimum limit
df_train['age'] = np.where(df_train['age'] > age_upper_limit, age_upper_limit,
                       np.where(df_train['age'] < age_lower_limit, age_lower_limit, df_train['age']))
df_train['age'].max()

# find_missing() return a table(dataframe) with columns's name , dtype and percentage of missing values for each column.

def find_missing(dataframe):
  cols_na = dataframe.columns[dataframe.isna().any()].tolist()
  dataframe_na = dataframe[cols_na].isnull().mean()
  dataframe_na = pd.DataFrame(dataframe_na.reset_index())
  dataframe_na.columns = ['column', 'NAN_percentage']
  dataframe_na.sort_values(by='NAN_percentage',  ascending=False, inplace=True)
  dataframe_na['dtype'] = [dataframe[col].dtype for col in dataframe_na.column]
  return dataframe_na

#print(find_missing(df_train))

# df_train['age']

median = df_train['age'].median()
df_train['age'].fillna(median, inplace = True)
df_train['age'].isnull().sum()

df_train['first_affiliate_tracked'].fillna('untracked', inplace = True)
df_train['first_affiliate_tracked'].isnull().sum()

#print(find_missing(df_sessions))

df_sessions['action_type'].fillna('-unknown-', inplace = True)
df_sessions['action_type'].isnull().sum()
#print(df_sessions['action_type'].isnull().sum())

df_sessions['action_detail'].fillna('-unknown-', inplace = True)
df_sessions['action_detail'].isnull().sum()
#print(df_sessions['action_detail'].isnull().sum())

#print(f'percentage of rows to be dropped: {round(1-len(df_sessions.dropna())/len(df_sessions),2)}')

# The total percentage of the remaining missing values are less than 5% and we will drop them.
df_sessions = df_sessions.dropna()
df_sessions.isnull().sum()

#print(df_sessions.isnull().sum())

# TODO 2.3 : Categorical variable encoding

train_cols_one_hot_encoding = ['gender', 'signup_method', 'signup_app']
train_cols_label_encoding = ['language', 'affiliate_channel', 'affiliate_provider',
                             'first_affiliate_tracked', 'first_device_type', 'first_browser',
                             'country_destination']
#One hot encoding

for col in train_cols_one_hot_encoding:
  tmp = pd.get_dummies(df_train[col], prefix=col)
  df_train = pd.concat([df_train,tmp],axis=1)

#Label encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in train_cols_label_encoding:
  label_encoders[col] = LabelEncoder()
  df_train[col] = label_encoders[col].fit_transform(df_train[col])

#print(df_train.head())


df_train['time_first_active'] = pd.to_datetime(df_train['timestamp_first_active'], format='%Y%m%d%H%M%S')
#print(df_train['time_first_active'].head())

#df_train['year_first_active'] = df_train['time_first_active'].dt.year
#df_train['quarter_first_active'] = df_train['time_first_active'].dt.quarter
#df_train['month_first_active'] = df_train['time_first_active'].dt.month
#df_train['week_first_active'] = df_train['time_first_active'].dt.week
#df_train['dayofweek_first_active'] = df_train['time_first_active'].dt.dayofweek
#df_train['date_first_active'] = df_train['time_first_active'].dt.date

print(df_train.columns)

#Index(['id', 'date_account_created', 'timestamp_first_active',
    #   'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow',
 #      'language', 'affiliate_channel', 'affiliate_provider',
#       'first_affiliate_tracked', 'signup_app', 'first_device_type',
#       'first_browser', 'country_destination', 'gender_-unknown-',
#       'gender_FEMALE', 'gender_MALE', 'gender_OTHER', 'signup_method_basic',
#       'signup_method_facebook', 'signup_method_google', 'signup_app_Android',
#       'signup_app_Moweb', 'signup_app_Web', 'signup_app_iOS',
 #      'time_first_active'],
 #       dtype='object')

from sklearn.model_selection import train_test_split

# y = df['country_destination'].values

# features = [f for f in df.columns.values if f not in ['country_destination']]
# X = df[features]

X = df_train[['language', 'affiliate_channel', 'affiliate_provider',
       'first_affiliate_tracked' , 'first_device_type',
      'first_browser', 'gender_-unknown-',
       'gender_FEMALE', 'gender_MALE', 'gender_OTHER', 'signup_method_basic',
       'signup_method_facebook', 'signup_method_google', 'signup_app_Android',
      'signup_app_Moweb', 'signup_app_Web', 'signup_app_iOS',
       'quarter_first_active',
      'month_first_active', 'week_first_active', 'dayofweek_first_active']]
#print(x)
#y = df_train['country_destination']

# split 25% of data as test data
##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2020)

# split another 20% as valid data
##X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=2020)

##print(f'X_train:{X_train.shape}\ny_train:{len(y_train)}\n\nX_valid:{X_valid.shape}\ny_valid:{len(y_valid)}\n\nX_test:{X_test.shape}\ny_test:{len(y_test)}')